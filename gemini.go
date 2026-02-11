package llmock

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// GeminiRequest represents a Google Gemini generateContent request.
type GeminiRequest struct {
	Contents          []GeminiContent          `json:"contents"`
	SystemInstruction *GeminiContent           `json:"systemInstruction,omitempty"`
	GenerationConfig  *GeminiGenerationConfig  `json:"generationConfig,omitempty"`
	Tools             []GeminiToolDef          `json:"tools,omitempty"`
}

// GeminiContent represents a content entry with a role and parts.
type GeminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []GeminiPart `json:"parts"`
}

// GeminiPart represents a part of a content entry.
// A part contains either text, a function call, or a function response.
type GeminiPart struct {
	Text             string                  `json:"text,omitempty"`
	FunctionCall     *GeminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *GeminiFunctionResponse `json:"functionResponse,omitempty"`
}

// GeminiFunctionCall represents a function call in a Gemini response part.
type GeminiFunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

// GeminiFunctionResponse represents a function result in a Gemini request part.
type GeminiFunctionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

// GeminiGenerationConfig holds generation parameters.
type GeminiGenerationConfig struct {
	Temperature     *float64 `json:"temperature,omitempty"`
	MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
	TopP            *float64 `json:"topP,omitempty"`
	TopK            *int     `json:"topK,omitempty"`
}

// GeminiToolDef represents a tool definition in a Gemini request.
type GeminiToolDef struct {
	FunctionDeclarations []GeminiFunctionDecl `json:"functionDeclarations,omitempty"`
}

// GeminiFunctionDecl describes a function tool in a Gemini request.
type GeminiFunctionDecl struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// GeminiResponse represents a Gemini generateContent response.
type GeminiResponse struct {
	Candidates    []GeminiCandidate    `json:"candidates"`
	UsageMetadata GeminiUsageMetadata  `json:"usageMetadata"`
}

// GeminiCandidate represents a candidate in a Gemini response.
type GeminiCandidate struct {
	Content       GeminiContent `json:"content"`
	FinishReason  string        `json:"finishReason,omitempty"`
}

// GeminiUsageMetadata represents token usage in a Gemini response.
type GeminiUsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

func geminiToInternal(contents []GeminiContent, sysInstruction *GeminiContent) []InternalMessage {
	internal := make([]InternalMessage, 0, len(contents)+1)

	// Add system instruction if present.
	if sysInstruction != nil {
		text := geminiContentText(*sysInstruction)
		if text != "" {
			internal = append(internal, InternalMessage{Role: "system", Content: text})
		}
	}

	for _, c := range contents {
		// Map Gemini roles to internal roles.
		role := c.Role
		if role == "model" {
			role = "assistant"
		}

		text := geminiContentText(c)
		// Skip model messages that only contain function calls (no text).
		if role == "assistant" && text == "" && hasFunctionCall(c) {
			continue
		}
		if text != "" {
			internal = append(internal, InternalMessage{Role: role, Content: text})
		}
	}
	return internal
}

// geminiContentText extracts text from a GeminiContent, concatenating text parts
// and function response content.
func geminiContentText(c GeminiContent) string {
	var parts []string
	for _, p := range c.Parts {
		if p.Text != "" {
			parts = append(parts, p.Text)
		}
		if p.FunctionResponse != nil {
			// Include function response as text for rule matching.
			if result, ok := p.FunctionResponse.Response["result"]; ok {
				if s, ok := result.(string); ok {
					parts = append(parts, s)
				}
			}
		}
	}
	return strings.Join(parts, "\n")
}

func hasFunctionCall(c GeminiContent) bool {
	for _, p := range c.Parts {
		if p.FunctionCall != nil {
			return true
		}
	}
	return false
}

func estimateGeminiTokens(contents []GeminiContent) int {
	total := 0
	for _, c := range contents {
		total += countTokens(geminiContentText(c))
		total += 4 // overhead per content entry
	}
	return total
}

// geminiToRequestTools extracts tool definitions from Gemini tool definitions.
func geminiToRequestTools(tools []GeminiToolDef) []RequestTool {
	var out []RequestTool
	for _, t := range tools {
		for _, fd := range t.FunctionDeclarations {
			out = append(out, RequestTool{
				Name:       fd.Name,
				Parameters: fd.Parameters,
			})
		}
	}
	return out
}

// handleGeminiRoute dispatches Gemini API requests based on the method suffix.
func (s *Server) handleGeminiRoute(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	switch {
	case strings.HasSuffix(path, ":generateContent"):
		s.handleGeminiGenerate(w, r)
	case strings.HasSuffix(path, ":streamGenerateContent"):
		s.handleGeminiStream(w, r)
	default:
		writeGeminiError(w, http.StatusNotFound, "unknown Gemini method")
	}
}

func (s *Server) handleGeminiGenerate(w http.ResponseWriter, r *http.Request) {
	// Extract model from path: /v1beta/models/{model}:generateContent
	model := extractGeminiModel(r.URL.Path)

	var req GeminiRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeGeminiError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}

	if len(req.Contents) == 0 {
		writeGeminiError(w, http.StatusBadRequest, "contents array is required and must not be empty")
		return
	}

	// Evaluate faults before normal processing.
	if f, ok := s.faults.evaluate(); ok {
		if s.executeFault(w, r, f, "gemini", false) {
			return
		}
	}

	internal := geminiToInternal(req.Contents, req.SystemInstruction)
	response, err := s.responder.Respond(internal)
	if err != nil {
		writeGeminiError(w, http.StatusBadRequest, err.Error())
		return
	}

	// If the conversation contains tool results, suppress tool call responses
	// to avoid infinite tool-call loops.
	hasToolResults := geminiHasToolResults(req.Contents)

	// Auto-generate a tool call if enabled and no rule produced one.
	if !hasToolResults && s.autoToolCalls && !response.IsToolCall() && len(req.Tools) > 0 {
		reqTools := geminiToRequestTools(req.Tools)
		if tc, ok := generateToolCallFromSchema(reqTools, s.rng); ok {
			response = Response{ToolCalls: []ToolCall{tc}}
		}
	}

	// Force text response when tool results are present.
	if hasToolResults && response.IsToolCall() {
		response = s.forceTextResponse(response, internal)
	}

	s.logAdminRequest(r, internal, response.Text)

	if model == "" {
		model = "llmock-1"
	}

	if response.IsToolCall() {
		// Validate tool calls against request tools.
		if len(req.Tools) > 0 {
			toolNames := make(map[string]bool)
			for _, t := range req.Tools {
				for _, fd := range t.FunctionDeclarations {
					toolNames[fd.Name] = true
				}
			}
			var validCalls []ToolCall
			for _, tc := range response.ToolCalls {
				if toolNames[tc.Name] {
					validCalls = append(validCalls, tc)
				}
			}
			if len(validCalls) == 0 {
				goto geminiTextResponse
			}
			response.ToolCalls = validCalls
		}

		promptTokens := estimateGeminiTokens(req.Contents)
		completionTokens := 5

		parts := make([]GeminiPart, len(response.ToolCalls))
		for i, tc := range response.ToolCalls {
			parts[i] = GeminiPart{
				FunctionCall: &GeminiFunctionCall{
					Name: tc.Name,
					Args: tc.Arguments,
				},
			}
		}

		resp := GeminiResponse{
			Candidates: []GeminiCandidate{
				{
					Content:      GeminiContent{Role: "model", Parts: parts},
					FinishReason: "STOP",
				},
			},
			UsageMetadata: GeminiUsageMetadata{
				PromptTokenCount:     promptTokens,
				CandidatesTokenCount: completionTokens,
				TotalTokenCount:      promptTokens + completionTokens,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
		return
	}

geminiTextResponse:
	responseText := response.Text
	promptTokens := estimateGeminiTokens(req.Contents)
	completionTokens := countTokens(responseText)

	resp := GeminiResponse{
		Candidates: []GeminiCandidate{
			{
				Content: GeminiContent{
					Role:  "model",
					Parts: []GeminiPart{{Text: responseText}},
				},
				FinishReason: "STOP",
			},
		},
		UsageMetadata: GeminiUsageMetadata{
			PromptTokenCount:     promptTokens,
			CandidatesTokenCount: completionTokens,
			TotalTokenCount:      promptTokens + completionTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleGeminiStream(w http.ResponseWriter, r *http.Request) {
	// Extract model from path: /v1beta/models/{model}:streamGenerateContent
	model := extractGeminiModel(r.URL.Path)

	var req GeminiRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeGeminiError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}

	if len(req.Contents) == 0 {
		writeGeminiError(w, http.StatusBadRequest, "contents array is required and must not be empty")
		return
	}

	// Evaluate faults before normal processing.
	if f, ok := s.faults.evaluate(); ok {
		if s.executeFault(w, r, f, "gemini", true) {
			return
		}
	}

	internal := geminiToInternal(req.Contents, req.SystemInstruction)
	response, err := s.responder.Respond(internal)
	if err != nil {
		writeGeminiError(w, http.StatusBadRequest, err.Error())
		return
	}

	// If the conversation contains tool results, suppress tool call responses
	// to avoid infinite tool-call loops.
	hasToolResults := geminiHasToolResults(req.Contents)

	// Auto-generate a tool call if enabled and no rule produced one.
	if !hasToolResults && s.autoToolCalls && !response.IsToolCall() && len(req.Tools) > 0 {
		reqTools := geminiToRequestTools(req.Tools)
		if tc, ok := generateToolCallFromSchema(reqTools, s.rng); ok {
			response = Response{ToolCalls: []ToolCall{tc}}
		}
	}

	// Force text response when tool results are present.
	if hasToolResults && response.IsToolCall() {
		response = s.forceTextResponse(response, internal)
	}

	s.logAdminRequest(r, internal, response.Text)

	if model == "" {
		model = "llmock-1"
	}

	promptTokens := estimateGeminiTokens(req.Contents)

	if response.IsToolCall() {
		// For tool calls, stream as a single chunk.
		s.streamGeminiToolCall(w, r, response.ToolCalls, promptTokens)
		return
	}

	s.streamGemini(w, r, response.Text, promptTokens)
}

// streamGemini writes the response as Gemini-format SSE chunks.
func (s *Server) streamGemini(w http.ResponseWriter, r *http.Request, responseText string, promptTokens int) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeGeminiError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	chunks := tokenize(responseText)
	outputTokens := countTokens(responseText)

	for i, chunk := range chunks {
		candidate := GeminiCandidate{
			Content: GeminiContent{
				Role:  "model",
				Parts: []GeminiPart{{Text: chunk}},
			},
		}

		// Last chunk gets finish reason and usage.
		resp := GeminiResponse{
			Candidates: []GeminiCandidate{candidate},
		}

		if i == len(chunks)-1 {
			resp.Candidates[0].FinishReason = "STOP"
			resp.UsageMetadata = GeminiUsageMetadata{
				PromptTokenCount:     promptTokens,
				CandidatesTokenCount: outputTokens,
				TotalTokenCount:      promptTokens + outputTokens,
			}
		}

		data, _ := json.Marshal(resp)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()

		if i < len(chunks)-1 {
			select {
			case <-r.Context().Done():
				return
			case <-time.After(s.getTokenDelay()):
			}
		}
	}
}

// streamGeminiToolCall streams a tool call response in Gemini SSE format.
func (s *Server) streamGeminiToolCall(w http.ResponseWriter, r *http.Request, toolCalls []ToolCall, promptTokens int) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeGeminiError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	parts := make([]GeminiPart, len(toolCalls))
	for i, tc := range toolCalls {
		parts[i] = GeminiPart{
			FunctionCall: &GeminiFunctionCall{
				Name: tc.Name,
				Args: tc.Arguments,
			},
		}
	}

	resp := GeminiResponse{
		Candidates: []GeminiCandidate{
			{
				Content:      GeminiContent{Role: "model", Parts: parts},
				FinishReason: "STOP",
			},
		},
		UsageMetadata: GeminiUsageMetadata{
			PromptTokenCount:     promptTokens,
			CandidatesTokenCount: 5,
			TotalTokenCount:      promptTokens + 5,
		},
	}

	data, _ := json.Marshal(resp)
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

// extractGeminiModel extracts the model name from Gemini API paths like
// /v1beta/models/{model}:generateContent or /v1beta/models/{model}:streamGenerateContent
func extractGeminiModel(path string) string {
	// Remove the method suffix.
	path = strings.TrimSuffix(path, ":generateContent")
	path = strings.TrimSuffix(path, ":streamGenerateContent")
	// Extract model name after /v1beta/models/
	const prefix = "/v1beta/models/"
	if strings.HasPrefix(path, prefix) {
		return path[len(prefix):]
	}
	return ""
}

// geminiHasToolResults returns true if any content contains a functionResponse part.
func geminiHasToolResults(contents []GeminiContent) bool {
	for _, c := range contents {
		for _, p := range c.Parts {
			if p.FunctionResponse != nil {
				return true
			}
		}
	}
	return false
}

func writeGeminiError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"code":    code,
			"message": msg,
			"status":  http.StatusText(code),
		},
	})
}
