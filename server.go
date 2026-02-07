package llmock

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	mrand "math/rand/v2"
	"net/http"
	"os"
	"strings"
	"time"
)

var errNoMessages = errors.New("no messages provided")

// Option configures a Server.
type Option func(*Server)

// InternalMessage is the internal representation of a chat message,
// used as the common format between API-specific types.
type InternalMessage struct {
	Role    string
	Content string
}

// Responder generates a response given a conversation.
type Responder interface {
	Respond(messages []InternalMessage) (Response, error)
}

// EchoResponder echoes the last user message (or last message if no user message).
type EchoResponder struct{}

// Respond returns the last user message content, or the last message if there is no user message.
func (e EchoResponder) Respond(messages []InternalMessage) (Response, error) {
	input := extractInput(messages)
	if input == "" {
		return Response{}, errNoMessages
	}
	return Response{Text: input}, nil
}

// Server is a mock LLM API server.
type Server struct {
	mux           *http.ServeMux
	responder     Responder
	tokenDelay    time.Duration
	adminEnabled  *bool
	admin         *adminState
	faults        *faultState
	initialFaults []Fault
	seed          *int64
	corpusText    string
	corpusFile    string
	markov        *MarkovResponder
	autoToolCalls bool
	rng           *mrand.Rand
}

// New creates a new Server with the given options.
func New(opts ...Option) *Server {
	s := &Server{}
	for _, opt := range opts {
		opt(s)
	}
	// Build the Markov responder.
	s.markov = NewMarkovResponder(s.seed)
	if s.corpusFile != "" {
		data, err := os.ReadFile(s.corpusFile)
		if err == nil {
			mc := NewMarkovChain(2)
			mc.Train(string(data))
			s.markov.chain = mc
		}
	} else if s.corpusText != "" {
		mc := NewMarkovChain(2)
		mc.Train(s.corpusText)
		s.markov.chain = mc
	}

	if s.responder == nil {
		s.responder = EchoResponder{}
	}

	// If the responder is a RuleResponder, set its markov fallback.
	if rr, ok := s.responder.(*RuleResponder); ok {
		rr.markov = s.markov
	}

	// Initialize RNG and fault state.
	var rng *mrand.Rand
	if s.seed != nil {
		rng = mrand.New(mrand.NewPCG(uint64(*s.seed), 0))
	} else {
		rng = mrand.New(mrand.NewPCG(mrand.Uint64(), mrand.Uint64()))
	}
	s.rng = rng
	s.faults = newFaultState(s.initialFaults, rng)

	// Admin API is enabled by default.
	adminOn := s.adminEnabled == nil || *s.adminEnabled
	if adminOn {
		// Extract initial rules from the responder if it's a RuleResponder.
		var rules []Rule
		if rr, ok := s.responder.(*RuleResponder); ok {
			rules = rr.rules
		}
		s.admin = newAdminState(rules, s.markov)
		// Wrap the responder: admin rules are tried first, then fallback
		// to the original responder.
		s.responder = &adminResponder{state: s.admin, fallback: s.responder}
	}

	s.mux = http.NewServeMux()
	s.mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	s.mux.HandleFunc("POST /v1/messages", s.handleMessages)

	if adminOn {
		registerAdminRoutes(s.mux, s.admin)
		registerFaultRoutes(s.mux, s.faults)
	}

	return s
}

// WithAdminAPI enables or disables the /_mock/ admin endpoints.
// The admin API is enabled by default.
func WithAdminAPI(enabled bool) Option {
	return func(s *Server) {
		s.adminEnabled = &enabled
	}
}

// WithAutoToolCalls enables auto-generation of tool calls. When enabled and
// a request includes tool definitions but no rule produces a tool call, the
// server picks a random tool and generates arguments from its JSON schema.
func WithAutoToolCalls(enabled bool) Option {
	return func(s *Server) {
		s.autoToolCalls = enabled
	}
}

// Handler returns the http.Handler for this server.
func (s *Server) Handler() http.Handler {
	return s.mux
}

// ChatCompletionRequest represents an OpenAI chat completion request.
type ChatCompletionRequest struct {
	Model       string           `json:"model"`
	Messages    []Message        `json:"messages"`
	Stream      bool             `json:"stream,omitempty"`
	Temperature *float64         `json:"temperature,omitempty"`
	MaxTokens   *int             `json:"max_tokens,omitempty"`
	Tools       []OpenAIToolDef  `json:"tools,omitempty"`
}

// OpenAIToolDef represents a tool definition in an OpenAI request.
type OpenAIToolDef struct {
	Type     string              `json:"type"`
	Function OpenAIFunctionDef   `json:"function"`
}

// OpenAIFunctionDef describes a function tool in an OpenAI request.
type OpenAIFunctionDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// Message represents a chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionResponse represents an OpenAI chat completion response.
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// ChoiceMessage represents the message in a response choice, which may
// contain either text content or tool calls.
type ChoiceMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content,omitempty"`
	ToolCalls []OpenAIToolCall `json:"tool_calls,omitempty"`
}

// OpenAIToolCall represents a tool call in an OpenAI response.
type OpenAIToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function OpenAIFunctionCall `json:"function"`
}

// OpenAIFunctionCall represents the function details in a tool call.
type OpenAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON-encoded arguments
}

// Choice represents a response choice.
type Choice struct {
	Index        int          `json:"index"`
	Message      ChoiceMessage `json:"message"`
	FinishReason string       `json:"finish_reason"`
}

// Usage represents token usage statistics.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func toInternalMessages(messages []Message) []InternalMessage {
	internal := make([]InternalMessage, len(messages))
	for i, m := range messages {
		internal[i] = InternalMessage{Role: m.Role, Content: m.Content}
	}
	return internal
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}

	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages array is required and must not be empty")
		return
	}

	// Evaluate faults before normal processing.
	if f, ok := s.faults.evaluate(); ok {
		if s.executeFault(w, r, f, "openai", req.Stream) {
			return
		}
	}

	internal := toInternalMessages(req.Messages)
	response, err := s.responder.Respond(internal)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	// Auto-generate a tool call if enabled and no rule produced one.
	if s.autoToolCalls && !response.IsToolCall() && len(req.Tools) > 0 {
		reqTools := openAIToRequestTools(req.Tools)
		if tc, ok := generateToolCallFromSchema(reqTools, s.rng); ok {
			response = Response{ToolCalls: []ToolCall{tc}}
		}
	}

	s.logAdminRequest(r, internal, response.Text)

	model := req.Model
	if model == "" {
		model = "llmock-1"
	}

	id := fmt.Sprintf("chatcmpl-mock-%d", time.Now().UnixNano())

	if response.IsToolCall() {
		// Tool call response: check that requested tools contain the called tool.
		if len(req.Tools) > 0 {
			toolNames := make(map[string]bool)
			for _, t := range req.Tools {
				if t.Function.Name != "" {
					toolNames[t.Function.Name] = true
				}
			}
			var validCalls []ToolCall
			for _, tc := range response.ToolCalls {
				if toolNames[tc.Name] {
					validCalls = append(validCalls, tc)
				}
			}
			if len(validCalls) == 0 {
				// No valid tool calls; fall through to text response.
				goto textResponse
			}
			response.ToolCalls = validCalls
		}

		promptTokens := estimateTokens(req.Messages)
		completionTokens := 5 // rough estimate for tool call tokens

		if req.Stream {
			s.streamOpenAIToolCall(w, r, response.ToolCalls, model, id)
			return
		}

		toolCalls := make([]OpenAIToolCall, len(response.ToolCalls))
		for i, tc := range response.ToolCalls {
			toolCalls[i] = openAIToolCallFromInternal(tc)
		}

		resp := ChatCompletionResponse{
			ID:      id,
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   model,
			Choices: []Choice{
				{
					Index: 0,
					Message: ChoiceMessage{
						Role:      "assistant",
						ToolCalls: toolCalls,
					},
					FinishReason: "tool_calls",
				},
			},
			Usage: Usage{
				PromptTokens:     promptTokens,
				CompletionTokens: completionTokens,
				TotalTokens:      promptTokens + completionTokens,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
		return
	}

textResponse:
	responseText := response.Text
	promptTokens := estimateTokens(req.Messages)
	completionTokens := countTokens(responseText)

	if req.Stream {
		s.streamOpenAI(w, r, responseText, model, id)
		return
	}

	resp := ChatCompletionResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []Choice{
			{
				Index: 0,
				Message: ChoiceMessage{
					Role:    "assistant",
					Content: responseText,
				},
				FinishReason: "stop",
			},
		},
		Usage: Usage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// AnthropicRequest represents an Anthropic Messages API request.
type AnthropicRequest struct {
	Model     string               `json:"model"`
	Messages  []AnthropicMessage   `json:"messages"`
	MaxTokens int                  `json:"max_tokens"`
	Stream    bool                 `json:"stream,omitempty"`
	Tools     []AnthropicToolDef   `json:"tools,omitempty"`
}

// AnthropicToolDef represents a tool definition in an Anthropic request.
type AnthropicToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema,omitempty"`
}

// AnthropicMessage represents a message in the Anthropic format.
type AnthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// AnthropicResponse represents an Anthropic Messages API response.
type AnthropicResponse struct {
	ID           string                 `json:"id"`
	Type         string                 `json:"type"`
	Role         string                 `json:"role"`
	Content      []AnthropicContentBlock `json:"content"`
	Model        string                 `json:"model"`
	StopReason   string                 `json:"stop_reason"`
	StopSequence *string                `json:"stop_sequence"`
	Usage        AnthropicUsage         `json:"usage"`
}

// AnthropicContentBlock represents a content block in an Anthropic response.
// For text blocks: Type="text", Text is set.
// For tool_use blocks: Type="tool_use", ID/Name/Input are set.
type AnthropicContentBlock struct {
	Type  string         `json:"type"`
	Text  string         `json:"text,omitempty"`
	ID    string         `json:"id,omitempty"`
	Name  string         `json:"name,omitempty"`
	Input map[string]any `json:"input,omitempty"`
}

// AnthropicUsage represents token usage in an Anthropic response.
type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

func randomHex(n int) string {
	b := make([]byte, n)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func anthropicToInternal(messages []AnthropicMessage) []InternalMessage {
	internal := make([]InternalMessage, len(messages))
	for i, m := range messages {
		internal[i] = InternalMessage{Role: m.Role, Content: m.Content}
	}
	return internal
}

func estimateAnthropicTokens(messages []AnthropicMessage) int {
	total := 0
	for _, m := range messages {
		total += countTokens(m.Content)
		total += 4
	}
	return total
}

func (s *Server) handleMessages(w http.ResponseWriter, r *http.Request) {
	var req AnthropicRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}

	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages array is required and must not be empty")
		return
	}

	// Evaluate faults before normal processing.
	if f, ok := s.faults.evaluate(); ok {
		if s.executeFault(w, r, f, "anthropic", req.Stream) {
			return
		}
	}

	internal := anthropicToInternal(req.Messages)
	response, err := s.responder.Respond(internal)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	// Auto-generate a tool call if enabled and no rule produced one.
	if s.autoToolCalls && !response.IsToolCall() && len(req.Tools) > 0 {
		reqTools := anthropicToRequestTools(req.Tools)
		if tc, ok := generateToolCallFromSchema(reqTools, s.rng); ok {
			response = Response{ToolCalls: []ToolCall{tc}}
		}
	}

	s.logAdminRequest(r, internal, response.Text)

	model := req.Model
	if model == "" {
		model = "llmock-1"
	}

	id := fmt.Sprintf("msg_%s", randomHex(12))

	if response.IsToolCall() {
		// Validate tool calls against request tools.
		if len(req.Tools) > 0 {
			toolNames := make(map[string]bool)
			for _, t := range req.Tools {
				if t.Name != "" {
					toolNames[t.Name] = true
				}
			}
			var validCalls []ToolCall
			for _, tc := range response.ToolCalls {
				if toolNames[tc.Name] {
					validCalls = append(validCalls, tc)
				}
			}
			if len(validCalls) == 0 {
				goto anthropicTextResponse
			}
			response.ToolCalls = validCalls
		}

		inputTokens := estimateAnthropicTokens(req.Messages)
		outputTokens := 5

		if req.Stream {
			s.streamAnthropicToolCall(w, r, response.ToolCalls, model, id, inputTokens)
			return
		}

		content := make([]AnthropicContentBlock, len(response.ToolCalls))
		for i, tc := range response.ToolCalls {
			// Use Anthropic-style ID
			tcID := generateToolCallID("toolu_")
			content[i] = AnthropicContentBlock{
				Type:  "tool_use",
				ID:    tcID,
				Name:  tc.Name,
				Input: tc.Arguments,
			}
		}

		resp := AnthropicResponse{
			ID:         id,
			Type:       "message",
			Role:       "assistant",
			Content:    content,
			Model:      model,
			StopReason: "tool_use",
			Usage:      AnthropicUsage{InputTokens: inputTokens, OutputTokens: outputTokens},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
		return
	}

anthropicTextResponse:
	responseText := response.Text
	inputTokens := estimateAnthropicTokens(req.Messages)
	outputTokens := countTokens(responseText)

	if req.Stream {
		s.streamAnthropic(w, r, responseText, model, id, inputTokens)
		return
	}

	resp := AnthropicResponse{
		ID:         id,
		Type:       "message",
		Role:       "assistant",
		Content:    []AnthropicContentBlock{{Type: "text", Text: responseText}},
		Model:      model,
		StopReason: "end_turn",
		Usage:      AnthropicUsage{InputTokens: inputTokens, OutputTokens: outputTokens},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func estimateTokens(messages []Message) int {
	total := 0
	for _, m := range messages {
		total += countTokens(m.Content)
		total += 4 // overhead per message (role, separators)
	}
	return total
}

func countTokens(s string) int {
	words := len(strings.Fields(s))
	// Rough approximation: ~1.3 tokens per word.
	tokens := int(float64(words) * 1.3)
	if tokens == 0 && len(s) > 0 {
		tokens = 1
	}
	return tokens
}

type errorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

// logAdminRequest records a request in the admin log if admin is enabled.
func (s *Server) logAdminRequest(r *http.Request, messages []InternalMessage, responseText string) {
	if s.admin == nil {
		return
	}
	matchedRule := ""
	if ar, ok := s.responder.(*adminResponder); ok {
		matchedRule = ar.getLastMatchedRule()
	}
	s.admin.logRequest(requestEntry{
		Timestamp:   time.Now(),
		Method:      r.Method,
		Path:        r.URL.Path,
		UserMessage: extractInput(messages),
		MatchedRule: matchedRule,
		Response:    responseText,
	})
}

// extractInput finds the last user message, or falls back to the last message.
func extractInput(messages []InternalMessage) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return messages[i].Content
		}
	}
	if len(messages) > 0 {
		return messages[len(messages)-1].Content
	}
	return ""
}

// openAIToolCallFromInternal converts an internal ToolCall to the OpenAI format.
func openAIToolCallFromInternal(tc ToolCall) OpenAIToolCall {
	argsJSON, _ := json.Marshal(tc.Arguments)
	return OpenAIToolCall{
		ID:   tc.ID,
		Type: "function",
		Function: OpenAIFunctionCall{
			Name:      tc.Name,
			Arguments: string(argsJSON),
		},
	}
}

// openAIToRequestTools converts OpenAI tool definitions to internal RequestTool format.
func openAIToRequestTools(tools []OpenAIToolDef) []RequestTool {
	out := make([]RequestTool, 0, len(tools))
	for _, t := range tools {
		out = append(out, RequestTool{
			Name:       t.Function.Name,
			Parameters: t.Function.Parameters,
		})
	}
	return out
}

// anthropicToRequestTools converts Anthropic tool definitions to internal RequestTool format.
func anthropicToRequestTools(tools []AnthropicToolDef) []RequestTool {
	out := make([]RequestTool, 0, len(tools))
	for _, t := range tools {
		out = append(out, RequestTool{
			Name:       t.Name,
			Parameters: t.InputSchema,
		})
	}
	return out
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	resp := errorResponse{}
	resp.Error.Message = msg
	resp.Error.Type = "invalid_request_error"
	json.NewEncoder(w).Encode(resp)
}
