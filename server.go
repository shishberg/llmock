package llmock

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

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
	Respond(messages []InternalMessage) (string, error)
}

// EchoResponder echoes the last user message (or last message if no user message).
type EchoResponder struct{}

// Respond returns the last user message content, or the last message if there is no user message.
func (e EchoResponder) Respond(messages []InternalMessage) (string, error) {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return messages[i].Content, nil
		}
	}
	if len(messages) > 0 {
		return messages[len(messages)-1].Content, nil
	}
	return "", fmt.Errorf("no messages provided")
}

// Server is a mock LLM API server.
type Server struct {
	mux        *http.ServeMux
	responder  Responder
	tokenDelay time.Duration
}

// New creates a new Server with the given options.
func New(opts ...Option) *Server {
	s := &Server{}
	for _, opt := range opts {
		opt(s)
	}
	if s.responder == nil {
		s.responder = EchoResponder{}
	}
	s.mux = http.NewServeMux()
	s.mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	s.mux.HandleFunc("POST /v1/messages", s.handleMessages)
	return s
}

// Handler returns the http.Handler for this server.
func (s *Server) Handler() http.Handler {
	return s.mux
}

// ChatCompletionRequest represents an OpenAI chat completion request.
type ChatCompletionRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Stream      bool      `json:"stream,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	MaxTokens   *int      `json:"max_tokens,omitempty"`
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

// Choice represents a response choice.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
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

	responseText, err := s.responder.Respond(toInternalMessages(req.Messages))
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	promptTokens := estimateTokens(req.Messages)
	completionTokens := countTokens(responseText)

	model := req.Model
	if model == "" {
		model = "llmock-1"
	}

	id := fmt.Sprintf("chatcmpl-mock-%d", time.Now().UnixNano())

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
				Index:        0,
				Message:      Message{Role: "assistant", Content: responseText},
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
	Model     string             `json:"model"`
	Messages  []AnthropicMessage `json:"messages"`
	MaxTokens int                `json:"max_tokens"`
	Stream    bool               `json:"stream,omitempty"`
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
type AnthropicContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
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

	responseText, err := s.responder.Respond(anthropicToInternal(req.Messages))
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	inputTokens := estimateAnthropicTokens(req.Messages)
	outputTokens := countTokens(responseText)

	model := req.Model
	if model == "" {
		model = "llmock-1"
	}

	id := fmt.Sprintf("msg_%s", randomHex(12))

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

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	resp := errorResponse{}
	resp.Error.Message = msg
	resp.Error.Type = "invalid_request_error"
	json.NewEncoder(w).Encode(resp)
}
