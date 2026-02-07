package llmock

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// Option configures a Server.
type Option func(*Server)

// Server is a mock LLM API server.
type Server struct {
	mux *http.ServeMux
}

// New creates a new Server with the given options.
func New(opts ...Option) *Server {
	s := &Server{}
	for _, opt := range opts {
		opt(s)
	}
	s.mux = http.NewServeMux()
	s.mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
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

	// Find the last user message to echo back.
	var lastUserMsg string
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "user" {
			lastUserMsg = req.Messages[i].Content
			break
		}
	}
	if lastUserMsg == "" {
		lastUserMsg = req.Messages[len(req.Messages)-1].Content
	}

	// Estimate token counts based on rough word counts (~0.75 words per token).
	promptTokens := estimateTokens(req.Messages)
	completionTokens := countTokens(lastUserMsg)

	model := req.Model
	if model == "" {
		model = "llmock-1"
	}

	resp := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-mock-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []Choice{
			{
				Index:        0,
				Message:      Message{Role: "assistant", Content: lastUserMsg},
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
