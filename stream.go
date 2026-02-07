package llmock

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"net/http"
	"strings"
	"time"
)

// WithTokenDelay sets the delay between streamed tokens.
// Default is 15ms.
func WithTokenDelay(d time.Duration) Option {
	return func(s *Server) {
		s.tokenDelay = d
	}
}

// tokenize splits text into chunks of 1-3 words to simulate token-by-token streaming.
func tokenize(text string) []string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}
	var chunks []string
	i := 0
	for i < len(words) {
		n := rand.IntN(3) + 1 // 1-3 words per chunk
		if i+n > len(words) {
			n = len(words) - i
		}
		chunk := strings.Join(words[i:i+n], " ")
		chunks = append(chunks, chunk)
		i += n
	}
	// Add a leading space to all chunks except the first, to preserve word boundaries.
	for i := 1; i < len(chunks); i++ {
		chunks[i] = " " + chunks[i]
	}
	return chunks
}

// streamOpenAI writes the response as OpenAI-format SSE chunks.
func (s *Server) streamOpenAI(w http.ResponseWriter, r *http.Request, responseText, model, id string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	chunks := tokenize(responseText)
	created := time.Now().Unix()

	for i, chunk := range chunks {
		delta := map[string]any{}
		if i == 0 {
			delta["role"] = "assistant"
		}
		delta["content"] = chunk

		event := map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []map[string]any{
				{
					"index":         0,
					"delta":         delta,
					"finish_reason": nil,
				},
			},
		}
		data, _ := json.Marshal(event)
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

	// Final chunk with finish_reason
	finalEvent := map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]any{
			{
				"index":         0,
				"delta":         map[string]any{},
				"finish_reason": "stop",
			},
		},
	}
	data, _ := json.Marshal(finalEvent)
	fmt.Fprintf(w, "data: %s\n\n", data)
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// streamAnthropic writes the response as Anthropic-format SSE events.
func (s *Server) streamAnthropic(w http.ResponseWriter, r *http.Request, responseText, model, id string, inputTokens int) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	outputTokens := countTokens(responseText)

	// message_start
	msgStart := map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            id,
			"type":          "message",
			"role":          "assistant",
			"content":       []any{},
			"model":         model,
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]any{
				"input_tokens":  inputTokens,
				"output_tokens": 0,
			},
		},
	}
	writeSSE(w, "message_start", msgStart)
	flusher.Flush()

	// content_block_start
	blockStart := map[string]any{
		"type":          "content_block_start",
		"index":         0,
		"content_block": map[string]any{"type": "text", "text": ""},
	}
	writeSSE(w, "content_block_start", blockStart)
	flusher.Flush()

	// content_block_delta events
	chunks := tokenize(responseText)
	for i, chunk := range chunks {
		delta := map[string]any{
			"type":  "content_block_delta",
			"index": 0,
			"delta": map[string]any{
				"type": "text_delta",
				"text": chunk,
			},
		}
		writeSSE(w, "content_block_delta", delta)
		flusher.Flush()

		if i < len(chunks)-1 {
			select {
			case <-r.Context().Done():
				return
			case <-time.After(s.getTokenDelay()):
			}
		}
	}

	// content_block_stop
	blockStop := map[string]any{
		"type":  "content_block_stop",
		"index": 0,
	}
	writeSSE(w, "content_block_stop", blockStop)
	flusher.Flush()

	// message_delta
	msgDelta := map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   "end_turn",
			"stop_sequence": nil,
		},
		"usage": map[string]any{
			"output_tokens": outputTokens,
		},
	}
	writeSSE(w, "message_delta", msgDelta)
	flusher.Flush()

	// message_stop
	msgStop := map[string]any{
		"type": "message_stop",
	}
	writeSSE(w, "message_stop", msgStop)
	flusher.Flush()
}

func writeSSE(w http.ResponseWriter, event string, data any) {
	b, _ := json.Marshal(data)
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
}

// streamOpenAIToolCall streams a tool call response in OpenAI format.
func (s *Server) streamOpenAIToolCall(w http.ResponseWriter, r *http.Request, toolCalls []ToolCall, model, id string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	created := time.Now().Unix()

	for i, tc := range toolCalls {
		argsJSON, _ := json.Marshal(tc.Arguments)
		argsStr := string(argsJSON)

		// First chunk: includes role and tool call start with function name.
		delta := map[string]any{
			"tool_calls": []map[string]any{
				{
					"index": i,
					"id":    tc.ID,
					"type":  "function",
					"function": map[string]any{
						"name":      tc.Name,
						"arguments": "",
					},
				},
			},
		}
		if i == 0 {
			delta["role"] = "assistant"
		}

		event := map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []map[string]any{
				{
					"index":         0,
					"delta":         delta,
					"finish_reason": nil,
				},
			},
		}
		data, _ := json.Marshal(event)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()

		// Stream the arguments in chunks.
		chunks := splitString(argsStr, 20)
		for _, chunk := range chunks {
			argDelta := map[string]any{
				"tool_calls": []map[string]any{
					{
						"index": i,
						"function": map[string]any{
							"arguments": chunk,
						},
					},
				},
			}
			argEvent := map[string]any{
				"id":      id,
				"object":  "chat.completion.chunk",
				"created": created,
				"model":   model,
				"choices": []map[string]any{
					{
						"index":         0,
						"delta":         argDelta,
						"finish_reason": nil,
					},
				},
			}
			data, _ := json.Marshal(argEvent)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()

			select {
			case <-r.Context().Done():
				return
			case <-time.After(s.getTokenDelay()):
			}
		}
	}

	// Final chunk with finish_reason.
	finalEvent := map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]any{
			{
				"index":         0,
				"delta":         map[string]any{},
				"finish_reason": "tool_calls",
			},
		},
	}
	data, _ := json.Marshal(finalEvent)
	fmt.Fprintf(w, "data: %s\n\n", data)
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// streamAnthropicToolCall streams a tool call response in Anthropic format.
func (s *Server) streamAnthropicToolCall(w http.ResponseWriter, r *http.Request, toolCalls []ToolCall, model, id string, inputTokens int) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	// message_start
	msgStart := map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            id,
			"type":          "message",
			"role":          "assistant",
			"content":       []any{},
			"model":         model,
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]any{
				"input_tokens":  inputTokens,
				"output_tokens": 0,
			},
		},
	}
	writeSSE(w, "message_start", msgStart)
	flusher.Flush()

	for i, tc := range toolCalls {
		tcID := generateToolCallID("toolu_")

		// content_block_start for tool_use
		blockStart := map[string]any{
			"type":  "content_block_start",
			"index": i,
			"content_block": map[string]any{
				"type":  "tool_use",
				"id":    tcID,
				"name":  tc.Name,
				"input": map[string]any{},
			},
		}
		writeSSE(w, "content_block_start", blockStart)
		flusher.Flush()

		// Stream input JSON as deltas.
		argsJSON, _ := json.Marshal(tc.Arguments)
		chunks := splitString(string(argsJSON), 20)
		for _, chunk := range chunks {
			delta := map[string]any{
				"type":  "content_block_delta",
				"index": i,
				"delta": map[string]any{
					"type":         "input_json_delta",
					"partial_json": chunk,
				},
			}
			writeSSE(w, "content_block_delta", delta)
			flusher.Flush()

			select {
			case <-r.Context().Done():
				return
			case <-time.After(s.getTokenDelay()):
			}
		}

		// content_block_stop
		blockStop := map[string]any{
			"type":  "content_block_stop",
			"index": i,
		}
		writeSSE(w, "content_block_stop", blockStop)
		flusher.Flush()
	}

	// message_delta
	msgDelta := map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   "tool_use",
			"stop_sequence": nil,
		},
		"usage": map[string]any{
			"output_tokens": 5,
		},
	}
	writeSSE(w, "message_delta", msgDelta)
	flusher.Flush()

	// message_stop
	msgStop := map[string]any{
		"type": "message_stop",
	}
	writeSSE(w, "message_stop", msgStop)
	flusher.Flush()
}

// splitString splits s into chunks of at most n bytes.
func splitString(s string, n int) []string {
	if len(s) == 0 {
		return nil
	}
	var chunks []string
	for len(s) > 0 {
		if len(s) <= n {
			chunks = append(chunks, s)
			break
		}
		chunks = append(chunks, s[:n])
		s = s[n:]
	}
	return chunks
}

func (s *Server) getTokenDelay() time.Duration {
	if s.tokenDelay > 0 {
		return s.tokenDelay
	}
	return 15 * time.Millisecond
}
