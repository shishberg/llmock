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

func (s *Server) getTokenDelay() time.Duration {
	if s.tokenDelay > 0 {
		return s.tokenDelay
	}
	return 15 * time.Millisecond
}
