package llmock

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

const defaultBridgeTimeout = 120 * time.Second

// bridgeRequest is a pending request waiting for an interactive response.
type bridgeRequest struct {
	ID   int64
	Body []byte // raw HTTP request body
}

// bridgeState holds the internal state for the bridge responder.
type bridgeState struct {
	mu      sync.Mutex
	seq     int64
	pending map[int64]chan []byte // response channels, keyed by request ID
	reqCh   chan *bridgeRequest   // incoming requests queue here
	timeout time.Duration
	logger  *log.Logger
}

// BridgeResponder implements Responder by forwarding requests to an external
// consumer via HTTP long-poll endpoints.
type BridgeResponder struct {
	state *bridgeState
}

// NewBridgeResponder creates a BridgeResponder with the given configuration.
func NewBridgeResponder(cfg BridgeConfig) *BridgeResponder {
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = defaultBridgeTimeout
	}
	return &BridgeResponder{
		state: &bridgeState{
			pending: make(map[int64]chan []byte),
			reqCh:   make(chan *bridgeRequest, 64),
			timeout: timeout,
		},
	}
}

// Respond queues the request for an external consumer and blocks until a
// response is provided or the context/timeout expires.
func (b *BridgeResponder) Respond(ctx context.Context, req Request) (Response, error) {
	st := b.state

	// Assign a sequential request ID.
	st.mu.Lock()
	st.seq++
	id := st.seq
	respCh := make(chan []byte, 1)
	st.pending[id] = respCh
	st.mu.Unlock()

	// Cleanup on exit.
	defer func() {
		st.mu.Lock()
		delete(st.pending, id)
		st.mu.Unlock()
	}()

	// Queue the request for the consumer.
	breq := &bridgeRequest{ID: id, Body: req.RawBody}
	select {
	case st.reqCh <- breq:
	case <-ctx.Done():
		return Response{}, ctx.Err()
	}

	// Wait for the response.
	timer := time.NewTimer(st.timeout)
	defer timer.Stop()

	select {
	case data := <-respCh:
		return parseBridgeResponse(data)
	case <-ctx.Done():
		return Response{}, ctx.Err()
	case <-timer.C:
		return Response{}, fmt.Errorf("bridge: timeout waiting for response to request %d", id)
	}
}

// registerRoutes adds the /_bridge/ endpoints to the mux.
func (b *BridgeResponder) registerRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /_bridge/next", b.handleNext)
	mux.HandleFunc("POST /_bridge/respond/", b.handleRespond)
}

// handleNext blocks until a request arrives, then returns a JSON envelope
// containing the response URL and the raw request body. The response URL
// is also set in the X-Response-URL header for convenience.
// Returns 204 if no request arrives before timeout.
func (b *BridgeResponder) handleNext(w http.ResponseWriter, r *http.Request) {
	timeout := b.state.timeout
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case breq := <-b.state.reqCh:
		respURL := fmt.Sprintf("/_bridge/respond/%d", breq.ID)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Response-URL", respURL)
		w.WriteHeader(http.StatusOK)

		// Wrap in envelope so consumers can get the response URL from
		// the JSON body (easier for scripts) instead of parsing headers.
		envelope := struct {
			ResponseURL string          `json:"response_url"`
			Body        json.RawMessage `json:"body"`
		}{
			ResponseURL: respURL,
			Body:        json.RawMessage(breq.Body),
		}
		json.NewEncoder(w).Encode(envelope)
	case <-timer.C:
		w.WriteHeader(http.StatusNoContent)
	case <-r.Context().Done():
		return
	}
}

// handleRespond receives a response for a pending bridge request.
// The request ID is extracted from the URL path.
func (b *BridgeResponder) handleRespond(w http.ResponseWriter, r *http.Request) {
	// Extract ID from path: /_bridge/respond/{id}
	path := r.URL.Path
	idStr := strings.TrimPrefix(path, "/_bridge/respond/")
	if idStr == path {
		writeError(w, http.StatusBadRequest, "missing request ID")
		return
	}
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid request ID")
		return
	}

	// Read and validate response body.
	buf, readErr := io.ReadAll(io.LimitReader(r.Body, 10<<20))
	if readErr != nil {
		writeError(w, http.StatusBadRequest, "reading body: "+readErr.Error())
		return
	}
	if len(buf) == 0 {
		writeError(w, http.StatusBadRequest, "empty response body")
		return
	}

	if !json.Valid(buf) {
		writeError(w, http.StatusBadRequest, "response body is not valid JSON")
		return
	}

	// Deliver to the waiting Respond goroutine.
	b.state.mu.Lock()
	ch, ok := b.state.pending[id]
	b.state.mu.Unlock()

	if !ok {
		writeError(w, http.StatusNotFound, fmt.Sprintf("no pending request with ID %d", id))
		return
	}

	select {
	case ch <- buf:
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	default:
		// Channel already has a response (shouldn't happen).
		writeError(w, http.StatusConflict, "response already provided")
	}
}

// bridgeResponsePayload is used to parse the content blocks from a bridge response.
type bridgeResponsePayload struct {
	Content []bridgeContentBlock `json:"content"`
}

type bridgeContentBlock struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

// parseBridgeResponse converts raw response bytes into a Response.
// Accepts two formats:
//   - A JSON string: "hello" → text response
//   - Anthropic-style content blocks: {"content":[{"type":"text","text":"hello"}]}
func parseBridgeResponse(data []byte) (Response, error) {
	// Try as a plain JSON string first (shorthand for text response).
	var plainText string
	if err := json.Unmarshal(data, &plainText); err == nil {
		return Response{Text: plainText, Final: true}, nil
	}

	var payload bridgeResponsePayload
	if err := json.Unmarshal(data, &payload); err != nil {
		return Response{}, fmt.Errorf("bridge: invalid response JSON: %w", err)
	}

	var texts []string
	var toolCalls []ToolCall

	for _, block := range payload.Content {
		switch block.Type {
		case "text":
			if block.Text != "" {
				texts = append(texts, block.Text)
			}
		case "tool_use":
			var args map[string]any
			if len(block.Input) > 0 {
				json.Unmarshal(block.Input, &args)
			}
			id := block.ID
			if id == "" {
				id = generateToolCallID("toolu_")
			}
			toolCalls = append(toolCalls, ToolCall{
				ID:        id,
				Name:      block.Name,
				Arguments: args,
			})
		}
	}

	return Response{
		Text:      strings.Join(texts, "\n"),
		ToolCalls: toolCalls,
		Final:     true,
	}, nil
}
