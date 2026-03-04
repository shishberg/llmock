package llmock

import (
	"context"
	"encoding/json"
	"fmt"
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

// handleNext blocks until a request arrives, then returns the raw request body.
// Sets X-Response-URL header so the consumer knows where to POST the response.
// Returns 204 if no request arrives before timeout.
func (b *BridgeResponder) handleNext(w http.ResponseWriter, r *http.Request) {
	timeout := b.state.timeout
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case breq := <-b.state.reqCh:
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Response-URL", fmt.Sprintf("/_bridge/respond/%d", breq.ID))
		w.WriteHeader(http.StatusOK)
		w.Write(breq.Body)
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
	var buf []byte
	buf, err = readLimited(r.Body, 10<<20) // 10MB limit
	if err != nil {
		writeError(w, http.StatusBadRequest, "reading body: "+err.Error())
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

// readLimited reads up to limit bytes from r.
func readLimited(r interface{ Read([]byte) (int, error) }, limit int64) ([]byte, error) {
	lr := &limitedReader{r: r, remaining: limit}
	var buf []byte
	tmp := make([]byte, 32*1024)
	for {
		n, err := lr.Read(tmp)
		if n > 0 {
			buf = append(buf, tmp[:n]...)
		}
		if err != nil {
			if err.Error() == "limit exceeded" {
				return nil, fmt.Errorf("body too large (limit %d bytes)", limit)
			}
			break
		}
	}
	return buf, nil
}

type limitedReader struct {
	r         interface{ Read([]byte) (int, error) }
	remaining int64
}

func (lr *limitedReader) Read(p []byte) (int, error) {
	if lr.remaining <= 0 {
		return 0, fmt.Errorf("limit exceeded")
	}
	if int64(len(p)) > lr.remaining {
		p = p[:lr.remaining]
	}
	n, err := lr.r.Read(p)
	lr.remaining -= int64(n)
	return n, err
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
// It expects Anthropic-style content blocks.
func parseBridgeResponse(data []byte) (Response, error) {
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
