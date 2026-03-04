package llmock_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/shishberg/llmock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// bridgeServer creates a test server with bridge mode enabled.
func bridgeServer(t *testing.T, opts ...llmock.Option) *httptest.Server {
	t.Helper()
	defaults := []llmock.Option{
		llmock.WithBridge(llmock.BridgeConfig{Timeout: 5 * time.Second}),
	}
	s := llmock.New(append(defaults, opts...)...)
	return httptest.NewServer(s.Handler())
}

func TestBridge_HappyPathText(t *testing.T) {
	ts := bridgeServer(t)
	defer ts.Close()

	// Goroutine acts as the interactive responder.
	go func() {
		resp, err := http.Get(ts.URL + "/_bridge/next")
		if err != nil {
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return
		}

		// Extract response URL from header.
		respURL := resp.Header.Get("X-Response-URL")

		// Send a text response.
		body := `{"content":[{"type":"text","text":"hello from bridge"}]}`
		http.Post(ts.URL+respURL, "application/json", strings.NewReader(body))
	}()

	// Send an Anthropic request.
	reqBody := `{"model":"test","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(reqBody))
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var result llmock.AnthropicResponse
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&result))
	require.Len(t, result.Content, 1)
	assert.Equal(t, "text", result.Content[0].Type)
	assert.Equal(t, "hello from bridge", result.Content[0].Text)
}

func TestBridge_HappyPathToolUse(t *testing.T) {
	ts := bridgeServer(t)
	defer ts.Close()

	go func() {
		resp, err := http.Get(ts.URL + "/_bridge/next")
		if err != nil {
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return
		}
		respURL := resp.Header.Get("X-Response-URL")

		// Send a tool_use response.
		body := `{"content":[{"type":"tool_use","id":"toolu_abc123","name":"read_file","input":{"path":"/tmp/test.txt"}}]}`
		http.Post(ts.URL+respURL, "application/json", strings.NewReader(body))
	}()

	// Send request with a tool definition so the tool call passes validation.
	reqBody := `{"model":"test","max_tokens":100,"messages":[{"role":"user","content":"read file"}],"tools":[{"name":"read_file","description":"Read a file","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}]}`
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(reqBody))
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var result llmock.AnthropicResponse
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&result))
	require.Len(t, result.Content, 1)
	assert.Equal(t, "tool_use", result.Content[0].Type)
	assert.Equal(t, "read_file", result.Content[0].Name)
	assert.Equal(t, "/tmp/test.txt", result.Content[0].Input["path"])
}

func TestBridge_ResponseURLHeader(t *testing.T) {
	ts := bridgeServer(t)
	defer ts.Close()

	// Send a request to the bridge in the background.
	go func() {
		reqBody := `{"model":"test","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
		http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(reqBody))
	}()

	// Poll for the next request.
	resp, err := http.Get(ts.URL + "/_bridge/next")
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusOK, resp.StatusCode)

	respURL := resp.Header.Get("X-Response-URL")
	assert.True(t, strings.HasPrefix(respURL, "/_bridge/respond/"), "expected response URL prefix, got %q", respURL)

	// Now POST a response to that URL and verify it works.
	body := `{"content":[{"type":"text","text":"responded via header URL"}]}`
	postResp, err := http.Post(ts.URL+respURL, "application/json", strings.NewReader(body))
	require.NoError(t, err)
	defer postResp.Body.Close()
	assert.Equal(t, http.StatusOK, postResp.StatusCode)
}

func TestBridge_Timeout(t *testing.T) {
	// Use a very short timeout to make the test fast.
	s := llmock.New(llmock.WithBridge(llmock.BridgeConfig{Timeout: 200 * time.Millisecond}))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// Send request — nobody responds, should timeout.
	reqBody := `{"model":"test","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(reqBody))
	require.NoError(t, err)
	defer resp.Body.Close()

	// Should get an error response (400 from the error handler).
	assert.Equal(t, http.StatusBadRequest, resp.StatusCode)

	bodyBytes, _ := io.ReadAll(resp.Body)
	assert.Contains(t, string(bodyBytes), "timeout")
}

func TestBridge_InvalidJSONResponse(t *testing.T) {
	ts := bridgeServer(t)
	defer ts.Close()

	go func() {
		resp, err := http.Get(ts.URL + "/_bridge/next")
		if err != nil {
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return
		}
		respURL := resp.Header.Get("X-Response-URL")

		// Send invalid JSON.
		http.Post(ts.URL+respURL, "application/json", strings.NewReader("not json at all"))
	}()

	// We need to first trigger a request so the bridge has something.
	// Send request concurrently — the invalid JSON should cause an error on the respond endpoint,
	// and the bridge request should timeout.
	s := llmock.New(llmock.WithBridge(llmock.BridgeConfig{Timeout: 500 * time.Millisecond}))
	ts2 := httptest.NewServer(s.Handler())
	defer ts2.Close()

	go func() {
		resp, err := http.Get(ts2.URL + "/_bridge/next")
		if err != nil {
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return
		}
		respURL := resp.Header.Get("X-Response-URL")
		resp2, err := http.Post(ts2.URL+respURL, "application/json", strings.NewReader("not json"))
		if err != nil {
			return
		}
		defer resp2.Body.Close()
		// Should be rejected.
		assert.Equal(t, http.StatusBadRequest, resp2.StatusCode)
	}()

	reqBody := `{"model":"test","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts2.URL+"/v1/messages", "application/json", strings.NewReader(reqBody))
	require.NoError(t, err)
	defer resp.Body.Close()
	// The bridge should timeout since the invalid JSON response was rejected.
	assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
}

func TestBridge_UnknownRequestID(t *testing.T) {
	ts := bridgeServer(t)
	defer ts.Close()

	// POST to a non-existent request ID.
	body := `{"content":[{"type":"text","text":"hello"}]}`
	resp, err := http.Post(ts.URL+"/_bridge/respond/999", "application/json", strings.NewReader(body))
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusNotFound, resp.StatusCode)
}

func TestBridge_ContextCancellation(t *testing.T) {
	s := llmock.New(llmock.WithBridge(llmock.BridgeConfig{Timeout: 10 * time.Second}))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// Create a cancellable context.
	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)
	go func() {
		reqBody := `{"model":"test","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
		req, _ := http.NewRequestWithContext(ctx, "POST", ts.URL+"/v1/messages", strings.NewReader(reqBody))
		req.Header.Set("Content-Type", "application/json")
		_, err := http.DefaultClient.Do(req)
		errCh <- err
	}()

	// Give the request time to reach the bridge, then cancel.
	time.Sleep(100 * time.Millisecond)
	cancel()

	// The request should return an error (context cancelled).
	err := <-errCh
	assert.Error(t, err)
}

func TestBridge_FaultsLayerOnTop(t *testing.T) {
	// Bridge with a fault configured — fault should fire first.
	s := llmock.New(
		llmock.WithBridge(llmock.BridgeConfig{Timeout: 2 * time.Second}),
		llmock.WithFault(llmock.Fault{Type: "error", Status: 503, Message: "service unavailable", Count: 1}),
	)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// First request should hit the fault.
	reqBody := `{"model":"test","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(reqBody))
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, 503, resp.StatusCode)
}

func TestBridge_FinalFlag(t *testing.T) {
	// Bridge responses should have Final=true, meaning auto-tool-calls don't fire.
	s := llmock.New(
		llmock.WithBridge(llmock.BridgeConfig{Timeout: 5 * time.Second}),
		llmock.WithAutoToolCalls(true),
	)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	go func() {
		resp, err := http.Get(ts.URL + "/_bridge/next")
		if err != nil {
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return
		}
		respURL := resp.Header.Get("X-Response-URL")

		// Send a text response (not a tool call).
		body := `{"content":[{"type":"text","text":"just text, no tool"}]}`
		http.Post(ts.URL+respURL, "application/json", strings.NewReader(body))
	}()

	// Send request with tools — auto-tool-calls should NOT override the bridge response.
	reqBody := `{"model":"test","max_tokens":100,"messages":[{"role":"user","content":"do it"}],"tools":[{"name":"run","description":"run","input_schema":{"type":"object"}}]}`
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(reqBody))
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var result llmock.AnthropicResponse
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&result))
	require.Len(t, result.Content, 1)
	assert.Equal(t, "text", result.Content[0].Type)
	assert.Equal(t, "just text, no tool", result.Content[0].Text)
}

func TestBridge_RawPassthrough(t *testing.T) {
	ts := bridgeServer(t)
	defer ts.Close()

	var receivedBody string
	go func() {
		resp, err := http.Get(ts.URL + "/_bridge/next")
		if err != nil {
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return
		}
		bodyBytes, _ := io.ReadAll(resp.Body)
		receivedBody = string(bodyBytes)
		respURL := resp.Header.Get("X-Response-URL")

		body := `{"content":[{"type":"text","text":"ok"}]}`
		http.Post(ts.URL+respURL, "application/json", strings.NewReader(body))
	}()

	// Send request with system prompt.
	reqBody := `{"model":"test","max_tokens":100,"system":"You are a helpful bot.","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(reqBody))
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusOK, resp.StatusCode)

	// Wait a moment for the goroutine to finish reading.
	time.Sleep(100 * time.Millisecond)

	// The raw body passed through the bridge should contain the system field.
	assert.Contains(t, receivedBody, `"system"`)
	assert.Contains(t, receivedBody, "You are a helpful bot.")
}

func TestBridge_NextTimeout(t *testing.T) {
	// Test that GET /_bridge/next returns 204 when no request arrives.
	s := llmock.New(llmock.WithBridge(llmock.BridgeConfig{Timeout: 200 * time.Millisecond}))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/_bridge/next")
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusNoContent, resp.StatusCode)
}

func TestBridge_MultipleSequentialRequests(t *testing.T) {
	ts := bridgeServer(t)
	defer ts.Close()

	for i := 0; i < 3; i++ {
		go func(n int) {
			resp, err := http.Get(ts.URL + "/_bridge/next")
			if err != nil {
				return
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				return
			}
			respURL := resp.Header.Get("X-Response-URL")
			body := fmt.Sprintf(`{"content":[{"type":"text","text":"response %d"}]}`, n)
			http.Post(ts.URL+respURL, "application/json", strings.NewReader(body))
		}(i)

		reqBody := `{"model":"test","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
		resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(reqBody))
		require.NoError(t, err)
		resp.Body.Close()
		assert.Equal(t, http.StatusOK, resp.StatusCode)
	}
}
