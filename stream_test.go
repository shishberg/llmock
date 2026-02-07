package llmock_test

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/shishberg/llmock"
)

func newStreamTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	s := llmock.New(llmock.WithTokenDelay(0))
	return httptest.NewServer(s.Handler())
}

// readSSEData reads all "data: " lines from an SSE response body.
func readSSEData(t *testing.T, resp *http.Response) []string {
	t.Helper()
	var lines []string
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			lines = append(lines, strings.TrimPrefix(line, "data: "))
		}
	}
	return lines
}

// readSSEEvents reads all SSE events (event + data pairs) from an Anthropic response.
type sseEvent struct {
	Event string
	Data  string
}

func readSSEEvents(t *testing.T, resp *http.Response) []sseEvent {
	t.Helper()
	var events []sseEvent
	scanner := bufio.NewScanner(resp.Body)
	var currentEvent string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
		} else if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			events = append(events, sseEvent{Event: currentEvent, Data: data})
			currentEvent = ""
		}
	}
	return events
}

func TestStreamOpenAI_ChunkFormat(t *testing.T) {
	ts := newStreamTestServer(t)
	defer ts.Close()

	body := `{
		"model": "test-model",
		"stream": true,
		"messages": [{"role": "user", "content": "Hello world"}]
	}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("expected Content-Type 'text/event-stream', got %q", ct)
	}

	lines := readSSEData(t, resp)
	if len(lines) < 2 {
		t.Fatalf("expected at least 2 data lines (content + [DONE]), got %d", len(lines))
	}

	// Last line should be [DONE]
	if lines[len(lines)-1] != "[DONE]" {
		t.Errorf("expected last data line to be '[DONE]', got %q", lines[len(lines)-1])
	}

	// Parse first content chunk
	var firstChunk struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Model   string `json:"model"`
		Created int64  `json:"created"`
		Choices []struct {
			Index int `json:"index"`
			Delta struct {
				Role    string `json:"role,omitempty"`
				Content string `json:"content,omitempty"`
			} `json:"delta"`
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(lines[0]), &firstChunk); err != nil {
		t.Fatalf("failed to parse first chunk: %v", err)
	}
	if firstChunk.Object != "chat.completion.chunk" {
		t.Errorf("expected object 'chat.completion.chunk', got %q", firstChunk.Object)
	}
	if firstChunk.Model != "test-model" {
		t.Errorf("expected model 'test-model', got %q", firstChunk.Model)
	}
	if !strings.HasPrefix(firstChunk.ID, "chatcmpl-mock-") {
		t.Errorf("expected id prefix 'chatcmpl-mock-', got %q", firstChunk.ID)
	}
	if len(firstChunk.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(firstChunk.Choices))
	}
	if firstChunk.Choices[0].Delta.Role != "assistant" {
		t.Errorf("expected first chunk delta role 'assistant', got %q", firstChunk.Choices[0].Delta.Role)
	}
	if firstChunk.Choices[0].FinishReason != nil {
		t.Errorf("expected nil finish_reason on content chunk, got %v", *firstChunk.Choices[0].FinishReason)
	}

	// Parse the final chunk (before [DONE])
	var finalChunk struct {
		Choices []struct {
			Delta        map[string]any `json:"delta"`
			FinishReason string         `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(lines[len(lines)-2]), &finalChunk); err != nil {
		t.Fatalf("failed to parse final chunk: %v", err)
	}
	if finalChunk.Choices[0].FinishReason != "stop" {
		t.Errorf("expected finish_reason 'stop' on final chunk, got %q", finalChunk.Choices[0].FinishReason)
	}
}

func TestStreamOpenAI_ReconstructedContent(t *testing.T) {
	ts := newStreamTestServer(t)
	defer ts.Close()

	body := `{
		"model": "test-model",
		"stream": true,
		"messages": [{"role": "user", "content": "Hello world"}]
	}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	lines := readSSEData(t, resp)

	// Reconstruct the full content from all content chunks (exclude [DONE] and the final stop chunk).
	var reconstructed strings.Builder
	for _, line := range lines {
		if line == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			t.Fatalf("failed to parse chunk: %v", err)
		}
		if len(chunk.Choices) > 0 {
			reconstructed.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	content := reconstructed.String()
	if content != "Hello world" {
		t.Errorf("expected reconstructed content 'Hello world', got %q", content)
	}
}

func TestStreamOpenAI_NonStreamStillWorks(t *testing.T) {
	ts := newStreamTestServer(t)
	defer ts.Close()

	// stream: false should produce normal JSON response.
	body := `{
		"model": "test-model",
		"messages": [{"role": "user", "content": "Hello world"}]
	}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("expected Content-Type 'application/json', got %q", ct)
	}

	var result llmock.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if result.Choices[0].Message.Content != "Hello world" {
		t.Errorf("expected 'Hello world', got %q", result.Choices[0].Message.Content)
	}
}

func TestStreamAnthropic_EventSequence(t *testing.T) {
	ts := newStreamTestServer(t)
	defer ts.Close()

	body := `{
		"model": "claude-3-opus",
		"max_tokens": 1024,
		"stream": true,
		"messages": [{"role": "user", "content": "Hello Anthropic"}]
	}`
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("expected Content-Type 'text/event-stream', got %q", ct)
	}

	events := readSSEEvents(t, resp)
	if len(events) < 5 {
		t.Fatalf("expected at least 5 events, got %d", len(events))
	}

	// Verify event sequence
	if events[0].Event != "message_start" {
		t.Errorf("expected first event 'message_start', got %q", events[0].Event)
	}
	if events[1].Event != "content_block_start" {
		t.Errorf("expected second event 'content_block_start', got %q", events[1].Event)
	}

	// Content deltas should be in the middle
	lastIdx := len(events) - 1
	for i := 2; i < lastIdx-2; i++ {
		if events[i].Event != "content_block_delta" {
			t.Errorf("expected 'content_block_delta' at index %d, got %q", i, events[i].Event)
		}
	}

	if events[lastIdx-2].Event != "content_block_stop" {
		t.Errorf("expected 'content_block_stop', got %q", events[lastIdx-2].Event)
	}
	if events[lastIdx-1].Event != "message_delta" {
		t.Errorf("expected 'message_delta', got %q", events[lastIdx-1].Event)
	}
	if events[lastIdx].Event != "message_stop" {
		t.Errorf("expected 'message_stop', got %q", events[lastIdx].Event)
	}

	// Verify message_start contains model and id
	var msgStart struct {
		Type    string `json:"type"`
		Message struct {
			ID    string `json:"id"`
			Model string `json:"model"`
			Role  string `json:"role"`
		} `json:"message"`
	}
	if err := json.Unmarshal([]byte(events[0].Data), &msgStart); err != nil {
		t.Fatalf("failed to parse message_start: %v", err)
	}
	if msgStart.Message.Model != "claude-3-opus" {
		t.Errorf("expected model 'claude-3-opus', got %q", msgStart.Message.Model)
	}
	if !strings.HasPrefix(msgStart.Message.ID, "msg_") {
		t.Errorf("expected id prefix 'msg_', got %q", msgStart.Message.ID)
	}
	if msgStart.Message.Role != "assistant" {
		t.Errorf("expected role 'assistant', got %q", msgStart.Message.Role)
	}
}

func TestStreamAnthropic_ReconstructedContent(t *testing.T) {
	ts := newStreamTestServer(t)
	defer ts.Close()

	body := `{
		"model": "claude-3-opus",
		"max_tokens": 1024,
		"stream": true,
		"messages": [{"role": "user", "content": "Hello Anthropic"}]
	}`
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	events := readSSEEvents(t, resp)

	var reconstructed strings.Builder
	for _, ev := range events {
		if ev.Event != "content_block_delta" {
			continue
		}
		var delta struct {
			Delta struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(ev.Data), &delta); err != nil {
			t.Fatalf("failed to parse content_block_delta: %v", err)
		}
		if delta.Delta.Type != "text_delta" {
			t.Errorf("expected delta type 'text_delta', got %q", delta.Delta.Type)
		}
		reconstructed.WriteString(delta.Delta.Text)
	}

	content := reconstructed.String()
	if content != "Hello Anthropic" {
		t.Errorf("expected reconstructed content 'Hello Anthropic', got %q", content)
	}
}

func TestStreamAnthropic_NonStreamStillWorks(t *testing.T) {
	ts := newStreamTestServer(t)
	defer ts.Close()

	body := `{
		"model": "claude-3-opus",
		"max_tokens": 1024,
		"messages": [{"role": "user", "content": "Hello Anthropic"}]
	}`
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("expected Content-Type 'application/json', got %q", ct)
	}

	var result llmock.AnthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if result.Content[0].Text != "Hello Anthropic" {
		t.Errorf("expected 'Hello Anthropic', got %q", result.Content[0].Text)
	}
}

func TestStreamAnthropic_MessageDeltaHasStopReason(t *testing.T) {
	ts := newStreamTestServer(t)
	defer ts.Close()

	body := `{
		"model": "claude-3-opus",
		"max_tokens": 1024,
		"stream": true,
		"messages": [{"role": "user", "content": "test"}]
	}`
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	events := readSSEEvents(t, resp)

	// Find message_delta event
	for _, ev := range events {
		if ev.Event != "message_delta" {
			continue
		}
		var delta struct {
			Delta struct {
				StopReason string `json:"stop_reason"`
			} `json:"delta"`
			Usage struct {
				OutputTokens int `json:"output_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(ev.Data), &delta); err != nil {
			t.Fatalf("failed to parse message_delta: %v", err)
		}
		if delta.Delta.StopReason != "end_turn" {
			t.Errorf("expected stop_reason 'end_turn', got %q", delta.Delta.StopReason)
		}
		if delta.Usage.OutputTokens <= 0 {
			t.Errorf("expected positive output_tokens, got %d", delta.Usage.OutputTokens)
		}
		return
	}
	t.Fatal("message_delta event not found")
}

func TestWithTokenDelay(t *testing.T) {
	s := llmock.New(llmock.WithTokenDelay(1 * time.Millisecond))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"model": "test",
		"stream": true,
		"messages": [{"role": "user", "content": "one two three four five six"}]
	}`

	start := time.Now()
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	// Read all data to ensure streaming completes.
	readSSEData(t, resp)
	elapsed := time.Since(start)

	// With a 1ms delay and multiple chunks, it should take at least some time.
	// We just verify it's not zero and not unreasonably long.
	if elapsed < 1*time.Millisecond {
		t.Errorf("streaming completed too quickly (%v), expected some delay", elapsed)
	}
}

func TestStreamOpenAI_ConsistentID(t *testing.T) {
	ts := newStreamTestServer(t)
	defer ts.Close()

	body := `{
		"model": "test-model",
		"stream": true,
		"messages": [{"role": "user", "content": "Hello world test message"}]
	}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	lines := readSSEData(t, resp)

	// All JSON chunks should have the same ID.
	var firstID string
	for _, line := range lines {
		if line == "[DONE]" {
			continue
		}
		var chunk struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			t.Fatalf("failed to parse chunk: %v", err)
		}
		if firstID == "" {
			firstID = chunk.ID
		} else if chunk.ID != firstID {
			t.Errorf("inconsistent IDs: first=%q, got=%q", firstID, chunk.ID)
		}
	}
}
