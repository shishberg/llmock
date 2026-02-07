package llmock_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/shishberg/llmock"
)

func newTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	s := llmock.New()
	return httptest.NewServer(s.Handler())
}

func TestChatCompletions_EchoesLastUserMessage(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	body := `{
		"model": "test-model",
		"messages": [
			{"role": "system", "content": "You are helpful."},
			{"role": "user", "content": "Hello, world!"}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var result llmock.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}

	if result.Object != "chat.completion" {
		t.Errorf("expected object 'chat.completion', got %q", result.Object)
	}
	if result.Model != "test-model" {
		t.Errorf("expected model 'test-model', got %q", result.Model)
	}
	if !strings.HasPrefix(result.ID, "chatcmpl-mock-") {
		t.Errorf("expected id prefix 'chatcmpl-mock-', got %q", result.ID)
	}
	if result.Created == 0 {
		t.Error("expected non-zero created timestamp")
	}
	if len(result.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(result.Choices))
	}

	choice := result.Choices[0]
	if choice.Message.Role != "assistant" {
		t.Errorf("expected role 'assistant', got %q", choice.Message.Role)
	}
	if choice.Message.Content != "Hello, world!" {
		t.Errorf("expected echoed content 'Hello, world!', got %q", choice.Message.Content)
	}
	if choice.FinishReason != "stop" {
		t.Errorf("expected finish_reason 'stop', got %q", choice.FinishReason)
	}
	if choice.Index != 0 {
		t.Errorf("expected index 0, got %d", choice.Index)
	}
}

func TestChatCompletions_UsageStats(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [
			{"role": "user", "content": "one two three four five"}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}

	if result.Usage.PromptTokens <= 0 {
		t.Errorf("expected positive prompt_tokens, got %d", result.Usage.PromptTokens)
	}
	if result.Usage.CompletionTokens <= 0 {
		t.Errorf("expected positive completion_tokens, got %d", result.Usage.CompletionTokens)
	}
	if result.Usage.TotalTokens != result.Usage.PromptTokens+result.Usage.CompletionTokens {
		t.Errorf("expected total_tokens = prompt + completion (%d + %d), got %d",
			result.Usage.PromptTokens, result.Usage.CompletionTokens, result.Usage.TotalTokens)
	}
}

func TestChatCompletions_EmptyMessages(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	body := `{"model": "test", "messages": []}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
}

func TestChatCompletions_InvalidJSON(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader("not json"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
}

func TestChatCompletions_WrongMethod(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/v1/chat/completions")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", resp.StatusCode)
	}
}

func TestUnknownPath_Returns404(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/v1/unknown", "application/json", bytes.NewReader(nil))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", resp.StatusCode)
	}
}

func TestChatCompletions_DefaultModel(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	body := `{"messages": [{"role": "user", "content": "hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}

	if result.Model != "llmock-1" {
		t.Errorf("expected default model 'llmock-1', got %q", result.Model)
	}
}

func TestChatCompletions_NoUserMessage_FallsBackToLastMessage(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	body := `{
		"model": "test",
		"messages": [
			{"role": "system", "content": "You are a bot."}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}

	if result.Choices[0].Message.Content != "You are a bot." {
		t.Errorf("expected fallback to last message content, got %q", result.Choices[0].Message.Content)
	}
}

func TestMessages_EchoesLastUserMessage(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	body := `{
		"model": "claude-3-opus",
		"max_tokens": 1024,
		"messages": [
			{"role": "user", "content": "Hello, Anthropic!"}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var result llmock.AnthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}

	if !strings.HasPrefix(result.ID, "msg_") {
		t.Errorf("expected id prefix 'msg_', got %q", result.ID)
	}
	if result.Type != "message" {
		t.Errorf("expected type 'message', got %q", result.Type)
	}
	if result.Role != "assistant" {
		t.Errorf("expected role 'assistant', got %q", result.Role)
	}
	if result.Model != "claude-3-opus" {
		t.Errorf("expected model 'claude-3-opus', got %q", result.Model)
	}
	if result.StopReason != "end_turn" {
		t.Errorf("expected stop_reason 'end_turn', got %q", result.StopReason)
	}
	if result.StopSequence != nil {
		t.Errorf("expected stop_sequence nil, got %v", result.StopSequence)
	}
	if len(result.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(result.Content))
	}
	if result.Content[0].Type != "text" {
		t.Errorf("expected content type 'text', got %q", result.Content[0].Type)
	}
	if result.Content[0].Text != "Hello, Anthropic!" {
		t.Errorf("expected echoed text 'Hello, Anthropic!', got %q", result.Content[0].Text)
	}
}

func TestMessages_UsageStats(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	body := `{
		"model": "claude-3-opus",
		"max_tokens": 1024,
		"messages": [
			{"role": "user", "content": "one two three four five"}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.AnthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}

	if result.Usage.InputTokens <= 0 {
		t.Errorf("expected positive input_tokens, got %d", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens <= 0 {
		t.Errorf("expected positive output_tokens, got %d", result.Usage.OutputTokens)
	}
}

func TestMessages_EmptyMessages(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	body := `{"model": "test", "max_tokens": 1024, "messages": []}`
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
}

func TestMessages_InvalidJSON(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader("not json"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
}

func TestBothEndpoints_SameContent(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	// Send the same logical message to both endpoints.
	openaiBody := `{
		"model": "test-model",
		"messages": [
			{"role": "user", "content": "Hello, both!"}
		]
	}`
	anthropicBody := `{
		"model": "test-model",
		"max_tokens": 1024,
		"messages": [
			{"role": "user", "content": "Hello, both!"}
		]
	}`

	openaiResp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(openaiBody))
	if err != nil {
		t.Fatal(err)
	}
	defer openaiResp.Body.Close()

	anthropicResp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(anthropicBody))
	if err != nil {
		t.Fatal(err)
	}
	defer anthropicResp.Body.Close()

	var openaiResult llmock.ChatCompletionResponse
	if err := json.NewDecoder(openaiResp.Body).Decode(&openaiResult); err != nil {
		t.Fatal(err)
	}

	var anthropicResult llmock.AnthropicResponse
	if err := json.NewDecoder(anthropicResp.Body).Decode(&anthropicResult); err != nil {
		t.Fatal(err)
	}

	openaiContent := openaiResult.Choices[0].Message.Content
	anthropicContent := anthropicResult.Content[0].Text

	if openaiContent != anthropicContent {
		t.Errorf("endpoints produced different content: OpenAI=%q, Anthropic=%q", openaiContent, anthropicContent)
	}
}
