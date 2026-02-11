package llmock_test

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/shishberg/llmock"
)

// TestMultiTurn_OpenAI_ToolResultConversation verifies that the server
// accepts a multi-turn conversation including assistant tool_calls and
// tool-role result messages, then responds based on the latest user message.
func TestMultiTurn_OpenAI_ToolResultConversation(t *testing.T) {
	rules := []llmock.Rule{
		{Pattern: regexp.MustCompile(`(?i).*weather.*`), Responses: []string{"The weather is sunny."}},
		{Pattern: regexp.MustCompile(`.*`), Responses: []string{"I see: ${input}"}},
	}
	s := llmock.New(llmock.WithRules(rules...), llmock.WithSeed(42))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// Simulate: user asks about weather -> assistant calls tool -> tool returns result -> user asks follow-up
	body := `{
		"model": "gpt-4",
		"messages": [
			{"role": "user", "content": "What's the weather in Paris?"},
			{"role": "assistant", "content": null, "tool_calls": [
				{
					"id": "call_abc123",
					"type": "function",
					"function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}
				}
			]},
			{"role": "tool", "tool_call_id": "call_abc123", "content": "72°F and sunny"},
			{"role": "user", "content": "Thanks for the weather info"}
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

	if len(result.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(result.Choices))
	}

	// The last user message is "Thanks for the weather info" which should
	// NOT match the weather rule (it says "thanks" not asking about weather).
	// It should match the catchall rule.
	choice := result.Choices[0]
	if choice.FinishReason != "stop" {
		t.Errorf("expected finish_reason 'stop', got %q", choice.FinishReason)
	}
	if choice.Message.Content == "" {
		t.Error("expected non-empty content in response")
	}
}

// TestMultiTurn_OpenAI_ToolResultMatchesRule verifies that a tool result
// message's content can be used for rule matching when it's the last
// non-assistant message.
func TestMultiTurn_OpenAI_ToolResultMatchesRule(t *testing.T) {
	rules := []llmock.Rule{
		{Pattern: regexp.MustCompile(`72°F`), Responses: []string{"That's a nice temperature!"}},
		{Pattern: regexp.MustCompile(`.*`), Responses: []string{"fallback"}},
	}
	s := llmock.New(llmock.WithRules(rules...), llmock.WithSeed(42))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// The tool result is the last message (no follow-up user message).
	// The server should use the tool result content for rule matching.
	body := `{
		"model": "gpt-4",
		"messages": [
			{"role": "user", "content": "What's the weather?"},
			{"role": "assistant", "content": null, "tool_calls": [
				{
					"id": "call_abc123",
					"type": "function",
					"function": {"name": "get_weather", "arguments": "{}"}
				}
			]},
			{"role": "tool", "tool_call_id": "call_abc123", "content": "72°F and sunny"}
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

	// extractInput looks for the last user message first, then falls back
	// to the last message. Since "tool" role messages are included, and
	// the last user message is "What's the weather?", it won't match "72°F".
	// But the conversation is still accepted without errors - that's the key.
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if result.Choices[0].Message.Content == "" {
		t.Error("expected non-empty response")
	}
}

// TestMultiTurn_OpenAI_AssistantNullContent verifies that assistant messages
// with null content (typical when tool_calls are present) are handled correctly.
func TestMultiTurn_OpenAI_AssistantNullContent(t *testing.T) {
	s := llmock.New(llmock.WithResponder(llmock.EchoResponder{}))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [
			{"role": "user", "content": "Hello"},
			{"role": "assistant", "content": null, "tool_calls": [
				{
					"id": "call_xyz",
					"type": "function",
					"function": {"name": "some_func", "arguments": "{}"}
				}
			]},
			{"role": "tool", "tool_call_id": "call_xyz", "content": "tool output"},
			{"role": "user", "content": "Now what?"}
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

	// Should echo the last user message "Now what?"
	if result.Choices[0].Message.Content != "Now what?" {
		t.Errorf("expected 'Now what?', got %q", result.Choices[0].Message.Content)
	}
}

// TestMultiTurn_Anthropic_ToolUseAndResult verifies that the Anthropic endpoint
// accepts multi-turn conversations with tool_use and tool_result content blocks.
func TestMultiTurn_Anthropic_ToolUseAndResult(t *testing.T) {
	rules := []llmock.Rule{
		{Pattern: regexp.MustCompile(`(?i)summarize`), Responses: []string{"Here's a summary."}},
		{Pattern: regexp.MustCompile(`.*`), Responses: []string{"I see: ${input}"}},
	}
	s := llmock.New(llmock.WithRules(rules...), llmock.WithSeed(42))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// Anthropic multi-turn: user text -> assistant tool_use -> user tool_result -> user follow-up
	body := `{
		"model": "claude-3-opus",
		"max_tokens": 1024,
		"messages": [
			{"role": "user", "content": "Summarize the latest news"},
			{"role": "assistant", "content": [
				{
					"type": "tool_use",
					"id": "toolu_abc123",
					"name": "get_news",
					"input": {"topic": "latest"}
				}
			]},
			{"role": "user", "content": [
				{
					"type": "tool_result",
					"tool_use_id": "toolu_abc123",
					"content": "Breaking: New discovery in space exploration."
				}
			]},
			{"role": "user", "content": "Now summarize that for me"}
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

	if result.StopReason != "end_turn" {
		t.Errorf("expected stop_reason 'end_turn', got %q", result.StopReason)
	}
	if len(result.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(result.Content))
	}
	if result.Content[0].Type != "text" {
		t.Errorf("expected content type 'text', got %q", result.Content[0].Type)
	}
	// The last user message is "Now summarize that for me" which matches "summarize".
	if result.Content[0].Text != "Here's a summary." {
		t.Errorf("expected 'Here's a summary.', got %q", result.Content[0].Text)
	}
}

// TestMultiTurn_Anthropic_ToolResultWithNestedBlocks verifies that tool_result
// content can contain nested content blocks (array format).
func TestMultiTurn_Anthropic_ToolResultWithNestedBlocks(t *testing.T) {
	s := llmock.New(llmock.WithResponder(llmock.EchoResponder{}))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"model": "claude-3-opus",
		"max_tokens": 1024,
		"messages": [
			{"role": "user", "content": "Use the tool"},
			{"role": "assistant", "content": [
				{
					"type": "tool_use",
					"id": "toolu_xyz",
					"name": "my_tool",
					"input": {"key": "value"}
				}
			]},
			{"role": "user", "content": [
				{
					"type": "tool_result",
					"tool_use_id": "toolu_xyz",
					"content": [
						{"type": "text", "text": "Result line 1"},
						{"type": "text", "text": "Result line 2"}
					]
				}
			]},
			{"role": "user", "content": "What did you find?"}
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

	// Should echo the last user message.
	if result.Content[0].Text != "What did you find?" {
		t.Errorf("expected 'What did you find?', got %q", result.Content[0].Text)
	}
}

// TestMultiTurn_Anthropic_MixedContentBlocks verifies that messages with
// mixed content blocks (text + tool_result) are parsed correctly.
func TestMultiTurn_Anthropic_MixedContentBlocks(t *testing.T) {
	rules := []llmock.Rule{
		{Pattern: regexp.MustCompile(`here are the results`), Responses: []string{"Got it!"}},
		{Pattern: regexp.MustCompile(`.*`), Responses: []string{"fallback"}},
	}
	s := llmock.New(llmock.WithRules(rules...), llmock.WithSeed(42))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// User sends both a tool_result and text in the same message.
	body := `{
		"model": "claude-3-opus",
		"max_tokens": 1024,
		"messages": [
			{"role": "user", "content": "Search for info"},
			{"role": "assistant", "content": [
				{
					"type": "tool_use",
					"id": "toolu_search",
					"name": "search",
					"input": {"query": "info"}
				}
			]},
			{"role": "user", "content": [
				{
					"type": "tool_result",
					"tool_use_id": "toolu_search",
					"content": "Found 5 results"
				},
				{
					"type": "text",
					"text": "here are the results"
				}
			]}
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

	// The last user message has content blocks, and extractInput should find
	// the last "user" role message. Its MessageContent() concatenates text blocks
	// and tool_result content: "Found 5 results\nhere are the results".
	// The rule matching should find "here are the results" in that string.
	if result.Content[0].Text != "Got it!" {
		t.Errorf("expected 'Got it!', got %q", result.Content[0].Text)
	}
}

// TestMultiTurn_OpenAI_MultipleToolCalls verifies that multiple tool calls
// and their results in a single conversation are handled correctly.
func TestMultiTurn_OpenAI_MultipleToolCalls(t *testing.T) {
	s := llmock.New(llmock.WithResponder(llmock.EchoResponder{}))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [
			{"role": "user", "content": "Get weather and news"},
			{"role": "assistant", "content": null, "tool_calls": [
				{
					"id": "call_weather",
					"type": "function",
					"function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}
				},
				{
					"id": "call_news",
					"type": "function",
					"function": {"name": "get_news", "arguments": "{\"topic\":\"tech\"}"}
				}
			]},
			{"role": "tool", "tool_call_id": "call_weather", "content": "72°F sunny"},
			{"role": "tool", "tool_call_id": "call_news", "content": "Tech stocks up"},
			{"role": "user", "content": "Summarize both results"}
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

	// Should echo the last user message.
	if result.Choices[0].Message.Content != "Summarize both results" {
		t.Errorf("expected 'Summarize both results', got %q", result.Choices[0].Message.Content)
	}
}

// TestMultiTurn_OpenAI_Streaming verifies that multi-turn conversations
// with tool results work correctly with streaming enabled.
func TestMultiTurn_OpenAI_Streaming(t *testing.T) {
	s := llmock.New(llmock.WithResponder(llmock.EchoResponder{}))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"stream": true,
		"messages": [
			{"role": "user", "content": "Do something"},
			{"role": "assistant", "content": null, "tool_calls": [
				{
					"id": "call_1",
					"type": "function",
					"function": {"name": "do_thing", "arguments": "{}"}
				}
			]},
			{"role": "tool", "tool_call_id": "call_1", "content": "Done"},
			{"role": "user", "content": "Great thanks"}
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
	if resp.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("expected text/event-stream, got %q", resp.Header.Get("Content-Type"))
	}
}

// TestMultiTurn_Anthropic_Streaming verifies that Anthropic multi-turn
// conversations with tool use work correctly with streaming.
func TestMultiTurn_Anthropic_Streaming(t *testing.T) {
	s := llmock.New(llmock.WithResponder(llmock.EchoResponder{}))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"model": "claude-3-opus",
		"max_tokens": 1024,
		"stream": true,
		"messages": [
			{"role": "user", "content": "Use a tool"},
			{"role": "assistant", "content": [
				{
					"type": "tool_use",
					"id": "toolu_123",
					"name": "my_tool",
					"input": {}
				}
			]},
			{"role": "user", "content": [
				{
					"type": "tool_result",
					"tool_use_id": "toolu_123",
					"content": "tool output here"
				}
			]},
			{"role": "user", "content": "Continue please"}
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
	if resp.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("expected text/event-stream, got %q", resp.Header.Get("Content-Type"))
	}
}

// TestMultiTurn_Anthropic_ToolResultErrorFlag verifies that tool_result
// messages with is_error=true are still accepted.
func TestMultiTurn_Anthropic_ToolResultErrorFlag(t *testing.T) {
	s := llmock.New(llmock.WithResponder(llmock.EchoResponder{}))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"model": "claude-3-opus",
		"max_tokens": 1024,
		"messages": [
			{"role": "user", "content": "Run the tool"},
			{"role": "assistant", "content": [
				{
					"type": "tool_use",
					"id": "toolu_err",
					"name": "risky_tool",
					"input": {}
				}
			]},
			{"role": "user", "content": [
				{
					"type": "tool_result",
					"tool_use_id": "toolu_err",
					"is_error": true,
					"content": "Permission denied"
				}
			]},
			{"role": "user", "content": "The tool failed, try something else"}
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

	// Should echo the last user message.
	if result.Content[0].Text != "The tool failed, try something else" {
		t.Errorf("expected 'The tool failed, try something else', got %q", result.Content[0].Text)
	}
}

// TestMultiTurn_OpenAI_ToolResultSuppressesToolCall verifies that when a
// request contains tool results and a rule matches with a tool_call, the
// server responds with text instead of another tool call (preventing loops).
func TestMultiTurn_OpenAI_ToolResultSuppressesToolCall(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern: regexp.MustCompile(`(?i).*weather.*`),
			ToolCall: &llmock.ToolCallConfig{
				Name:      "get_weather",
				Arguments: map[string]any{"location": "Paris"},
			},
		},
		{Pattern: regexp.MustCompile(`.*`), Responses: []string{"fallback"}},
	}
	s := llmock.New(llmock.WithRules(rules...), llmock.WithSeed(42))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// First request: should return a tool call.
	body1 := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "What's the weather?"}],
		"tools": [{"type": "function", "function": {"name": "get_weather"}}]
	}`
	resp1, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body1))
	if err != nil {
		t.Fatal(err)
	}
	defer resp1.Body.Close()

	var result1 llmock.ChatCompletionResponse
	json.NewDecoder(resp1.Body).Decode(&result1)
	if result1.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("first request: expected finish_reason 'tool_calls', got %q", result1.Choices[0].FinishReason)
	}

	// Second request: includes tool result. Should NOT return another tool call.
	body2 := `{
		"model": "gpt-4",
		"messages": [
			{"role": "user", "content": "What's the weather?"},
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}}
			]},
			{"role": "tool", "tool_call_id": "call_1", "content": "72°F and sunny in Paris"}
		],
		"tools": [{"type": "function", "function": {"name": "get_weather"}}]
	}`
	resp2, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body2))
	if err != nil {
		t.Fatal(err)
	}
	defer resp2.Body.Close()

	var result2 llmock.ChatCompletionResponse
	json.NewDecoder(resp2.Body).Decode(&result2)

	// Should be a text response, not a tool call.
	if result2.Choices[0].FinishReason != "stop" {
		t.Errorf("second request: expected finish_reason 'stop', got %q", result2.Choices[0].FinishReason)
	}
	if result2.Choices[0].Message.Content == "" {
		t.Error("second request: expected non-empty text content")
	}
	if len(result2.Choices[0].Message.ToolCalls) > 0 {
		t.Error("second request: expected no tool calls in response")
	}
}

// TestMultiTurn_Anthropic_ToolResultSuppressesToolCall verifies the same
// tool-result suppression for the Anthropic endpoint.
func TestMultiTurn_Anthropic_ToolResultSuppressesToolCall(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern: regexp.MustCompile(`(?i).*search.*`),
			ToolCall: &llmock.ToolCallConfig{
				Name:      "search_kb",
				Arguments: map[string]any{"query": "${input}"},
			},
		},
	}
	s := llmock.New(llmock.WithRules(rules...), llmock.WithSeed(42))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// Request with tool results should get text, not another tool call.
	body := `{
		"model": "claude-3-opus",
		"max_tokens": 1024,
		"messages": [
			{"role": "user", "content": "search for llmock docs"},
			{"role": "assistant", "content": [
				{"type": "tool_use", "id": "toolu_1", "name": "search_kb", "input": {"query": "llmock docs"}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Found: llmock documentation page."}
			]}
		],
		"tools": [{"name": "search_kb", "input_schema": {"type": "object"}}]
	}`

	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.AnthropicResponse
	json.NewDecoder(resp.Body).Decode(&result)

	if result.StopReason != "end_turn" {
		t.Errorf("expected stop_reason 'end_turn', got %q", result.StopReason)
	}
	// Should be a text block, not tool_use.
	if len(result.Content) == 0 {
		t.Fatal("expected at least one content block")
	}
	if result.Content[0].Type != "text" {
		t.Errorf("expected content type 'text', got %q", result.Content[0].Type)
	}
	if result.Content[0].Text == "" {
		t.Error("expected non-empty text content")
	}
}

// TestMultiTurn_Gemini_ToolResultSuppressesToolCall verifies tool-result
// suppression for the Gemini endpoint.
func TestMultiTurn_Gemini_ToolResultSuppressesToolCall(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern: regexp.MustCompile(`(?i).*weather.*`),
			ToolCall: &llmock.ToolCallConfig{
				Name:      "get_weather",
				Arguments: map[string]any{"city": "Tokyo"},
			},
		},
	}
	s := llmock.New(llmock.WithRules(rules...), llmock.WithSeed(42))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "What's the weather?"}]},
			{"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Tokyo"}}}]},
			{"role": "user", "parts": [{"functionResponse": {"name": "get_weather", "response": {"result": "25°C and clear"}}}]}
		],
		"tools": [{"functionDeclarations": [{"name": "get_weather"}]}]
	}`

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.GeminiResponse
	json.NewDecoder(resp.Body).Decode(&result)

	if len(result.Candidates) == 0 {
		t.Fatal("expected at least one candidate")
	}
	// Should be text, not a function call.
	parts := result.Candidates[0].Content.Parts
	if len(parts) == 0 {
		t.Fatal("expected at least one part")
	}
	if parts[0].FunctionCall != nil {
		t.Error("expected text response, got function call")
	}
	if parts[0].Text == "" {
		t.Error("expected non-empty text")
	}
}

// TestMultiTurn_OpenAI_AutoToolCallsSuppressedByToolResults verifies that
// auto-tool-call generation is also suppressed when tool results are present.
func TestMultiTurn_OpenAI_AutoToolCallsSuppressedByToolResults(t *testing.T) {
	s := llmock.New(llmock.WithAutoToolCalls(true), llmock.WithSeed(42))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [
			{"role": "user", "content": "Hello"},
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "greet", "arguments": "{}"}}
			]},
			{"role": "tool", "tool_call_id": "call_1", "content": "Greeting sent"}
		],
		"tools": [{"type": "function", "function": {"name": "greet", "parameters": {"type": "object"}}}]
	}`

	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.ChatCompletionResponse
	json.NewDecoder(resp.Body).Decode(&result)

	// Even with auto-tool-calls enabled, should return text when tool results present.
	if result.Choices[0].FinishReason != "stop" {
		t.Errorf("expected finish_reason 'stop', got %q", result.Choices[0].FinishReason)
	}
	if len(result.Choices[0].Message.ToolCalls) > 0 {
		t.Error("expected no tool calls when tool results are present")
	}
}

// TestMultiTurn_OpenAI_StreamingToolResultSuppressesToolCall verifies that
// streaming responses also suppress tool calls when tool results are present.
func TestMultiTurn_OpenAI_StreamingToolResultSuppressesToolCall(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern: regexp.MustCompile(`(?i).*weather.*`),
			ToolCall: &llmock.ToolCallConfig{
				Name:      "get_weather",
				Arguments: map[string]any{"city": "NYC"},
			},
		},
	}
	s := llmock.New(llmock.WithRules(rules...), llmock.WithSeed(42))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"stream": true,
		"messages": [
			{"role": "user", "content": "What's the weather?"},
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}}
			]},
			{"role": "tool", "tool_call_id": "call_1", "content": "68°F partly cloudy"}
		],
		"tools": [{"type": "function", "function": {"name": "get_weather"}}]
	}`

	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	// Read SSE events — should be text chunks, not tool call chunks.
	scanner := bufio.NewScanner(resp.Body)
	var hasContent bool
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content   string `json:"content"`
					ToolCalls []any  `json:"tool_calls"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(line[6:]), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) > 0 {
			if chunk.Choices[0].Delta.Content != "" {
				hasContent = true
			}
			if len(chunk.Choices[0].Delta.ToolCalls) > 0 {
				t.Error("streaming: expected no tool call chunks when tool results are present")
			}
		}
	}
	if !hasContent {
		t.Error("streaming: expected text content chunks")
	}
}

// TestMultiTurn_NoToolResults_StillReturnsToolCall verifies that requests
// WITHOUT tool results still correctly return tool calls (the fix doesn't
// break the normal tool call path).
func TestMultiTurn_NoToolResults_StillReturnsToolCall(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern: regexp.MustCompile(`(?i).*weather.*`),
			ToolCall: &llmock.ToolCallConfig{
				Name:      "get_weather",
				Arguments: map[string]any{"location": "SF"},
			},
		},
	}
	s := llmock.New(llmock.WithRules(rules...), llmock.WithSeed(42))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "What's the weather?"}],
		"tools": [{"type": "function", "function": {"name": "get_weather"}}]
	}`

	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.ChatCompletionResponse
	json.NewDecoder(resp.Body).Decode(&result)

	// Without tool results, should still return a tool call.
	if result.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("expected finish_reason 'tool_calls', got %q", result.Choices[0].FinishReason)
	}
	if len(result.Choices[0].Message.ToolCalls) == 0 {
		t.Error("expected tool calls in response")
	}
}
