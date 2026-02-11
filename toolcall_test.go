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

// newToolCallServer creates a test server with a tool-call rule.
func newToolCallServer(t *testing.T, rules ...llmock.Rule) *httptest.Server {
	t.Helper()
	s := llmock.New(llmock.WithRules(rules...), llmock.WithSeed(42))
	return httptest.NewServer(s.Handler())
}

func TestToolCall_OpenAI_BasicToolCallResponse(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern: regexp.MustCompile(`.*weather.*`),
			ToolCall: &llmock.ToolCallConfig{
				Name:      "get_weather",
				Arguments: map[string]any{"location": "San Francisco", "unit": "celsius"},
			},
		},
		{Pattern: regexp.MustCompile(`.*`), Responses: []string{"fallback"}},
	}
	ts := newToolCallServer(t, rules...)
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "What's the weather like?"}],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "get_weather",
					"description": "Get the current weather",
					"parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
				}
			}
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
	choice := result.Choices[0]
	if choice.FinishReason != "tool_calls" {
		t.Errorf("expected finish_reason 'tool_calls', got %q", choice.FinishReason)
	}
	if choice.Message.Role != "assistant" {
		t.Errorf("expected role 'assistant', got %q", choice.Message.Role)
	}
	if len(choice.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(choice.Message.ToolCalls))
	}

	tc := choice.Message.ToolCalls[0]
	if !strings.HasPrefix(tc.ID, "call_") {
		t.Errorf("expected tool call ID prefix 'call_', got %q", tc.ID)
	}
	if tc.Type != "function" {
		t.Errorf("expected type 'function', got %q", tc.Type)
	}
	if tc.Function.Name != "get_weather" {
		t.Errorf("expected function name 'get_weather', got %q", tc.Function.Name)
	}

	var args map[string]any
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		t.Fatalf("failed to parse arguments JSON: %v", err)
	}
	if args["location"] != "San Francisco" {
		t.Errorf("expected location 'San Francisco', got %v", args["location"])
	}
	if args["unit"] != "celsius" {
		t.Errorf("expected unit 'celsius', got %v", args["unit"])
	}
}

func TestToolCall_Anthropic_BasicToolCallResponse(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern: regexp.MustCompile(`.*weather.*`),
			ToolCall: &llmock.ToolCallConfig{
				Name:      "get_weather",
				Arguments: map[string]any{"location": "London"},
			},
		},
		{Pattern: regexp.MustCompile(`.*`), Responses: []string{"fallback"}},
	}
	ts := newToolCallServer(t, rules...)
	defer ts.Close()

	body := `{
		"model": "claude-3-opus",
		"max_tokens": 1024,
		"messages": [{"role": "user", "content": "What's the weather?"}],
		"tools": [
			{
				"name": "get_weather",
				"description": "Get weather",
				"input_schema": {"type": "object", "properties": {"location": {"type": "string"}}}
			}
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

	if result.StopReason != "tool_use" {
		t.Errorf("expected stop_reason 'tool_use', got %q", result.StopReason)
	}
	if len(result.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(result.Content))
	}

	block := result.Content[0]
	if block.Type != "tool_use" {
		t.Errorf("expected content type 'tool_use', got %q", block.Type)
	}
	if !strings.HasPrefix(block.ID, "toolu_") {
		t.Errorf("expected tool use ID prefix 'toolu_', got %q", block.ID)
	}
	if block.Name != "get_weather" {
		t.Errorf("expected name 'get_weather', got %q", block.Name)
	}
	if block.Input["location"] != "London" {
		t.Errorf("expected input location 'London', got %v", block.Input["location"])
	}
}

func TestToolCall_FallsThrough_WhenToolNotInRequest(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern: regexp.MustCompile(`.*weather.*`),
			ToolCall: &llmock.ToolCallConfig{
				Name:      "get_weather",
				Arguments: map[string]any{"location": "NYC"},
			},
		},
		{Pattern: regexp.MustCompile(`.*`), Responses: []string{"text fallback"}},
	}
	ts := newToolCallServer(t, rules...)
	defer ts.Close()

	// Request includes tools but NOT get_weather - should fall through.
	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "What's the weather?"}],
		"tools": [
			{
				"type": "function",
				"function": {"name": "different_tool", "parameters": {}}
			}
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

	// Should fall through to text response since tool not in request.
	if result.Choices[0].FinishReason != "stop" {
		t.Errorf("expected finish_reason 'stop' (text fallback), got %q", result.Choices[0].FinishReason)
	}
}

func TestToolCall_CaptureGroupsInArguments(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern: regexp.MustCompile(`.*weather in (.*)`),
			ToolCall: &llmock.ToolCallConfig{
				Name:      "get_weather",
				Arguments: map[string]any{"location": "$1", "unit": "celsius"},
			},
		},
	}
	ts := newToolCallServer(t, rules...)
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "What's the weather in Tokyo"}],
		"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
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

	tc := result.Choices[0].Message.ToolCalls[0]
	var args map[string]any
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		t.Fatal(err)
	}
	if args["location"] != "Tokyo" {
		t.Errorf("expected location 'Tokyo', got %v", args["location"])
	}
}

func TestToolCall_ConfigParsing(t *testing.T) {
	yamlData := []byte(`
rules:
  - pattern: ".*weather.*"
    tool_call:
      name: "get_weather"
      arguments:
        location: "$1"
        unit: "celsius"
`)

	cfg, err := llmock.ParseConfig(yamlData, "test.yaml")
	if err != nil {
		t.Fatal(err)
	}

	if len(cfg.Rules) != 1 {
		t.Fatalf("expected 1 rule, got %d", len(cfg.Rules))
	}
	if cfg.Rules[0].ToolCall == nil {
		t.Fatal("expected tool_call to be set")
	}
	if cfg.Rules[0].ToolCall.Name != "get_weather" {
		t.Errorf("expected tool_call name 'get_weather', got %q", cfg.Rules[0].ToolCall.Name)
	}
	if cfg.Rules[0].ToolCall.Arguments["unit"] != "celsius" {
		t.Errorf("expected argument unit 'celsius', got %v", cfg.Rules[0].ToolCall.Arguments["unit"])
	}

	// Compile rules from config.
	rules, err := llmock.CompileRules(cfg.Rules)
	if err != nil {
		t.Fatal(err)
	}
	if rules[0].ToolCall == nil {
		t.Fatal("expected compiled rule to have tool_call")
	}
}

func TestToolCall_OpenAI_StreamingToolCall(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern: regexp.MustCompile(`.*weather.*`),
			ToolCall: &llmock.ToolCallConfig{
				Name:      "get_weather",
				Arguments: map[string]any{"location": "Paris"},
			},
		},
	}
	ts := newToolCallServer(t, rules...)
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"stream": true,
		"messages": [{"role": "user", "content": "weather?"}],
		"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
	}`

	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("expected text/event-stream, got %q", resp.Header.Get("Content-Type"))
	}

	var events []map[string]any
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			t.Fatalf("failed to parse SSE event: %v", err)
		}
		events = append(events, event)
	}

	if len(events) < 2 {
		t.Fatalf("expected at least 2 events, got %d", len(events))
	}

	// Check that the last event has finish_reason "tool_calls".
	lastEvent := events[len(events)-1]
	choices := lastEvent["choices"].([]any)
	lastChoice := choices[0].(map[string]any)
	if lastChoice["finish_reason"] != "tool_calls" {
		t.Errorf("expected final finish_reason 'tool_calls', got %v", lastChoice["finish_reason"])
	}

	// The first event should have tool_calls with function name.
	firstEvent := events[0]
	firstChoices := firstEvent["choices"].([]any)
	firstDelta := firstChoices[0].(map[string]any)["delta"].(map[string]any)
	toolCalls := firstDelta["tool_calls"].([]any)
	if len(toolCalls) == 0 {
		t.Fatal("expected tool_calls in first delta")
	}
	firstTC := toolCalls[0].(map[string]any)
	fn := firstTC["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("expected function name 'get_weather', got %v", fn["name"])
	}
}

func TestToolCall_Anthropic_StreamingToolCall(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern: regexp.MustCompile(`.*weather.*`),
			ToolCall: &llmock.ToolCallConfig{
				Name:      "get_weather",
				Arguments: map[string]any{"location": "Berlin"},
			},
		},
	}
	ts := newToolCallServer(t, rules...)
	defer ts.Close()

	body := `{
		"model": "claude-3-opus",
		"max_tokens": 1024,
		"stream": true,
		"messages": [{"role": "user", "content": "weather?"}],
		"tools": [{"name": "get_weather", "input_schema": {"type": "object"}}]
	}`

	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("expected text/event-stream, got %q", resp.Header.Get("Content-Type"))
	}

	type sseEvent struct {
		Event string
		Data  map[string]any
	}

	var events []sseEvent
	scanner := bufio.NewScanner(resp.Body)
	var currentEvent string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
			continue
		}
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			var parsed map[string]any
			if err := json.Unmarshal([]byte(data), &parsed); err != nil {
				t.Fatalf("failed to parse SSE data: %v", err)
			}
			events = append(events, sseEvent{Event: currentEvent, Data: parsed})
		}
	}

	if len(events) < 4 {
		t.Fatalf("expected at least 4 events, got %d", len(events))
	}

	// Check message_start.
	if events[0].Event != "message_start" {
		t.Errorf("expected first event 'message_start', got %q", events[0].Event)
	}

	// Check content_block_start has tool_use type.
	if events[1].Event != "content_block_start" {
		t.Errorf("expected second event 'content_block_start', got %q", events[1].Event)
	}
	block := events[1].Data["content_block"].(map[string]any)
	if block["type"] != "tool_use" {
		t.Errorf("expected content_block type 'tool_use', got %v", block["type"])
	}
	if block["name"] != "get_weather" {
		t.Errorf("expected tool name 'get_weather', got %v", block["name"])
	}

	// Check that message_delta has stop_reason "tool_use".
	var foundDelta bool
	for _, ev := range events {
		if ev.Event == "message_delta" {
			delta := ev.Data["delta"].(map[string]any)
			if delta["stop_reason"] != "tool_use" {
				t.Errorf("expected stop_reason 'tool_use', got %v", delta["stop_reason"])
			}
			foundDelta = true
		}
	}
	if !foundDelta {
		t.Error("expected message_delta event")
	}
}

func TestToolCall_NoToolsInRequest_StillReturnsToolCall(t *testing.T) {
	// When no tools are in the request, a tool call rule still matches
	// but the server should still produce the tool call (no filtering needed
	// when there are no tools defined - the tool call is returned as-is).
	rules := []llmock.Rule{
		{
			Pattern: regexp.MustCompile(`.*weather.*`),
			ToolCall: &llmock.ToolCallConfig{
				Name:      "get_weather",
				Arguments: map[string]any{"location": "NYC"},
			},
		},
	}
	ts := newToolCallServer(t, rules...)
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "What's the weather?"}]
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

	// With no tools in request, tool call is returned as-is.
	if result.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("expected finish_reason 'tool_calls', got %q", result.Choices[0].FinishReason)
	}
	if len(result.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result.Choices[0].Message.ToolCalls))
	}
}

func TestToolCall_TextResponseUnchanged_WhenNoToolCallRule(t *testing.T) {
	// Verify normal text responses still work correctly.
	rules := []llmock.Rule{
		{Pattern: regexp.MustCompile(`^hello$`), Responses: []string{"Hello there!"}},
	}
	ts := newToolCallServer(t, rules...)
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "hello"}],
		"tools": [{"type": "function", "function": {"name": "some_tool", "parameters": {}}}]
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

	if result.Choices[0].FinishReason != "stop" {
		t.Errorf("expected finish_reason 'stop', got %q", result.Choices[0].FinishReason)
	}
	if result.Choices[0].Message.Content != "Hello there!" {
		t.Errorf("expected 'Hello there!', got %q", result.Choices[0].Message.Content)
	}
}

func TestToolCall_InputTemplateInArguments(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern: regexp.MustCompile(`.*`),
			ToolCall: &llmock.ToolCallConfig{
				Name:      "echo_tool",
				Arguments: map[string]any{"message": "${input}"},
			},
		},
	}
	ts := newToolCallServer(t, rules...)
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "test input here"}],
		"tools": [{"type": "function", "function": {"name": "echo_tool", "parameters": {}}}]
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

	tc := result.Choices[0].Message.ToolCalls[0]
	var args map[string]any
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		t.Fatal(err)
	}
	if args["message"] != "test input here" {
		t.Errorf("expected message 'test input here', got %v", args["message"])
	}
}

func intPtr(n int) *int { return &n }

func TestToolCall_MaxCalls_FallsToTextAfterLimit(t *testing.T) {
	// Rule fires tool call once, then falls through to text response.
	rules := []llmock.Rule{
		{
			Pattern:   regexp.MustCompile(`.*search.*`),
			ToolCall:  &llmock.ToolCallConfig{Name: "search_kb", Arguments: map[string]any{"query": "${input}"}},
			Responses: []string{"Here is a summary of the search results."},
			MaxCalls:  intPtr(1),
		},
	}
	ts := newToolCallServer(t, rules...)
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "search for cats"}],
		"tools": [{"type": "function", "function": {"name": "search_kb", "parameters": {}}}]
	}`

	// First request: should return a tool call.
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if result.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("first request: expected finish_reason 'tool_calls', got %q", result.Choices[0].FinishReason)
	}

	// Second request: should fall through to text response.
	resp2, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp2.Body.Close()

	var result2 llmock.ChatCompletionResponse
	if err := json.NewDecoder(resp2.Body).Decode(&result2); err != nil {
		t.Fatal(err)
	}
	if result2.Choices[0].FinishReason != "stop" {
		t.Fatalf("second request: expected finish_reason 'stop', got %q", result2.Choices[0].FinishReason)
	}
	if result2.Choices[0].Message.Content != "Here is a summary of the search results." {
		t.Errorf("expected text fallback, got %q", result2.Choices[0].Message.Content)
	}
}

func TestToolCall_MaxCalls_NoTextResponses_SkipsRule(t *testing.T) {
	// Rule with max_calls but no text responses: after exhaustion, rule is skipped
	// and the next rule is tried.
	rules := []llmock.Rule{
		{
			Pattern:  regexp.MustCompile(`.*search.*`),
			ToolCall: &llmock.ToolCallConfig{Name: "search_kb", Arguments: map[string]any{"q": "test"}},
			MaxCalls: intPtr(1),
		},
		{Pattern: regexp.MustCompile(`.*`), Responses: []string{"catchall response"}},
	}
	ts := newToolCallServer(t, rules...)
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "search for dogs"}],
		"tools": [{"type": "function", "function": {"name": "search_kb", "parameters": {}}}]
	}`

	// First request: tool call fires.
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	// Second request: tool call exhausted, no text responses, falls to catchall.
	resp2, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp2.Body.Close()

	var result2 llmock.ChatCompletionResponse
	if err := json.NewDecoder(resp2.Body).Decode(&result2); err != nil {
		t.Fatal(err)
	}
	if result2.Choices[0].FinishReason != "stop" {
		t.Fatalf("expected finish_reason 'stop', got %q", result2.Choices[0].FinishReason)
	}
	if result2.Choices[0].Message.Content != "catchall response" {
		t.Errorf("expected 'catchall response', got %q", result2.Choices[0].Message.Content)
	}
}

func TestToolCall_MaxCalls_Unlimited_ByDefault(t *testing.T) {
	// No MaxCalls set: tool call fires every time.
	rules := []llmock.Rule{
		{
			Pattern:  regexp.MustCompile(`.*weather.*`),
			ToolCall: &llmock.ToolCallConfig{Name: "get_weather", Arguments: map[string]any{"loc": "NYC"}},
		},
	}
	ts := newToolCallServer(t, rules...)
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "weather please"}],
		"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
	}`

	for i := 0; i < 3; i++ {
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		var result llmock.ChatCompletionResponse
		json.NewDecoder(resp.Body).Decode(&result)
		resp.Body.Close()
		if result.Choices[0].FinishReason != "tool_calls" {
			t.Fatalf("request %d: expected 'tool_calls', got %q", i+1, result.Choices[0].FinishReason)
		}
	}
}

func TestToolCall_MaxCalls_GreaterThanOne(t *testing.T) {
	// max_calls=2: tool call fires twice, then falls to text.
	rules := []llmock.Rule{
		{
			Pattern:   regexp.MustCompile(`.*fetch.*`),
			ToolCall:  &llmock.ToolCallConfig{Name: "fetch_data", Arguments: map[string]any{}},
			Responses: []string{"Done fetching."},
			MaxCalls:  intPtr(2),
		},
	}
	ts := newToolCallServer(t, rules...)
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "fetch results"}],
		"tools": [{"type": "function", "function": {"name": "fetch_data", "parameters": {}}}]
	}`

	// Requests 1 and 2: tool call fires.
	for i := 0; i < 2; i++ {
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		var result llmock.ChatCompletionResponse
		json.NewDecoder(resp.Body).Decode(&result)
		resp.Body.Close()
		if result.Choices[0].FinishReason != "tool_calls" {
			t.Fatalf("request %d: expected 'tool_calls', got %q", i+1, result.Choices[0].FinishReason)
		}
	}

	// Request 3: exhausted, text response.
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var result llmock.ChatCompletionResponse
	json.NewDecoder(resp.Body).Decode(&result)
	if result.Choices[0].FinishReason != "stop" {
		t.Fatalf("request 3: expected 'stop', got %q", result.Choices[0].FinishReason)
	}
	if result.Choices[0].Message.Content != "Done fetching." {
		t.Errorf("expected 'Done fetching.', got %q", result.Choices[0].Message.Content)
	}
}

func TestToolCall_MaxCalls_Anthropic(t *testing.T) {
	// Verify max_calls works on Anthropic endpoint too.
	rules := []llmock.Rule{
		{
			Pattern:   regexp.MustCompile(`.*search.*`),
			ToolCall:  &llmock.ToolCallConfig{Name: "search", Arguments: map[string]any{"q": "test"}},
			Responses: []string{"Search complete."},
			MaxCalls:  intPtr(1),
		},
	}
	ts := newToolCallServer(t, rules...)
	defer ts.Close()

	body := `{
		"model": "claude-3",
		"max_tokens": 1024,
		"messages": [{"role": "user", "content": "search for info"}],
		"tools": [{"name": "search", "input_schema": {"type": "object"}}]
	}`

	// First request: tool call.
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	var result llmock.AnthropicResponse
	json.NewDecoder(resp.Body).Decode(&result)
	resp.Body.Close()
	if result.StopReason != "tool_use" {
		t.Fatalf("first request: expected stop_reason 'tool_use', got %q", result.StopReason)
	}

	// Second request: text fallback.
	resp2, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp2.Body.Close()
	var result2 llmock.AnthropicResponse
	json.NewDecoder(resp2.Body).Decode(&result2)
	if result2.StopReason != "end_turn" {
		t.Fatalf("second request: expected stop_reason 'end_turn', got %q", result2.StopReason)
	}
	if len(result2.Content) == 0 || result2.Content[0].Text != "Search complete." {
		t.Errorf("expected 'Search complete.', got %+v", result2.Content)
	}
}

func TestToolCall_MaxCalls_ConfigParsing(t *testing.T) {
	yamlData := []byte(`
rules:
  - pattern: ".*search.*"
    tool_call:
      name: "search_kb"
      arguments:
        query: "${input}"
    responses:
      - "Here are the results."
    max_calls: 1
`)

	cfg, err := llmock.ParseConfig(yamlData, "test.yaml")
	if err != nil {
		t.Fatal(err)
	}

	if len(cfg.Rules) != 1 {
		t.Fatalf("expected 1 rule, got %d", len(cfg.Rules))
	}
	if cfg.Rules[0].MaxCalls == nil {
		t.Fatal("expected max_calls to be set")
	}
	if *cfg.Rules[0].MaxCalls != 1 {
		t.Errorf("expected max_calls=1, got %d", *cfg.Rules[0].MaxCalls)
	}

	rules, err := llmock.CompileRules(cfg.Rules)
	if err != nil {
		t.Fatal(err)
	}
	if rules[0].MaxCalls == nil || *rules[0].MaxCalls != 1 {
		t.Fatal("expected compiled rule to have max_calls=1")
	}
}

func TestToolCall_MaxCalls_ResetRestoresCounter(t *testing.T) {
	// After admin reset, the counter should be restored and tool call fires again.
	rules := []llmock.Rule{
		{
			Pattern:   regexp.MustCompile(`.*search.*`),
			ToolCall:  &llmock.ToolCallConfig{Name: "search", Arguments: map[string]any{}},
			Responses: []string{"text fallback"},
			MaxCalls:  intPtr(1),
		},
	}
	ts := newToolCallServer(t, rules...)
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "search now"}],
		"tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}]
	}`

	// First request: tool call.
	resp, _ := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	var r1 llmock.ChatCompletionResponse
	json.NewDecoder(resp.Body).Decode(&r1)
	resp.Body.Close()
	if r1.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("expected 'tool_calls', got %q", r1.Choices[0].FinishReason)
	}

	// Second request: text (exhausted).
	resp2, _ := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	var r2 llmock.ChatCompletionResponse
	json.NewDecoder(resp2.Body).Decode(&r2)
	resp2.Body.Close()
	if r2.Choices[0].FinishReason != "stop" {
		t.Fatalf("expected 'stop', got %q", r2.Choices[0].FinishReason)
	}

	// Reset via admin API.
	req, _ := http.NewRequest("POST", ts.URL+"/_mock/reset", nil)
	http.DefaultClient.Do(req)

	// Third request: tool call fires again after reset.
	resp3, _ := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	var r3 llmock.ChatCompletionResponse
	json.NewDecoder(resp3.Body).Decode(&r3)
	resp3.Body.Close()
	if r3.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("after reset: expected 'tool_calls', got %q", r3.Choices[0].FinishReason)
	}
}
