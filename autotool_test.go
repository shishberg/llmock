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

func newAutoToolServer(t *testing.T, opts ...llmock.Option) *httptest.Server {
	t.Helper()
	defaults := []llmock.Option{llmock.WithAutoToolCalls(true), llmock.WithSeed(42)}
	s := llmock.New(append(defaults, opts...)...)
	return httptest.NewServer(s.Handler())
}

func TestAutoTool_OpenAI_GeneratesToolCall(t *testing.T) {
	ts := newAutoToolServer(t)
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "hello"}],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "get_weather",
					"description": "Get the current weather",
					"parameters": {
						"type": "object",
						"properties": {
							"location": {"type": "string", "description": "The city name"},
							"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
						},
						"required": ["location"]
					}
				}
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

	if len(result.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(result.Choices))
	}
	choice := result.Choices[0]
	if choice.FinishReason != "tool_calls" {
		t.Errorf("expected finish_reason 'tool_calls', got %q", choice.FinishReason)
	}
	if len(choice.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(choice.Message.ToolCalls))
	}

	tc := choice.Message.ToolCalls[0]
	if tc.Function.Name != "get_weather" {
		t.Errorf("expected function name 'get_weather', got %q", tc.Function.Name)
	}
	if !strings.HasPrefix(tc.ID, "call_") {
		t.Errorf("expected ID prefix 'call_', got %q", tc.ID)
	}

	var args map[string]any
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		t.Fatalf("failed to parse arguments: %v", err)
	}
	// location is required and should be present.
	if _, ok := args["location"]; !ok {
		t.Error("expected 'location' in arguments")
	}
	// location should be a string.
	if _, ok := args["location"].(string); !ok {
		t.Errorf("expected 'location' to be a string, got %T", args["location"])
	}
	// If unit is present, it should be from the enum.
	if unit, ok := args["unit"]; ok {
		unitStr, ok := unit.(string)
		if !ok {
			t.Errorf("expected unit to be a string, got %T", unit)
		} else if unitStr != "celsius" && unitStr != "fahrenheit" {
			t.Errorf("expected unit from enum, got %q", unitStr)
		}
	}
}

func TestAutoTool_Anthropic_GeneratesToolCall(t *testing.T) {
	ts := newAutoToolServer(t)
	defer ts.Close()

	body := `{
		"model": "claude-3-opus",
		"max_tokens": 1024,
		"messages": [{"role": "user", "content": "hello"}],
		"tools": [
			{
				"name": "search_files",
				"description": "Search for files",
				"input_schema": {
					"type": "object",
					"properties": {
						"query": {"type": "string"},
						"max_results": {"type": "integer"}
					},
					"required": ["query"]
				}
			}
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

	if result.StopReason != "tool_use" {
		t.Errorf("expected stop_reason 'tool_use', got %q", result.StopReason)
	}
	if len(result.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(result.Content))
	}

	block := result.Content[0]
	if block.Type != "tool_use" {
		t.Errorf("expected type 'tool_use', got %q", block.Type)
	}
	if block.Name != "search_files" {
		t.Errorf("expected name 'search_files', got %q", block.Name)
	}
	if !strings.HasPrefix(block.ID, "toolu_") {
		t.Errorf("expected ID prefix 'toolu_', got %q", block.ID)
	}
	if _, ok := block.Input["query"]; !ok {
		t.Error("expected 'query' in input")
	}
}

func TestAutoTool_Disabled_ReturnsTextResponse(t *testing.T) {
	// Without WithAutoToolCalls, tools in request should not trigger auto-generation.
	s := llmock.New(llmock.WithSeed(42))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "hello"}],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "get_weather",
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

	var result llmock.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}

	// Should be a text response, not a tool call.
	if result.Choices[0].FinishReason != "stop" {
		t.Errorf("expected finish_reason 'stop', got %q", result.Choices[0].FinishReason)
	}
}

func TestAutoTool_RuleMatchTakesPrecedence(t *testing.T) {
	// When a rule matches with a tool call, auto-generation should not override it.
	rules := []llmock.Rule{
		{
			Pattern: regexp.MustCompile(`.*weather.*`),
			ToolCall: &llmock.ToolCallConfig{
				Name:      "get_weather",
				Arguments: map[string]any{"location": "ExplicitCity"},
			},
		},
	}
	ts := newAutoToolServer(t, llmock.WithRules(rules...))
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "What's the weather?"}],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "get_weather",
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

	var result llmock.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}

	tc := result.Choices[0].Message.ToolCalls[0]
	var args map[string]any
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		t.Fatal(err)
	}
	// Should use the explicit value from the rule, not auto-generated.
	if args["location"] != "ExplicitCity" {
		t.Errorf("expected location 'ExplicitCity', got %v", args["location"])
	}
}

func TestAutoTool_MultipleTools_PicksOne(t *testing.T) {
	ts := newAutoToolServer(t)
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "do something"}],
		"tools": [
			{
				"type": "function",
				"function": {"name": "tool_a", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}
			},
			{
				"type": "function",
				"function": {"name": "tool_b", "parameters": {"type": "object", "properties": {"y": {"type": "integer"}}}}
			},
			{
				"type": "function",
				"function": {"name": "tool_c", "parameters": {"type": "object", "properties": {"z": {"type": "boolean"}}}}
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

	if result.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("expected tool_calls, got %q", result.Choices[0].FinishReason)
	}

	tc := result.Choices[0].Message.ToolCalls[0]
	validNames := map[string]bool{"tool_a": true, "tool_b": true, "tool_c": true}
	if !validNames[tc.Function.Name] {
		t.Errorf("expected one of tool_a/b/c, got %q", tc.Function.Name)
	}
}

func TestAutoTool_ComplexSchema(t *testing.T) {
	ts := newAutoToolServer(t)
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "test"}],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "create_event",
					"parameters": {
						"type": "object",
						"properties": {
							"title": {"type": "string"},
							"date": {"type": "string", "format": "date"},
							"attendees": {
								"type": "array",
								"items": {"type": "string"}
							},
							"recurring": {"type": "boolean"},
							"priority": {"type": "integer"},
							"metadata": {
								"type": "object",
								"properties": {
									"category": {"type": "string", "enum": ["work", "personal"]},
									"tags": {"type": "array", "items": {"type": "string"}}
								}
							}
						},
						"required": ["title", "date"]
					}
				}
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

	tc := result.Choices[0].Message.ToolCalls[0]
	if tc.Function.Name != "create_event" {
		t.Fatalf("expected 'create_event', got %q", tc.Function.Name)
	}

	var args map[string]any
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		t.Fatal(err)
	}

	// Required fields must be present.
	if _, ok := args["title"]; !ok {
		t.Error("expected 'title' in arguments")
	}
	if _, ok := args["date"]; !ok {
		t.Error("expected 'date' in arguments")
	}
	// Date should look like a date (YYYY-MM-DD).
	if date, ok := args["date"].(string); ok {
		if len(date) != 10 || date[4] != '-' || date[7] != '-' {
			t.Errorf("expected date format YYYY-MM-DD, got %q", date)
		}
	}
}

func TestAutoTool_NoToolsInRequest_NoAutoGeneration(t *testing.T) {
	ts := newAutoToolServer(t)
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "hello"}]
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

	// No tools in request, so should be text response even with auto enabled.
	if result.Choices[0].FinishReason != "stop" {
		t.Errorf("expected 'stop', got %q", result.Choices[0].FinishReason)
	}
}

func TestAutoTool_Streaming_OpenAI(t *testing.T) {
	ts := newAutoToolServer(t)
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"stream": true,
		"messages": [{"role": "user", "content": "hello"}],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "greet",
					"parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
				}
			}
		]
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

	// Last event should have finish_reason "tool_calls".
	lastEvent := events[len(events)-1]
	choices := lastEvent["choices"].([]any)
	lastChoice := choices[0].(map[string]any)
	if lastChoice["finish_reason"] != "tool_calls" {
		t.Errorf("expected finish_reason 'tool_calls', got %v", lastChoice["finish_reason"])
	}

	// First event should contain the function name.
	firstEvent := events[0]
	firstChoices := firstEvent["choices"].([]any)
	firstDelta := firstChoices[0].(map[string]any)["delta"].(map[string]any)
	toolCalls := firstDelta["tool_calls"].([]any)
	firstTC := toolCalls[0].(map[string]any)
	fn := firstTC["function"].(map[string]any)
	if fn["name"] != "greet" {
		t.Errorf("expected function name 'greet', got %v", fn["name"])
	}
}

func TestAutoTool_ConfigParsing(t *testing.T) {
	yamlData := []byte(`
defaults:
  auto_tool_calls: true
  seed: 42
rules:
  - pattern: ".*"
    responses: ["hello"]
`)

	cfg, err := llmock.ParseConfig(yamlData, "test.yaml")
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Defaults.AutoToolCalls == nil {
		t.Fatal("expected auto_tool_calls to be set")
	}
	if !*cfg.Defaults.AutoToolCalls {
		t.Error("expected auto_tool_calls to be true")
	}

	opts, err := cfg.ToOptions()
	if err != nil {
		t.Fatal(err)
	}

	// Should produce at least a seed, auto_tool_calls, and rules option.
	if len(opts) < 3 {
		t.Errorf("expected at least 3 options, got %d", len(opts))
	}
}

func TestAutoTool_EmptyParameters_GeneratesEmptyArgs(t *testing.T) {
	ts := newAutoToolServer(t)
	defer ts.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "ping"}],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "ping",
					"parameters": {}
				}
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

	if result.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("expected tool_calls, got %q", result.Choices[0].FinishReason)
	}

	tc := result.Choices[0].Message.ToolCalls[0]
	if tc.Function.Name != "ping" {
		t.Errorf("expected 'ping', got %q", tc.Function.Name)
	}
}
