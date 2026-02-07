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

func newGeminiEchoServer(t *testing.T) *httptest.Server {
	t.Helper()
	s := llmock.New(llmock.WithResponder(llmock.EchoResponder{}))
	return httptest.NewServer(s.Handler())
}

func TestGemini_EchoesLastUserMessage(t *testing.T) {
	ts := newGeminiEchoServer(t)
	defer ts.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "Hello, Gemini!"}]}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var result llmock.GeminiResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}

	if len(result.Candidates) != 1 {
		t.Fatalf("expected 1 candidate, got %d", len(result.Candidates))
	}

	candidate := result.Candidates[0]
	if candidate.Content.Role != "model" {
		t.Errorf("expected role 'model', got %q", candidate.Content.Role)
	}
	if len(candidate.Content.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(candidate.Content.Parts))
	}
	if candidate.Content.Parts[0].Text != "Hello, Gemini!" {
		t.Errorf("expected echoed text 'Hello, Gemini!', got %q", candidate.Content.Parts[0].Text)
	}
	if candidate.FinishReason != "STOP" {
		t.Errorf("expected finishReason 'STOP', got %q", candidate.FinishReason)
	}
}

func TestGemini_UsageMetadata(t *testing.T) {
	ts := newGeminiEchoServer(t)
	defer ts.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "one two three"}]}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.GeminiResponse
	json.NewDecoder(resp.Body).Decode(&result)

	if result.UsageMetadata.PromptTokenCount == 0 {
		t.Error("expected non-zero prompt token count")
	}
	if result.UsageMetadata.CandidatesTokenCount == 0 {
		t.Error("expected non-zero candidates token count")
	}
	if result.UsageMetadata.TotalTokenCount != result.UsageMetadata.PromptTokenCount+result.UsageMetadata.CandidatesTokenCount {
		t.Error("expected total = prompt + candidates")
	}
}

func TestGemini_EmptyContents(t *testing.T) {
	ts := newGeminiEchoServer(t)
	defer ts.Close()

	body := `{"contents": []}`
	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
}

func TestGemini_InvalidJSON(t *testing.T) {
	ts := newGeminiEchoServer(t)
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader("not json"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
}

func TestGemini_ModelExtractedFromPath(t *testing.T) {
	ts := newGeminiEchoServer(t)
	defer ts.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "test"}]}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-1.5-flash:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var result llmock.GeminiResponse
	json.NewDecoder(resp.Body).Decode(&result)
	if len(result.Candidates) != 1 {
		t.Fatalf("expected 1 candidate, got %d", len(result.Candidates))
	}
}

func TestGemini_SystemInstruction(t *testing.T) {
	ts := newGeminiEchoServer(t)
	defer ts.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "What is 2+2?"}]}
		],
		"systemInstruction": {
			"parts": [{"text": "You are a math tutor."}]
		}
	}`

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.GeminiResponse
	json.NewDecoder(resp.Body).Decode(&result)

	if result.Candidates[0].Content.Parts[0].Text != "What is 2+2?" {
		t.Errorf("expected 'What is 2+2?', got %q", result.Candidates[0].Content.Parts[0].Text)
	}
}

func TestGemini_DefaultRulesResponse(t *testing.T) {
	s := llmock.New()
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "hello"}]}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.GeminiResponse
	json.NewDecoder(resp.Body).Decode(&result)

	text := result.Candidates[0].Content.Parts[0].Text
	if text == "hello" {
		t.Error("expected rule-based response, got echo")
	}
	if text == "" {
		t.Error("expected non-empty response")
	}
}

func TestGemini_MultiTurnConversation(t *testing.T) {
	ts := newGeminiEchoServer(t)
	defer ts.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "first message"}]},
			{"role": "model", "parts": [{"text": "model response"}]},
			{"role": "user", "parts": [{"text": "second message"}]}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.GeminiResponse
	json.NewDecoder(resp.Body).Decode(&result)

	if result.Candidates[0].Content.Parts[0].Text != "second message" {
		t.Errorf("expected 'second message', got %q", result.Candidates[0].Content.Parts[0].Text)
	}
}

func TestGemini_StreamGenerateContent(t *testing.T) {
	s := llmock.New(llmock.WithResponder(llmock.EchoResponder{}), llmock.WithTokenDelay(0))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "Hello streaming world"}]}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:streamGenerateContent?alt=sse", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("expected text/event-stream, got %q", ct)
	}

	var fullText strings.Builder
	scanner := bufio.NewScanner(resp.Body)
	var lastChunk llmock.GeminiResponse
	chunks := 0

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		var chunk llmock.GeminiResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("failed to parse chunk: %v", err)
		}
		if len(chunk.Candidates) > 0 && len(chunk.Candidates[0].Content.Parts) > 0 {
			fullText.WriteString(chunk.Candidates[0].Content.Parts[0].Text)
		}
		lastChunk = chunk
		chunks++
	}

	if chunks == 0 {
		t.Fatal("expected at least 1 chunk")
	}

	reconstructed := fullText.String()
	if reconstructed != "Hello streaming world" {
		t.Errorf("expected 'Hello streaming world', got %q", reconstructed)
	}

	if lastChunk.Candidates[0].FinishReason != "STOP" {
		t.Errorf("expected last chunk finishReason 'STOP', got %q", lastChunk.Candidates[0].FinishReason)
	}
	if lastChunk.UsageMetadata.PromptTokenCount == 0 {
		t.Error("expected non-zero prompt token count in last chunk")
	}
}

func TestGemini_StreamNonStreamStillWorks(t *testing.T) {
	ts := newGeminiEchoServer(t)
	defer ts.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "non-stream test"}]}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var result llmock.GeminiResponse
	json.NewDecoder(resp.Body).Decode(&result)

	if result.Candidates[0].Content.Parts[0].Text != "non-stream test" {
		t.Errorf("expected 'non-stream test', got %q", result.Candidates[0].Content.Parts[0].Text)
	}
}

func TestGemini_ToolCallResponse(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern:  regexp.MustCompile(`weather`),
			ToolCall: &llmock.ToolCallConfig{Name: "get_weather", Arguments: map[string]any{"city": "London"}},
		},
	}
	s := llmock.New(llmock.WithRules(rules...))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "What is the weather?"}]}
		],
		"tools": [
			{
				"functionDeclarations": [
					{"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
				]
			}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.GeminiResponse
	json.NewDecoder(resp.Body).Decode(&result)

	if len(result.Candidates) != 1 {
		t.Fatalf("expected 1 candidate, got %d", len(result.Candidates))
	}

	parts := result.Candidates[0].Content.Parts
	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}
	if parts[0].FunctionCall == nil {
		t.Fatal("expected a function call part")
	}
	if parts[0].FunctionCall.Name != "get_weather" {
		t.Errorf("expected function name 'get_weather', got %q", parts[0].FunctionCall.Name)
	}
	if parts[0].FunctionCall.Args["city"] != "London" {
		t.Errorf("expected city 'London', got %v", parts[0].FunctionCall.Args["city"])
	}
}

func TestGemini_ToolCallFallthrough(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern:  regexp.MustCompile(`weather`),
			ToolCall: &llmock.ToolCallConfig{Name: "get_weather", Arguments: map[string]any{}},
		},
	}
	s := llmock.New(llmock.WithRules(rules...))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "What is the weather?"}]}
		],
		"tools": [
			{
				"functionDeclarations": [
					{"name": "search", "description": "Search the web"}
				]
			}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.GeminiResponse
	json.NewDecoder(resp.Body).Decode(&result)

	if len(result.Candidates) == 0 || len(result.Candidates[0].Content.Parts) == 0 {
		t.Fatal("expected non-empty response")
	}
	if result.Candidates[0].Content.Parts[0].FunctionCall != nil {
		t.Error("expected text response, got function call")
	}
}

func TestGemini_AutoToolCalls(t *testing.T) {
	s := llmock.New(
		llmock.WithResponder(llmock.EchoResponder{}),
		llmock.WithAutoToolCalls(true),
		llmock.WithSeed(42),
	)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "search for cats"}]}
		],
		"tools": [
			{
				"functionDeclarations": [
					{
						"name": "search",
						"description": "Search the web",
						"parameters": {
							"type": "object",
							"properties": {"query": {"type": "string"}},
							"required": ["query"]
						}
					}
				]
			}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.GeminiResponse
	json.NewDecoder(resp.Body).Decode(&result)

	parts := result.Candidates[0].Content.Parts
	if len(parts) == 0 {
		t.Fatal("expected at least 1 part")
	}
	if parts[0].FunctionCall == nil {
		t.Fatal("expected auto-generated function call")
	}
	if parts[0].FunctionCall.Name != "search" {
		t.Errorf("expected function name 'search', got %q", parts[0].FunctionCall.Name)
	}
}

func TestGemini_FunctionResponseMultiTurn(t *testing.T) {
	ts := newGeminiEchoServer(t)
	defer ts.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "What's the weather?"}]},
			{"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "London"}}}]},
			{"role": "user", "parts": [{"functionResponse": {"name": "get_weather", "response": {"result": "sunny, 22C"}}}]}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.GeminiResponse
	json.NewDecoder(resp.Body).Decode(&result)

	got := result.Candidates[0].Content.Parts[0].Text
	if got != "sunny, 22C" {
		t.Errorf("expected 'sunny, 22C', got %q", got)
	}
}

func TestGemini_StreamToolCall(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern:  regexp.MustCompile(`weather`),
			ToolCall: &llmock.ToolCallConfig{Name: "get_weather", Arguments: map[string]any{"city": "Paris"}},
		},
	}
	s := llmock.New(llmock.WithRules(rules...), llmock.WithTokenDelay(0))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "What is the weather?"}]}
		],
		"tools": [
			{
				"functionDeclarations": [
					{"name": "get_weather", "description": "Get weather"}
				]
			}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:streamGenerateContent?alt=sse", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("expected text/event-stream, got %q", ct)
	}

	scanner := bufio.NewScanner(resp.Body)
	var chunk llmock.GeminiResponse
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("failed to parse chunk: %v", err)
		}
	}

	parts := chunk.Candidates[0].Content.Parts
	if len(parts) == 0 || parts[0].FunctionCall == nil {
		t.Fatal("expected function call in stream chunk")
	}
	if parts[0].FunctionCall.Name != "get_weather" {
		t.Errorf("expected 'get_weather', got %q", parts[0].FunctionCall.Name)
	}
}

func TestGemini_ErrorFault(t *testing.T) {
	s := llmock.New(
		llmock.WithResponder(llmock.EchoResponder{}),
		llmock.WithFault(llmock.Fault{Type: llmock.FaultError, Status: 503, Message: "service unavailable"}),
	)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "test"}]}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 503 {
		t.Fatalf("expected 503, got %d", resp.StatusCode)
	}

	var result map[string]any
	json.NewDecoder(resp.Body).Decode(&result)

	errObj, ok := result["error"].(map[string]any)
	if !ok {
		t.Fatal("expected error object")
	}
	if errObj["message"] != "service unavailable" {
		t.Errorf("expected 'service unavailable', got %v", errObj["message"])
	}
	code, ok := errObj["code"].(float64)
	if !ok || int(code) != 503 {
		t.Errorf("expected code 503, got %v", errObj["code"])
	}
}

func TestGemini_RateLimitFault(t *testing.T) {
	s := llmock.New(
		llmock.WithResponder(llmock.EchoResponder{}),
		llmock.WithFault(llmock.Fault{Type: llmock.FaultRateLimit}),
	)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "test"}]}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 429 {
		t.Fatalf("expected 429, got %d", resp.StatusCode)
	}
	if resp.Header.Get("Retry-After") != "1" {
		t.Errorf("expected Retry-After header")
	}
}

func TestGemini_AdminRuleInjection(t *testing.T) {
	ts := newGeminiEchoServer(t)
	defer ts.Close()

	ruleBody := `{"rules": [{"pattern": "greet", "responses": ["Bonjour!"]}]}`
	resp, err := http.Post(ts.URL+"/_mock/rules", "application/json", strings.NewReader(ruleBody))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "please greet me"}]}
		]
	}`

	resp, err = http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.GeminiResponse
	json.NewDecoder(resp.Body).Decode(&result)

	if result.Candidates[0].Content.Parts[0].Text != "Bonjour!" {
		t.Errorf("expected 'Bonjour!', got %q", result.Candidates[0].Content.Parts[0].Text)
	}
}

func TestGemini_CrossEndpointConsistency(t *testing.T) {
	s := llmock.New(llmock.WithResponder(llmock.EchoResponder{}))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	openaiBody := `{"model": "test", "messages": [{"role": "user", "content": "consistency check"}]}`
	openaiResp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(openaiBody))
	if err != nil {
		t.Fatal(err)
	}
	defer openaiResp.Body.Close()
	var openaiResult llmock.ChatCompletionResponse
	json.NewDecoder(openaiResp.Body).Decode(&openaiResult)

	geminiBody := `{"contents": [{"role": "user", "parts": [{"text": "consistency check"}]}]}`
	geminiResp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader(geminiBody))
	if err != nil {
		t.Fatal(err)
	}
	defer geminiResp.Body.Close()
	var geminiResult llmock.GeminiResponse
	json.NewDecoder(geminiResp.Body).Decode(&geminiResult)

	openaiText := openaiResult.Choices[0].Message.Content
	geminiText := geminiResult.Candidates[0].Content.Parts[0].Text

	if openaiText != geminiText {
		t.Errorf("expected same response from both endpoints, got OpenAI=%q, Gemini=%q", openaiText, geminiText)
	}
}

func TestGemini_RequestLogging(t *testing.T) {
	ts := newGeminiEchoServer(t)
	defer ts.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "log this request"}]}
		]
	}`

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	resp, err = http.Get(ts.URL + "/_mock/requests")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var logResult struct {
		Requests []struct {
			Path        string `json:"path"`
			UserMessage string `json:"user_message"`
		} `json:"requests"`
	}
	json.NewDecoder(resp.Body).Decode(&logResult)

	found := false
	for _, r := range logResult.Requests {
		if strings.Contains(r.Path, "generateContent") && r.UserMessage == "log this request" {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected Gemini request in admin request log")
	}
}

func TestGemini_StreamEmptyContents(t *testing.T) {
	ts := newGeminiEchoServer(t)
	defer ts.Close()

	body := `{"contents": []}`
	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:streamGenerateContent?alt=sse", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
}

func TestGemini_StreamInvalidJSON(t *testing.T) {
	ts := newGeminiEchoServer(t)
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:streamGenerateContent?alt=sse", "application/json", strings.NewReader("not json"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
}

func TestGemini_ConfigParsing(t *testing.T) {
	yamlData := []byte(`
rules:
  - pattern: "hello"
    responses: ["Hi from config!"]
`)
	cfg, err := llmock.ParseConfig(yamlData, "test.yaml")
	if err != nil {
		t.Fatal(err)
	}
	opts, err := cfg.ToOptions()
	if err != nil {
		t.Fatal(err)
	}
	s := llmock.New(opts...)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{
		"contents": [
			{"role": "user", "parts": [{"text": "hello there"}]}
		]
	}`
	resp, err := http.Post(ts.URL+"/v1beta/models/gemini-pro:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.GeminiResponse
	json.NewDecoder(resp.Body).Decode(&result)

	if result.Candidates[0].Content.Parts[0].Text != "Hi from config!" {
		t.Errorf("expected 'Hi from config!', got %q", result.Candidates[0].Content.Parts[0].Text)
	}
}
