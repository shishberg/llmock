package llmock_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"regexp"
	"strings"
	"testing"

	"github.com/shishberg/llmock"
)

func newTestServerWithRules(t *testing.T, rules ...llmock.Rule) *httptest.Server {
	t.Helper()
	s := llmock.New(llmock.WithRules(rules...))
	return httptest.NewServer(s.Handler())
}

func chatRequest(t *testing.T, ts *httptest.Server, content string) llmock.ChatCompletionResponse {
	t.Helper()
	body := `{"model":"test","messages":[{"role":"user","content":` + jsonString(content) + `}]}`
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
	return result
}

func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func TestRules_MatchPriority(t *testing.T) {
	rules := []llmock.Rule{
		{Pattern: regexp.MustCompile(`^hello$`), Responses: []string{"first rule"}},
		{Pattern: regexp.MustCompile(`hello`), Responses: []string{"second rule"}},
		{Pattern: regexp.MustCompile(`.*`), Responses: []string{"catchall"}},
	}
	ts := newTestServerWithRules(t, rules...)
	defer ts.Close()

	// Exact match should hit first rule, not second.
	result := chatRequest(t, ts, "hello")
	if result.Choices[0].Message.Content != "first rule" {
		t.Errorf("expected 'first rule', got %q", result.Choices[0].Message.Content)
	}

	// Partial match should hit second rule.
	result = chatRequest(t, ts, "say hello there")
	if result.Choices[0].Message.Content != "second rule" {
		t.Errorf("expected 'second rule', got %q", result.Choices[0].Message.Content)
	}

	// Non-matching should hit catchall.
	result = chatRequest(t, ts, "something else")
	if result.Choices[0].Message.Content != "catchall" {
		t.Errorf("expected 'catchall', got %q", result.Choices[0].Message.Content)
	}
}

func TestRules_CaptureGroupSubstitution(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern:   regexp.MustCompile(`deploy (.*) to (.*)`),
			Responses: []string{"Deploying $1 to $2 now."},
		},
	}
	ts := newTestServerWithRules(t, rules...)
	defer ts.Close()

	result := chatRequest(t, ts, "deploy myapp to production")
	expected := "Deploying myapp to production now."
	if result.Choices[0].Message.Content != expected {
		t.Errorf("expected %q, got %q", expected, result.Choices[0].Message.Content)
	}
}

func TestRules_InputTemplateExpansion(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern:   regexp.MustCompile(`.*`),
			Responses: []string{"You said: ${input}"},
		},
	}
	ts := newTestServerWithRules(t, rules...)
	defer ts.Close()

	result := chatRequest(t, ts, "testing 123")
	expected := "You said: testing 123"
	if result.Choices[0].Message.Content != expected {
		t.Errorf("expected %q, got %q", expected, result.Choices[0].Message.Content)
	}
}

func TestRules_NoMatchFallback(t *testing.T) {
	rules := []llmock.Rule{
		{Pattern: regexp.MustCompile(`^only this$`), Responses: []string{"matched"}},
	}
	ts := newTestServerWithRules(t, rules...)
	defer ts.Close()

	result := chatRequest(t, ts, "something else entirely")
	expected := "That's an interesting point. Could you tell me more?"
	if result.Choices[0].Message.Content != expected {
		t.Errorf("expected fallback %q, got %q", expected, result.Choices[0].Message.Content)
	}
}

func TestRules_RandomSelectionAmongTemplates(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern:   regexp.MustCompile(`.*`),
			Responses: []string{"response A", "response B", "response C"},
		},
	}
	ts := newTestServerWithRules(t, rules...)
	defer ts.Close()

	seen := make(map[string]bool)
	for i := 0; i < 50; i++ {
		result := chatRequest(t, ts, "test")
		content := result.Choices[0].Message.Content
		if content != "response A" && content != "response B" && content != "response C" {
			t.Fatalf("unexpected response %q", content)
		}
		seen[content] = true
	}
	// With 50 attempts and 3 options, we should see at least 2 different responses.
	if len(seen) < 2 {
		t.Errorf("expected multiple different responses from random selection, got only: %v", seen)
	}
}

func TestRules_DefaultRulesApplied(t *testing.T) {
	// WithRules() with no args should use default rules.
	ts := newTestServerWithRules(t)
	defer ts.Close()

	result := chatRequest(t, ts, "hello")
	content := result.Choices[0].Message.Content
	// Default greeting rules should produce a greeting-like response.
	if !strings.Contains(content, "Hello") && !strings.Contains(content, "Hi") && !strings.Contains(content, "Hey") {
		t.Errorf("expected greeting response from default rules, got %q", content)
	}
}

func TestRules_WorksWithAnthropicEndpoint(t *testing.T) {
	rules := []llmock.Rule{
		{
			Pattern:   regexp.MustCompile(`what is (.*)`),
			Responses: []string{"$1 is a concept."},
		},
	}
	s := llmock.New(llmock.WithRules(rules...))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{"model":"claude","max_tokens":1024,"messages":[{"role":"user","content":"what is Go"}]}`
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result llmock.AnthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}

	if result.Content[0].Text != "Go is a concept." {
		t.Errorf("expected 'Go is a concept.', got %q", result.Content[0].Text)
	}
}

func TestRules_WithResponderOption(t *testing.T) {
	rules := []llmock.Rule{
		{Pattern: regexp.MustCompile(`.*`), Responses: []string{"custom"}},
	}
	responder := llmock.NewRuleResponder(rules)
	s := llmock.New(llmock.WithResponder(responder))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	result := chatRequest(t, ts, "anything")
	if result.Choices[0].Message.Content != "custom" {
		t.Errorf("expected 'custom', got %q", result.Choices[0].Message.Content)
	}
}

func TestParseRulesYAML(t *testing.T) {
	yamlData := []byte(`
rules:
  - pattern: "deploy (.*) to (.*)"
    responses:
      - "Deploying $1 to $2."
      - "Starting deployment of $1 to $2."
  - pattern: ".*"
    responses:
      - "Catchall response."
`)

	rules, err := llmock.ParseRulesYAML(yamlData)
	if err != nil {
		t.Fatal(err)
	}
	if len(rules) != 2 {
		t.Fatalf("expected 2 rules, got %d", len(rules))
	}
	if rules[0].Pattern.String() != "deploy (.*) to (.*)" {
		t.Errorf("unexpected pattern: %s", rules[0].Pattern)
	}
	if len(rules[0].Responses) != 2 {
		t.Errorf("expected 2 responses, got %d", len(rules[0].Responses))
	}
}

func TestParseRulesYAML_InvalidRegex(t *testing.T) {
	yamlData := []byte(`
rules:
  - pattern: "[invalid"
    responses:
      - "test"
`)
	_, err := llmock.ParseRulesYAML(yamlData)
	if err == nil {
		t.Fatal("expected error for invalid regex")
	}
}

func TestParseRulesYAML_NoResponses(t *testing.T) {
	yamlData := []byte(`
rules:
  - pattern: ".*"
    responses: []
`)
	_, err := llmock.ParseRulesYAML(yamlData)
	if err == nil {
		t.Fatal("expected error for empty responses")
	}
}

func TestLoadRulesFile(t *testing.T) {
	content := `
rules:
  - pattern: "test (.*)"
    responses:
      - "Testing $1."
`
	f, err := os.CreateTemp("", "rules-*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())
	if _, err := f.WriteString(content); err != nil {
		t.Fatal(err)
	}
	f.Close()

	rules, err := llmock.LoadRulesFile(f.Name())
	if err != nil {
		t.Fatal(err)
	}
	if len(rules) != 1 {
		t.Fatalf("expected 1 rule, got %d", len(rules))
	}
}

func TestLoadRulesFile_FileNotFound(t *testing.T) {
	_, err := llmock.LoadRulesFile("/nonexistent/rules.yaml")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestExpandTemplate_DollarSignWithoutCapture(t *testing.T) {
	// A $ not followed by a digit or {input} should be kept as-is.
	rules := []llmock.Rule{
		{Pattern: regexp.MustCompile(`(.*)`), Responses: []string{"Cost: $5 for $1"}},
	}
	ts := newTestServerWithRules(t, rules...)
	defer ts.Close()

	result := chatRequest(t, ts, "item")
	if result.Choices[0].Message.Content != "Cost: $5 for item" {
		t.Errorf("expected 'Cost: $5 for item', got %q", result.Choices[0].Message.Content)
	}
}
