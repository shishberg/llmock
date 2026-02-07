package llmock_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/shishberg/llmock"
)

// newAdminServer creates a test server with rules and admin API enabled (default).
func newAdminServer(t *testing.T, rules ...llmock.Rule) *httptest.Server {
	t.Helper()
	s := llmock.New(llmock.WithRules(rules...))
	return httptest.NewServer(s.Handler())
}

func TestAdmin_InjectRule_MatchesRequest(t *testing.T) {
	ts := newAdminServer(t,
		llmock.Rule{Pattern: regexp.MustCompile(`^hello$`), Responses: []string{"original"}},
	)
	defer ts.Close()

	// Inject a new rule that matches "deploy".
	body := `{"rules":[{"pattern":".*deploy.*","responses":["Deploying now..."]}]}`
	resp, err := http.Post(ts.URL+"/_mock/rules", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("expected 201, got %d", resp.StatusCode)
	}

	// Send a request matching the injected rule.
	result := chatRequest(t, ts, "please deploy my app")
	if result.Choices[0].Message.Content != "Deploying now..." {
		t.Errorf("expected 'Deploying now...', got %q", result.Choices[0].Message.Content)
	}
}

func TestAdmin_InjectRule_PrependedByDefault(t *testing.T) {
	ts := newAdminServer(t,
		llmock.Rule{Pattern: regexp.MustCompile(`.*`), Responses: []string{"catchall"}},
	)
	defer ts.Close()

	// Inject a more specific rule; should be prepended (higher priority).
	body := `{"rules":[{"pattern":"^specific$","responses":["specific response"]}]}`
	resp, err := http.Post(ts.URL+"/_mock/rules", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	result := chatRequest(t, ts, "specific")
	if result.Choices[0].Message.Content != "specific response" {
		t.Errorf("expected 'specific response', got %q", result.Choices[0].Message.Content)
	}
}

func TestAdmin_InjectRule_AppendPriority(t *testing.T) {
	ts := newAdminServer(t,
		llmock.Rule{Pattern: regexp.MustCompile(`.*`), Responses: []string{"catchall"}},
	)
	defer ts.Close()

	// Inject a rule with priority -1 (append); catchall should still win.
	body := `{"rules":[{"pattern":".*","responses":["appended"], "priority":-1}]}`
	resp, err := http.Post(ts.URL+"/_mock/rules", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	result := chatRequest(t, ts, "anything")
	if result.Choices[0].Message.Content != "catchall" {
		t.Errorf("expected 'catchall' (original first), got %q", result.Choices[0].Message.Content)
	}
}

func TestAdmin_GetRules(t *testing.T) {
	ts := newAdminServer(t,
		llmock.Rule{Pattern: regexp.MustCompile(`^hello$`), Responses: []string{"hi"}},
		llmock.Rule{Pattern: regexp.MustCompile(`.*`), Responses: []string{"fallback"}},
	)
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/_mock/rules")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result struct {
		Rules []struct {
			Pattern   string   `json:"pattern"`
			Responses []string `json:"responses"`
		} `json:"rules"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}

	if len(result.Rules) != 2 {
		t.Fatalf("expected 2 rules, got %d", len(result.Rules))
	}
	if result.Rules[0].Pattern != "^hello$" {
		t.Errorf("expected first rule pattern '^hello$', got %q", result.Rules[0].Pattern)
	}
}

func TestAdmin_DeleteRules_ResetsToInitial(t *testing.T) {
	ts := newAdminServer(t,
		llmock.Rule{Pattern: regexp.MustCompile(`^hello$`), Responses: []string{"original"}},
	)
	defer ts.Close()

	// Inject a rule.
	body := `{"rules":[{"pattern":".*deploy.*","responses":["Deploying..."]}]}`
	resp, err := http.Post(ts.URL+"/_mock/rules", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	// Verify injected rule works.
	result := chatRequest(t, ts, "deploy")
	if result.Choices[0].Message.Content != "Deploying..." {
		t.Fatalf("expected injected rule to match, got %q", result.Choices[0].Message.Content)
	}

	// Delete rules (reset to initial).
	req, _ := http.NewRequest(http.MethodDelete, ts.URL+"/_mock/rules", nil)
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	// Injected rule should no longer match.
	result = chatRequest(t, ts, "deploy")
	if result.Choices[0].Message.Content == "Deploying..." {
		t.Error("expected injected rule to no longer match after reset")
	}
}

func TestAdmin_FullReset(t *testing.T) {
	ts := newAdminServer(t,
		llmock.Rule{Pattern: regexp.MustCompile(`.*`), Responses: []string{"matched"}},
	)
	defer ts.Close()

	// Send a request to populate the log.
	chatRequest(t, ts, "test")

	// Inject a rule.
	body := `{"rules":[{"pattern":"^injected$","responses":["injected response"]}]}`
	resp, err := http.Post(ts.URL+"/_mock/rules", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	// Full reset.
	resp, err = http.Post(ts.URL+"/_mock/reset", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	// Request log should be empty.
	resp, err = http.Get(ts.URL + "/_mock/requests")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var reqLog struct {
		Requests []json.RawMessage `json:"requests"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&reqLog); err != nil {
		t.Fatal(err)
	}
	if len(reqLog.Requests) != 0 {
		t.Errorf("expected empty request log after full reset, got %d entries", len(reqLog.Requests))
	}

	// Injected rule should no longer match (verify via GET /_mock/rules count).
	resp2, err := http.Get(ts.URL + "/_mock/rules")
	if err != nil {
		t.Fatal(err)
	}
	defer resp2.Body.Close()

	var rulesResp struct {
		Rules []json.RawMessage `json:"rules"`
	}
	json.NewDecoder(resp2.Body).Decode(&rulesResp)
	if len(rulesResp.Rules) != 1 {
		t.Errorf("expected 1 rule after reset (original), got %d", len(rulesResp.Rules))
	}
}

func TestAdmin_RequestLog(t *testing.T) {
	ts := newAdminServer(t,
		llmock.Rule{Pattern: regexp.MustCompile(`^hello$`), Responses: []string{"hi there"}},
	)
	defer ts.Close()

	// Send a matching request.
	chatRequest(t, ts, "hello")

	// Check the request log.
	resp, err := http.Get(ts.URL + "/_mock/requests")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result struct {
		Requests []struct {
			Method      string `json:"method"`
			Path        string `json:"path"`
			UserMessage string `json:"user_message"`
			MatchedRule string `json:"matched_rule"`
			Response    string `json:"response"`
		} `json:"requests"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}

	if len(result.Requests) != 1 {
		t.Fatalf("expected 1 request log entry, got %d", len(result.Requests))
	}

	entry := result.Requests[0]
	if entry.Method != "POST" {
		t.Errorf("expected method POST, got %q", entry.Method)
	}
	if entry.Path != "/v1/chat/completions" {
		t.Errorf("expected path '/v1/chat/completions', got %q", entry.Path)
	}
	if entry.UserMessage != "hello" {
		t.Errorf("expected user_message 'hello', got %q", entry.UserMessage)
	}
	if entry.MatchedRule != "^hello$" {
		t.Errorf("expected matched_rule '^hello$', got %q", entry.MatchedRule)
	}
	if entry.Response != "hi there" {
		t.Errorf("expected response 'hi there', got %q", entry.Response)
	}
}

func TestAdmin_DeleteRequests(t *testing.T) {
	ts := newAdminServer(t,
		llmock.Rule{Pattern: regexp.MustCompile(`.*`), Responses: []string{"response"}},
	)
	defer ts.Close()

	chatRequest(t, ts, "test")

	// Clear the log.
	req, _ := http.NewRequest(http.MethodDelete, ts.URL+"/_mock/requests", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	// Verify log is empty.
	resp, err = http.Get(ts.URL + "/_mock/requests")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result struct {
		Requests []json.RawMessage `json:"requests"`
	}
	json.NewDecoder(resp.Body).Decode(&result)
	if len(result.Requests) != 0 {
		t.Errorf("expected empty request log, got %d entries", len(result.Requests))
	}
}

func TestAdmin_WithAdminAPIDisabled(t *testing.T) {
	s := llmock.New(
		llmock.WithRules(llmock.Rule{Pattern: regexp.MustCompile(`.*`), Responses: []string{"test"}}),
		llmock.WithAdminAPI(false),
	)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// Admin endpoints should 404.
	resp, err := http.Get(ts.URL + "/_mock/rules")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("expected 404 when admin disabled, got %d", resp.StatusCode)
	}

	// Chat should still work.
	result := chatRequest(t, ts, "anything")
	if result.Choices[0].Message.Content != "test" {
		t.Errorf("expected 'test', got %q", result.Choices[0].Message.Content)
	}
}

func TestAdmin_InvalidRulePattern(t *testing.T) {
	ts := newAdminServer(t)
	defer ts.Close()

	body := `{"rules":[{"pattern":"[invalid","responses":["test"]}]}`
	resp, err := http.Post(ts.URL+"/_mock/rules", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400 for invalid regex, got %d", resp.StatusCode)
	}
}

func TestAdmin_EmptyRulesArray(t *testing.T) {
	ts := newAdminServer(t)
	defer ts.Close()

	body := `{"rules":[]}`
	resp, err := http.Post(ts.URL+"/_mock/rules", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400 for empty rules, got %d", resp.StatusCode)
	}
}

func TestAdmin_RuleWithNoResponses(t *testing.T) {
	ts := newAdminServer(t)
	defer ts.Close()

	body := `{"rules":[{"pattern":".*","responses":[]}]}`
	resp, err := http.Post(ts.URL+"/_mock/rules", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400 for empty responses, got %d", resp.StatusCode)
	}
}

func TestAdmin_RequestLogLimit100(t *testing.T) {
	ts := newAdminServer(t,
		llmock.Rule{Pattern: regexp.MustCompile(`.*`), Responses: []string{"ok"}},
	)
	defer ts.Close()

	// Send 110 requests.
	for i := 0; i < 110; i++ {
		chatRequest(t, ts, "test")
	}

	resp, err := http.Get(ts.URL + "/_mock/requests")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result struct {
		Requests []json.RawMessage `json:"requests"`
	}
	json.NewDecoder(resp.Body).Decode(&result)
	if len(result.Requests) != 100 {
		t.Errorf("expected request log capped at 100, got %d", len(result.Requests))
	}
}

func TestAdmin_DefaultServerHasAdminEndpoints(t *testing.T) {
	// A server with no explicit options should have admin endpoints.
	s := llmock.New()
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/_mock/rules")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected 200 for admin endpoint on default server, got %d", resp.StatusCode)
	}

	// Default rules should produce a non-empty response.
	result := chatRequest(t, ts, "Hello, world!")
	if result.Choices[0].Message.Content == "" {
		t.Error("expected non-empty response from default server")
	}
}

func TestAdmin_InjectRuleOnDefaultServer(t *testing.T) {
	// Default server should allow rule injection.
	s := llmock.New()
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// Inject a rule.
	body := `{"rules":[{"pattern":".*error.*","responses":["Something went wrong"]}]}`
	resp, err := http.Post(ts.URL+"/_mock/rules", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	// Now it should match the injected rule.
	result := chatRequest(t, ts, "trigger error please")
	if result.Choices[0].Message.Content != "Something went wrong" {
		t.Errorf("expected 'Something went wrong', got %q", result.Choices[0].Message.Content)
	}

	// Non-matching input should fall back to default rules (not echo).
	result = chatRequest(t, ts, "normal message")
	if result.Choices[0].Message.Content == "" {
		t.Error("expected non-empty response for non-matching input")
	}
}

func TestAdmin_ConcurrentAccess(t *testing.T) {
	ts := newAdminServer(t,
		llmock.Rule{Pattern: regexp.MustCompile(`.*`), Responses: []string{"ok"}},
	)
	defer ts.Close()

	// Run concurrent requests and rule injections.
	done := make(chan struct{})
	for i := 0; i < 10; i++ {
		go func() {
			defer func() { done <- struct{}{} }()
			for j := 0; j < 10; j++ {
				chatRequest(t, ts, "concurrent test")
			}
		}()
	}
	for i := 0; i < 5; i++ {
		go func() {
			defer func() { done <- struct{}{} }()
			body := `{"rules":[{"pattern":"^concurrent$","responses":["found"]}]}`
			resp, err := http.Post(ts.URL+"/_mock/rules", "application/json", strings.NewReader(body))
			if err != nil {
				return
			}
			resp.Body.Close()
		}()
	}

	for i := 0; i < 15; i++ {
		<-done
	}
	// If we get here without data races or panics, the test passes.
}
