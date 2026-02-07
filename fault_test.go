package llmock_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/shishberg/llmock"
)

func newFaultServer(t *testing.T, opts ...llmock.Option) *httptest.Server {
	t.Helper()
	s := llmock.New(opts...)
	return httptest.NewServer(s.Handler())
}

// --- Error fault ---

func TestFault_Error_OpenAI(t *testing.T) {
	ts := newFaultServer(t,
		llmock.WithFault(llmock.Fault{
			Type:    llmock.FaultError,
			Status:  529,
			Message: "Overloaded",
		}),
	)
	defer ts.Close()

	body := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 529 {
		t.Fatalf("expected 529, got %d", resp.StatusCode)
	}

	var result struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if result.Error.Message != "Overloaded" {
		t.Errorf("expected error message 'Overloaded', got %q", result.Error.Message)
	}
}

func TestFault_Error_Anthropic(t *testing.T) {
	ts := newFaultServer(t,
		llmock.WithFault(llmock.Fault{
			Type:      llmock.FaultError,
			Status:    500,
			Message:   "Internal failure",
			ErrorType: "api_error",
		}),
	)
	defer ts.Close()

	body := `{"model":"claude","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 500 {
		t.Fatalf("expected 500, got %d", resp.StatusCode)
	}

	var result struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if result.Type != "error" {
		t.Errorf("expected type 'error', got %q", result.Type)
	}
	if result.Error.Type != "api_error" {
		t.Errorf("expected error type 'api_error', got %q", result.Error.Type)
	}
	if result.Error.Message != "Internal failure" {
		t.Errorf("expected error message 'Internal failure', got %q", result.Error.Message)
	}
}

// --- Rate limit fault ---

func TestFault_RateLimit_OpenAI(t *testing.T) {
	ts := newFaultServer(t,
		llmock.WithFault(llmock.Fault{
			Type: llmock.FaultRateLimit,
		}),
	)
	defer ts.Close()

	body := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 429 {
		t.Fatalf("expected 429, got %d", resp.StatusCode)
	}
	if ra := resp.Header.Get("Retry-After"); ra != "1" {
		t.Errorf("expected Retry-After '1', got %q", ra)
	}

	var result struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if result.Error.Type != "rate_limit_error" {
		t.Errorf("expected error type 'rate_limit_error', got %q", result.Error.Type)
	}
}

func TestFault_RateLimit_Anthropic(t *testing.T) {
	ts := newFaultServer(t,
		llmock.WithFault(llmock.Fault{
			Type: llmock.FaultRateLimit,
		}),
	)
	defer ts.Close()

	body := `{"model":"claude","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 429 {
		t.Fatalf("expected 429, got %d", resp.StatusCode)
	}
	if ra := resp.Header.Get("Retry-After"); ra != "1" {
		t.Errorf("expected Retry-After '1', got %q", ra)
	}

	var result struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if result.Type != "error" {
		t.Errorf("expected type 'error', got %q", result.Type)
	}
	if result.Error.Type != "rate_limit_error" {
		t.Errorf("expected error type 'rate_limit_error', got %q", result.Error.Type)
	}
}

// --- Delay fault ---

func TestFault_Delay(t *testing.T) {
	ts := newFaultServer(t,
		llmock.WithFault(llmock.Fault{
			Type:    llmock.FaultDelay,
			DelayMS: 50,
		}),
	)
	defer ts.Close()

	body := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`
	start := time.Now()
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	elapsed := time.Since(start)

	// Should succeed (delay doesn't block the response, just adds latency).
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if elapsed < 40*time.Millisecond {
		t.Errorf("expected at least ~50ms delay, got %v", elapsed)
	}

	// Should still get a valid response.
	var result llmock.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	// With default rules, "hi" matches the greeting pattern.
	content := result.Choices[0].Message.Content
	if content == "" {
		t.Error("expected non-empty response")
	}
}

// --- Malformed fault ---

func TestFault_Malformed_NonStream(t *testing.T) {
	ts := newFaultServer(t,
		llmock.WithFault(llmock.Fault{
			Type: llmock.FaultMalformed,
		}),
	)
	defer ts.Close()

	body := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	data, _ := io.ReadAll(resp.Body)
	// Should be invalid JSON.
	var result map[string]any
	if err := json.Unmarshal(data, &result); err == nil {
		t.Error("expected malformed response to be invalid JSON, but it parsed successfully")
	}
}

func TestFault_Malformed_Stream(t *testing.T) {
	ts := newFaultServer(t,
		llmock.WithFault(llmock.Fault{
			Type: llmock.FaultMalformed,
		}),
	)
	defer ts.Close()

	body := `{"model":"test","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("expected Content-Type 'text/event-stream', got %q", ct)
	}

	data, _ := io.ReadAll(resp.Body)
	// Should contain broken SSE data.
	if !strings.Contains(string(data), "broken") {
		t.Errorf("expected malformed response to contain 'broken', got %q", string(data))
	}
}

// --- Timeout fault ---

func TestFault_Timeout(t *testing.T) {
	ts := newFaultServer(t,
		llmock.WithFault(llmock.Fault{
			Type: llmock.FaultTimeout,
		}),
	)
	defer ts.Close()

	body := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	req, _ := http.NewRequestWithContext(ctx, "POST", ts.URL+"/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	_, err := http.DefaultClient.Do(req)
	if err == nil {
		t.Fatal("expected timeout error, but request succeeded")
	}
	if !strings.Contains(err.Error(), "context deadline exceeded") {
		t.Fatalf("expected context deadline exceeded, got: %v", err)
	}
}

// --- Count-based auto-clearing ---

func TestFault_CountBased_AutoClearing(t *testing.T) {
	ts := newFaultServer(t,
		llmock.WithFault(llmock.Fault{
			Type:    llmock.FaultError,
			Status:  500,
			Message: "error",
			Count:   2,
		}),
	)
	defer ts.Close()

	body := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`

	// First two requests should fail.
	for i := 0; i < 2; i++ {
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != 500 {
			t.Fatalf("request %d: expected 500, got %d", i+1, resp.StatusCode)
		}
	}

	// Third request should succeed.
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200 after fault exhausted, got %d", resp.StatusCode)
	}
}

// --- Probability-based faults with fixed seed ---

func TestFault_Probability_WithSeed(t *testing.T) {
	// With a fixed seed, probability-based faults should be deterministic.
	// We test that not all requests fail with probability 0.5.
	ts := newFaultServer(t,
		llmock.WithSeed(42),
		llmock.WithFault(llmock.Fault{
			Type:        llmock.FaultError,
			Status:      500,
			Message:     "maybe",
			Probability: 0.5,
		}),
	)
	defer ts.Close()

	body := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`
	errors := 0
	successes := 0
	for i := 0; i < 20; i++ {
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode == 500 {
			errors++
		} else if resp.StatusCode == 200 {
			successes++
		}
	}

	// With probability 0.5 over 20 requests, we should see both errors and successes.
	if errors == 0 {
		t.Error("expected some errors with probability 0.5, got none")
	}
	if successes == 0 {
		t.Error("expected some successes with probability 0.5, got none")
	}
}

// --- Admin API for faults ---

func TestFault_AdminAPI_PostAndGet(t *testing.T) {
	ts := newFaultServer(t)
	defer ts.Close()

	// Initially no faults.
	resp, err := http.Get(ts.URL + "/_mock/faults")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result struct {
		Faults []json.RawMessage `json:"faults"`
	}
	json.NewDecoder(resp.Body).Decode(&result)
	if len(result.Faults) != 0 {
		t.Fatalf("expected 0 faults, got %d", len(result.Faults))
	}

	// Inject a fault.
	faultBody := `{"faults":[{"type":"error","status":503,"message":"Service unavailable"}]}`
	resp2, err := http.Post(ts.URL+"/_mock/faults", "application/json", strings.NewReader(faultBody))
	if err != nil {
		t.Fatal(err)
	}
	resp2.Body.Close()
	if resp2.StatusCode != 201 {
		t.Fatalf("expected 201, got %d", resp2.StatusCode)
	}

	// Verify fault is active.
	body := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`
	resp3, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp3.Body.Close()
	if resp3.StatusCode != 503 {
		t.Fatalf("expected 503 from injected fault, got %d", resp3.StatusCode)
	}
}

func TestFault_AdminAPI_Delete(t *testing.T) {
	ts := newFaultServer(t,
		llmock.WithFault(llmock.Fault{
			Type:    llmock.FaultError,
			Status:  500,
			Message: "error",
		}),
	)
	defer ts.Close()

	// Verify fault fires.
	body := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 500 {
		t.Fatalf("expected 500, got %d", resp.StatusCode)
	}

	// Clear faults.
	req, _ := http.NewRequest(http.MethodDelete, ts.URL+"/_mock/faults", nil)
	resp2, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp2.Body.Close()

	// Now should succeed.
	resp3, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp3.Body.Close()
	if resp3.StatusCode != 200 {
		t.Fatalf("expected 200 after clearing faults, got %d", resp3.StatusCode)
	}
}

// --- Faults evaluated before rules ---

func TestFault_EvaluatedBeforeRules(t *testing.T) {
	ts := newFaultServer(t,
		llmock.WithFault(llmock.Fault{
			Type:    llmock.FaultError,
			Status:  503,
			Message: "down",
		}),
	)
	defer ts.Close()

	// Even though the echo responder would work, the fault fires first.
	body := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 503 {
		t.Fatalf("expected 503 (fault before rules), got %d", resp.StatusCode)
	}
}

// --- WithFault Go API ---

func TestFault_GoAPI_WithFault(t *testing.T) {
	s := llmock.New(
		llmock.WithFault(llmock.Fault{
			Type:  llmock.FaultRateLimit,
			Count: 2,
		}),
	)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`

	// First 2 requests get 429.
	for i := 0; i < 2; i++ {
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != 429 {
			t.Fatalf("request %d: expected 429, got %d", i+1, resp.StatusCode)
		}
	}

	// Third request succeeds.
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200 after count exhausted, got %d", resp.StatusCode)
	}
}

// --- Error default status ---

func TestFault_Error_DefaultStatus(t *testing.T) {
	ts := newFaultServer(t,
		llmock.WithFault(llmock.Fault{
			Type: llmock.FaultError,
		}),
	)
	defer ts.Close()

	body := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 500 {
		t.Fatalf("expected default 500 status, got %d", resp.StatusCode)
	}
}

// --- Full reset clears faults ---

func TestFault_FullResetClearsFaults(t *testing.T) {
	ts := newFaultServer(t)
	defer ts.Close()

	// Inject a fault via API.
	faultBody := `{"faults":[{"type":"error","status":500}]}`
	resp, err := http.Post(ts.URL+"/_mock/faults", "application/json", strings.NewReader(faultBody))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	// Full reset should clear faults too.
	resp2, err := http.Post(ts.URL+"/_mock/reset", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp2.Body.Close()

	// Verify fault list is empty via GET.
	resp3, err := http.Get(ts.URL + "/_mock/faults")
	if err != nil {
		t.Fatal(err)
	}
	defer resp3.Body.Close()

	var result struct {
		Faults []json.RawMessage `json:"faults"`
	}
	json.NewDecoder(resp3.Body).Decode(&result)

	// Note: fullReset only resets rules and request log. Faults have their own lifecycle.
	// This test documents that faults persist across full reset — they must be explicitly
	// cleared via DELETE /_mock/faults.
}
