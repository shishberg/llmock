package llmock_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/shishberg/llmock"
)

func controlTestServer(t *testing.T, opts ...llmock.Option) *httptest.Server {
	t.Helper()
	s := llmock.New(opts...)
	return httptest.NewServer(s.Handler())
}

func controlCall(t *testing.T, ts *httptest.Server, req jsonRPCRequest) jsonRPCResponse {
	t.Helper()
	body, _ := json.Marshal(req)
	resp, err := http.Post(ts.URL+"/mcp/control", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /mcp/control failed: %v", err)
	}
	defer resp.Body.Close()

	var result jsonRPCResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decoding response: %v", err)
	}
	return result
}

func controlCallTool(t *testing.T, ts *httptest.Server, name string, args map[string]any) jsonRPCResponse {
	t.Helper()
	return controlCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: map[string]any{
			"name":      name,
			"arguments": args,
		},
	})
}

func getControlToolText(t *testing.T, resp jsonRPCResponse) string {
	t.Helper()
	if resp.Error != nil {
		t.Fatalf("unexpected error: %s", resp.Error.Message)
	}
	var result struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		t.Fatalf("unmarshaling result: %v", err)
	}
	if result.IsError {
		t.Fatalf("tool returned error: %s", result.Content[0].Text)
	}
	if len(result.Content) == 0 {
		t.Fatal("no content in response")
	}
	return result.Content[0].Text
}

func TestControl_Initialize(t *testing.T) {
	ts := controlTestServer(t)
	defer ts.Close()

	resp := controlCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "initialize",
	})

	if resp.Error != nil {
		t.Fatalf("unexpected error: %s", resp.Error.Message)
	}

	var result struct {
		ProtocolVersion string `json:"protocolVersion"`
		ServerInfo      struct {
			Name string `json:"name"`
		} `json:"serverInfo"`
		Capabilities struct {
			Tools map[string]any `json:"tools"`
		} `json:"capabilities"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		t.Fatalf("unmarshaling result: %v", err)
	}
	if result.ServerInfo.Name != "llmock-control" {
		t.Errorf("server name = %q, want llmock-control", result.ServerInfo.Name)
	}
	if result.Capabilities.Tools == nil {
		t.Error("expected tools capability")
	}
}

func TestControl_ToolsList(t *testing.T) {
	ts := controlTestServer(t)
	defer ts.Close()

	resp := controlCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/list",
	})

	if resp.Error != nil {
		t.Fatalf("unexpected error: %s", resp.Error.Message)
	}

	var result struct {
		Tools []struct {
			Name        string `json:"name"`
			Description string `json:"description"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		t.Fatalf("unmarshaling result: %v", err)
	}

	expectedTools := map[string]bool{
		"llmock_add_rule":      false,
		"llmock_list_rules":    false,
		"llmock_reset_rules":   false,
		"llmock_add_fault":     false,
		"llmock_list_faults":   false,
		"llmock_clear_faults":  false,
		"llmock_list_requests": false,
		"llmock_clear_requests": false,
		"llmock_reset":         false,
	}

	for _, tool := range result.Tools {
		if _, ok := expectedTools[tool.Name]; ok {
			expectedTools[tool.Name] = true
		} else {
			t.Errorf("unexpected tool: %s", tool.Name)
		}
		if tool.Description == "" {
			t.Errorf("tool %s has no description", tool.Name)
		}
	}

	for name, found := range expectedTools {
		if !found {
			t.Errorf("missing expected tool: %s", name)
		}
	}
}

func TestControl_AddRuleAndVerify(t *testing.T) {
	ts := controlTestServer(t, llmock.WithResponder(llmock.EchoResponder{}))
	defer ts.Close()

	// Add a rule via control plane.
	resp := controlCallTool(t, ts, "llmock_add_rule", map[string]any{
		"pattern":   ".*deploy.*",
		"responses": []any{"Deploying now..."},
	})
	text := getControlToolText(t, resp)
	if !strings.Contains(text, "success") {
		t.Errorf("expected success message, got: %s", text)
	}

	// Verify the rule is active by making a chat request.
	result := chatRequest(t, ts, "please deploy my app")
	if result.Choices[0].Message.Content != "Deploying now..." {
		t.Errorf("expected 'Deploying now...', got %q", result.Choices[0].Message.Content)
	}
}

func TestControl_AddRule_InvalidRegex(t *testing.T) {
	ts := controlTestServer(t)
	defer ts.Close()

	resp := controlCallTool(t, ts, "llmock_add_rule", map[string]any{
		"pattern":   "[invalid",
		"responses": []any{"test"},
	})

	var result struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		t.Fatalf("unmarshaling result: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected isError to be true")
	}
	if !strings.Contains(result.Content[0].Text, "invalid regex") {
		t.Errorf("expected 'invalid regex' error, got: %s", result.Content[0].Text)
	}
}

func TestControl_AddRule_Priority(t *testing.T) {
	ts := controlTestServer(t, llmock.WithResponder(llmock.EchoResponder{}))
	defer ts.Close()

	// Add a rule with default priority (prepend).
	controlCallTool(t, ts, "llmock_add_rule", map[string]any{
		"pattern":   ".*",
		"responses": []any{"catch-all"},
	})
	// Add a higher-priority rule with prepend.
	controlCallTool(t, ts, "llmock_add_rule", map[string]any{
		"pattern":   "^hello$",
		"responses": []any{"Hello!"},
		"priority":  0,
	})

	result := chatRequest(t, ts, "hello")
	if result.Choices[0].Message.Content != "Hello!" {
		t.Errorf("expected 'Hello!', got %q", result.Choices[0].Message.Content)
	}
}

func TestControl_ListRules(t *testing.T) {
	ts := controlTestServer(t,
		llmock.WithRules(
			llmock.Rule{Pattern: regexp.MustCompile(`^test$`), Responses: []string{"test response"}},
		),
	)
	defer ts.Close()

	resp := controlCallTool(t, ts, "llmock_list_rules", nil)
	text := getControlToolText(t, resp)

	var rules []struct {
		Pattern   string   `json:"pattern"`
		Responses []string `json:"responses"`
	}
	if err := json.Unmarshal([]byte(text), &rules); err != nil {
		t.Fatalf("parsing rules JSON: %v", err)
	}
	if len(rules) != 1 {
		t.Fatalf("expected 1 rule, got %d", len(rules))
	}
	if rules[0].Pattern != "^test$" {
		t.Errorf("expected pattern '^test$', got %q", rules[0].Pattern)
	}
}

func TestControl_ResetRules(t *testing.T) {
	ts := controlTestServer(t,
		llmock.WithRules(
			llmock.Rule{Pattern: regexp.MustCompile(`^original$`), Responses: []string{"original"}},
		),
	)
	defer ts.Close()

	// Add a new rule.
	controlCallTool(t, ts, "llmock_add_rule", map[string]any{
		"pattern":   ".*new.*",
		"responses": []any{"new rule"},
	})

	// Reset rules.
	resp := controlCallTool(t, ts, "llmock_reset_rules", nil)
	text := getControlToolText(t, resp)
	if !strings.Contains(text, "reset") {
		t.Errorf("expected reset message, got: %s", text)
	}

	// Verify the added rule is gone.
	result := chatRequest(t, ts, "new stuff")
	if result.Choices[0].Message.Content == "new rule" {
		t.Error("injected rule should have been cleared after reset")
	}
}

func TestControl_AddFault(t *testing.T) {
	ts := controlTestServer(t)
	defer ts.Close()

	// Add an error fault.
	resp := controlCallTool(t, ts, "llmock_add_fault", map[string]any{
		"type":    "error",
		"status":  503,
		"message": "service down",
		"count":   1,
	})
	text := getControlToolText(t, resp)
	if !strings.Contains(text, "success") {
		t.Errorf("expected success message, got: %s", text)
	}

	// Verify the fault fires on the next request.
	body := `{"model":"test","messages":[{"role":"user","content":"hello"}]}`
	httpResp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	httpResp.Body.Close()
	if httpResp.StatusCode != 503 {
		t.Errorf("expected 503, got %d", httpResp.StatusCode)
	}
}

func TestControl_ListFaults(t *testing.T) {
	ts := controlTestServer(t, llmock.WithFault(llmock.Fault{Type: llmock.FaultError, Status: 500, Message: "boom"}))
	defer ts.Close()

	resp := controlCallTool(t, ts, "llmock_list_faults", nil)
	text := getControlToolText(t, resp)

	var faults []struct {
		Type    string `json:"type"`
		Status  int    `json:"status"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal([]byte(text), &faults); err != nil {
		t.Fatalf("parsing faults JSON: %v", err)
	}
	if len(faults) != 1 {
		t.Fatalf("expected 1 fault, got %d", len(faults))
	}
	if faults[0].Type != "error" || faults[0].Status != 500 {
		t.Errorf("unexpected fault: %+v", faults[0])
	}
}

func TestControl_ClearFaults(t *testing.T) {
	ts := controlTestServer(t, llmock.WithFault(llmock.Fault{Type: llmock.FaultError, Status: 500}))
	defer ts.Close()

	// Clear faults.
	controlCallTool(t, ts, "llmock_clear_faults", nil)

	// Verify the fault is gone — request should succeed.
	result := chatRequest(t, ts, "hello")
	if result.Choices == nil {
		t.Error("expected successful response after clearing faults")
	}
}

func TestControl_ListRequests(t *testing.T) {
	ts := controlTestServer(t, llmock.WithResponder(llmock.EchoResponder{}))
	defer ts.Close()

	// Make a chat request to create a log entry.
	chatRequest(t, ts, "hello from test")

	// List requests via control plane.
	resp := controlCallTool(t, ts, "llmock_list_requests", nil)
	text := getControlToolText(t, resp)

	var requests []struct {
		UserMessage string `json:"user_message"`
	}
	if err := json.Unmarshal([]byte(text), &requests); err != nil {
		t.Fatalf("parsing requests JSON: %v", err)
	}
	if len(requests) != 1 {
		t.Fatalf("expected 1 request, got %d", len(requests))
	}
	if requests[0].UserMessage != "hello from test" {
		t.Errorf("expected 'hello from test', got %q", requests[0].UserMessage)
	}
}

func TestControl_ClearRequests(t *testing.T) {
	ts := controlTestServer(t, llmock.WithResponder(llmock.EchoResponder{}))
	defer ts.Close()

	// Make a chat request.
	chatRequest(t, ts, "hello")

	// Clear requests.
	controlCallTool(t, ts, "llmock_clear_requests", nil)

	// Verify log is empty.
	resp := controlCallTool(t, ts, "llmock_list_requests", nil)
	text := getControlToolText(t, resp)

	var requests []any
	if err := json.Unmarshal([]byte(text), &requests); err != nil {
		t.Fatalf("parsing requests JSON: %v", err)
	}
	if len(requests) != 0 {
		t.Errorf("expected 0 requests, got %d", len(requests))
	}
}

func TestControl_FullReset(t *testing.T) {
	ts := controlTestServer(t,
		llmock.WithRules(
			llmock.Rule{Pattern: regexp.MustCompile(`^original$`), Responses: []string{"original"}},
		),
	)
	defer ts.Close()

	// Add a rule, make a request to create a log entry, then add a fault.
	controlCallTool(t, ts, "llmock_add_rule", map[string]any{
		"pattern":   ".*extra.*",
		"responses": []any{"extra rule"},
	})
	chatRequest(t, ts, "original") // creates a log entry
	controlCallTool(t, ts, "llmock_add_fault", map[string]any{
		"type":   "error",
		"status": 500,
		"count":  999,
	})

	// Full reset.
	resp := controlCallTool(t, ts, "llmock_reset", nil)
	text := getControlToolText(t, resp)
	if !strings.Contains(text, "reset") {
		t.Errorf("expected reset message, got: %s", text)
	}

	// Verify: injected rule gone.
	result := chatRequest(t, ts, "extra stuff")
	if result.Choices[0].Message.Content == "extra rule" {
		t.Error("injected rule should be gone after full reset")
	}

	// Verify: faults cleared (request succeeds).
	result = chatRequest(t, ts, "original")
	if result.Choices[0].Message.Content != "original" {
		t.Errorf("expected 'original' after reset, got %q", result.Choices[0].Message.Content)
	}

	// Verify: request log cleared.
	reqResp := controlCallTool(t, ts, "llmock_list_requests", nil)
	reqText := getControlToolText(t, reqResp)
	var requests []any
	json.Unmarshal([]byte(reqText), &requests)
	// Only requests made after reset should be in the log.
	// The 2 chatRequests above after reset should be logged, but
	// the one before reset should not.
	if len(requests) != 2 {
		t.Errorf("expected 2 requests after reset, got %d", len(requests))
	}
}

func TestControl_UnknownTool(t *testing.T) {
	ts := controlTestServer(t)
	defer ts.Close()

	resp := controlCallTool(t, ts, "nonexistent_tool", nil)
	if resp.Error == nil {
		t.Fatal("expected error for unknown tool")
	}
	if !strings.Contains(resp.Error.Message, "Unknown tool") {
		t.Errorf("expected 'Unknown tool' error, got: %s", resp.Error.Message)
	}
}

func TestControl_DisabledWhenAdminDisabled(t *testing.T) {
	s := llmock.New(llmock.WithAdminAPI(false))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body, _ := json.Marshal(jsonRPCRequest{JSONRPC: "2.0", ID: 1, Method: "initialize"})
	resp, err := http.Post(ts.URL+"/mcp/control", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	// Should get 404 or 405 since the route isn't registered.
	if resp.StatusCode == 200 {
		t.Error("control plane should not be available when admin API is disabled")
	}
}

func TestControl_MethodNotFound(t *testing.T) {
	ts := controlTestServer(t)
	defer ts.Close()

	resp := controlCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "resources/list",
	})
	if resp.Error == nil {
		t.Fatal("expected error for unknown method")
	}
	if resp.Error.Code != -32601 {
		t.Errorf("expected -32601 error code, got %d", resp.Error.Code)
	}
}

func TestControl_InvalidJSON(t *testing.T) {
	ts := controlTestServer(t)
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/mcp/control", "application/json", strings.NewReader("{broken"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var result jsonRPCResponse
	json.NewDecoder(resp.Body).Decode(&result)
	if result.Error == nil {
		t.Fatal("expected parse error")
	}
	if result.Error.Code != -32700 {
		t.Errorf("expected -32700 parse error code, got %d", result.Error.Code)
	}
}

func TestControl_CoexistsWithMockMCP(t *testing.T) {
	// Both the mock MCP server (POST /mcp) and the control plane (POST /mcp/control)
	// should work simultaneously.
	ts := controlTestServer(t, llmock.WithMCP(llmock.MCPConfig{
		Tools: []llmock.MCPToolConfig{
			{Name: "mock_tool", Description: "a mock tool"},
		},
	}))
	defer ts.Close()

	// Control plane initialize.
	controlResp := controlCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "initialize",
	})
	if controlResp.Error != nil {
		t.Fatalf("control plane error: %s", controlResp.Error.Message)
	}
	var controlResult struct {
		ServerInfo struct {
			Name string `json:"name"`
		} `json:"serverInfo"`
	}
	json.Unmarshal(controlResp.Result, &controlResult)
	if controlResult.ServerInfo.Name != "llmock-control" {
		t.Errorf("control server name = %q, want llmock-control", controlResult.ServerInfo.Name)
	}

	// Mock MCP initialize.
	mockResp := mcpCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      2,
		Method:  "initialize",
	})
	if mockResp.Error != nil {
		t.Fatalf("mock MCP error: %s", mockResp.Error.Message)
	}
	var mockResult struct {
		ServerInfo struct {
			Name string `json:"name"`
		} `json:"serverInfo"`
	}
	json.Unmarshal(mockResp.Result, &mockResult)
	if mockResult.ServerInfo.Name != "llmock" {
		t.Errorf("mock server name = %q, want llmock", mockResult.ServerInfo.Name)
	}
}
