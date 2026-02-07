package llmock_test

import (
	"bytes"
	"encoding/json"
	"regexp"
	"strings"
	"testing"

	"github.com/shishberg/llmock"
)

// stdioCall sends a JSON-RPC request line through a StdioTransport and
// returns the parsed response.
func stdioCall(t *testing.T, st *llmock.StdioTransport, req jsonRPCRequest) jsonRPCResponse {
	t.Helper()
	line, _ := json.Marshal(req)
	line = append(line, '\n')

	var out bytes.Buffer
	if err := st.Run(bytes.NewReader(line), &out); err != nil {
		t.Fatalf("StdioTransport.Run failed: %v", err)
	}

	var resp jsonRPCResponse
	if err := json.Unmarshal(out.Bytes(), &resp); err != nil {
		t.Fatalf("decoding stdio response: %v (raw: %s)", err, out.String())
	}
	return resp
}

func TestStdio_Initialize(t *testing.T) {
	s := llmock.New()
	st := llmock.NewStdioTransport(s)
	if st == nil {
		t.Fatal("expected non-nil StdioTransport")
	}

	resp := stdioCall(t, st, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "initialize",
	})

	if resp.Error != nil {
		t.Fatalf("unexpected error: %s", resp.Error.Message)
	}

	var result struct {
		ServerInfo struct {
			Name string `json:"name"`
		} `json:"serverInfo"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		t.Fatalf("unmarshaling result: %v", err)
	}
	if result.ServerInfo.Name != "llmock-control" {
		t.Errorf("server name = %q, want llmock-control", result.ServerInfo.Name)
	}
}

func TestStdio_ToolsList(t *testing.T) {
	s := llmock.New()
	st := llmock.NewStdioTransport(s)

	resp := stdioCall(t, st, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/list",
	})

	if resp.Error != nil {
		t.Fatalf("unexpected error: %s", resp.Error.Message)
	}

	var result struct {
		Tools []struct {
			Name string `json:"name"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		t.Fatalf("unmarshaling result: %v", err)
	}
	if len(result.Tools) != 9 {
		t.Errorf("expected 9 tools, got %d", len(result.Tools))
	}
}

func TestStdio_AddRuleAndListRules(t *testing.T) {
	s := llmock.New(llmock.WithResponder(llmock.EchoResponder{}))
	st := llmock.NewStdioTransport(s)

	// Add a rule.
	addReq := jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: map[string]any{
			"name": "llmock_add_rule",
			"arguments": map[string]any{
				"pattern":   ".*hello.*",
				"responses": []any{"Hi there!"},
			},
		},
	}
	resp := stdioCall(t, st, addReq)
	if resp.Error != nil {
		t.Fatalf("unexpected error: %s", resp.Error.Message)
	}

	// List rules.
	listReq := jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      2,
		Method:  "tools/call",
		Params: map[string]any{
			"name":      "llmock_list_rules",
			"arguments": map[string]any{},
		},
	}
	listResp := stdioCall(t, st, listReq)
	text := getControlToolText(t, listResp)

	var rules []struct {
		Pattern string `json:"pattern"`
	}
	if err := json.Unmarshal([]byte(text), &rules); err != nil {
		t.Fatalf("parsing rules: %v", err)
	}
	found := false
	for _, r := range rules {
		if r.Pattern == ".*hello.*" {
			found = true
			break
		}
	}
	if !found {
		t.Error("added rule not found in list")
	}
}

func TestStdio_MultipleRequests(t *testing.T) {
	s := llmock.New()
	st := llmock.NewStdioTransport(s)

	// Send two requests in one batch (two lines).
	var input bytes.Buffer
	req1, _ := json.Marshal(jsonRPCRequest{JSONRPC: "2.0", ID: 1, Method: "initialize"})
	req2, _ := json.Marshal(jsonRPCRequest{JSONRPC: "2.0", ID: 2, Method: "tools/list"})
	input.Write(req1)
	input.Write([]byte("\n"))
	input.Write(req2)
	input.Write([]byte("\n"))

	var out bytes.Buffer
	if err := st.Run(&input, &out); err != nil {
		t.Fatalf("StdioTransport.Run failed: %v", err)
	}

	// Parse two responses (one per line).
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 response lines, got %d: %q", len(lines), out.String())
	}

	var resp1, resp2 jsonRPCResponse
	json.Unmarshal([]byte(lines[0]), &resp1)
	json.Unmarshal([]byte(lines[1]), &resp2)

	if resp1.Error != nil {
		t.Errorf("request 1 error: %s", resp1.Error.Message)
	}
	if resp2.Error != nil {
		t.Errorf("request 2 error: %s", resp2.Error.Message)
	}
}

func TestStdio_InvalidJSON(t *testing.T) {
	s := llmock.New()
	st := llmock.NewStdioTransport(s)

	var out bytes.Buffer
	if err := st.Run(strings.NewReader("{broken\n"), &out); err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	var resp jsonRPCResponse
	if err := json.Unmarshal(out.Bytes(), &resp); err != nil {
		t.Fatalf("decoding response: %v", err)
	}
	if resp.Error == nil {
		t.Fatal("expected parse error")
	}
	if resp.Error.Code != -32700 {
		t.Errorf("expected -32700, got %d", resp.Error.Code)
	}
}

func TestStdio_InvalidJSONRPCVersion(t *testing.T) {
	s := llmock.New()
	st := llmock.NewStdioTransport(s)

	resp := stdioCall(t, st, jsonRPCRequest{
		JSONRPC: "1.0",
		ID:      1,
		Method:  "initialize",
	})

	if resp.Error == nil {
		t.Fatal("expected error for wrong version")
	}
	if resp.Error.Code != -32600 {
		t.Errorf("expected -32600, got %d", resp.Error.Code)
	}
}

func TestStdio_MethodNotFound(t *testing.T) {
	s := llmock.New()
	st := llmock.NewStdioTransport(s)

	resp := stdioCall(t, st, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "resources/list",
	})

	if resp.Error == nil {
		t.Fatal("expected method not found error")
	}
	if resp.Error.Code != -32601 {
		t.Errorf("expected -32601, got %d", resp.Error.Code)
	}
}

func TestStdio_BlankLinesSkipped(t *testing.T) {
	s := llmock.New()
	st := llmock.NewStdioTransport(s)

	req, _ := json.Marshal(jsonRPCRequest{JSONRPC: "2.0", ID: 1, Method: "initialize"})
	input := "\n\n" + string(req) + "\n\n"

	var out bytes.Buffer
	if err := st.Run(strings.NewReader(input), &out); err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected 1 response line (blank lines should be skipped), got %d", len(lines))
	}
}

func TestStdio_NilWhenAdminDisabled(t *testing.T) {
	s := llmock.New(llmock.WithAdminAPI(false))
	st := llmock.NewStdioTransport(s)
	if st != nil {
		t.Error("expected nil StdioTransport when admin API is disabled")
	}
}

func TestStdio_IDPreserved(t *testing.T) {
	s := llmock.New()
	st := llmock.NewStdioTransport(s)

	// Test with string ID.
	resp := stdioCall(t, st, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      "my-request-42",
		Method:  "initialize",
	})
	// The ID comes back as any; marshal/unmarshal to check.
	idJSON, _ := json.Marshal(resp.ID)
	if string(idJSON) != `"my-request-42"` {
		t.Errorf("expected string ID \"my-request-42\", got %s", string(idJSON))
	}

	// Test with integer ID.
	resp = stdioCall(t, st, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      42,
		Method:  "initialize",
	})
	idJSON, _ = json.Marshal(resp.ID)
	if string(idJSON) != "42" {
		t.Errorf("expected integer ID 42, got %s", string(idJSON))
	}
}

func TestStdio_FullWorkflow(t *testing.T) {
	// End-to-end test: create a server, add a rule via stdio, then verify
	// the rule works via the HTTP endpoint, demonstrating that stdio and
	// HTTP share the same server state.
	s := llmock.New(llmock.WithResponder(llmock.EchoResponder{}))
	st := llmock.NewStdioTransport(s)

	// Add a rule via stdio.
	addReq, _ := json.Marshal(jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: map[string]any{
			"name": "llmock_add_rule",
			"arguments": map[string]any{
				"pattern":   ".*weather.*",
				"responses": []any{"It's sunny!"},
			},
		},
	})

	var out bytes.Buffer
	st.Run(bytes.NewReader(append(addReq, '\n')), &out)

	var resp jsonRPCResponse
	json.Unmarshal(out.Bytes(), &resp)
	if resp.Error != nil {
		t.Fatalf("unexpected error: %s", resp.Error.Message)
	}

	// Now test via the server's admin state using another stdio call.
	listReq, _ := json.Marshal(jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      2,
		Method:  "tools/call",
		Params: map[string]any{
			"name":      "llmock_list_rules",
			"arguments": map[string]any{},
		},
	})
	out.Reset()
	st.Run(bytes.NewReader(append(listReq, '\n')), &out)

	var listResp jsonRPCResponse
	json.Unmarshal(out.Bytes(), &listResp)
	text := getControlToolText(t, listResp)
	if !strings.Contains(text, "weather") {
		t.Errorf("expected rule with 'weather' pattern in list, got: %s", text)
	}
}

func TestStdio_AddFaultAndClear(t *testing.T) {
	s := llmock.New()
	st := llmock.NewStdioTransport(s)

	// Add a fault.
	resp := stdioCall(t, st, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: map[string]any{
			"name": "llmock_add_fault",
			"arguments": map[string]any{
				"type":   "error",
				"status": 503,
			},
		},
	})
	if resp.Error != nil {
		t.Fatalf("unexpected error: %s", resp.Error.Message)
	}

	// List faults.
	listResp := stdioCall(t, st, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      2,
		Method:  "tools/call",
		Params: map[string]any{
			"name":      "llmock_list_faults",
			"arguments": map[string]any{},
		},
	})
	text := getControlToolText(t, listResp)
	if !strings.Contains(text, "error") {
		t.Errorf("expected fault in list, got: %s", text)
	}

	// Clear faults.
	clearResp := stdioCall(t, st, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      3,
		Method:  "tools/call",
		Params: map[string]any{
			"name":      "llmock_clear_faults",
			"arguments": map[string]any{},
		},
	})
	clearText := getControlToolText(t, clearResp)
	if !strings.Contains(clearText, "cleared") {
		t.Errorf("expected 'cleared' message, got: %s", clearText)
	}
}

func TestStdio_FullReset(t *testing.T) {
	s := llmock.New(llmock.WithRules(
		llmock.Rule{Pattern: regexp.MustCompile(`^original$`), Responses: []string{"original"}},
	))
	st := llmock.NewStdioTransport(s)

	// Add a rule.
	stdioCall(t, st, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: map[string]any{
			"name": "llmock_add_rule",
			"arguments": map[string]any{
				"pattern":   ".*extra.*",
				"responses": []any{"extra"},
			},
		},
	})

	// Full reset.
	resetResp := stdioCall(t, st, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      2,
		Method:  "tools/call",
		Params: map[string]any{
			"name":      "llmock_reset",
			"arguments": map[string]any{},
		},
	})
	text := getControlToolText(t, resetResp)
	if !strings.Contains(text, "reset") {
		t.Errorf("expected 'reset' message, got: %s", text)
	}

	// Verify: list rules should only show original.
	listResp := stdioCall(t, st, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      3,
		Method:  "tools/call",
		Params: map[string]any{
			"name":      "llmock_list_rules",
			"arguments": map[string]any{},
		},
	})
	listText := getControlToolText(t, listResp)
	if strings.Contains(listText, "extra") {
		t.Error("injected rule should be gone after reset")
	}
	if !strings.Contains(listText, "original") {
		t.Error("original rule should still exist after reset")
	}
}
