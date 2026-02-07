package llmock_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/shishberg/llmock"
)

// jsonRPCRequest mirrors the internal type for test payloads.
type jsonRPCRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      any    `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

// jsonRPCResponse mirrors the internal type for test assertions.
type jsonRPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func mcpTestServer(cfg llmock.MCPConfig) *httptest.Server {
	s := llmock.New(llmock.WithMCP(cfg))
	return httptest.NewServer(s.Handler())
}

func mcpCall(t *testing.T, ts *httptest.Server, req jsonRPCRequest) jsonRPCResponse {
	t.Helper()
	body, _ := json.Marshal(req)
	resp, err := http.Post(ts.URL+"/mcp", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /mcp failed: %v", err)
	}
	defer resp.Body.Close()

	var result jsonRPCResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decoding response: %v", err)
	}
	return result
}

func TestMCPDisabledByDefault(t *testing.T) {
	s := llmock.New()
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body, _ := json.Marshal(jsonRPCRequest{JSONRPC: "2.0", ID: 1, Method: "initialize"})
	resp, err := http.Post(ts.URL+"/mcp", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /mcp failed: %v", err)
	}
	defer resp.Body.Close()

	// MCP is not enabled, so should get 404 or 405.
	if resp.StatusCode == http.StatusOK {
		t.Fatalf("expected non-200 when MCP disabled, got %d", resp.StatusCode)
	}
}

func TestMCPInitialize(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{})
	defer ts.Close()

	result := mcpCall(t, ts, jsonRPCRequest{JSONRPC: "2.0", ID: 1, Method: "initialize"})

	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}
	if result.JSONRPC != "2.0" {
		t.Errorf("expected jsonrpc=2.0, got %s", result.JSONRPC)
	}

	var initResult map[string]any
	if err := json.Unmarshal(result.Result, &initResult); err != nil {
		t.Fatalf("parsing result: %v", err)
	}

	if v, ok := initResult["protocolVersion"].(string); !ok || v == "" {
		t.Error("expected protocolVersion in result")
	}

	serverInfo, ok := initResult["serverInfo"].(map[string]any)
	if !ok {
		t.Fatal("expected serverInfo in result")
	}
	if serverInfo["name"] != "llmock" {
		t.Errorf("expected server name=llmock, got %v", serverInfo["name"])
	}

	caps, ok := initResult["capabilities"].(map[string]any)
	if !ok {
		t.Fatal("expected capabilities in result")
	}
	for _, cap := range []string{"tools", "resources", "prompts"} {
		if _, ok := caps[cap]; !ok {
			t.Errorf("expected %s capability", cap)
		}
	}
}

func TestMCPInvalidJSON(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{})
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/mcp", "application/json", bytes.NewReader([]byte("not json")))
	if err != nil {
		t.Fatalf("POST /mcp failed: %v", err)
	}
	defer resp.Body.Close()

	var result jsonRPCResponse
	json.NewDecoder(resp.Body).Decode(&result)
	if result.Error == nil {
		t.Fatal("expected error for invalid JSON")
	}
	if result.Error.Code != -32700 {
		t.Errorf("expected parse error code -32700, got %d", result.Error.Code)
	}
}

func TestMCPInvalidJSONRPCVersion(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{})
	defer ts.Close()

	result := mcpCall(t, ts, jsonRPCRequest{JSONRPC: "1.0", ID: 1, Method: "initialize"})
	if result.Error == nil {
		t.Fatal("expected error for invalid jsonrpc version")
	}
	if result.Error.Code != -32600 {
		t.Errorf("expected invalid request code -32600, got %d", result.Error.Code)
	}
}

func TestMCPMethodNotFound(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{})
	defer ts.Close()

	result := mcpCall(t, ts, jsonRPCRequest{JSONRPC: "2.0", ID: 1, Method: "nonexistent"})
	if result.Error == nil {
		t.Fatal("expected error for unknown method")
	}
	if result.Error.Code != -32601 {
		t.Errorf("expected method not found code -32601, got %d", result.Error.Code)
	}
}

func TestMCPToolsList(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{
		Tools: []llmock.MCPToolConfig{
			{
				Name:        "get_weather",
				Description: "Get current weather",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{"type": "string"},
					},
					"required": []any{"location"},
				},
			},
			{
				Name:        "search",
				Description: "Search the web",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"query": map[string]any{"type": "string"},
					},
				},
			},
		},
	})
	defer ts.Close()

	result := mcpCall(t, ts, jsonRPCRequest{JSONRPC: "2.0", ID: 1, Method: "tools/list"})
	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}

	var listResult map[string]any
	json.Unmarshal(result.Result, &listResult)

	tools, ok := listResult["tools"].([]any)
	if !ok {
		t.Fatal("expected tools array in result")
	}
	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(tools))
	}

	tool0 := tools[0].(map[string]any)
	if tool0["name"] != "get_weather" {
		t.Errorf("expected first tool name=get_weather, got %v", tool0["name"])
	}
	if tool0["description"] != "Get current weather" {
		t.Errorf("expected description='Get current weather', got %v", tool0["description"])
	}
	if tool0["inputSchema"] == nil {
		t.Error("expected inputSchema in tool")
	}
}

func TestMCPToolsCall(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{
		Tools: []llmock.MCPToolConfig{
			{
				Name:        "get_weather",
				Description: "Get current weather",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{"type": "string"},
					},
				},
				Responses: []llmock.MCPToolResponse{
					{Pattern: `"location":\s*"San Francisco"`, Result: `{"temperature": 72, "condition": "sunny"}`},
					{Pattern: `"location":\s*"London"`, Result: `{"temperature": 55, "condition": "rainy"}`},
					{Pattern: ".*", Result: `{"temperature": 70, "condition": "unknown"}`},
				},
			},
		},
	})
	defer ts.Close()

	// Call with matching pattern.
	result := mcpCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: map[string]any{
			"name":      "get_weather",
			"arguments": map[string]any{"location": "San Francisco"},
		},
	})

	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}

	var callResult map[string]any
	json.Unmarshal(result.Result, &callResult)

	content, ok := callResult["content"].([]any)
	if !ok || len(content) == 0 {
		t.Fatal("expected content array in result")
	}
	block := content[0].(map[string]any)
	if block["type"] != "text" {
		t.Errorf("expected type=text, got %v", block["type"])
	}
	text := block["text"].(string)
	if text != `{"temperature": 72, "condition": "sunny"}` {
		t.Errorf("unexpected result text: %s", text)
	}
}

func TestMCPToolsCallSecondPattern(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{
		Tools: []llmock.MCPToolConfig{
			{
				Name: "get_weather",
				Responses: []llmock.MCPToolResponse{
					{Pattern: `"location":\s*"San Francisco"`, Result: `{"temperature": 72}`},
					{Pattern: `"location":\s*"London"`, Result: `{"temperature": 55}`},
				},
			},
		},
	})
	defer ts.Close()

	result := mcpCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      2,
		Method:  "tools/call",
		Params: map[string]any{
			"name":      "get_weather",
			"arguments": map[string]any{"location": "London"},
		},
	})

	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}

	var callResult map[string]any
	json.Unmarshal(result.Result, &callResult)
	content := callResult["content"].([]any)
	block := content[0].(map[string]any)
	if block["text"] != `{"temperature": 55}` {
		t.Errorf("expected London result, got %v", block["text"])
	}
}

func TestMCPToolsCallUnknownTool(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{
		Tools: []llmock.MCPToolConfig{
			{Name: "get_weather", Responses: []llmock.MCPToolResponse{{Pattern: ".*", Result: "ok"}}},
		},
	})
	defer ts.Close()

	result := mcpCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params:  map[string]any{"name": "nonexistent"},
	})

	if result.Error == nil {
		t.Fatal("expected error for unknown tool")
	}
	if result.Error.Code != -32602 {
		t.Errorf("expected invalid params code -32602, got %d", result.Error.Code)
	}
}

func TestMCPToolsCallNoName(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{})
	defer ts.Close()

	result := mcpCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params:  map[string]any{},
	})

	if result.Error == nil {
		t.Fatal("expected error for missing name")
	}
	if result.Error.Code != -32602 {
		t.Errorf("expected invalid params code -32602, got %d", result.Error.Code)
	}
}

func TestMCPResourcesList(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{
		Resources: []llmock.MCPResourceConfig{
			{
				URI:     "file:///project/README.md",
				Name:    "Project README",
				Content: "# My Project",
			},
			{
				URI:      "file:///project/config.yaml",
				Name:     "Config",
				MimeType: "application/yaml",
				Content:  "key: value",
			},
		},
	})
	defer ts.Close()

	result := mcpCall(t, ts, jsonRPCRequest{JSONRPC: "2.0", ID: 1, Method: "resources/list"})
	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}

	var listResult map[string]any
	json.Unmarshal(result.Result, &listResult)

	resources, ok := listResult["resources"].([]any)
	if !ok {
		t.Fatal("expected resources array")
	}
	if len(resources) != 2 {
		t.Fatalf("expected 2 resources, got %d", len(resources))
	}

	r0 := resources[0].(map[string]any)
	if r0["uri"] != "file:///project/README.md" {
		t.Errorf("expected uri=file:///project/README.md, got %v", r0["uri"])
	}
	if r0["name"] != "Project README" {
		t.Errorf("expected name=Project README, got %v", r0["name"])
	}
	// First resource has no mimeType set.
	if _, hasMime := r0["mimeType"]; hasMime {
		t.Error("expected no mimeType for first resource")
	}

	r1 := resources[1].(map[string]any)
	if r1["mimeType"] != "application/yaml" {
		t.Errorf("expected mimeType=application/yaml, got %v", r1["mimeType"])
	}
}

func TestMCPResourcesRead(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{
		Resources: []llmock.MCPResourceConfig{
			{
				URI:     "file:///project/README.md",
				Name:    "Project README",
				Content: "# My Project\nThis is a test.",
			},
		},
	})
	defer ts.Close()

	result := mcpCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "resources/read",
		Params:  map[string]any{"uri": "file:///project/README.md"},
	})

	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}

	var readResult map[string]any
	json.Unmarshal(result.Result, &readResult)

	contents, ok := readResult["contents"].([]any)
	if !ok || len(contents) == 0 {
		t.Fatal("expected contents array")
	}
	c := contents[0].(map[string]any)
	if c["uri"] != "file:///project/README.md" {
		t.Errorf("expected uri in content, got %v", c["uri"])
	}
	if c["text"] != "# My Project\nThis is a test." {
		t.Errorf("unexpected text: %v", c["text"])
	}
	if c["mimeType"] != "text/plain" {
		t.Errorf("expected default mimeType=text/plain, got %v", c["mimeType"])
	}
}

func TestMCPResourcesReadNotFound(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{
		Resources: []llmock.MCPResourceConfig{
			{URI: "file:///project/README.md", Name: "README", Content: "hi"},
		},
	})
	defer ts.Close()

	result := mcpCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "resources/read",
		Params:  map[string]any{"uri": "file:///nonexistent"},
	})

	if result.Error == nil {
		t.Fatal("expected error for nonexistent resource")
	}
}

func TestMCPResourcesReadNoURI(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{})
	defer ts.Close()

	result := mcpCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "resources/read",
		Params:  map[string]any{},
	})

	if result.Error == nil {
		t.Fatal("expected error for missing uri")
	}
}

func TestMCPPromptsList(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{
		Prompts: []llmock.MCPPromptConfig{
			{
				Name:        "review_code",
				Description: "Review code for issues",
				Arguments: []llmock.MCPPromptArgument{
					{Name: "language", Required: true},
				},
				Template: "Please review the following {{language}} code...",
			},
		},
	})
	defer ts.Close()

	result := mcpCall(t, ts, jsonRPCRequest{JSONRPC: "2.0", ID: 1, Method: "prompts/list"})
	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}

	var listResult map[string]any
	json.Unmarshal(result.Result, &listResult)

	prompts, ok := listResult["prompts"].([]any)
	if !ok {
		t.Fatal("expected prompts array")
	}
	if len(prompts) != 1 {
		t.Fatalf("expected 1 prompt, got %d", len(prompts))
	}

	p := prompts[0].(map[string]any)
	if p["name"] != "review_code" {
		t.Errorf("expected name=review_code, got %v", p["name"])
	}
	if p["description"] != "Review code for issues" {
		t.Errorf("unexpected description: %v", p["description"])
	}

	args, ok := p["arguments"].([]any)
	if !ok || len(args) != 1 {
		t.Fatal("expected 1 argument")
	}
	arg := args[0].(map[string]any)
	if arg["name"] != "language" {
		t.Errorf("expected arg name=language, got %v", arg["name"])
	}
	if arg["required"] != true {
		t.Errorf("expected required=true, got %v", arg["required"])
	}
}

func TestMCPPromptsGet(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{
		Prompts: []llmock.MCPPromptConfig{
			{
				Name:     "review_code",
				Template: "Please review the following {{language}} code: {{code}}",
				Arguments: []llmock.MCPPromptArgument{
					{Name: "language", Required: true},
					{Name: "code", Required: true},
				},
			},
		},
	})
	defer ts.Close()

	result := mcpCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "prompts/get",
		Params: map[string]any{
			"name": "review_code",
			"arguments": map[string]string{
				"language": "Go",
				"code":     "func main() {}",
			},
		},
	})

	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}

	var getResult map[string]any
	json.Unmarshal(result.Result, &getResult)

	messages, ok := getResult["messages"].([]any)
	if !ok || len(messages) == 0 {
		t.Fatal("expected messages array")
	}
	msg := messages[0].(map[string]any)
	if msg["role"] != "user" {
		t.Errorf("expected role=user, got %v", msg["role"])
	}
	content := msg["content"].(map[string]any)
	text := content["text"].(string)
	expected := "Please review the following Go code: func main() {}"
	if text != expected {
		t.Errorf("expected %q, got %q", expected, text)
	}
}

func TestMCPPromptsGetNotFound(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{
		Prompts: []llmock.MCPPromptConfig{
			{Name: "review_code", Template: "test"},
		},
	})
	defer ts.Close()

	result := mcpCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "prompts/get",
		Params:  map[string]any{"name": "nonexistent"},
	})

	if result.Error == nil {
		t.Fatal("expected error for nonexistent prompt")
	}
}

func TestMCPPromptsGetNoName(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{})
	defer ts.Close()

	result := mcpCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "prompts/get",
		Params:  map[string]any{},
	})

	if result.Error == nil {
		t.Fatal("expected error for missing name")
	}
}

func TestMCPRequestIDPreserved(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{})
	defer ts.Close()

	// Test with integer ID.
	result := mcpCall(t, ts, jsonRPCRequest{JSONRPC: "2.0", ID: 42, Method: "initialize"})
	// JSON numbers are decoded as float64.
	if result.ID != float64(42) {
		t.Errorf("expected ID=42, got %v", result.ID)
	}

	// Test with string ID.
	result = mcpCall(t, ts, jsonRPCRequest{JSONRPC: "2.0", ID: "abc-123", Method: "initialize"})
	if result.ID != "abc-123" {
		t.Errorf("expected ID=abc-123, got %v", result.ID)
	}
}

func TestMCPFullHandshake(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{
		Tools: []llmock.MCPToolConfig{
			{
				Name:        "get_weather",
				Description: "Get weather",
				InputSchema: map[string]any{
					"type":       "object",
					"properties": map[string]any{"location": map[string]any{"type": "string"}},
				},
				Responses: []llmock.MCPToolResponse{
					{Pattern: ".*", Result: `{"temp": 72}`},
				},
			},
		},
		Resources: []llmock.MCPResourceConfig{
			{URI: "file:///README.md", Name: "README", Content: "# Hello"},
		},
		Prompts: []llmock.MCPPromptConfig{
			{Name: "greet", Template: "Hello {{name}}!", Arguments: []llmock.MCPPromptArgument{{Name: "name", Required: true}}},
		},
	})
	defer ts.Close()

	// Step 1: Initialize.
	init := mcpCall(t, ts, jsonRPCRequest{JSONRPC: "2.0", ID: 1, Method: "initialize"})
	if init.Error != nil {
		t.Fatalf("initialize failed: %v", init.Error)
	}

	// Step 2: List tools.
	toolsList := mcpCall(t, ts, jsonRPCRequest{JSONRPC: "2.0", ID: 2, Method: "tools/list"})
	if toolsList.Error != nil {
		t.Fatalf("tools/list failed: %v", toolsList.Error)
	}

	// Step 3: Call a tool.
	toolCall := mcpCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0", ID: 3, Method: "tools/call",
		Params: map[string]any{"name": "get_weather", "arguments": map[string]any{"location": "NYC"}},
	})
	if toolCall.Error != nil {
		t.Fatalf("tools/call failed: %v", toolCall.Error)
	}

	// Step 4: List resources.
	resList := mcpCall(t, ts, jsonRPCRequest{JSONRPC: "2.0", ID: 4, Method: "resources/list"})
	if resList.Error != nil {
		t.Fatalf("resources/list failed: %v", resList.Error)
	}

	// Step 5: Read a resource.
	resRead := mcpCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0", ID: 5, Method: "resources/read",
		Params: map[string]any{"uri": "file:///README.md"},
	})
	if resRead.Error != nil {
		t.Fatalf("resources/read failed: %v", resRead.Error)
	}

	// Step 6: List prompts.
	promptList := mcpCall(t, ts, jsonRPCRequest{JSONRPC: "2.0", ID: 6, Method: "prompts/list"})
	if promptList.Error != nil {
		t.Fatalf("prompts/list failed: %v", promptList.Error)
	}

	// Step 7: Get a prompt.
	promptGet := mcpCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0", ID: 7, Method: "prompts/get",
		Params: map[string]any{"name": "greet", "arguments": map[string]string{"name": "World"}},
	})
	if promptGet.Error != nil {
		t.Fatalf("prompts/get failed: %v", promptGet.Error)
	}
}

// Admin endpoint tests.

func TestMCPAdminToolsEndpoints(t *testing.T) {
	s := llmock.New(llmock.WithMCP(llmock.MCPConfig{
		Tools: []llmock.MCPToolConfig{
			{Name: "initial_tool", Description: "Initial tool"},
		},
	}))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// GET: should have initial tool.
	resp, err := http.Get(ts.URL + "/_mock/mcp/tools")
	if err != nil {
		t.Fatalf("GET failed: %v", err)
	}
	defer resp.Body.Close()
	var getResult struct {
		Tools []llmock.MCPToolConfig `json:"tools"`
	}
	json.NewDecoder(resp.Body).Decode(&getResult)
	if len(getResult.Tools) != 1 || getResult.Tools[0].Name != "initial_tool" {
		t.Errorf("expected initial_tool, got %+v", getResult.Tools)
	}

	// POST: add a tool.
	addBody, _ := json.Marshal(map[string]any{
		"tools": []map[string]any{
			{"name": "added_tool", "description": "Added via admin"},
		},
	})
	postResp, err := http.Post(ts.URL+"/_mock/mcp/tools", "application/json", bytes.NewReader(addBody))
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	postResp.Body.Close()
	if postResp.StatusCode != http.StatusCreated {
		t.Errorf("expected 201, got %d", postResp.StatusCode)
	}

	// Verify added via MCP protocol.
	toolsList := mcpCall(t, ts, jsonRPCRequest{JSONRPC: "2.0", ID: 1, Method: "tools/list"})
	var listResult map[string]any
	json.Unmarshal(toolsList.Result, &listResult)
	tools := listResult["tools"].([]any)
	if len(tools) != 2 {
		t.Errorf("expected 2 tools after add, got %d", len(tools))
	}

	// DELETE: clear tools.
	delReq, _ := http.NewRequest("DELETE", ts.URL+"/_mock/mcp/tools", nil)
	delResp, err := http.DefaultClient.Do(delReq)
	if err != nil {
		t.Fatalf("DELETE failed: %v", err)
	}
	delResp.Body.Close()

	// Verify cleared.
	toolsList = mcpCall(t, ts, jsonRPCRequest{JSONRPC: "2.0", ID: 2, Method: "tools/list"})
	json.Unmarshal(toolsList.Result, &listResult)
	tools = listResult["tools"].([]any)
	if len(tools) != 0 {
		t.Errorf("expected 0 tools after delete, got %d", len(tools))
	}
}

func TestMCPAdminResourcesEndpoints(t *testing.T) {
	s := llmock.New(llmock.WithMCP(llmock.MCPConfig{}))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// POST: add resources.
	addBody, _ := json.Marshal(map[string]any{
		"resources": []map[string]any{
			{"uri": "file:///test.txt", "name": "Test", "content": "test content"},
		},
	})
	postResp, err := http.Post(ts.URL+"/_mock/mcp/resources", "application/json", bytes.NewReader(addBody))
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	postResp.Body.Close()
	if postResp.StatusCode != http.StatusCreated {
		t.Errorf("expected 201, got %d", postResp.StatusCode)
	}

	// Read via MCP.
	readResult := mcpCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0", ID: 1, Method: "resources/read",
		Params: map[string]any{"uri": "file:///test.txt"},
	})
	if readResult.Error != nil {
		t.Fatalf("resources/read failed: %v", readResult.Error)
	}

	// DELETE.
	delReq, _ := http.NewRequest("DELETE", ts.URL+"/_mock/mcp/resources", nil)
	delResp, _ := http.DefaultClient.Do(delReq)
	delResp.Body.Close()

	// Verify cleared.
	readResult = mcpCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0", ID: 2, Method: "resources/read",
		Params: map[string]any{"uri": "file:///test.txt"},
	})
	if readResult.Error == nil {
		t.Error("expected error after resources cleared")
	}
}

func TestMCPAdminPromptsEndpoints(t *testing.T) {
	s := llmock.New(llmock.WithMCP(llmock.MCPConfig{}))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// POST: add prompts.
	addBody, _ := json.Marshal(map[string]any{
		"prompts": []map[string]any{
			{"name": "greet", "description": "A greeting", "template": "Hello {{name}}!",
				"arguments": []map[string]any{{"name": "name", "required": true}}},
		},
	})
	postResp, err := http.Post(ts.URL+"/_mock/mcp/prompts", "application/json", bytes.NewReader(addBody))
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	postResp.Body.Close()
	if postResp.StatusCode != http.StatusCreated {
		t.Errorf("expected 201, got %d", postResp.StatusCode)
	}

	// Get via MCP.
	getResult := mcpCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0", ID: 1, Method: "prompts/get",
		Params: map[string]any{"name": "greet", "arguments": map[string]string{"name": "Test"}},
	})
	if getResult.Error != nil {
		t.Fatalf("prompts/get failed: %v", getResult.Error)
	}

	// DELETE.
	delReq, _ := http.NewRequest("DELETE", ts.URL+"/_mock/mcp/prompts", nil)
	delResp, _ := http.DefaultClient.Do(delReq)
	delResp.Body.Close()

	// Verify cleared.
	listResult := mcpCall(t, ts, jsonRPCRequest{JSONRPC: "2.0", ID: 2, Method: "prompts/list"})
	var list map[string]any
	json.Unmarshal(listResult.Result, &list)
	prompts := list["prompts"].([]any)
	if len(prompts) != 0 {
		t.Errorf("expected 0 prompts after delete, got %d", len(prompts))
	}
}

func TestMCPConfigIntegration(t *testing.T) {
	cfgYAML := `
mcp:
  tools:
    - name: "get_weather"
      description: "Get weather"
      input_schema:
        type: object
        properties:
          location:
            type: string
      responses:
        - pattern: ".*"
          result: '{"temp": 72}'
  resources:
    - uri: "file:///README.md"
      name: "README"
      content: "# Hello"
  prompts:
    - name: "greet"
      description: "Greet someone"
      arguments:
        - name: "name"
          required: true
      template: "Hello {{name}}!"
`
	cfg, err := llmock.ParseConfig([]byte(cfgYAML), "test.yaml")
	if err != nil {
		t.Fatalf("parsing config: %v", err)
	}

	if cfg.MCP == nil {
		t.Fatal("expected MCP config to be parsed")
	}
	if len(cfg.MCP.Tools) != 1 {
		t.Errorf("expected 1 MCP tool, got %d", len(cfg.MCP.Tools))
	}
	if len(cfg.MCP.Resources) != 1 {
		t.Errorf("expected 1 MCP resource, got %d", len(cfg.MCP.Resources))
	}
	if len(cfg.MCP.Prompts) != 1 {
		t.Errorf("expected 1 MCP prompt, got %d", len(cfg.MCP.Prompts))
	}

	opts, err := cfg.ToOptions()
	if err != nil {
		t.Fatalf("converting to options: %v", err)
	}

	s := llmock.New(opts...)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// Verify MCP endpoint works.
	result := mcpCall(t, ts, jsonRPCRequest{JSONRPC: "2.0", ID: 1, Method: "initialize"})
	if result.Error != nil {
		t.Fatalf("initialize failed: %v", result.Error)
	}
}

func TestMCPToolsCallMarkovFallback(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{
		Tools: []llmock.MCPToolConfig{
			{
				Name: "search",
				// No responses configured - should fall back to Markov.
			},
		},
	})
	defer ts.Close()

	result := mcpCall(t, ts, jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params:  map[string]any{"name": "search", "arguments": map[string]any{"q": "test"}},
	})

	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}

	var callResult map[string]any
	json.Unmarshal(result.Result, &callResult)

	content := callResult["content"].([]any)
	block := content[0].(map[string]any)
	text := block["text"].(string)
	// Should have some text (Markov generated or fallback).
	if text == "" {
		t.Error("expected non-empty result text from Markov fallback")
	}
}

func TestMCPEmptyToolsList(t *testing.T) {
	ts := mcpTestServer(llmock.MCPConfig{})
	defer ts.Close()

	result := mcpCall(t, ts, jsonRPCRequest{JSONRPC: "2.0", ID: 1, Method: "tools/list"})
	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}

	var listResult map[string]any
	json.Unmarshal(result.Result, &listResult)

	tools := listResult["tools"].([]any)
	if len(tools) != 0 {
		t.Errorf("expected 0 tools, got %d", len(tools))
	}
}
