package llmock

import (
	"encoding/json"
	"net/http"
	"regexp"
)

// controlPlane handles MCP control plane requests (POST /mcp/control).
// It exposes llmock's admin functionality as MCP tools so that an AI agent
// can control llmock's behavior via MCP.
type controlPlane struct {
	admin  *adminState
	faults *faultState
}

// controlToolDef describes an MCP tool for the tools/list response.
type controlToolDef struct {
	name        string
	description string
	inputSchema map[string]any
}

var controlTools = []controlToolDef{
	{
		name:        "llmock_add_rule",
		description: "Add a response rule. The rule's regex pattern is matched against incoming user messages; when matched, one of the responses is returned.",
		inputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"pattern":   map[string]any{"type": "string", "description": "Regex pattern to match against user messages"},
				"responses": map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Response templates (one is chosen randomly)"},
				"priority":  map[string]any{"type": "integer", "description": "0=prepend (default), -1=append, N=insert at index N"},
			},
			"required": []string{"pattern", "responses"},
		},
	},
	{
		name:        "llmock_list_rules",
		description: "List all current response rules with their patterns and responses.",
		inputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	},
	{
		name:        "llmock_reset_rules",
		description: "Reset rules to the initial startup configuration.",
		inputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	},
	{
		name:        "llmock_add_fault",
		description: "Add a fault injection. Types: error (HTTP error), delay (latency), timeout (hang), malformed (bad response), rate_limit (429).",
		inputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"type":        map[string]any{"type": "string", "enum": []string{"error", "delay", "timeout", "malformed", "rate_limit"}, "description": "Fault type"},
				"status":      map[string]any{"type": "integer", "description": "HTTP status code (for error faults)"},
				"message":     map[string]any{"type": "string", "description": "Error message"},
				"delay_ms":    map[string]any{"type": "integer", "description": "Delay in milliseconds (for delay faults)"},
				"probability": map[string]any{"type": "number", "description": "Probability of firing (0-1, default 1)"},
				"count":       map[string]any{"type": "integer", "description": "Auto-clear after N triggers (0=unlimited)"},
			},
			"required": []string{"type"},
		},
	},
	{
		name:        "llmock_list_faults",
		description: "List all active fault injections.",
		inputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	},
	{
		name:        "llmock_clear_faults",
		description: "Clear all active fault injections.",
		inputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	},
	{
		name:        "llmock_list_requests",
		description: "View the recent request log (last 100 requests).",
		inputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	},
	{
		name:        "llmock_clear_requests",
		description: "Clear the request log.",
		inputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	},
	{
		name:        "llmock_reset",
		description: "Full reset: restore rules to initial config, clear all faults, and clear the request log.",
		inputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	},
}

// handleControl handles POST /mcp/control using JSON-RPC 2.0.
func (cp *controlPlane) handleControl(w http.ResponseWriter, r *http.Request) {
	var req jsonRPCRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSONRPC(w, jsonRPCResponse{
			JSONRPC: "2.0",
			Error: &jsonRPCErr{
				Code:    jsonRPCParseError,
				Message: "Parse error: " + err.Error(),
			},
		})
		return
	}

	if req.JSONRPC != "2.0" {
		writeJSONRPC(w, jsonRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error: &jsonRPCErr{
				Code:    jsonRPCInvalidRequest,
				Message: "Invalid Request: jsonrpc must be \"2.0\"",
			},
		})
		return
	}

	resp := cp.dispatch(req)
	writeJSONRPC(w, resp)
}

func (cp *controlPlane) dispatch(req jsonRPCRequest) jsonRPCResponse {
	switch req.Method {
	case "initialize":
		return cp.initialize(req)
	case "tools/list":
		return cp.toolsList(req)
	case "tools/call":
		return cp.toolsCall(req)
	default:
		return jsonRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error: &jsonRPCErr{
				Code:    jsonRPCMethodNotFound,
				Message: "Method not found: " + req.Method,
			},
		}
	}
}

func (cp *controlPlane) initialize(req jsonRPCRequest) jsonRPCResponse {
	return jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
		Result: map[string]any{
			"protocolVersion": "2025-03-26",
			"serverInfo": map[string]any{
				"name":    "llmock-control",
				"version": "1.0.0",
			},
			"capabilities": map[string]any{
				"tools": map[string]any{},
			},
		},
	}
}

func (cp *controlPlane) toolsList(req jsonRPCRequest) jsonRPCResponse {
	tools := make([]map[string]any, len(controlTools))
	for i, t := range controlTools {
		tools[i] = map[string]any{
			"name":        t.name,
			"description": t.description,
			"inputSchema": t.inputSchema,
		}
	}
	return jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
		Result: map[string]any{
			"tools": tools,
		},
	}
}

func (cp *controlPlane) toolsCall(req jsonRPCRequest) jsonRPCResponse {
	var params struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	}
	if req.Params != nil {
		if err := json.Unmarshal(req.Params, &params); err != nil {
			return jsonRPCResponse{
				JSONRPC: "2.0",
				ID:      req.ID,
				Error: &jsonRPCErr{
					Code:    jsonRPCInvalidParams,
					Message: "Invalid params: " + err.Error(),
				},
			}
		}
	}

	if params.Name == "" {
		return jsonRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error: &jsonRPCErr{
				Code:    jsonRPCInvalidParams,
				Message: "Invalid params: name is required",
			},
		}
	}

	var result string
	var callErr error

	switch params.Name {
	case "llmock_add_rule":
		result, callErr = cp.callAddRule(params.Arguments)
	case "llmock_list_rules":
		result, callErr = cp.callListRules()
	case "llmock_reset_rules":
		result, callErr = cp.callResetRules()
	case "llmock_add_fault":
		result, callErr = cp.callAddFault(params.Arguments)
	case "llmock_list_faults":
		result, callErr = cp.callListFaults()
	case "llmock_clear_faults":
		result, callErr = cp.callClearFaults()
	case "llmock_list_requests":
		result, callErr = cp.callListRequests()
	case "llmock_clear_requests":
		result, callErr = cp.callClearRequests()
	case "llmock_reset":
		result, callErr = cp.callReset()
	default:
		return jsonRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error: &jsonRPCErr{
				Code:    jsonRPCInvalidParams,
				Message: "Unknown tool: " + params.Name,
			},
		}
	}

	if callErr != nil {
		return jsonRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result: map[string]any{
				"content": []map[string]any{
					{"type": "text", "text": callErr.Error()},
				},
				"isError": true,
			},
		}
	}

	return jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
		Result: map[string]any{
			"content": []map[string]any{
				{"type": "text", "text": result},
			},
		},
	}
}

func (cp *controlPlane) callAddRule(args map[string]any) (string, error) {
	patternStr, _ := args["pattern"].(string)
	if patternStr == "" {
		return "", &controlError{"pattern is required"}
	}
	re, err := regexp.Compile(patternStr)
	if err != nil {
		return "", &controlError{"invalid regex: " + err.Error()}
	}

	responsesRaw, _ := args["responses"].([]any)
	if len(responsesRaw) == 0 {
		return "", &controlError{"responses is required and must not be empty"}
	}
	responses := make([]string, len(responsesRaw))
	for i, r := range responsesRaw {
		s, ok := r.(string)
		if !ok {
			return "", &controlError{"responses must be an array of strings"}
		}
		responses[i] = s
	}

	priority := 0
	if p, ok := args["priority"]; ok {
		// JSON numbers are float64.
		if pf, ok := p.(float64); ok {
			priority = int(pf)
		}
	}

	cp.admin.addRules([]Rule{
		{Pattern: re, Responses: responses},
	}, priority)

	return "Rule added successfully", nil
}

func (cp *controlPlane) callListRules() (string, error) {
	rules := cp.admin.getRulesJSON()
	data, _ := json.Marshal(rules)
	return string(data), nil
}

func (cp *controlPlane) callResetRules() (string, error) {
	cp.admin.resetRules()
	return "Rules reset to initial configuration", nil
}

func (cp *controlPlane) callAddFault(args map[string]any) (string, error) {
	typeStr, _ := args["type"].(string)
	if typeStr == "" {
		return "", &controlError{"type is required"}
	}

	f := Fault{Type: FaultType(typeStr)}
	if v, ok := args["status"].(float64); ok {
		f.Status = int(v)
	}
	if v, ok := args["message"].(string); ok {
		f.Message = v
	}
	if v, ok := args["error_type"].(string); ok {
		f.ErrorType = v
	}
	if v, ok := args["delay_ms"].(float64); ok {
		f.DelayMS = int(v)
	}
	if v, ok := args["probability"].(float64); ok {
		f.Probability = v
	}
	if v, ok := args["count"].(float64); ok {
		f.Count = int(v)
	}

	cp.faults.addFaults([]Fault{f})
	return "Fault added successfully", nil
}

func (cp *controlPlane) callListFaults() (string, error) {
	faults := cp.faults.getFaults()
	data, _ := json.Marshal(faults)
	return string(data), nil
}

func (cp *controlPlane) callClearFaults() (string, error) {
	cp.faults.clear()
	return "All faults cleared", nil
}

func (cp *controlPlane) callListRequests() (string, error) {
	requests := cp.admin.getRequests()
	data, _ := json.Marshal(requests)
	return string(data), nil
}

func (cp *controlPlane) callClearRequests() (string, error) {
	cp.admin.clearRequests()
	return "Request log cleared", nil
}

func (cp *controlPlane) callReset() (string, error) {
	cp.admin.fullReset()
	cp.faults.clear()
	return "Full reset complete", nil
}

// controlError is a simple error type for tool call validation errors.
type controlError struct {
	msg string
}

func (e *controlError) Error() string {
	return e.msg
}
