package llmock

import (
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"sync"
)

// JSON-RPC 2.0 types for MCP protocol.

// jsonRPCRequest represents a JSON-RPC 2.0 request message.
type jsonRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// jsonRPCResponse represents a JSON-RPC 2.0 response message.
type jsonRPCResponse struct {
	JSONRPC string       `json:"jsonrpc"`
	ID      any          `json:"id,omitempty"`
	Result  any          `json:"result,omitempty"`
	Error   *jsonRPCErr  `json:"error,omitempty"`
}

// jsonRPCErr represents a JSON-RPC 2.0 error object.
type jsonRPCErr struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

// Standard JSON-RPC 2.0 error codes.
const (
	jsonRPCParseError     = -32700
	jsonRPCInvalidRequest = -32600
	jsonRPCMethodNotFound = -32601
	jsonRPCInvalidParams  = -32602
)

// MCP configuration types.

// MCPToolConfig describes a tool advertised by the MCP server.
type MCPToolConfig struct {
	Name        string         `yaml:"name" json:"name"`
	Description string         `yaml:"description" json:"description"`
	InputSchema map[string]any `yaml:"input_schema" json:"input_schema"`
	Responses   []MCPToolResponse `yaml:"responses" json:"responses"`
}

// MCPToolResponse is a pattern-matched response for an MCP tool call.
type MCPToolResponse struct {
	Pattern string `yaml:"pattern" json:"pattern"`
	Result  string `yaml:"result" json:"result"`
}

// MCPResourceConfig describes a resource advertised by the MCP server.
type MCPResourceConfig struct {
	URI      string `yaml:"uri" json:"uri"`
	Name     string `yaml:"name" json:"name"`
	MimeType string `yaml:"mime_type,omitempty" json:"mime_type,omitempty"`
	Content  string `yaml:"content" json:"content"`
}

// MCPPromptConfig describes a prompt advertised by the MCP server.
type MCPPromptConfig struct {
	Name        string              `yaml:"name" json:"name"`
	Description string              `yaml:"description" json:"description"`
	Arguments   []MCPPromptArgument `yaml:"arguments" json:"arguments"`
	Template    string              `yaml:"template" json:"template"`
}

// MCPPromptArgument describes an argument to an MCP prompt.
type MCPPromptArgument struct {
	Name     string `yaml:"name" json:"name"`
	Required bool   `yaml:"required" json:"required"`
}

// MCPConfig holds the full MCP configuration section.
type MCPConfig struct {
	Tools     []MCPToolConfig     `yaml:"tools" json:"tools"`
	Resources []MCPResourceConfig `yaml:"resources" json:"resources"`
	Prompts   []MCPPromptConfig   `yaml:"prompts" json:"prompts"`
}

// mcpState holds the runtime MCP state (tools, resources, prompts).
type mcpState struct {
	mu        sync.RWMutex
	tools     []MCPToolConfig
	resources []MCPResourceConfig
	prompts   []MCPPromptConfig
	// Keep initial state for reset support.
	initialTools     []MCPToolConfig
	initialResources []MCPResourceConfig
	initialPrompts   []MCPPromptConfig
}

func newMCPState(cfg MCPConfig) *mcpState {
	return &mcpState{
		tools:            cloneSlice(cfg.Tools),
		resources:        cloneSlice(cfg.Resources),
		prompts:          cloneSlice(cfg.Prompts),
		initialTools:     cloneSlice(cfg.Tools),
		initialResources: cloneSlice(cfg.Resources),
		initialPrompts:   cloneSlice(cfg.Prompts),
	}
}

func cloneSlice[T any](s []T) []T {
	if s == nil {
		return nil
	}
	cp := make([]T, len(s))
	copy(cp, s)
	return cp
}

func (m *mcpState) getTools() []MCPToolConfig {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return cloneSlice(m.tools)
}

func (m *mcpState) getResources() []MCPResourceConfig {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return cloneSlice(m.resources)
}

func (m *mcpState) getPrompts() []MCPPromptConfig {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return cloneSlice(m.prompts)
}

func (m *mcpState) setTools(tools []MCPToolConfig) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.tools = cloneSlice(tools)
}

func (m *mcpState) setResources(resources []MCPResourceConfig) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.resources = cloneSlice(resources)
}

func (m *mcpState) setPrompts(prompts []MCPPromptConfig) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.prompts = cloneSlice(prompts)
}

func (m *mcpState) addTools(tools []MCPToolConfig) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.tools = append(m.tools, tools...)
}

func (m *mcpState) addResources(resources []MCPResourceConfig) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.resources = append(m.resources, resources...)
}

func (m *mcpState) addPrompts(prompts []MCPPromptConfig) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.prompts = append(m.prompts, prompts...)
}

func (m *mcpState) reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.tools = cloneSlice(m.initialTools)
	m.resources = cloneSlice(m.initialResources)
	m.prompts = cloneSlice(m.initialPrompts)
}

// WithMCP enables the MCP server with the given configuration.
func WithMCP(cfg MCPConfig) Option {
	return func(s *Server) {
		s.mcpEnabled = true
		s.mcpConfig = cfg
	}
}

// handleMCP handles POST /mcp requests using JSON-RPC 2.0.
func (s *Server) handleMCP(w http.ResponseWriter, r *http.Request) {
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

	resp := s.dispatchMCP(req)
	writeJSONRPC(w, resp)
}

// dispatchMCP routes an MCP JSON-RPC request to the appropriate handler.
func (s *Server) dispatchMCP(req jsonRPCRequest) jsonRPCResponse {
	switch req.Method {
	case "initialize":
		return s.mcpInitialize(req)
	case "tools/list":
		return s.mcpToolsList(req)
	case "tools/call":
		return s.mcpToolsCall(req)
	case "resources/list":
		return s.mcpResourcesList(req)
	case "resources/read":
		return s.mcpResourcesRead(req)
	case "prompts/list":
		return s.mcpPromptsList(req)
	case "prompts/get":
		return s.mcpPromptsGet(req)
	default:
		return jsonRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error: &jsonRPCErr{
				Code:    jsonRPCMethodNotFound,
				Message: fmt.Sprintf("Method not found: %s", req.Method),
			},
		}
	}
}

func (s *Server) mcpInitialize(req jsonRPCRequest) jsonRPCResponse {
	return jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
		Result: map[string]any{
			"protocolVersion": "2025-03-26",
			"serverInfo": map[string]any{
				"name":    "llmock",
				"version": "1.0.0",
			},
			"capabilities": map[string]any{
				"tools":     map[string]any{},
				"resources": map[string]any{},
				"prompts":   map[string]any{},
			},
		},
	}
}

func (s *Server) mcpToolsList(req jsonRPCRequest) jsonRPCResponse {
	tools := s.mcp.getTools()
	toolList := make([]map[string]any, len(tools))
	for i, t := range tools {
		entry := map[string]any{
			"name":        t.Name,
			"description": t.Description,
		}
		if t.InputSchema != nil {
			entry["inputSchema"] = t.InputSchema
		}
		toolList[i] = entry
	}
	return jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
		Result: map[string]any{
			"tools": toolList,
		},
	}
}

func (s *Server) mcpToolsCall(req jsonRPCRequest) jsonRPCResponse {
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

	// Find the tool.
	tools := s.mcp.getTools()
	var tool *MCPToolConfig
	for i := range tools {
		if tools[i].Name == params.Name {
			tool = &tools[i]
			break
		}
	}
	if tool == nil {
		return jsonRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error: &jsonRPCErr{
				Code:    jsonRPCInvalidParams,
				Message: fmt.Sprintf("Unknown tool: %s", params.Name),
			},
		}
	}

	// Match arguments against response patterns.
	argsJSON, _ := json.Marshal(params.Arguments)
	argsStr := string(argsJSON)

	resultText := ""
	for _, resp := range tool.Responses {
		re, err := regexp.Compile(resp.Pattern)
		if err != nil {
			continue
		}
		if re.MatchString(argsStr) {
			resultText = resp.Result
			break
		}
	}

	// If no pattern matched, use Markov fallback.
	if resultText == "" && s.markov != nil {
		resultText = s.markov.GenerateMarkov(50)
	}
	if resultText == "" {
		resultText = "{}"
	}

	return jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
		Result: map[string]any{
			"content": []map[string]any{
				{
					"type": "text",
					"text": resultText,
				},
			},
		},
	}
}

func (s *Server) mcpResourcesList(req jsonRPCRequest) jsonRPCResponse {
	resources := s.mcp.getResources()
	resourceList := make([]map[string]any, len(resources))
	for i, r := range resources {
		entry := map[string]any{
			"uri":  r.URI,
			"name": r.Name,
		}
		if r.MimeType != "" {
			entry["mimeType"] = r.MimeType
		}
		resourceList[i] = entry
	}
	return jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
		Result: map[string]any{
			"resources": resourceList,
		},
	}
}

func (s *Server) mcpResourcesRead(req jsonRPCRequest) jsonRPCResponse {
	var params struct {
		URI string `json:"uri"`
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

	if params.URI == "" {
		return jsonRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error: &jsonRPCErr{
				Code:    jsonRPCInvalidParams,
				Message: "Invalid params: uri is required",
			},
		}
	}

	resources := s.mcp.getResources()
	for _, r := range resources {
		if r.URI == params.URI {
			mimeType := r.MimeType
			if mimeType == "" {
				mimeType = "text/plain"
			}
			return jsonRPCResponse{
				JSONRPC: "2.0",
				ID:      req.ID,
				Result: map[string]any{
					"contents": []map[string]any{
						{
							"uri":      r.URI,
							"mimeType": mimeType,
							"text":     r.Content,
						},
					},
				},
			}
		}
	}

	return jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
		Error: &jsonRPCErr{
			Code:    jsonRPCInvalidParams,
			Message: fmt.Sprintf("Resource not found: %s", params.URI),
		},
	}
}

func (s *Server) mcpPromptsList(req jsonRPCRequest) jsonRPCResponse {
	prompts := s.mcp.getPrompts()
	promptList := make([]map[string]any, len(prompts))
	for i, p := range prompts {
		args := make([]map[string]any, len(p.Arguments))
		for j, a := range p.Arguments {
			args[j] = map[string]any{
				"name":     a.Name,
				"required": a.Required,
			}
		}
		promptList[i] = map[string]any{
			"name":        p.Name,
			"description": p.Description,
			"arguments":   args,
		}
	}
	return jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
		Result: map[string]any{
			"prompts": promptList,
		},
	}
}

func (s *Server) mcpPromptsGet(req jsonRPCRequest) jsonRPCResponse {
	var params struct {
		Name      string            `json:"name"`
		Arguments map[string]string `json:"arguments"`
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

	prompts := s.mcp.getPrompts()
	for _, p := range prompts {
		if p.Name == params.Name {
			// Expand template: replace {{argName}} with argument values.
			text := p.Template
			for k, v := range params.Arguments {
				text = replaceAll(text, "{{"+k+"}}", v)
			}
			return jsonRPCResponse{
				JSONRPC: "2.0",
				ID:      req.ID,
				Result: map[string]any{
					"messages": []map[string]any{
						{
							"role": "user",
							"content": map[string]any{
								"type": "text",
								"text": text,
							},
						},
					},
				},
			}
		}
	}

	return jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
		Error: &jsonRPCErr{
			Code:    jsonRPCInvalidParams,
			Message: fmt.Sprintf("Prompt not found: %s", params.Name),
		},
	}
}

// replaceAll is a simple string replacement (avoids importing strings just for this).
func replaceAll(s, old, new string) string {
	result := make([]byte, 0, len(s))
	i := 0
	for i < len(s) {
		if i+len(old) <= len(s) && s[i:i+len(old)] == old {
			result = append(result, new...)
			i += len(old)
		} else {
			result = append(result, s[i])
			i++
		}
	}
	return string(result)
}

func writeJSONRPC(w http.ResponseWriter, resp jsonRPCResponse) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// registerMCPAdminRoutes adds the /_mock/mcp/* endpoints to the mux.
func registerMCPAdminRoutes(mux *http.ServeMux, state *mcpState) {
	// Tools
	mux.HandleFunc("GET /_mock/mcp/tools", func(w http.ResponseWriter, r *http.Request) {
		tools := state.getTools()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"tools": tools})
	})

	mux.HandleFunc("POST /_mock/mcp/tools", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Tools []MCPToolConfig `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
			return
		}
		if len(req.Tools) == 0 {
			writeError(w, http.StatusBadRequest, "tools array is required and must not be empty")
			return
		}
		state.addTools(req.Tools)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("DELETE /_mock/mcp/tools", func(w http.ResponseWriter, r *http.Request) {
		state.setTools(nil)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	// Resources
	mux.HandleFunc("GET /_mock/mcp/resources", func(w http.ResponseWriter, r *http.Request) {
		resources := state.getResources()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"resources": resources})
	})

	mux.HandleFunc("POST /_mock/mcp/resources", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Resources []MCPResourceConfig `json:"resources"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
			return
		}
		if len(req.Resources) == 0 {
			writeError(w, http.StatusBadRequest, "resources array is required and must not be empty")
			return
		}
		state.addResources(req.Resources)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("DELETE /_mock/mcp/resources", func(w http.ResponseWriter, r *http.Request) {
		state.setResources(nil)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	// Prompts
	mux.HandleFunc("GET /_mock/mcp/prompts", func(w http.ResponseWriter, r *http.Request) {
		prompts := state.getPrompts()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"prompts": prompts})
	})

	mux.HandleFunc("POST /_mock/mcp/prompts", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Prompts []MCPPromptConfig `json:"prompts"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
			return
		}
		if len(req.Prompts) == 0 {
			writeError(w, http.StatusBadRequest, "prompts array is required and must not be empty")
			return
		}
		state.addPrompts(req.Prompts)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("DELETE /_mock/mcp/prompts", func(w http.ResponseWriter, r *http.Request) {
		state.setPrompts(nil)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})
}
