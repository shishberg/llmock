package llmock

import (
	"bufio"
	"encoding/json"
	"io"
	"sync"
)

// StdioTransport runs the MCP control plane over newline-delimited JSON-RPC
// on a reader/writer pair (typically stdin/stdout). It reads one JSON-RPC
// request per line, dispatches it through the control plane, and writes the
// response as a single line followed by a newline.
type StdioTransport struct {
	cp *controlPlane
	mu sync.Mutex // serializes writes
}

// NewStdioTransport creates a StdioTransport backed by the given Server's
// control plane. The server must have admin API enabled (the default).
// Returns nil if the server has no control plane (admin API disabled).
func NewStdioTransport(s *Server) *StdioTransport {
	if s.control == nil {
		return nil
	}
	return &StdioTransport{cp: s.control}
}

// Run reads JSON-RPC requests from r line by line and writes responses to w.
// It blocks until r is exhausted (EOF) or a read error occurs. The returned
// error is nil on normal EOF.
func (st *StdioTransport) Run(r io.Reader, w io.Writer) error {
	scanner := bufio.NewScanner(r)
	// Allow up to 1 MB per line for large JSON-RPC messages.
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue // skip blank lines
		}

		resp := st.handleLine(line)
		st.writeResponse(w, resp)
	}

	if err := scanner.Err(); err != nil {
		return err
	}
	return nil
}

func (st *StdioTransport) handleLine(line []byte) jsonRPCResponse {
	var req jsonRPCRequest
	if err := json.Unmarshal(line, &req); err != nil {
		return jsonRPCResponse{
			JSONRPC: "2.0",
			Error: &jsonRPCErr{
				Code:    jsonRPCParseError,
				Message: "Parse error: " + err.Error(),
			},
		}
	}

	if req.JSONRPC != "2.0" {
		return jsonRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error: &jsonRPCErr{
				Code:    jsonRPCInvalidRequest,
				Message: "Invalid Request: jsonrpc must be \"2.0\"",
			},
		}
	}

	return st.cp.dispatch(req)
}

func (st *StdioTransport) writeResponse(w io.Writer, resp jsonRPCResponse) {
	data, _ := json.Marshal(resp)
	st.mu.Lock()
	defer st.mu.Unlock()
	w.Write(data)
	w.Write([]byte("\n"))
}
