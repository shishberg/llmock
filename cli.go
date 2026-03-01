package llmock

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// CLITransport mocks the claude CLI binary. It parses command-line arguments,
// generates a response using the server's responder, and writes newline-delimited
// JSON events in claude's stream-json format.
type CLITransport struct {
	server    *Server
	sessionID string // auto-generated if not provided
	exitCode  int    // configurable exit code (default 0)
}

// CLIOption configures a CLITransport.
type CLIOption func(*CLITransport)

// WithExitCode sets the exit code returned by Run.
func WithExitCode(code int) CLIOption {
	return func(t *CLITransport) {
		t.exitCode = code
	}
}

// WithSessionID sets a fixed session ID instead of auto-generating one.
func WithSessionID(id string) CLIOption {
	return func(t *CLITransport) {
		t.sessionID = id
	}
}

// NewCLITransport creates a CLITransport backed by the given Server.
func NewCLITransport(s *Server, opts ...CLIOption) *CLITransport {
	t := &CLITransport{server: s}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// cliStreamEvent is a JSON event in claude's stream-json output format.
type cliStreamEvent struct {
	Type       string `json:"type"`
	Subtype    string `json:"subtype"`
	SessionID  string `json:"session_id,omitempty"`
	Tools      []any  `json:"tools,omitempty"`
	Model      string `json:"model,omitempty"`
	CWD        string `json:"cwd,omitempty"`
	Text       string `json:"text,omitempty"`
	Result     string `json:"result,omitempty"`
	IsError    bool   `json:"is_error,omitempty"`
	DurationMS int    `json:"duration_ms,omitempty"`
	DurationAPI int   `json:"duration_api_ms,omitempty"`
	NumTurns   int    `json:"num_turns,omitempty"`
	CostUSD    float64 `json:"cost_usd,omitempty"`
}

// Run parses CLI args, generates a response, and writes stream-json events to w.
// It returns the configured exit code.
func (t *CLITransport) Run(args []string, w io.Writer) int {
	prompt, model, sessionID, _ := parseCLIArgs(args)

	// Determine session ID: arg flag > CLIOption > auto-generate.
	sid := sessionID
	if sid == "" {
		sid = t.sessionID
	}
	if sid == "" {
		sid = fmt.Sprintf("cli-%s", randomHex(16))
	}

	if model == "" {
		model = "claude-sonnet-4-20250514"
	}

	// Generate response from the server's responder.
	var responseText string
	if prompt != "" {
		messages := []InternalMessage{{Role: "user", Content: prompt}}
		resp, err := t.server.responder.Respond(messages)
		if err != nil {
			responseText = "I understand. Could you tell me more about that?"
		} else {
			responseText = resp.Text
		}
	} else {
		responseText = "Hello! How can I help you today?"
	}

	cwd, _ := os.Getwd()
	enc := json.NewEncoder(w)

	// System event.
	enc.Encode(cliStreamEvent{
		Type:      "system",
		Subtype:   "init",
		SessionID: sid,
		Tools:     []any{},
		Model:     model,
		CWD:       cwd,
	})

	// Assistant event.
	enc.Encode(cliStreamEvent{
		Type:    "assistant",
		Subtype: "text",
		Text:    responseText,
	})

	// Result event.
	enc.Encode(cliStreamEvent{
		Type:        "result",
		Subtype:     "",
		SessionID:   sid,
		Result:      responseText,
		DurationMS:  100,
		DurationAPI: 80,
		NumTurns:    1,
		CostUSD:     0.001,
	})

	return t.exitCode
}

// parseCLIArgs extracts relevant flags from claude CLI arguments.
// It returns prompt, model, sessionID, and outputFormat.
func parseCLIArgs(args []string) (prompt, model, sessionID, outputFormat string) {
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "-p":
			if i+1 < len(args) {
				i++
				prompt = args[i]
			}
		case "--model":
			if i+1 < len(args) {
				i++
				model = args[i]
			}
		case "--session-id":
			if i+1 < len(args) {
				i++
				sessionID = args[i]
			}
		case "--resume":
			if i+1 < len(args) {
				i++
				sessionID = args[i]
			}
		case "--output-format":
			if i+1 < len(args) {
				i++
				outputFormat = args[i]
			}
		case "--permission-mode", "--max-turns", "--allowedTools":
			// Skip the value for flags we parse but don't use.
			if i+1 < len(args) {
				i++
			}
		}
	}
	return
}
