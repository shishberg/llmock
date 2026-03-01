package llmock

import (
	"bytes"
	"encoding/json"
	"regexp"
	"strings"
	"testing"
)

func TestCLITransportBasic(t *testing.T) {
	s := New(WithAdminAPI(false))
	tr := NewCLITransport(s)

	var buf bytes.Buffer
	code := tr.Run([]string{"-p", "hello", "--output-format", "stream-json"}, &buf)
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d", code)
	}

	events := parseStreamEvents(t, buf.String())
	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(events))
	}

	// System event.
	if events[0]["type"] != "system" {
		t.Errorf("event 0: expected type=system, got %v", events[0]["type"])
	}
	if events[0]["subtype"] != "init" {
		t.Errorf("event 0: expected subtype=init, got %v", events[0]["subtype"])
	}
	if events[0]["session_id"] == nil || events[0]["session_id"] == "" {
		t.Error("event 0: session_id should not be empty")
	}

	// Assistant event.
	if events[1]["type"] != "assistant" {
		t.Errorf("event 1: expected type=assistant, got %v", events[1]["type"])
	}
	if events[1]["subtype"] != "text" {
		t.Errorf("event 1: expected subtype=text, got %v", events[1]["subtype"])
	}
	if events[1]["text"] == nil || events[1]["text"] == "" {
		t.Error("event 1: text should not be empty")
	}

	// Result event.
	if events[2]["type"] != "result" {
		t.Errorf("event 2: expected type=result, got %v", events[2]["type"])
	}
	if events[2]["result"] == nil || events[2]["result"] == "" {
		t.Error("event 2: result should not be empty")
	}
}

func TestCLITransportSessionIDConsistency(t *testing.T) {
	s := New(WithAdminAPI(false))
	tr := NewCLITransport(s)

	var buf bytes.Buffer
	tr.Run([]string{"-p", "hello", "--output-format", "stream-json"}, &buf)

	events := parseStreamEvents(t, buf.String())
	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(events))
	}

	systemSID := events[0]["session_id"]
	resultSID := events[2]["session_id"]
	if systemSID != resultSID {
		t.Errorf("session_id mismatch: system=%v result=%v", systemSID, resultSID)
	}
}

func TestCLITransportSessionIDFlag(t *testing.T) {
	s := New(WithAdminAPI(false))
	tr := NewCLITransport(s)

	var buf bytes.Buffer
	tr.Run([]string{"-p", "hello", "--session-id", "my-custom-session-123", "--output-format", "stream-json"}, &buf)

	events := parseStreamEvents(t, buf.String())
	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(events))
	}

	if events[0]["session_id"] != "my-custom-session-123" {
		t.Errorf("expected session_id=my-custom-session-123, got %v", events[0]["session_id"])
	}
	if events[2]["session_id"] != "my-custom-session-123" {
		t.Errorf("expected result session_id=my-custom-session-123, got %v", events[2]["session_id"])
	}
}

func TestCLITransportResumeFlag(t *testing.T) {
	s := New(WithAdminAPI(false))
	tr := NewCLITransport(s)

	var buf bytes.Buffer
	tr.Run([]string{"-p", "hello", "--resume", "resume-session-456", "--output-format", "stream-json"}, &buf)

	events := parseStreamEvents(t, buf.String())
	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(events))
	}

	if events[0]["session_id"] != "resume-session-456" {
		t.Errorf("expected session_id=resume-session-456, got %v", events[0]["session_id"])
	}
}

func TestCLITransportModelFlag(t *testing.T) {
	s := New(WithAdminAPI(false))
	tr := NewCLITransport(s)

	var buf bytes.Buffer
	tr.Run([]string{"-p", "hello", "--model", "claude-haiku-2025", "--output-format", "stream-json"}, &buf)

	events := parseStreamEvents(t, buf.String())
	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(events))
	}

	if events[0]["model"] != "claude-haiku-2025" {
		t.Errorf("expected model=claude-haiku-2025, got %v", events[0]["model"])
	}
}

func TestCLITransportExitCode(t *testing.T) {
	s := New(WithAdminAPI(false))
	tr := NewCLITransport(s, WithExitCode(42))

	var buf bytes.Buffer
	code := tr.Run([]string{"-p", "hello", "--output-format", "stream-json"}, &buf)
	if code != 42 {
		t.Errorf("expected exit code 42, got %d", code)
	}
}

func TestCLITransportWithSessionIDOption(t *testing.T) {
	s := New(WithAdminAPI(false))
	tr := NewCLITransport(s, WithSessionID("option-session-789"))

	var buf bytes.Buffer
	tr.Run([]string{"-p", "hello", "--output-format", "stream-json"}, &buf)

	events := parseStreamEvents(t, buf.String())
	if events[0]["session_id"] != "option-session-789" {
		t.Errorf("expected session_id=option-session-789, got %v", events[0]["session_id"])
	}
}

func TestCLITransportArgSessionIDOverridesOption(t *testing.T) {
	s := New(WithAdminAPI(false))
	tr := NewCLITransport(s, WithSessionID("option-session"))

	var buf bytes.Buffer
	tr.Run([]string{"-p", "hello", "--session-id", "arg-session", "--output-format", "stream-json"}, &buf)

	events := parseStreamEvents(t, buf.String())
	// --session-id flag should override the CLIOption.
	if events[0]["session_id"] != "arg-session" {
		t.Errorf("expected session_id=arg-session, got %v", events[0]["session_id"])
	}
}

func TestCLITransportRuleMatching(t *testing.T) {
	rules := []Rule{
		{
			Pattern:   regexp.MustCompile(`(?i)hello`),
			Responses: []string{"Greetings from the mock!"},
		},
	}
	s := New(WithRules(rules...), WithAdminAPI(false))
	tr := NewCLITransport(s)

	var buf bytes.Buffer
	tr.Run([]string{"-p", "hello", "--output-format", "stream-json"}, &buf)

	events := parseStreamEvents(t, buf.String())
	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(events))
	}

	if events[1]["text"] != "Greetings from the mock!" {
		t.Errorf("expected response 'Greetings from the mock!', got %v", events[1]["text"])
	}
	if events[2]["result"] != "Greetings from the mock!" {
		t.Errorf("expected result 'Greetings from the mock!', got %v", events[2]["result"])
	}
}

func TestCLITransportNoPrompt(t *testing.T) {
	s := New(WithAdminAPI(false))
	tr := NewCLITransport(s)

	var buf bytes.Buffer
	code := tr.Run([]string{"--output-format", "stream-json"}, &buf)
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}

	events := parseStreamEvents(t, buf.String())
	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(events))
	}

	// Should get default response.
	if events[1]["text"] == nil || events[1]["text"] == "" {
		t.Error("expected non-empty text for no-prompt case")
	}
}

func TestCLITransportDefaultModel(t *testing.T) {
	s := New(WithAdminAPI(false))
	tr := NewCLITransport(s)

	var buf bytes.Buffer
	tr.Run([]string{"-p", "hello", "--output-format", "stream-json"}, &buf)

	events := parseStreamEvents(t, buf.String())
	if events[0]["model"] != "claude-sonnet-4-20250514" {
		t.Errorf("expected default model claude-sonnet-4-20250514, got %v", events[0]["model"])
	}
}

// parseStreamEvents parses newline-delimited JSON events from stream-json output.
func parseStreamEvents(t *testing.T, output string) []map[string]any {
	t.Helper()
	var events []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(output), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			t.Fatalf("failed to parse event JSON %q: %v", line, err)
		}
		events = append(events, event)
	}
	return events
}
