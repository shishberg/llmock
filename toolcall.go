package llmock

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
)

// ToolCallConfig specifies a tool call to include in the response when a rule matches.
type ToolCallConfig struct {
	Name      string         `yaml:"name" json:"name"`
	Arguments map[string]any `yaml:"arguments" json:"arguments"`
}

// ToolCall represents a resolved tool call in a response.
type ToolCall struct {
	ID        string
	Name      string
	Arguments map[string]any
}

// Response is the result from a Responder. It carries either text content
// or one or more tool calls (but not both).
type Response struct {
	Text      string
	ToolCalls []ToolCall
}

// IsToolCall returns true if this response contains tool calls.
func (r Response) IsToolCall() bool {
	return len(r.ToolCalls) > 0
}

// RequestTool describes a tool definition provided in the API request.
type RequestTool struct {
	Name       string
	Parameters map[string]any // JSON Schema for the tool's parameters
}

// generateToolCallID generates a realistic-looking tool call ID.
// OpenAI uses "call_" + alphanumeric, Anthropic uses "toolu_" + alphanumeric.
func generateToolCallID(prefix string) string {
	b := make([]byte, 12)
	rand.Read(b)
	return fmt.Sprintf("%s%s", prefix, hex.EncodeToString(b))
}

// resolveToolCall creates a ToolCall from a ToolCallConfig, expanding
// argument templates with capture groups from the rule match.
func resolveToolCall(cfg ToolCallConfig, matches []string, input string) ToolCall {
	args := make(map[string]any, len(cfg.Arguments))
	for k, v := range cfg.Arguments {
		if s, ok := v.(string); ok {
			args[k] = expandToolArg(s, matches, input)
		} else {
			args[k] = v
		}
	}
	return ToolCall{
		ID:        generateToolCallID("call_"),
		Name:      cfg.Name,
		Arguments: args,
	}
}

// expandToolArg expands $1, $2, ... and ${input} in a tool argument string.
func expandToolArg(s string, matches []string, input string) string {
	result := make([]byte, 0, len(s)*2)
	i := 0
	for i < len(s) {
		if s[i] != '$' {
			result = append(result, s[i])
			i++
			continue
		}
		if i+len("${input}") <= len(s) && s[i:i+len("${input}")] == "${input}" {
			result = append(result, input...)
			i += len("${input}")
			continue
		}
		if i+1 < len(s) && s[i+1] >= '1' && s[i+1] <= '9' {
			idx := int(s[i+1] - '0')
			if idx < len(matches) {
				result = append(result, matches[idx]...)
				i += 2
				continue
			}
		}
		result = append(result, s[i])
		i++
	}
	return string(result)
}
