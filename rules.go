package llmock

import (
	"fmt"
	"math/rand/v2"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"
)

// Rule is a compiled regex pattern with one or more response templates.
// Templates may use $1, $2, etc. for capture groups and ${input} for the
// full original user message. A rule may also specify a ToolCall instead
// of (or in addition to) text responses.
//
// MaxCalls limits how many times this rule's tool call fires. After that
// many invocations, the rule falls through to its text Responses instead
// (or is skipped if it has no text responses). Nil means unlimited.
type Rule struct {
	Pattern   *regexp.Regexp
	Responses []string
	ToolCall  *ToolCallConfig
	MaxCalls  *int
}

// RuleResponder matches messages against an ordered list of rules.
// The first matching rule wins. If no rule matches, the Markov fallback
// responder is used.
type RuleResponder struct {
	rules      []Rule
	markov     *MarkovResponder
	mu         sync.Mutex
	callCounts map[int]int // rule index → number of tool call invocations
}

// NewRuleResponder creates a RuleResponder from the given rules.
// If rules is empty, the built-in default rules are used.
func NewRuleResponder(rules []Rule) *RuleResponder {
	if len(rules) == 0 {
		rules = DefaultRules()
	}
	return &RuleResponder{rules: rules, callCounts: make(map[int]int)}
}

// Respond finds the first rule matching the last user message and expands
// its response template with capture groups.
func (r *RuleResponder) Respond(messages []InternalMessage) (Response, error) {
	input := extractInput(messages)
	if input == "" {
		return Response{}, errNoMessages
	}

	for i, rule := range r.rules {
		matches := rule.Pattern.FindStringSubmatch(input)
		if matches == nil {
			continue
		}
		// If this rule specifies a tool call, return a tool call response.
		if rule.ToolCall != nil {
			if rule.MaxCalls != nil {
				r.mu.Lock()
				count := r.callCounts[i]
				if count >= *rule.MaxCalls {
					r.mu.Unlock()
					// Exhausted: fall through to text responses if available.
					if len(rule.Responses) > 0 {
						template := rule.Responses[rand.IntN(len(rule.Responses))]
						return Response{Text: expandTemplate(template, matches, input, r.markov)}, nil
					}
					continue
				}
				r.callCounts[i]++
				r.mu.Unlock()
			}
			tc := resolveToolCall(*rule.ToolCall, matches, input)
			return Response{ToolCalls: []ToolCall{tc}}, nil
		}
		template := rule.Responses[rand.IntN(len(rule.Responses))]
		return Response{Text: expandTemplate(template, matches, input, r.markov)}, nil
	}

	if r.markov != nil {
		return r.markov.Respond(messages)
	}
	return Response{Text: "That's an interesting point. Could you tell me more?"}, nil
}

// expandTemplate replaces $1, $2, ... with capture group values,
// ${input} with the full original message, and {{markov}} or {{markov:N}}
// with Markov-generated text.
func expandTemplate(template string, matches []string, input string, markov *MarkovResponder) string {
	// Handle {{markov}} and {{markov:N}} placeholders first.
	if markov != nil && strings.Contains(template, "{{markov") {
		template = expandMarkovPlaceholders(template, markov)
	}

	result := make([]byte, 0, len(template)*2)
	i := 0
	for i < len(template) {
		if template[i] != '$' {
			result = append(result, template[i])
			i++
			continue
		}
		// Check for ${input}
		if i+len("${input}") <= len(template) && template[i:i+len("${input}")] == "${input}" {
			result = append(result, input...)
			i += len("${input}")
			continue
		}
		// Check for $N capture group reference (only substitute if within bounds)
		if i+1 < len(template) && template[i+1] >= '1' && template[i+1] <= '9' {
			idx := int(template[i+1] - '0')
			if idx < len(matches) {
				result = append(result, matches[idx]...)
				i += 2
				continue
			}
		}
		result = append(result, template[i])
		i++
	}
	return string(result)
}

// expandMarkovPlaceholders replaces {{markov}} and {{markov:N}} in the template.
func expandMarkovPlaceholders(template string, markov *MarkovResponder) string {
	var result strings.Builder
	i := 0
	for i < len(template) {
		if i+len("{{markov}}") <= len(template) && template[i:i+len("{{markov}}")] == "{{markov}}" {
			result.WriteString(markov.GenerateMarkov(100))
			i += len("{{markov}}")
			continue
		}
		if i+len("{{markov:") <= len(template) && template[i:i+len("{{markov:")] == "{{markov:" {
			end := strings.Index(template[i:], "}}")
			if end != -1 {
				numStr := template[i+len("{{markov:") : i+end]
				if n, err := strconv.Atoi(numStr); err == nil && n > 0 {
					result.WriteString(markov.GenerateMarkov(n))
					i += end + 2
					continue
				}
			}
		}
		result.WriteByte(template[i])
		i++
	}
	return result.String()
}

// WithRules configures the server to use a RuleResponder with the given rules.
func WithRules(rules ...Rule) Option {
	return func(s *Server) {
		s.responder = NewRuleResponder(rules)
	}
}

// WithResponder configures the server to use the given Responder.
func WithResponder(r Responder) Option {
	return func(s *Server) {
		s.responder = r
	}
}

// ruleConfig is the YAML representation of a rule (used by LoadRulesFile).
type ruleConfig struct {
	Pattern   string          `yaml:"pattern"`
	Responses []string        `yaml:"responses"`
	ToolCall  *ToolCallConfig `yaml:"tool_call,omitempty"`
	MaxCalls  *int            `yaml:"max_calls,omitempty"`
}

// rulesFileConfig is the top-level YAML structure.
type rulesFileConfig struct {
	Rules []ruleConfig `yaml:"rules"`
}

// LoadRulesFile reads a YAML file and returns compiled Rules.
func LoadRulesFile(path string) ([]Rule, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading rules file: %w", err)
	}
	return ParseRulesYAML(data)
}

// ParseRulesYAML parses YAML bytes into compiled Rules.
func ParseRulesYAML(data []byte) ([]Rule, error) {
	var cfg rulesFileConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parsing rules YAML: %w", err)
	}
	rules := make([]Rule, len(cfg.Rules))
	for i, rc := range cfg.Rules {
		re, err := regexp.Compile(rc.Pattern)
		if err != nil {
			return nil, fmt.Errorf("compiling rule %d pattern %q: %w", i, rc.Pattern, err)
		}
		if len(rc.Responses) == 0 && rc.ToolCall == nil {
			return nil, fmt.Errorf("rule %d pattern %q has no responses or tool_call", i, rc.Pattern)
		}
		rules[i] = Rule{Pattern: re, Responses: rc.Responses, ToolCall: rc.ToolCall, MaxCalls: rc.MaxCalls}
	}
	return rules, nil
}

// DefaultRules returns a set of built-in rules that produce helpful
// AI-assistant-like responses.
func DefaultRules() []Rule {
	return []Rule{
		{
			Pattern: regexp.MustCompile(`(?i)^(?:hi|hello|hey|greetings|good (?:morning|afternoon|evening))[\s!.,]*$`),
			Responses: []string{
				"Hello! How can I help you today?",
				"Hi there! What can I assist you with?",
				"Hey! What would you like to work on?",
			},
		},
		{
			Pattern: regexp.MustCompile(`(?i)I need (.*)`),
			Responses: []string{
				"I understand you need $1. Let me help you with that.",
				"Sure, I can help with $1. What specifically would you like to know?",
				"Let's work on getting you $1. Can you give me more details?",
			},
		},
		{
			Pattern: regexp.MustCompile(`(?i)how do I (.*)`),
			Responses: []string{
				"Here's how you can approach $1: first, break it down into smaller steps.",
				"To $1, I'd recommend starting with the basics and building from there.",
				"Great question! There are several ways to $1. Let me walk you through the most common approach.",
			},
		},
		{
			Pattern: regexp.MustCompile(`(?i)what is (.*)`),
			Responses: []string{
				"That's a great question. $1 refers to a concept that I can explain in detail.",
				"Good question! $1 is something worth understanding well. Let me break it down.",
				"Let me explain $1 for you in a clear and concise way.",
			},
		},
		{
			Pattern: regexp.MustCompile(`(?i)help me (.*)`),
			Responses: []string{
				"I'd be happy to help you $1. Let me break this down step by step.",
				"Of course! Let me assist you with $1. Here's what I suggest.",
				"Sure thing! To $1, here's what we should do.",
			},
		},
		{
			Pattern: regexp.MustCompile(`(?i)(?:can you|could you|would you) (.*)`),
			Responses: []string{
				"Absolutely! I can $1. Let me work on that.",
				"Sure, I'd be happy to $1. Here's what I've got.",
				"Of course! Let me $1 for you.",
			},
		},
		{
			Pattern: regexp.MustCompile(`(?i)(?:thanks|thank you|thx)[\s!.,]*$`),
			Responses: []string{
				"You're welcome! Let me know if you need anything else.",
				"Happy to help! Is there anything else I can assist with?",
				"Glad I could help! Don't hesitate to ask if you have more questions.",
			},
		},
		{
			Pattern: regexp.MustCompile(`(?i)(?:bye|goodbye|see you|farewell)[\s!.,]*$`),
			Responses: []string{
				"Goodbye! Feel free to come back anytime.",
				"See you later! Have a great day.",
				"Take care! I'm here whenever you need me.",
			},
		},
		{
			Pattern: regexp.MustCompile(`.*`),
			Responses: []string{
				"That's an interesting point about '${input}'. Could you tell me more?",
				"I see what you mean. Let me think about '${input}' and get back to you with some ideas.",
				"Interesting! Can you elaborate on '${input}'?",
			},
		},
	}
}
