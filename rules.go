package llmock

import (
	"fmt"
	"math/rand/v2"
	"os"
	"regexp"

	"gopkg.in/yaml.v3"
)

// Rule is a compiled regex pattern with one or more response templates.
// Templates may use $1, $2, etc. for capture groups and ${input} for the
// full original user message.
type Rule struct {
	Pattern   *regexp.Regexp
	Responses []string
}

// RuleResponder matches messages against an ordered list of rules.
// The first matching rule wins. If no rule matches, a default fallback
// response is returned.
type RuleResponder struct {
	rules []Rule
}

// NewRuleResponder creates a RuleResponder from the given rules.
// If rules is empty, the built-in default rules are used.
func NewRuleResponder(rules []Rule) *RuleResponder {
	if len(rules) == 0 {
		rules = DefaultRules()
	}
	return &RuleResponder{rules: rules}
}

// Respond finds the first rule matching the last user message and expands
// its response template with capture groups.
func (r *RuleResponder) Respond(messages []InternalMessage) (string, error) {
	var input string
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			input = messages[i].Content
			break
		}
	}
	if input == "" {
		if len(messages) > 0 {
			input = messages[len(messages)-1].Content
		} else {
			return "", fmt.Errorf("no messages provided")
		}
	}

	for _, rule := range r.rules {
		matches := rule.Pattern.FindStringSubmatch(input)
		if matches == nil {
			continue
		}
		template := rule.Responses[rand.IntN(len(rule.Responses))]
		return expandTemplate(template, matches, input), nil
	}

	return "That's an interesting point. Could you tell me more?", nil
}

// expandTemplate replaces $1, $2, ... with capture group values
// and ${input} with the full original message.
func expandTemplate(template string, matches []string, input string) string {
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

// ruleConfig is the YAML representation of a rule.
type ruleConfig struct {
	Pattern   string   `yaml:"pattern"`
	Responses []string `yaml:"responses"`
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
		if len(rc.Responses) == 0 {
			return nil, fmt.Errorf("rule %d pattern %q has no responses", i, rc.Pattern)
		}
		rules[i] = Rule{Pattern: re, Responses: rc.Responses}
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
