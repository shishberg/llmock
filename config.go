package llmock

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Config represents the full configuration for an llmock server.
// It can be loaded from a YAML or JSON file, or built programmatically
// using functional options.
type Config struct {
	Server   ServerConfig  `yaml:"server" json:"server"`
	Defaults DefaultConfig `yaml:"defaults" json:"defaults"`
	Rules    []RuleConfig  `yaml:"rules" json:"rules"`

	CorpusFile string  `yaml:"corpus_file" json:"corpus_file"`
	Faults     []Fault `yaml:"faults" json:"faults"`
}

// ServerConfig holds server-level settings.
type ServerConfig struct {
	Port     int  `yaml:"port" json:"port"`
	AdminAPI *bool `yaml:"admin_api" json:"admin_api"`
}

// DefaultConfig holds default response behavior settings.
type DefaultConfig struct {
	TokenDelayMS int    `yaml:"token_delay_ms" json:"token_delay_ms"`
	Seed         *int64 `yaml:"seed" json:"seed"`
	Model        string `yaml:"model" json:"model"`
}

// RuleConfig is the config-file representation of a rule.
type RuleConfig struct {
	Pattern   string          `yaml:"pattern" json:"pattern"`
	Responses []string        `yaml:"responses" json:"responses"`
	DelayMS   int             `yaml:"delay_ms,omitempty" json:"delay_ms,omitempty"`
	ToolCall  *ToolCallConfig `yaml:"tool_call,omitempty" json:"tool_call,omitempty"`
}

// LoadConfig reads a config file (YAML or JSON) from the given path.
// The format is detected by file extension: .json for JSON, anything else
// is treated as YAML.
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config file: %w", err)
	}
	return ParseConfig(data, path)
}

// ParseConfig parses config data. The path is used only to detect format
// by extension (.json for JSON, otherwise YAML).
func ParseConfig(data []byte, path string) (*Config, error) {
	var cfg Config
	if strings.HasSuffix(strings.ToLower(path), ".json") {
		if err := json.Unmarshal(data, &cfg); err != nil {
			return nil, fmt.Errorf("parsing JSON config: %w", err)
		}
	} else {
		if err := yaml.Unmarshal(data, &cfg); err != nil {
			return nil, fmt.Errorf("parsing YAML config: %w", err)
		}
	}
	return &cfg, nil
}

// FindDefaultConfig looks for llmock.yaml or llmock.json in the current
// directory. Returns the path if found, or empty string if neither exists.
func FindDefaultConfig() string {
	for _, name := range []string{"llmock.yaml", "llmock.json"} {
		if _, err := os.Stat(name); err == nil {
			return name
		}
	}
	return ""
}

// CompileRules converts RuleConfig entries to compiled Rule values.
func CompileRules(configs []RuleConfig) ([]Rule, error) {
	rules := make([]Rule, len(configs))
	for i, rc := range configs {
		re, err := regexp.Compile(rc.Pattern)
		if err != nil {
			return nil, fmt.Errorf("compiling rule %d pattern %q: %w", i, rc.Pattern, err)
		}
		if len(rc.Responses) == 0 && rc.ToolCall == nil {
			return nil, fmt.Errorf("rule %d pattern %q has no responses or tool_call", i, rc.Pattern)
		}
		rules[i] = Rule{Pattern: re, Responses: rc.Responses, ToolCall: rc.ToolCall}
	}
	return rules, nil
}

// ToOptions converts a Config into functional Options for New().
// This allows config files and functional options to use the same
// underlying code path. Options passed directly to New() will override
// config file values when applied after config options.
func (c *Config) ToOptions() ([]Option, error) {
	var opts []Option

	if c.Defaults.TokenDelayMS > 0 {
		opts = append(opts, WithTokenDelay(
			durationFromMS(c.Defaults.TokenDelayMS),
		))
	}

	if c.Defaults.Seed != nil {
		opts = append(opts, WithSeed(*c.Defaults.Seed))
	}

	if c.Server.AdminAPI != nil {
		opts = append(opts, WithAdminAPI(*c.Server.AdminAPI))
	}

	if c.CorpusFile != "" {
		opts = append(opts, WithCorpusFile(c.CorpusFile))
	}

	if len(c.Rules) > 0 {
		rules, err := CompileRules(c.Rules)
		if err != nil {
			return nil, err
		}
		opts = append(opts, WithRules(rules...))
	}

	for _, f := range c.Faults {
		opts = append(opts, WithFault(f))
	}

	return opts, nil
}

func durationFromMS(ms int) time.Duration {
	return time.Duration(ms) * time.Millisecond
}
