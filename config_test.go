package llmock

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestParseConfigYAML(t *testing.T) {
	data := []byte(`
server:
  port: 8080
  admin_api: false

defaults:
  token_delay_ms: 50
  seed: 42
  model: "test-model"

rules:
  - pattern: "hello"
    responses: ["Hi there!"]
  - pattern: "how do I (.*)"
    responses:
      - "Here is how you can $1"
    delay_ms: 200

corpus_file: "./my-corpus.txt"
`)
	cfg, err := ParseConfig(data, "test.yaml")
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}

	if cfg.Server.Port != 8080 {
		t.Errorf("port = %d, want 8080", cfg.Server.Port)
	}
	if cfg.Server.AdminAPI == nil || *cfg.Server.AdminAPI {
		t.Error("admin_api should be false")
	}
	if cfg.Defaults.TokenDelayMS != 50 {
		t.Errorf("token_delay_ms = %d, want 50", cfg.Defaults.TokenDelayMS)
	}
	if cfg.Defaults.Seed == nil || *cfg.Defaults.Seed != 42 {
		t.Error("seed should be 42")
	}
	if cfg.Defaults.Model != "test-model" {
		t.Errorf("model = %q, want %q", cfg.Defaults.Model, "test-model")
	}
	if len(cfg.Rules) != 2 {
		t.Fatalf("rules count = %d, want 2", len(cfg.Rules))
	}
	if cfg.Rules[0].Pattern != "hello" {
		t.Errorf("rule 0 pattern = %q, want %q", cfg.Rules[0].Pattern, "hello")
	}
	if cfg.Rules[1].DelayMS != 200 {
		t.Errorf("rule 1 delay_ms = %d, want 200", cfg.Rules[1].DelayMS)
	}
	if cfg.CorpusFile != "./my-corpus.txt" {
		t.Errorf("corpus_file = %q, want %q", cfg.CorpusFile, "./my-corpus.txt")
	}
}

func TestParseConfigJSON(t *testing.T) {
	data := []byte(`{
		"server": {"port": 7070},
		"defaults": {"token_delay_ms": 25},
		"rules": [{"pattern": "test", "responses": ["ok"]}]
	}`)
	cfg, err := ParseConfig(data, "config.json")
	if err != nil {
		t.Fatalf("ParseConfig JSON: %v", err)
	}
	if cfg.Server.Port != 7070 {
		t.Errorf("port = %d, want 7070", cfg.Server.Port)
	}
	if cfg.Defaults.TokenDelayMS != 25 {
		t.Errorf("token_delay_ms = %d, want 25", cfg.Defaults.TokenDelayMS)
	}
	if len(cfg.Rules) != 1 {
		t.Fatalf("rules count = %d, want 1", len(cfg.Rules))
	}
}

func TestParseConfigInvalidYAML(t *testing.T) {
	data := []byte(`{not valid yaml: [`)
	_, err := ParseConfig(data, "bad.yaml")
	if err == nil {
		t.Error("expected error for invalid YAML")
	}
}

func TestParseConfigInvalidJSON(t *testing.T) {
	data := []byte(`{not valid json`)
	_, err := ParseConfig(data, "bad.json")
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestLoadConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.yaml")
	os.WriteFile(path, []byte(`
server:
  port: 3000
rules:
  - pattern: "hi"
    responses: ["hello"]
`), 0644)

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.Server.Port != 3000 {
		t.Errorf("port = %d, want 3000", cfg.Server.Port)
	}
}

func TestLoadConfigMissingFile(t *testing.T) {
	_, err := LoadConfig("/nonexistent/config.yaml")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestFindDefaultConfig(t *testing.T) {
	// Create a temp dir and chdir to it.
	dir := t.TempDir()
	orig, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(orig)

	// No config files exist.
	if got := FindDefaultConfig(); got != "" {
		t.Errorf("FindDefaultConfig() = %q, want empty", got)
	}

	// Create llmock.yaml.
	os.WriteFile(filepath.Join(dir, "llmock.yaml"), []byte("server:\n  port: 1234\n"), 0644)
	if got := FindDefaultConfig(); got != "llmock.yaml" {
		t.Errorf("FindDefaultConfig() = %q, want %q", got, "llmock.yaml")
	}

	// Remove yaml, create json.
	os.Remove(filepath.Join(dir, "llmock.yaml"))
	os.WriteFile(filepath.Join(dir, "llmock.json"), []byte(`{"server":{"port":5678}}`), 0644)
	if got := FindDefaultConfig(); got != "llmock.json" {
		t.Errorf("FindDefaultConfig() = %q, want %q", got, "llmock.json")
	}
}

func TestCompileRules(t *testing.T) {
	configs := []RuleConfig{
		{Pattern: "hello", Responses: []string{"hi"}},
		{Pattern: "(?i)world", Responses: []string{"earth", "globe"}},
	}
	rules, err := CompileRules(configs)
	if err != nil {
		t.Fatalf("CompileRules: %v", err)
	}
	if len(rules) != 2 {
		t.Fatalf("got %d rules, want 2", len(rules))
	}
	if !rules[0].Pattern.MatchString("hello") {
		t.Error("rule 0 should match 'hello'")
	}
	if !rules[1].Pattern.MatchString("WORLD") {
		t.Error("rule 1 should match 'WORLD' (case insensitive)")
	}
}

func TestCompileRulesInvalidRegex(t *testing.T) {
	configs := []RuleConfig{
		{Pattern: "[invalid", Responses: []string{"ok"}},
	}
	_, err := CompileRules(configs)
	if err == nil {
		t.Error("expected error for invalid regex")
	}
}

func TestCompileRulesNoResponses(t *testing.T) {
	configs := []RuleConfig{
		{Pattern: "hello", Responses: nil},
	}
	_, err := CompileRules(configs)
	if err == nil {
		t.Error("expected error for rule with no responses")
	}
}

func TestConfigToOptions(t *testing.T) {
	seed := int64(42)
	adminFalse := false
	cfg := &Config{
		Server: ServerConfig{
			AdminAPI: &adminFalse,
		},
		Defaults: DefaultConfig{
			TokenDelayMS: 100,
			Seed:         &seed,
		},
		Rules: []RuleConfig{
			{Pattern: "hello", Responses: []string{"hi"}},
		},
		CorpusFile: "corpus.txt",
	}

	opts, err := cfg.ToOptions()
	if err != nil {
		t.Fatalf("ToOptions: %v", err)
	}

	// Verify options produce a working server.
	s := New(opts...)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// Admin should be disabled.
	resp, err := http.Get(ts.URL + "/_mock/rules")
	if err != nil {
		t.Fatalf("GET /_mock/rules: %v", err)
	}
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("admin API should be disabled, got status %d", resp.StatusCode)
	}
}

func TestConfigToOptionsInvalidRule(t *testing.T) {
	cfg := &Config{
		Rules: []RuleConfig{
			{Pattern: "[bad", Responses: []string{"oops"}},
		},
	}
	_, err := cfg.ToOptions()
	if err == nil {
		t.Error("expected error for invalid rule pattern in ToOptions")
	}
}

func TestConfigToOptionsEmpty(t *testing.T) {
	cfg := &Config{}
	opts, err := cfg.ToOptions()
	if err != nil {
		t.Fatalf("ToOptions: %v", err)
	}
	if len(opts) != 0 {
		t.Errorf("empty config should produce 0 options, got %d", len(opts))
	}
}

func TestConfigWithFaults(t *testing.T) {
	cfg := &Config{
		Faults: []Fault{
			{Type: FaultError, Status: 500, Message: "boom", Count: 1},
		},
	}
	opts, err := cfg.ToOptions()
	if err != nil {
		t.Fatalf("ToOptions: %v", err)
	}
	s := New(opts...)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// First request should get a 500.
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	if resp.StatusCode != 500 {
		t.Errorf("status = %d, want 500", resp.StatusCode)
	}
}

func TestConfigTokenDelay(t *testing.T) {
	cfg := &Config{
		Defaults: DefaultConfig{
			TokenDelayMS: 50,
		},
	}
	opts, err := cfg.ToOptions()
	if err != nil {
		t.Fatalf("ToOptions: %v", err)
	}
	// Verify the option sets tokenDelay on the server.
	s := &Server{}
	for _, opt := range opts {
		opt(s)
	}
	if s.tokenDelay != 50*time.Millisecond {
		t.Errorf("tokenDelay = %v, want 50ms", s.tokenDelay)
	}
}

func TestConfigVerbose(t *testing.T) {
	data := []byte(`
server:
  verbose: true
rules:
  - pattern: "hello"
    responses: ["Hi!"]
`)
	cfg, err := ParseConfig(data, "test.yaml")
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if cfg.Server.Verbose == nil || !*cfg.Server.Verbose {
		t.Error("verbose should be true")
	}
	opts, err := cfg.ToOptions()
	if err != nil {
		t.Fatalf("ToOptions: %v", err)
	}
	s := &Server{}
	for _, opt := range opts {
		opt(s)
	}
	if !s.verbose {
		t.Error("server verbose should be true after applying options")
	}
}
