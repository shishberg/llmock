package llmock_test

import (
	"encoding/json"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"os"
	"regexp"
	"strings"
	"testing"

	"github.com/shishberg/llmock"
)

func TestMarkovChain_DeterministicWithSeed(t *testing.T) {
	mc := llmock.NewMarkovChain(2)
	mc.Train("one two three four five six seven eight nine ten one two four five")

	rng1 := rand.New(rand.NewPCG(42, 0))
	rng2 := rand.New(rand.NewPCG(42, 0))

	out1 := mc.Generate(20, rng1)
	out2 := mc.Generate(20, rng2)

	if out1 != out2 {
		t.Errorf("expected deterministic output with same seed, got %q and %q", out1, out2)
	}
	if out1 == "" {
		t.Error("expected non-empty output")
	}
}

func TestMarkovChain_DifferentSeeds(t *testing.T) {
	corpus := "the quick brown fox jumps over the lazy dog the quick red fox runs through the green field the lazy cat sleeps on the warm mat"
	mc := llmock.NewMarkovChain(2)
	mc.Train(corpus)

	seen := make(map[string]bool)
	for seed := int64(0); seed < 20; seed++ {
		rng := rand.New(rand.NewPCG(uint64(seed), 0))
		out := mc.Generate(20, rng)
		seen[out] = true
	}
	// With 20 different seeds, we should see multiple distinct outputs.
	if len(seen) < 2 {
		t.Errorf("expected multiple distinct outputs from different seeds, got %d", len(seen))
	}
}

func TestMarkovChain_OutputContainsOnlyCorpusWords(t *testing.T) {
	corpus := "alpha beta gamma delta alpha beta delta gamma alpha gamma beta delta"
	mc := llmock.NewMarkovChain(2)
	mc.Train(corpus)

	corpusWords := make(map[string]bool)
	for _, w := range strings.Fields(corpus) {
		corpusWords[w] = true
	}

	for seed := int64(0); seed < 50; seed++ {
		rng := rand.New(rand.NewPCG(uint64(seed), 0))
		out := mc.Generate(50, rng)
		for _, word := range strings.Fields(out) {
			if !corpusWords[word] {
				t.Errorf("seed %d: word %q not in corpus, output was %q", seed, word, out)
			}
		}
	}
}

func TestMarkovChain_DefaultCorpusWordsOnly(t *testing.T) {
	// The default MarkovResponder uses the embedded corpus.
	// All generated words should appear in the corpus.
	seed := int64(99)
	mr := llmock.NewMarkovResponder(&seed)

	corpusWords := make(map[string]bool)
	for _, w := range strings.Fields(llmock.DefaultCorpusText()) {
		corpusWords[w] = true
	}

	for i := 0; i < 20; i++ {
		text := mr.GenerateMarkov(100)
		for _, word := range strings.Fields(text) {
			if !corpusWords[word] {
				t.Errorf("word %q not in default corpus", word)
			}
		}
	}
}

func TestMarkovChain_RespectsMaxTokens(t *testing.T) {
	corpus := "a b c d e f g h i j k l m n o p q r s t u v w x y z a b c d e f g h i j k l m n o p"
	mc := llmock.NewMarkovChain(1)
	mc.Train(corpus)

	rng := rand.New(rand.NewPCG(42, 0))
	out := mc.Generate(5, rng)
	words := strings.Fields(out)
	if len(words) > 5 {
		t.Errorf("expected at most 5 tokens, got %d: %q", len(words), out)
	}
}

func TestMarkovChain_EmptyCorpus(t *testing.T) {
	mc := llmock.NewMarkovChain(2)
	rng := rand.New(rand.NewPCG(42, 0))
	out := mc.Generate(10, rng)
	if out != "" {
		t.Errorf("expected empty output from empty chain, got %q", out)
	}
}

func TestMarkovChain_Order1(t *testing.T) {
	mc := llmock.NewMarkovChain(1)
	mc.Train("hello world hello earth hello mars")
	rng := rand.New(rand.NewPCG(42, 0))
	out := mc.Generate(10, rng)
	if out == "" {
		t.Error("expected non-empty output with order 1")
	}
}

func TestMarkovResponder_IntegrationWithServer(t *testing.T) {
	// A server with rules that don't match should fall back to Markov.
	rules := []llmock.Rule{
		{Pattern: regexp.MustCompile(`^specific phrase$`), Responses: []string{"matched!"}},
	}
	s := llmock.New(llmock.WithRules(rules...), llmock.WithSeed(42))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// This should match the rule.
	body := `{"model":"test","messages":[{"role":"user","content":"specific phrase"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var result llmock.ChatCompletionResponse
	json.NewDecoder(resp.Body).Decode(&result)
	if result.Choices[0].Message.Content != "matched!" {
		t.Errorf("expected 'matched!', got %q", result.Choices[0].Message.Content)
	}

	// This should NOT match and fall back to Markov.
	body2 := `{"model":"test","messages":[{"role":"user","content":"something unmatched"}]}`
	resp2, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body2))
	if err != nil {
		t.Fatal(err)
	}
	defer resp2.Body.Close()
	var result2 llmock.ChatCompletionResponse
	json.NewDecoder(resp2.Body).Decode(&result2)
	markovOutput := result2.Choices[0].Message.Content
	if markovOutput == "" {
		t.Error("expected non-empty Markov fallback response")
	}
	if markovOutput == "matched!" {
		t.Error("expected Markov fallback, not the rule response")
	}
}

func TestMarkovResponder_DeterministicWithSeed(t *testing.T) {
	seed1 := int64(42)
	seed2 := int64(42)
	mr1 := llmock.NewMarkovResponder(&seed1)
	mr2 := llmock.NewMarkovResponder(&seed2)

	msgs := []llmock.InternalMessage{{Role: "user", Content: "hello"}}
	out1, _ := mr1.Respond(msgs)
	out2, _ := mr2.Respond(msgs)

	if out1.Text != out2.Text {
		t.Errorf("expected deterministic output, got %q and %q", out1.Text, out2.Text)
	}
}

func TestMarkovResponder_NoMessages(t *testing.T) {
	seed := int64(42)
	mr := llmock.NewMarkovResponder(&seed)
	_, err := mr.Respond(nil)
	if err == nil {
		t.Error("expected error for nil messages")
	}
}

func TestMarkovTemplate_InRuleResponse(t *testing.T) {
	rules := []llmock.Rule{
		{Pattern: regexp.MustCompile(`.*`), Responses: []string{"Here is some text: {{markov}}"}},
	}
	s := llmock.New(llmock.WithRules(rules...), llmock.WithSeed(42))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{"model":"test","messages":[{"role":"user","content":"hello"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var result llmock.ChatCompletionResponse
	json.NewDecoder(resp.Body).Decode(&result)
	content := result.Choices[0].Message.Content
	if !strings.HasPrefix(content, "Here is some text: ") {
		t.Errorf("expected prefix 'Here is some text: ', got %q", content)
	}
	markovPart := strings.TrimPrefix(content, "Here is some text: ")
	if markovPart == "" || markovPart == "{{markov}}" {
		t.Errorf("expected expanded markov text, got %q", markovPart)
	}
}

func TestMarkovTemplate_WithTokenLimit(t *testing.T) {
	rules := []llmock.Rule{
		{Pattern: regexp.MustCompile(`.*`), Responses: []string{"Short: {{markov:10}}"}},
	}
	s := llmock.New(llmock.WithRules(rules...), llmock.WithSeed(42))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{"model":"test","messages":[{"role":"user","content":"hello"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var result llmock.ChatCompletionResponse
	json.NewDecoder(resp.Body).Decode(&result)
	content := result.Choices[0].Message.Content
	if !strings.HasPrefix(content, "Short: ") {
		t.Errorf("expected prefix 'Short: ', got %q", content)
	}
	markovPart := strings.TrimPrefix(content, "Short: ")
	words := strings.Fields(markovPart)
	if len(words) > 10 {
		t.Errorf("expected at most 10 words from {{markov:10}}, got %d: %q", len(words), markovPart)
	}
}

func TestWithCorpus(t *testing.T) {
	customCorpus := "foo bar baz foo bar baz foo baz bar foo bar baz qux foo bar"
	rules := []llmock.Rule{
		{Pattern: regexp.MustCompile(`^nomatch$`), Responses: []string{"nope"}},
	}
	s := llmock.New(
		llmock.WithRules(rules...),
		llmock.WithCorpus(strings.NewReader(customCorpus)),
		llmock.WithSeed(42),
	)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{"model":"test","messages":[{"role":"user","content":"anything"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var result llmock.ChatCompletionResponse
	json.NewDecoder(resp.Body).Decode(&result)
	content := result.Choices[0].Message.Content

	// Output should only contain words from the custom corpus.
	allowed := map[string]bool{"foo": true, "bar": true, "baz": true, "qux": true}
	for _, word := range strings.Fields(content) {
		if !allowed[word] {
			t.Errorf("word %q not in custom corpus, output was %q", word, content)
		}
	}
}

func TestWithCorpusFile(t *testing.T) {
	corpusContent := "apple banana cherry apple banana cherry apple cherry banana apple banana cherry date apple banana"
	f, err := os.CreateTemp("", "corpus-*.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())
	f.WriteString(corpusContent)
	f.Close()

	rules := []llmock.Rule{
		{Pattern: regexp.MustCompile(`^nomatch$`), Responses: []string{"nope"}},
	}
	s := llmock.New(
		llmock.WithRules(rules...),
		llmock.WithCorpusFile(f.Name()),
		llmock.WithSeed(42),
	)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{"model":"test","messages":[{"role":"user","content":"anything"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var result llmock.ChatCompletionResponse
	json.NewDecoder(resp.Body).Decode(&result)
	content := result.Choices[0].Message.Content

	allowed := map[string]bool{"apple": true, "banana": true, "cherry": true, "date": true}
	for _, word := range strings.Fields(content) {
		if !allowed[word] {
			t.Errorf("word %q not in custom corpus file, output was %q", word, content)
		}
	}
}

func TestMarkov_AnthropicEndpoint(t *testing.T) {
	rules := []llmock.Rule{
		{Pattern: regexp.MustCompile(`^nomatch$`), Responses: []string{"nope"}},
	}
	s := llmock.New(llmock.WithRules(rules...), llmock.WithSeed(42))
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{"model":"claude","max_tokens":1024,"messages":[{"role":"user","content":"tell me something"}]}`
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var result llmock.AnthropicResponse
	json.NewDecoder(resp.Body).Decode(&result)
	if len(result.Content) == 0 || result.Content[0].Text == "" {
		t.Error("expected non-empty Markov response via Anthropic endpoint")
	}
}
