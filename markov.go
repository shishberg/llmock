package llmock

import (
	_ "embed"
	"io"
	"math/rand/v2"
	"slices"
	"strings"
	"sync"
)

//go:embed corpus.txt
var defaultCorpus string

// DefaultCorpusText returns the embedded default corpus text.
func DefaultCorpusText() string {
	return defaultCorpus
}

// MarkovChain generates text using a Markov chain trained on a corpus.
// It is safe for concurrent reads after training.
type MarkovChain struct {
	order int
	chain map[string][]string
	mu    sync.RWMutex
}

// NewMarkovChain creates a MarkovChain with the given order (prefix length in words).
func NewMarkovChain(order int) *MarkovChain {
	if order < 1 {
		order = 1
	}
	return &MarkovChain{
		order: order,
		chain: make(map[string][]string),
	}
}

// Train adds text to the chain's model. The text is split into whitespace-delimited tokens.
func (mc *MarkovChain) Train(text string) {
	words := strings.Fields(text)
	if len(words) <= mc.order {
		return
	}
	mc.mu.Lock()
	defer mc.mu.Unlock()
	for i := 0; i <= len(words)-mc.order-1; i++ {
		prefix := strings.Join(words[i:i+mc.order], " ")
		next := words[i+mc.order]
		mc.chain[prefix] = append(mc.chain[prefix], next)
	}
}

// Generate produces text of up to maxTokens words using the given random source.
// Generation stops at maxTokens or when it hits a natural sentence ending.
func (mc *MarkovChain) Generate(maxTokens int, rng *rand.Rand) string {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	if len(mc.chain) == 0 || maxTokens <= 0 {
		return ""
	}

	// Pick a random starting prefix (sorted for determinism).
	keys := make([]string, 0, len(mc.chain))
	for k := range mc.chain {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	prefix := keys[rng.IntN(len(keys))]
	words := strings.Fields(prefix)
	result := make([]string, len(words))
	copy(result, words)

	for len(result) < maxTokens {
		followers, ok := mc.chain[prefix]
		if !ok || len(followers) == 0 {
			break
		}
		next := followers[rng.IntN(len(followers))]
		result = append(result, next)

		// Update prefix.
		prefixWords := strings.Fields(prefix)
		prefixWords = append(prefixWords[1:], next)
		prefix = strings.Join(prefixWords, " ")

		// Stop at natural sentence ending if we've produced a reasonable amount.
		if len(result) >= mc.order+4 && endsWithSentence(next) {
			break
		}
	}

	return strings.Join(result, " ")
}

// endsWithSentence returns true if the word ends with sentence-ending punctuation.
func endsWithSentence(word string) bool {
	if len(word) == 0 {
		return false
	}
	last := word[len(word)-1]
	return last == '.' || last == '!' || last == '?'
}

// MarkovResponder uses a MarkovChain to generate responses.
type MarkovResponder struct {
	chain *MarkovChain
	rng   *rand.Rand
	mu    sync.Mutex
}

// NewMarkovResponder creates a MarkovResponder trained on the default corpus.
func NewMarkovResponder(seed *int64) *MarkovResponder {
	mc := NewMarkovChain(2)
	mc.Train(defaultCorpus)

	var rng *rand.Rand
	if seed != nil {
		rng = rand.New(rand.NewPCG(uint64(*seed), 0))
	} else {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}

	return &MarkovResponder{chain: mc, rng: rng}
}

// Respond generates a Markov chain response.
func (mr *MarkovResponder) Respond(messages []InternalMessage) (string, error) {
	if extractInput(messages) == "" {
		return "", errNoMessages
	}
	mr.mu.Lock()
	text := mr.chain.Generate(100, mr.rng)
	mr.mu.Unlock()
	if text == "" {
		return "I understand. Could you tell me more about that?", nil
	}
	return text, nil
}

// GenerateMarkov produces Markov text with the given token limit, for use in templates.
func (mr *MarkovResponder) GenerateMarkov(maxTokens int) string {
	mr.mu.Lock()
	defer mr.mu.Unlock()
	text := mr.chain.Generate(maxTokens, mr.rng)
	if text == "" {
		return "I understand. Could you tell me more about that?"
	}
	return text
}

// WithCorpus provides a custom training corpus via an io.Reader.
func WithCorpus(r io.Reader) Option {
	return func(s *Server) {
		data, err := io.ReadAll(r)
		if err != nil {
			return
		}
		s.corpusText = string(data)
	}
}

// WithCorpusFile provides a custom training corpus from a file path.
func WithCorpusFile(path string) Option {
	return func(s *Server) {
		s.corpusFile = path
	}
}
