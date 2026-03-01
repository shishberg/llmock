---
id: llmock-w2p
status: open
deps: [llmock-64l]
links: []
created: 2026-02-07T10:24:31.26535+11:00
type: task
priority: 2
---
# mockllm: Markov chain text generator

Add a Markov chain text generator that produces LLM-ish filler text, used as
a fallback and for padding template responses.

## What this does
When no rule matches, or when a template includes a `{{markov}}` placeholder,
the server generates plausible-sounding "helpful assistant" text using a
Markov chain.

## Requirements
- A `MarkovChain` type that can be trained on a text corpus and generate text
- Configurable chain order (default 2 — bigram prefix)
- Generation stops at a configurable max token count or when it hits a natural
  sentence ending
- Ship a default corpus embedded via `//go:embed` containing ~2000 words of
  "helpful AI assistant" style text. Write this corpus yourself — paragraphs like:
  "That is a great question. Let me break this down step by step. First, you will
  want to consider the overall architecture of your system. There are several
  approaches you could take, each with different tradeoffs..."
  The goal is that Markov-generated output from this corpus reads like a
  stereotypical LLM response at a glance.
- `mockllm.WithCorpus(r io.Reader)` option to provide a custom training corpus
- `mockllm.WithCorpusFile(path string)` convenience option
- The Markov generator becomes the default fallback Responder when no rules match
- Template responses can include `{{markov}}` or `{{markov:50}}` (with token limit)
  to splice in generated text
- Tests: deterministic output with a fixed seed, statistical tests that output
  only contains words from the corpus, integration test showing it plugs into
  the response pipeline

## Design notes
- The chain should be built once at startup and be safe for concurrent reads
- Use a deterministic seed option for testing: `mockllm.WithSeed(int64)`
- Token splitting can just be whitespace-based, nothing fancy



