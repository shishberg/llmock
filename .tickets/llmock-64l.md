---
id: llmock-64l
status: open
deps: [llmock-bw3]
links: []
created: 2026-02-07T10:24:31.003651+11:00
type: task
priority: 2
---
# mockllm: Regex rule matching with template expansion

Add the core rule-matching engine: configurable regex rules that match against
user messages and produce templated responses.

## What this does
Instead of echoing, the server now matches user messages against an ordered list
of regex rules and expands response templates with capture groups.

## Requirements
- A `Rule` type: compiled regex pattern + response template string (or list of
  strings to pick from randomly)
- Rules are evaluated in order; first match wins
- Template expansion supports:
  - `$1`, `$2`, etc for regex capture groups
  - `${input}` for the full original user message
- If no rule matches, fall back to a default response (hardcoded for now,
  will become Markov later)
- Server accepts rules via functional options: `mockllm.WithRules(rules...)`
- Ship a small set of built-in default rules that produce ELIZA-like responses:
  - "I need (.*)" -> "Why do you need $1?" / "What would it mean if you got $1?"
  - "how do I (.*)" -> "Here's how you can approach $1: first, ..."
  - "what is (.*)" -> "That's a great question. $1 refers to ..."
  - "help me (.*)" -> "I'd be happy to help you $1. Let me break this down..."
  - General greetings, farewells, etc.
  Make these feel more like a helpful AI assistant than a psychotherapist.
- Rules should be loadable from a YAML config file:
  ```yaml
  rules:
    - pattern: "deploy (.*) to (.*)"
      responses:
        - "To deploy $1 to $2, you will want to follow these steps..."
        - "Deploying $1 to $2 requires careful planning. Here is what I recommend..."
    - pattern: ".*"
      responses:
        - "That is an interesting point. Could you tell me more?"
  ```
- Tests covering: match priority, capture group substitution, no-match fallback,
  random selection among multiple response templates

## Design notes
- Rules should be safe for concurrent access (they are read-only after init,
  but runtime injection is coming later)
- Use `regexp.MustCompile` at config time, not per-request



