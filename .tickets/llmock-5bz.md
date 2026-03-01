---
id: llmock-5bz
status: open
deps: [llmock-6dk, llmock-w2p]
links: []
created: 2026-02-07T10:24:32.418727+11:00
type: task
priority: 2
---
# mockllm: YAML/JSON config file and CLI polish

Add config file support and make the standalone CLI server production-ready.

## What this does
The standalone `mockllm` binary can be fully configured from a YAML or JSON
config file, with sensible defaults and good DX.

## Requirements
- Config file format (YAML and JSON both supported):
  ```yaml
  server:
    port: 9090
    admin_api: true

  defaults:
    token_delay_ms: 15
    seed: 0
    model: "mock-llm-1"

  rules:
    - pattern: ".*hello.*"
      responses: ["Hi there! How can I help you today?"]
    - pattern: "how do I (.*)"
      responses:
        - "Here is how you can $1: {{markov:50}}"
      delay_ms: 200

  corpus_file: "./my-corpus.txt"

  faults: []
  ```
- CLI flags: `--config`, `--port` (overrides config), `--verbose` (log all
  requests/responses to stderr)
- If no config file specified, look for `mockllm.yaml` or `mockllm.json` in
  the current directory, otherwise use defaults
- Verbose mode logs: timestamp, method, path, matched rule (or "fallback"),
  response status, response time
- Startup banner showing: port, number of rules loaded, corpus size, admin API
  status
- Graceful shutdown on SIGINT/SIGTERM
- README.md with:
  - Quick start (go install + run)
  - Library usage with httptest
  - Config file reference
  - Examples for common test scenarios
- Tests for config loading, CLI flag parsing, default config behavior

## Design notes
- Use a single Config struct that both the YAML loader and functional options
  populate, so there is one source of truth
- The functional options (WithRules, WithCorpus, etc) should override config
  file values when both are provided



