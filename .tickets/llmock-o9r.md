---
id: llmock-o9r
status: open
deps: []
links: []
created: 2026-02-07T10:24:30.559764+11:00
type: task
priority: 2
---
# mockllm: Core handler with hardcoded echo response (OpenAI format)

Build the foundation of a mock LLM API server in Go.

## What this does
Create a Go module `mockllm` that exposes an `http.Handler` implementing the
OpenAI `/v1/chat/completions` endpoint. For now it returns a hardcoded/echo
response — the intelligence comes later.

## Requirements
- Go module at `github.com/OWNER/mockllm` (use a placeholder module path for now)
- A `Server` struct with a `Handler() http.Handler` method
- `POST /v1/chat/completions` accepts an OpenAI ChatCompletion request
  (model, messages array with role/content, optional stream bool, temperature, max_tokens)
- For now: respond with the last user message echoed back, wrapped in a valid
  OpenAI ChatCompletion response JSON (id, object, created, model, choices, usage)
- Generate plausible-looking usage stats (prompt_tokens, completion_tokens, total_tokens)
  based on rough word counts
- Ignore the `stream` field for now (always return non-streaming)
- Wire up a `cmd/mockllm/main.go` that starts the server on a configurable port
  (flag or env var, default 9090)
- Include tests that use `httptest.NewServer` with the handler, send a request
  via raw HTTP, and validate the response structure

## Design notes
- Keep the Server struct ready to accept configuration (rules, corpus, etc) even
  though we are not using them yet — use an Options pattern or functional options
- The handler should use a mux (stdlib `http.ServeMux` is fine) so we can add
  more routes later
- Return proper HTTP error codes for malformed requests (400), wrong method (405),
  unknown paths (404)



