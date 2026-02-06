#!/usr/bin/env bash
set -euo pipefail

# Creates bead tasks for building mockllm - a mock LLM API server
# with ELIZA-like response generation for testing.
#
# Uses bd q to capture issue IDs, bd update to add descriptions,
# and bd dep to wire up dependencies so they're picked up in order.
#
# Heredoc delimiters are quoted ('EOF') to prevent shell expansion
# of $1, $2, backticks, etc. in the task descriptions.

# --- 1. Core handler (OpenAI format) ---
CORE=$(bd q "mockllm: Core handler with hardcoded echo response (OpenAI format)")
bd update "$CORE" --body-file - <<'EOF'
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
EOF

# --- 2. Anthropic Messages API format ---
ANTHROPIC=$(bd q "mockllm: Anthropic Messages API format")
bd update "$ANTHROPIC" --body-file - <<'EOF'
Add support for the Anthropic `/v1/messages` endpoint alongside the existing
OpenAI endpoint.

## What this does
The server now speaks both OpenAI and Anthropic API formats.

## Requirements
- `POST /v1/messages` accepts an Anthropic Messages API request
  (model, messages array, max_tokens, optional stream bool)
- Return a valid Anthropic Messages response (id, type:"message", role:"assistant",
  content array with type:"text" blocks, model, stop_reason:"end_turn",
  usage with input_tokens/output_tokens)
- The actual response content uses the same internal generation logic as the OpenAI
  endpoint (currently just echo) — both endpoints should call into the same
  response generator interface
- Define a clean internal interface/type for "generate a response given a conversation"
  that both API formats call into. Something like:
  `type Responder interface { Respond(messages []Message) (string, error) }`
  where Message is an internal type that both API formats convert to/from
- Tests for the Anthropic endpoint validating response structure
- Tests confirming both endpoints produce the same logical content for the same input

## Design notes
- The Anthropic format uses `content: [{"type":"text","text":"..."}]` not a plain string
- `stop_reason` is `end_turn` not `stop`
- Message id format should look like `msg_` prefix + random hex
EOF

bd dep "$CORE" --blocks "$ANTHROPIC"

# --- 3. Regex rule matching with template expansion ---
RULES=$(bd q "mockllm: Regex rule matching with template expansion")
bd update "$RULES" --body-file - <<'EOF'
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
EOF

bd dep "$ANTHROPIC" --blocks "$RULES"

# --- 4. Markov chain text generator ---
MARKOV=$(bd q "mockllm: Markov chain text generator")
bd update "$MARKOV" --body-file - <<'EOF'
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
EOF

bd dep "$RULES" --blocks "$MARKOV"

# --- 5. SSE streaming responses ---
STREAM=$(bd q "mockllm: SSE streaming responses")
bd update "$STREAM" --body-file - <<'EOF'
Add Server-Sent Events streaming support for both OpenAI and Anthropic formats.

## What this does
When `stream: true` is set in the request, the server streams the response
token-by-token using the appropriate SSE format for each API.

## Requirements
- OpenAI streaming format:
  - Content-Type: text/event-stream
  - Each chunk: `data: {"id":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"word "},"finish_reason":null}]}\n\n`
  - Final chunk has `finish_reason: "stop"` and empty delta
  - Ends with `data: [DONE]\n\n`
- Anthropic streaming format:
  - Content-Type: text/event-stream
  - Event sequence: `message_start` -> `content_block_start` -> multiple
    `content_block_delta` events -> `content_block_stop` -> `message_delta` -> `message_stop`
  - Each content_block_delta: `event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"word "}}\n\n`
- Tokenization: split the generated response into chunks of 1-3 words
  (randomized to feel natural)
- Configurable inter-token delay (default 15ms) via `mockllm.WithTokenDelay(d)`
  to simulate generation speed
- Flush after each chunk (ensure streaming actually streams, not buffered)
- Tests using a streaming HTTP client that reads chunks incrementally and
  validates the event format
- Both streaming and non-streaming should work for both API formats

## Design notes
- Set appropriate headers: Cache-Control: no-cache, Connection: keep-alive
- Use `http.Flusher` interface to flush after each write
- Handle client disconnection gracefully (context cancellation)
EOF

bd dep "$ANTHROPIC" --blocks "$STREAM"

# --- 6. Runtime rule injection API ---
INJECT=$(bd q "mockllm: Runtime rule injection API")
bd update "$INJECT" --body-file - <<'EOF'
Add HTTP endpoints to inject, inspect, and reset rules at runtime — the
escape hatch for integration tests.

## What this does
Tests can configure the mock server's behavior on the fly without restarting,
using a simple HTTP API under the `/_mock/` prefix.

## Requirements
- `POST /_mock/rules` — add one or more rules. Request body:
  ```json
  {
    "rules": [
      {
        "pattern": ".*deploy.*",
        "responses": ["Deploying now..."],
        "priority": 0
      }
    ]
  }
  ```
  Rules added via API are prepended (higher priority) to the rule list by default.
  Optional `priority` field: 0 = prepend (default), -1 = append, or an integer
  index to insert at.
- `GET /_mock/rules` — return the current rule list as JSON (patterns as strings,
  not compiled regexps obviously)
- `DELETE /_mock/rules` — reset to the initial rules from config/startup
- `POST /_mock/reset` — full reset: rules, request log, everything back to
  startup state
- `GET /_mock/requests` — return a log of recent requests (last 100) with
  timestamps, matched rule (if any), and response summary. Invaluable for
  debugging integration tests.
- `DELETE /_mock/requests` — clear the request log
- Thread safety: all of this must be safe under concurrent access. The rule list
  is now mutable, so use appropriate synchronization (RWMutex on the rule list).
- Tests: inject a rule, send a matching request, verify it matches; reset,
  verify it no longer matches; inspect the request log

## Design notes
- The /_mock/ endpoints should be optionally disableable for production-like
  usage via `mockllm.WithAdminAPI(false)`
- Consider a Go helper for tests:
  ```go
  mock := mockllm.NewTestHelper(ts.URL)
  mock.AddRule(".*error.*", "Something went wrong")
  defer mock.Reset()
  ```
EOF

bd dep "$RULES" --blocks "$INJECT"

# --- 7. Failure and delay injection ---
FAULTS=$(bd q "mockllm: Failure and delay injection")
bd update "$FAULTS" --body-file - <<'EOF'
Add the ability to simulate failures, errors, and latency — critical for
testing retry logic, timeouts, and error handling.

## What this does
The server can be configured to return errors, inject delays, or behave
badly in controllable ways.

## Requirements
- Per-rule failure injection in the rule config:
  ```yaml
  rules:
    - pattern: ".*deploy.*prod.*"
      error:
        status: 529
        type: "overloaded_error"
        message: "Overloaded"
      probability: 0.5  # 50% chance of error vs normal response
  ```
- Global failure modes settable via API:
  - `POST /_mock/faults` with body:
    ```json
    {
      "type": "error",
      "status": 500,
      "message": "Internal server error",
      "delay_ms": 5000,
      "probability": 1.0,
      "count": 3
    }
    ```
  - `DELETE /_mock/faults` — clear all faults
- Fault types:
  - `error`: return the specified HTTP status + error body in the correct API format
    (OpenAI and Anthropic have different error response schemas)
  - `delay`: add latency before responding (works with both streaming and non-streaming)
  - `timeout`: accept the connection, start streaming (if applicable), then hang
    and never finish — simulates a mid-response timeout
  - `malformed`: return invalid JSON / broken SSE stream — for testing parser resilience
  - `rate_limit`: return 429 with Retry-After header and appropriate rate limit
    error bodies for each API format
- Per-rule delays: any rule can have a `delay_ms` field
- Faults are evaluated before rules: if a global fault matches (by probability),
  it fires instead of the normal pipeline
- The Go API should support this too:
  ```go
  srv := mockllm.New(
      mockllm.WithFault(mockllm.Fault{
          Type: mockllm.FaultRateLimit,
          Count: 2,  // first 2 requests get 429, then normal
      }),
  )
  ```
- Tests: verify each fault type produces the correct output, test probability-based
  faults with a fixed seed, test count-based auto-clearing

## Design notes
- Error response formats differ between OpenAI and Anthropic — make sure both
  are correct:
  - OpenAI: `{"error":{"message":"...","type":"...","code":"..."}}`
  - Anthropic: `{"type":"error","error":{"type":"...","message":"..."}}`
- The `timeout` fault is the trickiest — you need to hold the connection open.
  Use a context with a very long timer or wait for client disconnect.
EOF

bd dep "$INJECT" --blocks "$FAULTS"
bd dep "$STREAM" --blocks "$FAULTS"

# --- 8. YAML/JSON config file and CLI polish ---
CONFIG=$(bd q "mockllm: YAML/JSON config file and CLI polish")
bd update "$CONFIG" --body-file - <<'EOF'
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
EOF

bd dep "$FAULTS" --blocks "$CONFIG"
bd dep "$MARKOV" --blocks "$CONFIG"

# --- 9. Tool use / function calling simulation ---
TOOLS=$(bd q "mockllm: Tool use / function calling simulation")
bd update "$TOOLS" --body-file - <<'EOF'
Add support for simulating tool/function calls in responses — when the request
includes tool definitions, the mock can respond with tool calls.

## What this does
The server can inspect tool definitions provided in the request and generate
responses that include tool calls, enabling testing of tool-use workflows.

## Requirements
- Parse tool/function definitions from requests:
  - OpenAI format: `tools` array with `type:"function"` and function name/description/parameters
  - Anthropic format: `tools` array with name/description/input_schema
- New rule fields for tool call responses:
  ```yaml
  rules:
    - pattern: ".*weather.*"
      tool_call:
        name: "get_weather"
        arguments:
          location: "$1"
          unit: "celsius"
  ```
  If the named tool is not in the request, fall through to next rule.
- If a rule specifies a tool_call, respond in the correct format:
  - OpenAI: `choices[0].message.tool_calls` array with id, type, function name/arguments
  - Anthropic: `content` array with `type:"tool_use"` block with id, name, input
- Support multi-turn tool use: if the request includes a tool_result message
  (user providing tool output), subsequent rules can match against the tool
  output content
- Auto-generation mode: if `auto_tools: true` is set and no rule matches but
  tools are defined in the request, pick a random tool and generate plausible
  arguments based on the JSON schema (strings get Markov text, numbers get
  random values in range, booleans get random true/false, enums pick a random value)
- Streaming tool calls: both OpenAI and Anthropic have specific streaming formats
  for tool calls — implement these
- Tests: tool call response format validation for both APIs, multi-turn tool use
  conversation, auto-generation with schema-based arguments, streaming tool calls

## Design notes
- Tool call IDs should look realistic: OpenAI uses `call_` + alphanumeric,
  Anthropic uses `toolu_` + alphanumeric
- The auto-generation of arguments from JSON schema does not need to be perfect —
  this is for testing that your code handles the shape correctly, not the content
- For multi-turn, you need to handle the message history, not just the last message
EOF

bd dep "$CONFIG" --blocks "$TOOLS"

# --- 10. MCP server simulation ---
MCP=$(bd q "mockllm: MCP server simulation")
bd update "$MCP" --body-file - <<'EOF'
Add a simulated MCP (Model Context Protocol) server that can be used to test
MCP client integrations.

## What this does
The server can act as an MCP server, advertising tools and resources and
responding to MCP protocol messages — enabling testing of MCP client code
without a real MCP server.

## Requirements
- Implement the MCP protocol over HTTP+SSE transport (streamable HTTP):
  - `POST /mcp` — main MCP endpoint accepting JSON-RPC 2.0 messages
  - Support the core MCP methods:
    - `initialize` — return server capabilities and info
    - `tools/list` — return configured tools
    - `tools/call` — execute a tool call and return results
    - `resources/list` — return configured resources
    - `resources/read` — return resource content
    - `prompts/list` — return configured prompts
    - `prompts/get` — return prompt content
- MCP tools, resources, and prompts are configurable:
  ```yaml
  mcp:
    tools:
      - name: "get_weather"
        description: "Get current weather for a location"
        input_schema:
          type: object
          properties:
            location: { type: string }
          required: [location]
        responses:
          - pattern: ".*"
            result: '{"temperature": 72, "condition": "sunny"}'
    resources:
      - uri: "file:///project/README.md"
        name: "Project README"
        content: "# My Project\nThis is a mock project."
    prompts:
      - name: "review_code"
        description: "Review code for issues"
        arguments:
          - name: "language"
            required: true
        template: "Please review the following {{language}} code..."
  ```
- MCP tool call responses support the same rule-matching as the main API:
  pattern matching on the tool arguments, template expansion, Markov filler
- Runtime injection via `/_mock/mcp/tools`, `/_mock/mcp/resources`, etc.
- Tests: full MCP handshake, tool listing, tool calling with pattern matching,
  resource reading, prompt retrieval

## Design notes
- MCP uses JSON-RPC 2.0 — make sure to handle request IDs correctly
- The streamable HTTP transport uses SSE for server-to-client messages
- MCP is optional — disabled by default, enabled via config or
  `mockllm.WithMCP(true)`
- This can share the rule-matching and Markov infrastructure with the main
  chat API handlers
EOF

bd dep "$TOOLS" --blocks "$MCP"

# --- Summary ---
echo ""
echo "✅ All beads created with dependencies."
echo ""
echo "Issues:"
echo "  CORE      = $CORE"
echo "  ANTHROPIC = $ANTHROPIC"
echo "  RULES     = $RULES"
echo "  MARKOV    = $MARKOV"
echo "  STREAM    = $STREAM"
echo "  INJECT    = $INJECT"
echo "  FAULTS    = $FAULTS"
echo "  CONFIG    = $CONFIG"
echo "  TOOLS     = $TOOLS"
echo "  MCP       = $MCP"
echo ""
echo "Dependency graph:"
echo ""
echo "  $CORE (Core OpenAI handler)"
echo "    └─► $ANTHROPIC (Anthropic format)"
echo "          ├─► $RULES (Rule matching)"
echo "          │     ├─► $MARKOV (Markov generator) ──────┐"
echo "          │     └─► $INJECT (Runtime injection)      │"
echo "          │           └─► $FAULTS (Fault injection) ◄┤"
echo "          └─► $STREAM (SSE streaming) ──────────────►┘"
echo "                                  $CONFIG (Config/CLI)"
echo "                                    └─► $TOOLS (Tool use)"
echo "                                          └─► $MCP (MCP server)"
