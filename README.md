# llmock

A mock LLM API server for testing. Simulates OpenAI and Anthropic chat completion APIs so you can develop and test LLM-integrated applications without hitting real endpoints or spending tokens.

## Features

- **OpenAI & Anthropic API compatibility** &mdash; drop-in replacement for `/v1/chat/completions` (OpenAI) and `/v1/messages` (Anthropic)
- **Streaming** &mdash; Server-Sent Events in both OpenAI and Anthropic formats
- **Rule-based responses** &mdash; regex pattern matching with capture groups and template expansion
- **Tool calling / function calling** &mdash; simulates tool use with auto-generation from JSON schemas
- **Multi-turn conversations** &mdash; handles tool call/result message sequences
- **Markov chain fallback** &mdash; generates plausible-looking LLM text when no rule matches
- **Fault injection** &mdash; simulate errors, delays, timeouts, rate limits, and malformed responses
- **MCP protocol** &mdash; Model Context Protocol server with tools, resources, and prompts
- **Admin API** &mdash; inject rules, faults, and inspect requests at runtime
- **Config files** &mdash; YAML or JSON configuration with auto-discovery

## Install

```bash
go install github.com/shishberg/llmock/cmd/llmock@latest
```

Or build from source:

```bash
git clone https://github.com/shishberg/llmock.git
cd llmock
go build -o llmock ./cmd/llmock
```

## Quick start

Start the server with defaults (port 9090, echo responder):

```bash
./llmock
```

Send a request using the OpenAI format:

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello, world!"}]
  }'
```

Response:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "gpt-4",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello, world!"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

Or use the Anthropic format:

```bash
curl http://localhost:9090/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello, world!"}]
  }'
```

## CLI flags

```
-config string   Path to config file (YAML or JSON)
-port int        Port to listen on (overrides config)
-verbose         Log all requests/responses to stderr
```

Port resolution order: `-port` flag > config file > `PORT` env var > `9090`.

If no `-config` is given, llmock looks for `llmock.yaml` or `llmock.json` in the current directory.

## Configuration

### Example config file (`llmock.yaml`)

```yaml
server:
  port: 8080
  admin_api: true

defaults:
  token_delay_ms: 20
  seed: 42
  model: "gpt-4"
  auto_tool_calls: true

corpus_file: "./my-corpus.txt"

rules:
  - pattern: "(?i)hello"
    responses:
      - "Hi there! How can I help you today?"
      - "Hello! What can I do for you?"

  - pattern: "(?i)weather in (\\w+)"
    responses:
      - "The weather in $1 is sunny and 72°F."
    tool_call:
      name: "get_weather"
      arguments:
        location: "$1"

  - pattern: ".*"
    responses:
      - "{{markov:50}}"

faults:
  - type: error
    status: 500
    count: 1
  - type: delay
    delay_ms: 3000
    probability: 0.1

mcp:
  tools:
    - name: "lookup"
      description: "Look up information"
      input_schema:
        type: object
        properties:
          query:
            type: string
      responses:
        - pattern: ".*"
          result: "Result for: ${input}"
  resources:
    - uri: "memory://notes"
      name: "Notes"
      content: "Some stored notes"
  prompts:
    - name: "summarize"
      description: "Summarize text"
      template: "Please summarize: ${text}"
      arguments:
        - name: "text"
          required: true
```

### Config reference

| Section | Field | Description |
|---|---|---|
| `server.port` | int | Port to listen on |
| `server.admin_api` | bool | Enable `/_mock/` admin endpoints (default: true) |
| `defaults.token_delay_ms` | int | Delay between streamed tokens in ms |
| `defaults.seed` | int | RNG seed for deterministic output |
| `defaults.model` | string | Model name in responses |
| `defaults.auto_tool_calls` | bool | Auto-generate tool calls from request schemas |
| `corpus_file` | string | Path to custom Markov training text |
| `rules` | list | Response rules (see below) |
| `faults` | list | Fault injection config (see below) |
| `mcp` | object | MCP server config (tools, resources, prompts) |

## Rules

Rules match user messages by regex and return templated responses.

```yaml
rules:
  - pattern: "(?i)translate (.*) to (\\w+)"
    responses:
      - "Here is '$1' translated to $2: {{markov:20}}"
```

**Pattern**: A Go regex applied to the last user message.

**Responses**: One is chosen at random. Supports:
- `$1`, `$2`, ... &mdash; regex capture groups
- `${input}` &mdash; the full user message
- `{{markov}}` &mdash; Markov-generated text (default ~50 words)
- `{{markov:N}}` &mdash; Markov-generated text of ~N words

**Tool calls**: Optionally attach a tool call to the response:

```yaml
rules:
  - pattern: "(?i)search (.*)"
    responses:
      - "Searching for $1..."
    tool_call:
      name: "web_search"
      arguments:
        query: "$1"
```

## Streaming

Request streaming with `"stream": true`:

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

Tokens are sent as Server-Sent Events with a configurable delay (`token_delay_ms`).

## Tool calling

### Rule-based tool calls

Pair a rule with a `tool_call` to simulate function calling:

```yaml
rules:
  - pattern: "(?i)what.*weather"
    tool_call:
      name: "get_weather"
      arguments:
        location: "San Francisco"
```

### Auto-generated tool calls

When `auto_tool_calls` is enabled and a request includes tool definitions but no rule produces a tool call, llmock picks a random tool and generates arguments from its JSON schema:

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Do something"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["location"]
        }
      }
    }]
  }'
```

The server generates schema-compliant arguments, respecting types, enums, formats, and required fields.

### Multi-turn conversations

llmock handles the full tool-use conversation loop:

```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":\"NYC\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "72°F and sunny"},
    {"role": "user", "content": "Thanks!"}
  ]
}
```

## Fault injection

Simulate failure modes to test your application's error handling:

```yaml
faults:
  - type: error       # Return an HTTP error
    status: 503
    probability: 0.5  # 50% chance per request

  - type: delay       # Add latency
    delay_ms: 5000

  - type: timeout     # Hang forever (no response)
    count: 1          # Only the first request

  - type: rate_limit  # Return 429 Too Many Requests

  - type: malformed   # Return invalid JSON / broken SSE
```

Each fault supports `probability` (0.0&ndash;1.0) and `count` (trigger N times, 0 = unlimited).

## Admin API

The admin API at `/_mock/` lets you modify server behavior at runtime.

### Rules

```bash
# List rules
curl http://localhost:9090/_mock/rules

# Add a rule (appended to the end)
curl -X POST http://localhost:9090/_mock/rules \
  -d '{"pattern": "(?i)test", "responses": ["This is a test response"]}'

# Add a high-priority rule (prepended to the front)
curl -X POST http://localhost:9090/_mock/rules \
  -d '{"pattern": ".*", "responses": ["Override!"], "priority": true}'

# Reset rules to initial config
curl -X DELETE http://localhost:9090/_mock/rules
```

### Faults

```bash
# List active faults
curl http://localhost:9090/_mock/faults

# Add a fault
curl -X POST http://localhost:9090/_mock/faults \
  -d '{"type": "error", "status": 500}'

# Clear all faults
curl -X DELETE http://localhost:9090/_mock/faults
```

### Request log

```bash
# View last 100 requests
curl http://localhost:9090/_mock/requests

# Clear log
curl -X DELETE http://localhost:9090/_mock/requests
```

### Reset everything

```bash
curl -X POST http://localhost:9090/_mock/reset
```

## MCP (Model Context Protocol)

llmock includes an MCP server at `POST /mcp` using JSON-RPC 2.0:

```bash
# List tools
curl -X POST http://localhost:9090/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}'

# Call a tool
curl -X POST http://localhost:9090/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "lookup", "arguments": {"query": "test"}}}'

# List resources
curl -X POST http://localhost:9090/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 3, "method": "resources/list"}'
```

MCP tools, resources, and prompts can also be managed at runtime via the admin API at `/_mock/mcp/tools`, `/_mock/mcp/resources`, and `/_mock/mcp/prompts`.

## Go library usage

Use llmock as a library in Go tests:

```go
package myapp_test

import (
    "net/http/httptest"
    "regexp"
    "testing"

    "github.com/shishberg/llmock"
)

func TestWithMockLLM(t *testing.T) {
    rules := []llmock.Rule{
        {
            Pattern:   regexp.MustCompile(`(?i)hello`),
            Responses: []string{"Hi from the mock!"},
        },
    }
    s := llmock.New(
        llmock.WithRules(rules...),
        llmock.WithSeed(42),
    )
    ts := httptest.NewServer(s.Handler())
    defer ts.Close()

    // Point your LLM client at ts.URL instead of the real API
    // client := openai.NewClient(ts.URL, "fake-key")
}
```

### Available options

```go
llmock.WithRules(rules...)              // Add response rules
llmock.WithSeed(42)                     // Deterministic RNG
llmock.WithTokenDelay(50*time.Millisecond) // Streaming token delay
llmock.WithAutoToolCalls(true)          // Auto-generate tool calls
llmock.WithAdminAPI(true)               // Enable admin endpoints
llmock.WithCorpusFile("corpus.txt")     // Custom Markov training text
llmock.WithMCP(mcpConfig)              // Enable MCP server
llmock.WithFault(fault)                 // Add fault injection
```

## API endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI chat completions |
| POST | `/v1/messages` | Anthropic messages |
| POST | `/mcp` | MCP JSON-RPC 2.0 (when enabled) |
| GET | `/_mock/rules` | List rules |
| POST | `/_mock/rules` | Add a rule |
| DELETE | `/_mock/rules` | Reset rules |
| GET | `/_mock/faults` | List faults |
| POST | `/_mock/faults` | Add a fault |
| DELETE | `/_mock/faults` | Clear faults |
| GET | `/_mock/requests` | View request log |
| DELETE | `/_mock/requests` | Clear request log |
| POST | `/_mock/reset` | Full reset |

## Running tests

```bash
go test ./...
```

## License

See [LICENSE](LICENSE) for details.
