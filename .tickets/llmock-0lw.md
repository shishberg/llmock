---
id: llmock-0lw
status: open
deps: [llmock-lp1]
links: []
created: 2026-02-07T10:24:33.03007+11:00
type: task
priority: 2
---
# mockllm: MCP server simulation

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



