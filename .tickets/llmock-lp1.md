---
id: llmock-lp1
status: open
deps: [llmock-5bz]
links: []
created: 2026-02-07T10:24:32.768858+11:00
type: task
priority: 2
---
# mockllm: Tool use / function calling simulation

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



