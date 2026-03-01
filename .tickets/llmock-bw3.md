---
id: llmock-bw3
status: open
deps: [llmock-o9r]
links: []
created: 2026-02-07T10:24:30.732526+11:00
type: task
priority: 2
---
# mockllm: Anthropic Messages API format

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



