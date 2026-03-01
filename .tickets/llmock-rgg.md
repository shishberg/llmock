---
id: llmock-rgg
status: open
deps: [llmock-bw3]
links: []
created: 2026-02-07T10:24:31.529535+11:00
type: task
priority: 2
---
# mockllm: SSE streaming responses

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



