---
id: llmock-6dk
status: open
deps: [llmock-0we, llmock-rgg]
links: []
created: 2026-02-07T10:24:32.059954+11:00
type: task
priority: 2
---
# mockllm: Failure and delay injection

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



