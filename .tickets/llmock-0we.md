---
id: llmock-0we
status: open
deps: [llmock-64l]
links: []
created: 2026-02-07T10:24:31.788717+11:00
type: task
priority: 2
---
# mockllm: Runtime rule injection API

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



