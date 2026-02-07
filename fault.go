package llmock

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"net/http"
	"sync"
	"time"
)

// FaultType identifies the kind of fault to inject.
type FaultType string

const (
	// FaultError returns an HTTP error response with the specified status and message.
	FaultError FaultType = "error"
	// FaultDelay adds latency before responding normally.
	FaultDelay FaultType = "delay"
	// FaultTimeout accepts the connection, optionally starts streaming, then hangs.
	FaultTimeout FaultType = "timeout"
	// FaultMalformed returns invalid JSON or a broken SSE stream.
	FaultMalformed FaultType = "malformed"
	// FaultRateLimit returns a 429 with Retry-After header and appropriate error body.
	FaultRateLimit FaultType = "rate_limit"
)

// Fault describes a fault to inject into the request pipeline.
type Fault struct {
	Type        FaultType `json:"type"`
	Status      int       `json:"status,omitempty"`
	Message     string    `json:"message,omitempty"`
	ErrorType   string    `json:"error_type,omitempty"`
	DelayMS     int       `json:"delay_ms,omitempty"`
	Probability float64   `json:"probability,omitempty"`
	Count       int       `json:"count,omitempty"`
}

// faultState manages the global fault configuration.
type faultState struct {
	mu     sync.Mutex
	faults []activeFault
	rng    *rand.Rand
}

// activeFault is a Fault with remaining count tracking.
type activeFault struct {
	Fault
	remaining int // 0 means unlimited
}

func newFaultState(initial []Fault, rng *rand.Rand) *faultState {
	fs := &faultState{rng: rng}
	for _, f := range initial {
		fs.faults = append(fs.faults, activeFault{Fault: f, remaining: f.Count})
	}
	return fs
}

// evaluate checks if a fault should fire. Returns the fault and true if so.
// Decrements count-based faults and removes exhausted ones.
func (fs *faultState) evaluate() (Fault, bool) {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	for i := range fs.faults {
		f := &fs.faults[i]
		prob := f.Probability
		if prob <= 0 {
			prob = 1.0
		}
		if prob < 1.0 && fs.rng.Float64() >= prob {
			continue
		}
		result := f.Fault
		if f.remaining > 0 {
			f.remaining--
			if f.remaining == 0 {
				// Remove exhausted fault.
				fs.faults = append(fs.faults[:i], fs.faults[i+1:]...)
			}
		}
		return result, true
	}
	return Fault{}, false
}

// addFaults appends faults to the active list.
func (fs *faultState) addFaults(faults []Fault) {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	for _, f := range faults {
		fs.faults = append(fs.faults, activeFault{Fault: f, remaining: f.Count})
	}
}

// clear removes all active faults.
func (fs *faultState) clear() {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	fs.faults = nil
}

// getFaults returns a copy of the current faults for inspection.
func (fs *faultState) getFaults() []Fault {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	out := make([]Fault, len(fs.faults))
	for i, f := range fs.faults {
		out[i] = f.Fault
	}
	return out
}

// WithFault configures a global fault to be active at server startup.
func WithFault(f Fault) Option {
	return func(s *Server) {
		s.initialFaults = append(s.initialFaults, f)
	}
}

// WithSeed sets a deterministic random seed for fault probability evaluation.
func WithSeed(seed int64) Option {
	return func(s *Server) {
		s.seed = &seed
	}
}

// executeFault handles writing the fault response for an already-triggered fault.
// It returns true if the fault was fully handled (caller should return).
func (s *Server) executeFault(w http.ResponseWriter, r *http.Request, f Fault, apiFormat string, isStream bool) bool {
	switch f.Type {
	case FaultDelay:
		if f.DelayMS > 0 {
			select {
			case <-time.After(time.Duration(f.DelayMS) * time.Millisecond):
			case <-r.Context().Done():
				return true
			}
		}
		return false // Continue to normal handling after delay.

	case FaultError:
		status := f.Status
		if status == 0 {
			status = http.StatusInternalServerError
		}
		writeFaultError(w, status, f.Message, f.ErrorType, apiFormat)
		return true

	case FaultRateLimit:
		w.Header().Set("Retry-After", "1")
		writeFaultError(w, http.StatusTooManyRequests, faultMsg(f.Message, "rate limit exceeded"), "rate_limit_error", apiFormat)
		return true

	case FaultTimeout:
		flusher, ok := w.(http.Flusher)
		if isStream && ok {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			if apiFormat == "anthropic" {
				writeSSE(w, "message_start", map[string]any{
					"type": "message_start",
					"message": map[string]any{
						"id":   "msg_timeout",
						"type": "message",
						"role": "assistant",
					},
				})
			} else if apiFormat == "gemini" {
				partial := map[string]any{
					"candidates": []map[string]any{
						{"content": map[string]any{"role": "model", "parts": []map[string]any{{"text": ""}}}},
					},
				}
				data, _ := json.Marshal(partial)
				fmt.Fprintf(w, "data: %s\n\n", data)
			} else {
				fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-timeout\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"index\":0}]}\n\n")
			}
			flusher.Flush()
		}
		// Block until client disconnects.
		<-r.Context().Done()
		return true

	case FaultMalformed:
		if isStream {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Write([]byte("data: {\"broken json\n\ndata: not-valid\n\n"))
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
		} else {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"broken": json, not valid}`))
		}
		return true

	default:
		return false
	}
}

// writeFaultError writes an error response in the appropriate API format.
func writeFaultError(w http.ResponseWriter, status int, message, errType, apiFormat string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	if errType == "" {
		errType = "server_error"
	}
	if message == "" {
		message = http.StatusText(status)
	}

	if apiFormat == "anthropic" {
		json.NewEncoder(w).Encode(map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    errType,
				"message": message,
			},
		})
	} else if apiFormat == "gemini" {
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"code":    status,
				"message": message,
				"status":  http.StatusText(status),
			},
		})
	} else {
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": message,
				"type":    errType,
				"code":    nil,
			},
		})
	}
}

func faultMsg(msg, fallback string) string {
	if msg != "" {
		return msg
	}
	return fallback
}
