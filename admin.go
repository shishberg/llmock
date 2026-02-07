package llmock

import (
	"encoding/json"
	"math/rand/v2"
	"net/http"
	"regexp"
	"sync"
	"time"
)

// requestEntry records a single incoming request for the request log.
type requestEntry struct {
	Timestamp   time.Time `json:"timestamp"`
	Method      string    `json:"method"`
	Path        string    `json:"path"`
	UserMessage string    `json:"user_message"`
	MatchedRule string    `json:"matched_rule,omitempty"`
	Response    string    `json:"response"`
}

// adminState holds the mutable state for the admin API: the live rule list,
// the initial (startup) rules for resets, and the request log.
type adminState struct {
	mu           sync.RWMutex
	rules        []Rule
	initialRules []Rule
	requestLog   []requestEntry
}

func newAdminState(initial []Rule) *adminState {
	cp := make([]Rule, len(initial))
	copy(cp, initial)
	return &adminState{
		rules:        cp,
		initialRules: initial,
	}
}

// snapshot returns a copy of the current rule list safe for reading.
func (a *adminState) snapshot() []Rule {
	a.mu.RLock()
	defer a.mu.RUnlock()
	cp := make([]Rule, len(a.rules))
	copy(cp, a.rules)
	return cp
}

// matchRules tries each rule in order; returns the response and pattern on
// match, or empty strings if nothing matched.
func (a *adminState) matchRules(input string) (responseText, matchedPattern string) {
	a.mu.RLock()
	defer a.mu.RUnlock()

	for _, rule := range a.rules {
		matches := rule.Pattern.FindStringSubmatch(input)
		if matches == nil {
			continue
		}
		matchedPattern = rule.Pattern.String()
		template := rule.Responses[rand.IntN(len(rule.Responses))]
		responseText = expandTemplate(template, matches, input)
		return
	}
	return "", ""
}

// logRequest appends an entry to the request log, keeping the last 100.
func (a *adminState) logRequest(entry requestEntry) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.requestLog = append(a.requestLog, entry)
	if len(a.requestLog) > 100 {
		a.requestLog = a.requestLog[len(a.requestLog)-100:]
	}
}

// resetRules restores the rule list to the initial startup rules.
func (a *adminState) resetRules() {
	a.mu.Lock()
	defer a.mu.Unlock()
	cp := make([]Rule, len(a.initialRules))
	copy(cp, a.initialRules)
	a.rules = cp
}

// fullReset restores rules and clears the request log.
func (a *adminState) fullReset() {
	a.mu.Lock()
	defer a.mu.Unlock()
	cp := make([]Rule, len(a.initialRules))
	copy(cp, a.initialRules)
	a.rules = cp
	a.requestLog = nil
}

// addRules inserts rules at the given priority position.
// priority 0 = prepend (default), -1 = append, positive int = insert at index.
func (a *adminState) addRules(rules []Rule, priority int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	switch {
	case priority == 0:
		a.rules = append(rules, a.rules...)
	case priority == -1:
		a.rules = append(a.rules, rules...)
	default:
		idx := priority
		if idx > len(a.rules) {
			idx = len(a.rules)
		}
		if idx < 0 {
			idx = 0
		}
		result := make([]Rule, 0, len(a.rules)+len(rules))
		result = append(result, a.rules[:idx]...)
		result = append(result, rules...)
		result = append(result, a.rules[idx:]...)
		a.rules = result
	}
}

// getRequests returns a copy of the request log.
func (a *adminState) getRequests() []requestEntry {
	a.mu.RLock()
	defer a.mu.RUnlock()
	cp := make([]requestEntry, len(a.requestLog))
	copy(cp, a.requestLog)
	return cp
}

// clearRequests empties the request log.
func (a *adminState) clearRequests() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.requestLog = nil
}

// getRulesJSON returns the current rules in a JSON-friendly format.
func (a *adminState) getRulesJSON() []ruleJSON {
	a.mu.RLock()
	defer a.mu.RUnlock()
	out := make([]ruleJSON, len(a.rules))
	for i, r := range a.rules {
		out[i] = ruleJSON{
			Pattern:   r.Pattern.String(),
			Responses: r.Responses,
		}
	}
	return out
}

// ruleJSON is the JSON representation of a rule for the admin API.
type ruleJSON struct {
	Pattern   string   `json:"pattern"`
	Responses []string `json:"responses"`
}

// addRulesRequest is the JSON body for POST /_mock/rules.
type addRulesRequest struct {
	Rules []addRuleEntry `json:"rules"`
}

type addRuleEntry struct {
	Pattern   string   `json:"pattern"`
	Responses []string `json:"responses"`
	Priority  *int     `json:"priority,omitempty"`
}

// adminResponder is a Responder that uses the adminState for rule matching
// and request logging. When no rule matches, it delegates to the fallback.
type adminResponder struct {
	state    *adminState
	fallback Responder

	mu              sync.Mutex
	lastMatchedRule string
}

func (ar *adminResponder) Respond(messages []InternalMessage) (string, error) {
	input := extractInput(messages)
	if input == "" {
		return "", errNoMessages
	}
	text, matched := ar.state.matchRules(input)
	ar.mu.Lock()
	ar.lastMatchedRule = matched
	ar.mu.Unlock()
	if text != "" {
		return text, nil
	}
	return ar.fallback.Respond(messages)
}

func (ar *adminResponder) getLastMatchedRule() string {
	ar.mu.Lock()
	defer ar.mu.Unlock()
	return ar.lastMatchedRule
}

// registerFaultRoutes adds the /_mock/faults endpoints to the mux.
func registerFaultRoutes(mux *http.ServeMux, fs *faultState) {
	mux.HandleFunc("GET /_mock/faults", func(w http.ResponseWriter, r *http.Request) {
		faults := fs.getFaults()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"faults": faults})
	})

	mux.HandleFunc("POST /_mock/faults", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Faults []Fault `json:"faults"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			// Try single fault for convenience.
			writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
			return
		}

		// Support both single fault (top-level fields) and array.
		if len(req.Faults) == 0 {
			writeError(w, http.StatusBadRequest, "faults array is required and must not be empty")
			return
		}

		fs.addFaults(req.Faults)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("DELETE /_mock/faults", func(w http.ResponseWriter, r *http.Request) {
		fs.clear()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})
}

// registerAdminRoutes adds the /_mock/ endpoints to the mux.
func registerAdminRoutes(mux *http.ServeMux, state *adminState) {
	mux.HandleFunc("GET /_mock/rules", func(w http.ResponseWriter, r *http.Request) {
		rules := state.getRulesJSON()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"rules": rules})
	})

	mux.HandleFunc("POST /_mock/rules", func(w http.ResponseWriter, r *http.Request) {
		var req addRulesRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
			return
		}
		if len(req.Rules) == 0 {
			writeError(w, http.StatusBadRequest, "rules array is required and must not be empty")
			return
		}

		compiled := make([]Rule, 0, len(req.Rules))
		priority := 0 // default: prepend
		for i, entry := range req.Rules {
			re, err := regexp.Compile(entry.Pattern)
			if err != nil {
				writeError(w, http.StatusBadRequest, "invalid regex in rule "+string(rune('0'+i))+": "+err.Error())
				return
			}
			if len(entry.Responses) == 0 {
				writeError(w, http.StatusBadRequest, "rule must have at least one response")
				return
			}
			compiled = append(compiled, Rule{Pattern: re, Responses: entry.Responses})
			if entry.Priority != nil {
				priority = *entry.Priority
			}
		}

		state.addRules(compiled, priority)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("DELETE /_mock/rules", func(w http.ResponseWriter, r *http.Request) {
		state.resetRules()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("POST /_mock/reset", func(w http.ResponseWriter, r *http.Request) {
		state.fullReset()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("GET /_mock/requests", func(w http.ResponseWriter, r *http.Request) {
		requests := state.getRequests()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"requests": requests})
	})

	mux.HandleFunc("DELETE /_mock/requests", func(w http.ResponseWriter, r *http.Request) {
		state.clearRequests()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})
}
