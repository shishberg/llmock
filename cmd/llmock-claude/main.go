// llmock-claude is a mock claude CLI binary for testing.
// It accepts the same flags as the real claude CLI and outputs
// newline-delimited JSON events in claude's stream-json format.
//
// Usage:
//
//	llmock-claude -p "hello" --output-format stream-json
//
// Configuration is loaded from LLMOCK_CONFIG env var, or auto-discovered
// from llmock.yaml in the current directory. Works with zero config using
// default rules/markov for responses.
package main

import (
	"log"
	"os"

	"github.com/shishberg/llmock"
)

func main() {
	// Load config: LLMOCK_CONFIG env, or auto-discover, or defaults.
	var cfg *llmock.Config
	cfgPath := os.Getenv("LLMOCK_CONFIG")
	if cfgPath == "" {
		cfgPath = llmock.FindDefaultConfig()
	}
	if cfgPath != "" {
		var err error
		cfg, err = llmock.LoadConfig(cfgPath)
		if err != nil {
			log.Fatalf("llmock-claude: loading config %s: %v", cfgPath, err)
		}
	} else {
		cfg = &llmock.Config{}
	}

	// Convert config to server options.
	opts, err := cfg.ToOptions()
	if err != nil {
		log.Fatalf("llmock-claude: invalid config: %v", err)
	}

	// Disable admin API for CLI mode — it's not needed.
	opts = append(opts, llmock.WithAdminAPI(false))

	s := llmock.New(opts...)
	t := llmock.NewCLITransport(s)
	os.Exit(t.Run(os.Args[1:], os.Stdout))
}
