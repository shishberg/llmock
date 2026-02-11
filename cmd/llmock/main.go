package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/shishberg/llmock"
)

func main() {
	configPath := flag.String("config", "", "path to config file (YAML or JSON)")
	port := flag.Int("port", 0, "port to listen on (overrides config)")
	verbose := flag.Bool("verbose", false, "log all requests/responses to stderr")
	mcpStdio := flag.Bool("mcp-stdio", false, "run MCP control plane over stdin/stdout (no HTTP server)")
	flag.Parse()

	// Load config: explicit --config, or auto-discover, or defaults.
	var cfg *llmock.Config
	cfgPath := *configPath
	if cfgPath == "" {
		cfgPath = llmock.FindDefaultConfig()
	}
	if cfgPath != "" {
		var err error
		cfg, err = llmock.LoadConfig(cfgPath)
		if err != nil {
			log.Fatalf("loading config %s: %v", cfgPath, err)
		}
	} else {
		cfg = &llmock.Config{}
	}

	// Apply --verbose flag.
	if *verbose {
		v := true
		cfg.Server.Verbose = &v
	}

	// Convert config to options.
	opts, err := cfg.ToOptions()
	if err != nil {
		log.Fatalf("invalid config: %v", err)
	}

	// Resolve port: --port flag > config > PORT env > 9090.
	p := *port
	if p == 0 {
		p = cfg.Server.Port
	}
	if p == 0 {
		if env := os.Getenv("PORT"); env != "" {
			fmt.Sscanf(env, "%d", &p)
		}
	}
	if p == 0 {
		p = 9090
	}

	s := llmock.New(opts...)

	// MCP stdio mode: run control plane over stdin/stdout instead of HTTP.
	if *mcpStdio {
		st := llmock.NewStdioTransport(s)
		if st == nil {
			log.Fatal("llmock: --mcp-stdio requires admin API to be enabled")
		}
		log.Printf("llmock: MCP control plane running on stdio")
		if err := st.Run(os.Stdin, os.Stdout); err != nil {
			log.Fatalf("llmock: stdio error: %v", err)
		}
		return
	}

	handler := s.Handler()

	// Startup banner.
	adminStatus := "enabled"
	if cfg.Server.AdminAPI != nil && !*cfg.Server.AdminAPI {
		adminStatus = "disabled"
	}
	ruleCount := len(cfg.Rules)
	corpusInfo := "default"
	if cfg.CorpusFile != "" {
		corpusInfo = cfg.CorpusFile
	}
	if cfgPath != "" {
		log.Printf("llmock: loaded config from %s", cfgPath)
	}
	log.Printf("llmock: port=%d rules=%d corpus=%s admin=%s",
		p, ruleCount, corpusInfo, adminStatus)

	// Set up server with graceful shutdown.
	addr := fmt.Sprintf(":%d", p)
	srv := &http.Server{
		Addr:    addr,
		Handler: handler,
	}

	// Listen for shutdown signals.
	done := make(chan struct{})
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Printf("llmock: shutting down...")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := srv.Shutdown(ctx); err != nil {
			log.Printf("llmock: shutdown error: %v", err)
		}
		close(done)
	}()

	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatal(err)
	}
	<-done
}

