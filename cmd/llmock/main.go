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

	var handler http.Handler = s.Handler()
	if *verbose {
		handler = verboseMiddleware(handler)
	}

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

// verboseMiddleware logs timestamp, method, path, status, and response time for each request.
func verboseMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &responseWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rw, r)
		elapsed := time.Since(start)
		log.Printf("llmock: %s %s -> %d (%s)",
			r.Method, r.URL.Path, rw.status, elapsed.Round(time.Millisecond))
	})
}

// responseWriter wraps http.ResponseWriter to capture the status code.
type responseWriter struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
}

func (rw *responseWriter) WriteHeader(code int) {
	if !rw.wroteHeader {
		rw.status = code
		rw.wroteHeader = true
	}
	rw.ResponseWriter.WriteHeader(code)
}

func (rw *responseWriter) Flush() {
	if f, ok := rw.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}
