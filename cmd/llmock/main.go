package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/shishberg/llmock"
)

func main() {
	port := flag.Int("port", 0, "port to listen on (default: 9090, or PORT env var)")
	flag.Parse()

	p := *port
	if p == 0 {
		if env := os.Getenv("PORT"); env != "" {
			fmt.Sscanf(env, "%d", &p)
		}
	}
	if p == 0 {
		p = 9090
	}

	s := llmock.New()
	addr := fmt.Sprintf(":%d", p)
	log.Printf("llmock listening on %s", addr)
	if err := http.ListenAndServe(addr, s.Handler()); err != nil {
		log.Fatal(err)
	}
}
