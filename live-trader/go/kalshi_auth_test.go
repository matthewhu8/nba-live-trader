package main

import (
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestKalshiSignPath(t *testing.T) {
	tests := map[string]string{
		"/markets?event_ticker=KXNBA":        "/trade-api/v2/markets",
		"markets":                            "/trade-api/v2/markets",
		"/trade-api/v2/portfolio/orders?x=1": "/trade-api/v2/portfolio/orders",
		"/trade-api/ws/v2":                   "/trade-api/ws/v2",
	}

	for input, want := range tests {
		if got := kalshiSignPath(input); got != want {
			t.Fatalf("kalshiSignPath(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestKalshiLiveAuth(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping live Kalshi auth test in short mode")
	}
	if _, err := LoadEnvCandidates(".env", "../.env", "../../.env"); err != nil {
		t.Fatalf("load env: %v", err)
	}

	req, err := http.NewRequest("GET", kalshiRESTURL("/api_keys"), nil)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}

	headers, err := GetKalshiAuthHeaders("GET", "/api_keys")
	if err != nil {
		t.Fatalf("build auth headers: %v", err)
	}
	for key, values := range headers {
		req.Header[key] = values
	}

	client := http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Kalshi auth request failed: %v", err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("Kalshi auth returned HTTP %d; check key id, private key, host, and signed path", resp.StatusCode)
	}
}

func TestKalshiLiveWebSocketAuth(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping live Kalshi WebSocket auth test in short mode")
	}
	if _, err := LoadEnvCandidates(".env", "../.env", "../../.env"); err != nil {
		t.Fatalf("load env: %v", err)
	}

	headers, err := GetKalshiAuthHeaders("GET", kalshiWebSocketSignPath)
	if err != nil {
		t.Fatalf("build auth headers: %v", err)
	}

	dialer := websocket.Dialer{HandshakeTimeout: 10 * time.Second}
	conn, resp, err := dialer.Dial(kalshiWebSocketURL(), headers)
	if resp != nil && resp.Body != nil {
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
	}
	if err != nil {
		status := 0
		if resp != nil {
			status = resp.StatusCode
		}
		t.Fatalf("Kalshi WebSocket auth failed status=%d: %v", status, err)
	}
	_ = conn.Close()
}
