// KalshiFeed maintains a live WebSocket connection to the Kalshi market feed
// for a single game's market ticker.
//
// Emits KalshiTick structs to the provided channel on every orderbook update.
// Reconnects automatically on disconnect (exponential backoff, max 30s).
// If feed is silent for >30s during a live game: emits a StaleTick sentinel
// so the ring buffer can zero out has_market_data.
package main

import (
	"context"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"math"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/gorilla/websocket"
	"github.com/joho/godotenv" // need to import this once online again
)

const (
	kalshiWSURL  = "wss://api.elections.kalshi.com/trade-api/ws/v2"
	kalshiWSPath = "/trade-api/ws/v2"
)

// KalshiTick is one orderbook snapshot from the Kalshi WebSocket feed.
type KalshiTick struct {
	TS           time.Time
	YesBid       int  // cents
	YesAsk       int  // cents
	YesLast      int  // cents — last traded price
	Volume       int  // contracts traded this update
	OpenInterest int  // total open contracts
	IsStale      bool // true if emitted as a timeout sentinel (no real data)
}

// KalshiFeed subscribes to a Kalshi market and streams ticks.
type KalshiFeed struct {
	marketTicker string
	apiKey       string
}

func NewKalshiFeed(marketTicker, apiKey string) *KalshiFeed {
	return &KalshiFeed{marketTicker: marketTicker, apiKey: apiKey}
}

// Run connects to the Kalshi WebSocket and emits ticks until ctx is cancelled.
func (f *KalshiFeed) Run(ctx context.Context, out chan<- KalshiTick) {
	backoff := time.Second
	maxBackoff := 30 * time.Second

	for {
		connected, err := f.runSession(ctx, out)
		if ctx.Err() != nil {
			return
		}

		if connected {
			backoff = time.Second
		}

		if err != nil {
			zlog.Error().Err(err).
				Str("market_ticker", f.marketTicker).
				Dur("retry_in", backoff).
				Msg("kalshi feed disconnected — reconnecting")
		}

		select {
		case <-ctx.Done():
			return
		case <-time.After(backoff):
		}

		// backoff strategy helps manage WS connection retries
		backoff *= 2
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
	}
}

// runSession runs one WebSocket connection lifetime.
// Returns (connected, err): connected=true if at least one valid tick was received.
func (f *KalshiFeed) runSession(ctx context.Context, out chan<- KalshiTick) (connected bool, err error) {
	headers, err := f.authHeaders()
	if err != nil {
		return false, fmt.Errorf("build auth headers: %w", err)
	}

	dialer := websocket.Dialer{HandshakeTimeout: 6 * time.Second}
	conn, _, err := dialer.DialContext(ctx, kalshiWSURL, headers)
	if err != nil {
		return false, fmt.Errorf("dial %s: %w", kalshiWSURL, err)
	}
	defer conn.Close()

	zlog.Info().Str("market_ticker", f.marketTicker).Msg("kalshi feed connected")

	subMsg, _ := json.Marshal(map[string]any{
		"id":  1,
		"cmd": "subscribe",
		"params": map[string]any{
			"channels":       []string{"ticker"},
			"market_tickers": []string{f.marketTicker},
		},
	})
	if err := conn.WriteMessage(websocket.TextMessage, subMsg); err != nil {
		return false, fmt.Errorf("send subscribe: %w", err)
	}

	staleDuration := 30 * time.Second
	staleTimer := time.NewTimer(staleDuration)
	defer staleTimer.Stop()

	type readResult struct {
		tick KalshiTick
		err  error
	}
	msgCh := make(chan readResult, 10)

	go func() {
		for {
			_, data, err := conn.ReadMessage()
			if err != nil {
				msgCh <- readResult{err: err}
				return
			}
			tick, ok := parseTickerMessage(data)
			if ok {
				msgCh <- readResult{tick: tick}
			}
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return connected, nil

		case <-staleTimer.C:
			select {
			case out <- KalshiTick{TS: time.Now(), IsStale: true}:
			default:
			}
			staleTimer.Reset(staleDuration)

		case res := <-msgCh:
			if res.err != nil {
				return connected, res.err
			}
			connected = true
			staleTimer.Reset(staleDuration)
			select {
			case out <- res.tick:
			default:
				// Channel full — drop rather than block.
			}
		}
	}
}

// authHeaders generates the three RSA-PSS signed request headers.
func (f *KalshiFeed) authHeaders() (http.Header, error) {
	err := godotenv.Load(".env")
	if err != nil {
		return nil, fmt.Errorf("trouble importing Kalshi keys from .env file")
	}

	keyID := os.Getenv("KALSHI_API_KEY")
	rsaKeyStr := os.Getenv("RSA_KEY_KALSHI")

	if keyID == "" || rsaKeyStr == "" {
		return nil, fmt.Errorf("env var(s) not set")
	}

	// Decode the base64 encoded RSA key
	decodedKeyBytes, err := base64.StdEncoding.DecodeString(rsaKeyStr)
	if err != nil {
		return nil, fmt.Errorf("failed to decode RSA key: %w", err)
	}

	// CODE BELOW IS LEGACY - we directly paste base64 string in the environment variable
	// pemBytes, err := os.ReadFile(pemPath)
	// if err != nil {
	// 	return nil, fmt.Errorf("read PEM %q: %w", pemPath, err)
	// }

	block, _ := pem.Decode(decodedKeyBytes)
	if block == nil {
		return nil, fmt.Errorf("no PEM block found in %q", decodedKeyBytes)
	}

	rsaKey, err := parseRSAKey(block.Bytes)
	if err != nil {
		return nil, err
	}

	tsMS := strconv.FormatInt(time.Now().UnixMilli(), 10)
	message := []byte(tsMS + "GET" + kalshiWSPath)
	digest := sha256.Sum256(message)

	sig, err := rsa.SignPSS(rand.Reader, rsaKey, crypto.SHA256, digest[:], &rsa.PSSOptions{
		SaltLength: rsa.PSSSaltLengthEqualsHash,
	})
	if err != nil {
		return nil, fmt.Errorf("sign PSS: %w", err)
	}

	h := http.Header{}
	h.Set("KALSHI-ACCESS-KEY", keyID)
	h.Set("KALSHI-ACCESS-TIMESTAMP", tsMS)
	h.Set("KALSHI-ACCESS-SIGNATURE", base64.StdEncoding.EncodeToString(sig))
	return h, nil
}

// parseRSAKey tries PKCS8 first, then PKCS1.
func parseRSAKey(derBytes []byte) (*rsa.PrivateKey, error) {
	key, err := x509.ParsePKCS8PrivateKey(derBytes)
	if err == nil {
		rsaKey, ok := key.(*rsa.PrivateKey)
		if !ok {
			return nil, fmt.Errorf("PKCS8 key is not RSA")
		}
		return rsaKey, nil
	}

	rsaKey, pkcs1Err := x509.ParsePKCS1PrivateKey(derBytes)
	if pkcs1Err != nil {
		return nil, fmt.Errorf("parse private key (PKCS8: %v, PKCS1: %v)", err, pkcs1Err)
	}
	return rsaKey, nil
}

// tickerMsg is the outer envelope for Kalshi WebSocket messages.
type tickerMsg struct {
	Type string          `json:"type"`
	Msg  json.RawMessage `json:"msg"`
}

// tickerInner is the inner payload for type="ticker" messages.
type tickerInner struct {
	MarketTicker   string `json:"market_ticker"`
	YesBidDollars  string `json:"yes_bid_dollars"`
	YesAskDollars  string `json:"yes_ask_dollars"`
	PriceDollars   string `json:"price_dollars"`
	VolumeFP       string `json:"volume_fp"`
	OpenInterestFP string `json:"open_interest_fp"`
}

// parseTickerMessage decodes a raw WebSocket frame into a KalshiTick.
// Returns (tick, true) on success; (zero, false) if the message should be skipped.
func parseTickerMessage(data []byte) (KalshiTick, bool) {
	var outer tickerMsg
	if err := json.Unmarshal(data, &outer); err != nil {
		return KalshiTick{}, false
	}
	if outer.Type != "ticker" {
		return KalshiTick{}, false
	}

	var inner tickerInner
	if err := json.Unmarshal(outer.Msg, &inner); err != nil {
		return KalshiTick{}, false
	}

	yesBid := dollarsStrToCents(inner.YesBidDollars)
	if yesBid == 0 {
		// Pre/post-game — market not actively quoted.
		return KalshiTick{}, false
	}

	return KalshiTick{
		TS:           time.Now(),
		YesBid:       yesBid,
		YesAsk:       dollarsStrToCents(inner.YesAskDollars),
		YesLast:      dollarsStrToCents(inner.PriceDollars),
		Volume:       fpStrToInt(inner.VolumeFP),
		OpenInterest: fpStrToInt(inner.OpenInterestFP),
	}, true
}

func dollarsStrToCents(s string) int {
	if s == "" {
		return 0
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0
	}
	return int(math.Round(f * 100))
}

func fpStrToInt(s string) int {
	if s == "" {
		return 0
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0
	}
	return int(f)
}
