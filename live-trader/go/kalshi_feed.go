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
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/gorilla/websocket"
)

// KalshiTick is one orderbook snapshot from the Kalshi WebSocket feed.
type KalshiTick struct {
	TS           time.Time
	MarketTicker string
	YesBid       int  // cents
	YesAsk       int  // cents
	YesLast      int  // cents — last traded price
	Volume       int  // contracts traded this update
	OpenInterest int  // total open contracts
	IsStale      bool // true if emitted as a timeout sentinel (no real data)
}

// KalshiFeed subscribes to a Kalshi market and streams ticks.
type KalshiFeed struct {
	currentTicker string
}

func NewKalshiFeed(initialTicker string) *KalshiFeed {
	return &KalshiFeed{currentTicker: initialTicker}
}

// Run connects to the Kalshi WebSocket and emits ticks until ctx is cancelled.
func (f *KalshiFeed) Run(ctx context.Context, marketCh <-chan string, out chan<- KalshiTick) {
	backoff := time.Second
	maxBackoff := 30 * time.Second

	for {
		connected, err := f.runSession(ctx, marketCh, out)
		if ctx.Err() != nil {
			return
		}

		if connected {
			backoff = time.Second
		}

		if err != nil {
			zlog.Error().Err(err).
				Str("market_ticker", f.currentTicker).
				Dur("retry_in", backoff).
				Msg("kalshi feed disconnected — reconnecting")
		}

		select {
		case <-ctx.Done():
			return
		case <-time.After(backoff):
		}

		backoff *= 2
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
	}
}

// runSession runs one WebSocket connection lifetime.
// Returns (connected, err): connected=true if at least one valid tick was received.
func (f *KalshiFeed) runSession(ctx context.Context, marketCh <-chan string, out chan<- KalshiTick) (connected bool, err error) {
	headers, err := GetKalshiAuthHeaders("GET", kalshiWebSocketSignPath)
	if err != nil {
		return false, fmt.Errorf("build auth headers: %w", err)
	}

	dialer := websocket.Dialer{HandshakeTimeout: 10 * time.Second}
	wsURL := kalshiWebSocketURL()
	conn, _, err := dialer.DialContext(ctx, wsURL, headers)
	if err != nil {
		return false, fmt.Errorf("dial %s: %w", wsURL, err)
	}
	defer conn.Close()

	zlog.Info().Str("market_ticker", f.currentTicker).Msg("kalshi feed connected")

	subMsg, _ := json.Marshal(map[string]any{
		"id":  1,
		"cmd": "subscribe",
		"params": map[string]any{
			"channels":       []string{"ticker"},
			"market_tickers": []string{f.currentTicker},
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

		case newTicker := <-marketCh:
			if newTicker != f.currentTicker {
				// Swap subscriptions dynamically without dropping conn
				unsubMsg, _ := json.Marshal(map[string]any{
					"id":  2,
					"cmd": "update_subscription",
					"params": map[string]any{
						"action":         "delete_markets",
						"channels":       []string{"ticker"},
						"market_tickers": []string{f.currentTicker},
					},
				})
				_ = conn.WriteMessage(websocket.TextMessage, unsubMsg)

				subMsg, _ := json.Marshal(map[string]any{
					"id":  3,
					"cmd": "update_subscription",
					"params": map[string]any{
						"action":         "add_markets",
						"channels":       []string{"ticker"},
						"market_tickers": []string{newTicker},
					},
				})
				_ = conn.WriteMessage(websocket.TextMessage, subMsg)

				zlog.Info().Str("old", f.currentTicker).Str("new", newTicker).Msg("hot-swapped kalshi feed market")
				f.currentTicker = newTicker
			}

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
		MarketTicker: inner.MarketTicker,
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
