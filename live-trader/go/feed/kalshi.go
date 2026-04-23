// KalshiFeed maintains a live WebSocket connection to the Kalshi market feed
// for a single game's market ticker.
//
// Emits KalshiTick structs to the provided channel on every orderbook update.
// Reconnects automatically on disconnect (exponential backoff, max 30s).
// If feed is silent for >30s during a live game: emits a StaleTick sentinel
// so the ring buffer can zero out has_market_data.
package feed

import (
	"context"
	"time"
)

// KalshiTick is one orderbook snapshot from the Kalshi WebSocket feed.
type KalshiTick struct {
	TS           time.Time
	YesBid       int // cents
	YesAsk       int // cents
	YesLast      int // cents — last traded price
	Volume       int // contracts traded this update
	OpenInterest int // total open contracts
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
	// TODO: dial wss://trading-api.kalshi.com/trade-api/ws/v2
	// TODO: send subscribe message for f.marketTicker
	// TODO: on each orderbook_snapshot / orderbook_delta: parse and emit KalshiTick
	// TODO: reconnect loop with exponential backoff on disconnect
	// TODO: emit IsStale=true sentinel if no tick received in 30s
}
