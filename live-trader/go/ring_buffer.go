// KalshiRingBuffer maintains a rolling window of Kalshi ticks (last 120s)
// and computes the 14 MARKET_COLS features on demand at Snapshot() time.
//
// This is the Go side of the X vector computation. It produces the market
// feature slice that Go passes to the Python inference service alongside
// each raw NBA event. Python never sees raw ticks — only this snapshot.
//
// Feature definitions (must match models/mmoe/feature_config.py MARKET_COLS order):
//   [0]  yes_bid                  — current best bid (cents)
//   [1]  yes_ask                  — current best ask (cents)
//   [2]  spread                   — yes_ask - yes_bid
//   [3]  yes_last                 — last traded price (cents)
//   [4]  open_interest            — total open contracts
//   [5]  trade_volume_60s         — sum of volume in last 60s
//   [6]  time_since_last_trade_ms — ms since last volume > 0 tick
//   [7]  open_interest_change_60s — open_interest now - open_interest 60s ago
//   [8]  d_yes_bid                — yes_bid now - yes_bid at last possession snapshot
//   [9]  d_spread                 — spread now - spread at last possession snapshot
//   [10] bid_velocity_30s         — (bid_now - bid_30s_ago) / 30
//   [11] bid_acceleration_30s     — velocity_now - velocity_30s_ago
//   [12] bid_vs_last_divergence   — yes_bid - yes_last
//   [13] has_market_data          — 1.0 if feed active, 0.0 if stale
package main

import (
	"sync"
	"time"

)

const ringWindowSecs = 120

// MarketSnapshot holds the 14 computed market features ready to send to Python.
type MarketSnapshot struct {
	Features    [14]float32
	HasData     bool
	SnapshotTS  time.Time
}

type RingBuffer struct {
	mu          sync.RWMutex
	ticks       []KalshiTick // rolling window, newest last
	prevSnap    MarketSnapshot    // snapshot taken at last possession (for d_yes_bid, d_spread)
}

func NewRingBuffer() *RingBuffer {
	return &RingBuffer{}
}

// Update adds a new tick to the buffer. Called from the Kalshi feed goroutine.
func (rb *RingBuffer) Update(tick KalshiTick) {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	// TODO: append tick, evict entries older than ringWindowSecs
}

// Snapshot computes all 14 market features from current buffer state.
// Called from the game engine main loop on each possession event.
// Updates prevSnap so the next call can compute d_yes_bid / d_spread.
func (rb *RingBuffer) Snapshot() MarketSnapshot {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	// TODO: compute all 14 features
	// TODO: update rb.prevSnap
	return MarketSnapshot{}
}

// helper: sum volume in last nSecs seconds
func (rb *RingBuffer) volumeWindow(nSecs int) int { return 0 }

// helper: bid at approximately nSecs ago (nearest tick)
func (rb *RingBuffer) bidAtOffset(nSecs int) int { return 0 }
