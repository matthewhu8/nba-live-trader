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
	Features      [14]float32
	HasData       bool
	SnapshotTS    time.Time
	CurrentTicker string
}

type RingBuffer struct {
	mu           sync.RWMutex
	ticks        []KalshiTick  // rolling window, newest last
	prevSnap     MarketSnapshot // snapshot taken at last possession (for d_yes_bid, d_spread)
	prevVelocity float32        // velocity at last possession snapshot (for acceleration)
}

func NewRingBuffer() *RingBuffer {
	return &RingBuffer{}
}

// Update adds a new tick to the buffer. Called from the Kalshi feed goroutine.
func (rb *RingBuffer) Update(tick KalshiTick) {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	rb.ticks = append(rb.ticks, tick)

	// Evict ticks older than ringWindowSecs from the front.
	cutoff := tick.TS.Add(-ringWindowSecs * time.Second)
	evict := 0
	for evict < len(rb.ticks) && rb.ticks[evict].TS.Before(cutoff) {
		evict++
	}
	if evict > 0 {
		rb.ticks = rb.ticks[evict:]
	}
}

// Snapshot computes all 14 market features from current buffer state.
// Called from the game engine main loop on each possession event.
// Updates prevSnap so the next call can compute d_yes_bid / d_spread.
func (rb *RingBuffer) Snapshot() MarketSnapshot {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	if len(rb.ticks) == 0 {
		return MarketSnapshot{HasData: false}
	}

	latest := rb.ticks[len(rb.ticks)-1]
	if time.Since(latest.TS) > 30*time.Second {
		return MarketSnapshot{HasData: false}
	}

	now := time.Now()

	spread := latest.YesAsk - latest.YesBid

	// [5] volume in last 60s
	vol60 := rb.volumeWindow(60)

	// [6] ms since a tick actually carried a trade.
	// The predicate is the DELTA in the cumulative counter, not the counter
	// itself — `Volume > 0` is true on 99.96% of ticks and made this ~0 always.
	timeSinceLastTrade := float32(999999)
	for i := len(rb.ticks) - 1; i >= 1; i-- {
		if rb.tradedAt(i) > 0 {
			timeSinceLastTrade = float32(now.Sub(rb.ticks[i].TS).Milliseconds())
			break
		}
	}

	// [7] open interest change vs 60s ago
	oi60sAgo := rb.openInterestAtOffset(60)
	oiDelta := float32(latest.OpenInterest - oi60sAgo)

	// [8] d_yes_bid vs last possession snapshot
	dYesBid := float32(latest.YesBid) - rb.prevSnap.Features[0]

	// [9] d_spread vs last possession snapshot
	dSpread := float32(spread) - rb.prevSnap.Features[2]

	// [10] bid velocity: (bid_now - bid_30s_ago) / 30
	bid30s := rb.bidAtOffset(30)
	velocity := float32(latest.YesBid-bid30s) / 30.0

	// [11] acceleration: velocity_now - velocity at last snapshot
	acceleration := velocity - rb.prevVelocity

	// [12] bid vs last traded
	bidVsLast := float32(latest.YesBid - latest.YesLast)

	var features [14]float32
	features[0] = float32(latest.YesBid)
	features[1] = float32(latest.YesAsk)
	features[2] = float32(spread)
	features[3] = float32(latest.YesLast)
	features[4] = float32(latest.OpenInterest)
	features[5] = float32(vol60)
	features[6] = timeSinceLastTrade
	features[7] = oiDelta
	features[8] = dYesBid
	features[9] = dSpread
	features[10] = velocity
	features[11] = acceleration
	features[12] = bidVsLast
	features[13] = 1.0

	snap := MarketSnapshot{
		Features:      features,
		HasData:       true,
		SnapshotTS:    now,
		CurrentTicker: latest.MarketTicker,
	}

	rb.prevVelocity = velocity
	rb.prevSnap = snap

	return snap
}

// volumeWindow sums the volume actually TRADED in the last nSecs seconds.
//
// Kalshi's `volume` field is a cumulative lifetime counter, not per-tick size, so
// this sums first differences between consecutive ticks rather than the raw field.
// Summing the raw field yielded roughly (ticks in window) x (lifetime volume) —
// a number with no relationship to recent trading activity. Mirrors
// models/mmoe/dataset.py::_compute_market_features_for_game.
func (rb *RingBuffer) volumeWindow(nSecs int) int {
	cutoff := time.Now().Add(-time.Duration(nSecs) * time.Second)
	total := 0
	for i := len(rb.ticks) - 1; i >= 1; i-- {
		if rb.ticks[i].TS.Before(cutoff) {
			break
		}
		if d := rb.ticks[i].Volume - rb.ticks[i-1].Volume; d > 0 {
			total += d
		}
	}
	return total
}

// tradedAt reports the volume traded at tick i, i.e. the increase in the
// cumulative counter since the previous tick. The oldest retained tick has no
// predecessor and is reported as no trade rather than as its whole lifetime total.
func (rb *RingBuffer) tradedAt(i int) int {
	if i <= 0 {
		return 0
	}
	if d := rb.ticks[i].Volume - rb.ticks[i-1].Volume; d > 0 {
		return d
	}
	return 0
}

// bidAtOffset returns the YesBid of the tick nearest to now - nSecs*time.Second.
// Falls back to current bid if no tick is found.
func (rb *RingBuffer) bidAtOffset(nSecs int) int {
	if len(rb.ticks) == 0 {
		return 0
	}
	target := time.Now().Add(-time.Duration(nSecs) * time.Second)
	best := rb.ticks[len(rb.ticks)-1]
	bestDiff := absDuration(best.TS.Sub(target))
	for i := len(rb.ticks) - 2; i >= 0; i-- {
		d := absDuration(rb.ticks[i].TS.Sub(target))
		if d < bestDiff {
			bestDiff = d
			best = rb.ticks[i]
		}
		// Ticks are sorted oldest-first; once we pass the target going backwards,
		// further ticks are only getting farther away.
		if rb.ticks[i].TS.Before(target) {
			break
		}
	}
	return best.YesBid
}

// openInterestAtOffset returns the OpenInterest of the tick nearest to now - nSecs*time.Second.
func (rb *RingBuffer) openInterestAtOffset(nSecs int) int {
	if len(rb.ticks) == 0 {
		return 0
	}
	target := time.Now().Add(-time.Duration(nSecs) * time.Second)
	best := rb.ticks[len(rb.ticks)-1]
	bestDiff := absDuration(best.TS.Sub(target))
	for i := len(rb.ticks) - 2; i >= 0; i-- {
		d := absDuration(rb.ticks[i].TS.Sub(target))
		if d < bestDiff {
			bestDiff = d
			best = rb.ticks[i]
		}
		if rb.ticks[i].TS.Before(target) {
			break
		}
	}
	return best.OpenInterest
}

func absDuration(d time.Duration) time.Duration {
	if d < 0 {
		return -d
	}
	return d
}
