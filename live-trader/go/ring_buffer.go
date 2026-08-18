// RingBuffer keeps the last 120s of Kalshi ticks and computes the 14 MARKET_COLS
// features on demand. Python never sees raw ticks, only these snapshots.
//
// Feature order must match MARKET_COLS in models/mmoe/feature_config.py:
//
//	[0]  yes_bid                   current best bid (cents)
//	[1]  yes_ask                   current best ask (cents)
//	[2]  spread                    yes_ask - yes_bid
//	[3]  yes_last                  last traded price (cents)
//	[4]  open_interest             total open contracts
//	[5]  trade_volume_60s          volume traded in the last 60s
//	[6]  time_since_last_trade_ms  ms since a tick carried a trade
//	[7]  open_interest_change_60s  open interest now minus 60s ago
//	[8]  d_yes_bid                 yes_bid change since the last possession
//	[9]  d_spread                  spread change since the last possession
//	[10] bid_velocity_30s          (bid_now - bid_30s_ago) / 30
//	[11] bid_acceleration_30s      velocity now minus velocity last possession
//	[12] bid_vs_last_divergence    yes_bid - yes_last
//	[13] has_market_data           1.0 if the feed is active, 0.0 if stale
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
	ticks        []KalshiTick   // rolling window, newest last
	prevSnap     MarketSnapshot // last possession's snapshot, for d_yes_bid and d_spread
	prevVelocity float32        // last possession's velocity, for acceleration
}

func NewRingBuffer() *RingBuffer {
	return &RingBuffer{}
}

// Update adds a tick to the buffer. Called from the Kalshi feed goroutine.
func (rb *RingBuffer) Update(tick KalshiTick) {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	rb.ticks = append(rb.ticks, tick)

	cutoff := tick.TS.Add(-ringWindowSecs * time.Second)
	evict := 0
	for evict < len(rb.ticks) && rb.ticks[evict].TS.Before(cutoff) {
		evict++
	}
	if evict > 0 {
		rb.ticks = rb.ticks[evict:]
	}
}

// Snapshot computes all 14 market features from the current buffer. The game loop
// calls it once per possession; it updates prevSnap so the next call can diff.
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

	// [6] ms since a tick actually carried a trade. The test is the delta in the
	// cumulative counter, not the counter itself, which is non-zero on nearly every tick.
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

// volumeWindow sums the volume actually traded in the last nSecs seconds. Kalshi's
// `volume` is a cumulative lifetime counter, so this sums first differences between
// consecutive ticks. Mirrors dataset.py::_compute_market_features_for_game.
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

// tradedAt reports the volume traded at tick i: the rise in the cumulative counter
// since the previous tick. The oldest tick has no predecessor and reports no trade.
func (rb *RingBuffer) tradedAt(i int) int {
	if i <= 0 {
		return 0
	}
	if d := rb.ticks[i].Volume - rb.ticks[i-1].Volume; d > 0 {
		return d
	}
	return 0
}

// bidAtOffset returns the YesBid of the tick nearest to nSecs ago.
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
		// Ticks are oldest-first, so once we pass the target walking backwards the
		// rest are only getting farther away.
		if rb.ticks[i].TS.Before(target) {
			break
		}
	}
	return best.YesBid
}

// openInterestAtOffset returns the OpenInterest of the tick nearest to nSecs ago.
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
