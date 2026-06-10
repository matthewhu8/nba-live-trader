// Tests for the exit path rewritten after the 2026-05-28 OKC@SAS post-mortem:
//   - exits cross the book with a bounded slippage budget (exitLimitPrice)
//   - CheckExit tracks the peak price and supports a trailing take-profit
//   - the hard stop-loss is the same trigger but is now actually fillable
package main

import (
	"context"
	"testing"
)

// exitCfg builds a Config with explicit exit knobs for CheckExit tests.
func exitCfg(tp, sl, maxHold, trailActivate, trailGiveback int) *Config {
	cfg := &Config{}
	cfg.Agent.TakeProfitCents = tp
	cfg.Agent.StopLossCents = sl
	cfg.Agent.MaxHoldPossessions = maxHold
	cfg.Agent.TrailActivateCents = trailActivate
	cfg.Agent.TrailGivebackCents = trailGiveback
	return cfg
}

// exitResp builds a PossessionResponse whose marketable price for a YES
// position equals yesBid (we use YES so the marketable side is just YesBid).
func exitResp(yesBid int) *PossessionResponse {
	return &PossessionResponse{YesBid: yesBid, YesAsk: yesBid + 1}
}

func TestExitLimitPrice(t *testing.T) {
	cases := []struct {
		name       string
		marketable int
		budget     int
		want       int
	}{
		{"no_budget_sells_at_bid", 40, 0, 40},
		{"budget_crosses_through_bid", 40, 2, 38},
		{"clamps_to_floor_1", 3, 5, 1},
		{"clamps_to_ceiling_99", 99, -5, 99}, // negative budget would exceed 99
		{"floor_exact", 1, 0, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := exitLimitPrice(tc.marketable, tc.budget); got != tc.want {
				t.Errorf("exitLimitPrice(%d,%d)=%d want %d", tc.marketable, tc.budget, got, tc.want)
			}
		})
	}
}

func TestCheckExitHardThresholds(t *testing.T) {
	r := NewRouter("", true)
	cfg := exitCfg(5, 3, 20, 5, 0) // trailing disabled (giveback 0)

	// YES entry at 45. Marketable price = YesBid.
	newPos := func() *PaperPosition {
		return &PaperPosition{Direction: "YES", EntryPrice: 45, Size: 5, EntryPossID: 0, PeakPrice: 45}
	}

	cases := []struct {
		name       string
		yesBid     int
		possID     int
		wantExit   bool
		wantReason string
	}{
		{"take_profit_at_plus5", 50, 1, true, "TAKE_PROFIT"},
		{"take_profit_beyond", 60, 1, true, "TAKE_PROFIT"},
		{"stop_loss_at_minus3", 42, 1, true, "STOP_LOSS"},
		{"stop_loss_gap_down", 30, 1, true, "STOP_LOSS"}, // would have been the -12¢ trade
		{"hold_inside_band", 47, 1, false, ""},
		{"time_stop_at_max_hold", 47, 20, true, "TIME_STOP"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			exit, reason, _ := r.CheckExit(context.TODO(), newPos(), exitResp(tc.yesBid), tc.possID, cfg)
			if exit != tc.wantExit || reason != tc.wantReason {
				t.Errorf("got (exit=%v reason=%q) want (exit=%v reason=%q)", exit, reason, tc.wantExit, tc.wantReason)
			}
		})
	}
}

// TestCheckExitTrailing reproduces the 2026-05-28 +15¢ excursion that the broken
// maker exit gave back to +3¢. With trailing on, the winner is allowed to run
// and is banked near the peak instead of capped at a flat +5.
func TestCheckExitTrailing(t *testing.T) {
	r := NewRouter("", true)
	cfg := exitCfg(5, 3, 20, 5, 3) // trailing on: activate +5, give back 3 from peak
	pos := &PaperPosition{Direction: "YES", EntryPrice: 29, Size: 5, EntryPossID: 0, PeakPrice: 29}

	// Price climbs 29 -> 44 (+15) without ever giving back 3 -> hold the whole way.
	for _, bid := range []int{34, 38, 43, 44} {
		exit, reason, _ := r.CheckExit(context.TODO(), pos, exitResp(bid), 1, cfg)
		if exit {
			t.Fatalf("trailing exited early at bid=%d (reason=%q); should ride the run", bid, reason)
		}
	}
	if pos.PeakPrice != 44 {
		t.Fatalf("peak not tracked: got %d want 44", pos.PeakPrice)
	}

	// Retrace from peak 44 by 3 -> exit at 41 as TRAIL_STOP (banks +12, not +3).
	exit, reason, pnl := r.CheckExit(context.TODO(), pos, exitResp(41), 2, cfg)
	if !exit || reason != "TRAIL_STOP" {
		t.Fatalf("expected TRAIL_STOP at 41, got (exit=%v reason=%q)", exit, reason)
	}
	if pnl <= 0 {
		t.Fatalf("trailing exit should bank a profit, got pnl=%v", pnl)
	}

	// Hard stop still wins over trailing when the move reverses past the stop.
	pos2 := &PaperPosition{Direction: "YES", EntryPrice: 29, Size: 5, EntryPossID: 0, PeakPrice: 29}
	r.CheckExit(context.TODO(), pos2, exitResp(31), 1, cfg) // peak +2, below activation
	exit, reason, _ = r.CheckExit(context.TODO(), pos2, exitResp(26), 2, cfg)
	if !exit || reason != "STOP_LOSS" {
		t.Fatalf("expected STOP_LOSS, got (exit=%v reason=%q)", exit, reason)
	}
}

// TestCheckExitNoPosition_NOdirection verifies the marketable price for a NO
// position is computed from the ask and trailing peak tracking still works.
func TestCheckExitNODirection(t *testing.T) {
	r := NewRouter("", true)
	cfg := exitCfg(5, 3, 20, 5, 0)
	// NO position entered at 30 (= 100 - YesAsk when YesAsk=70).
	pos := &PaperPosition{Direction: "NO", EntryPrice: 30, Size: 5, EntryPossID: 0, PeakPrice: 30}
	// YesAsk drops to 67 -> NO marketable = 33 (+3) -> still holding (TP is +5).
	exit, _, _ := r.CheckExit(context.TODO(), pos, &PossessionResponse{YesBid: 65, YesAsk: 67}, 1, cfg)
	if exit {
		t.Fatalf("should hold at +3 with TP=5")
	}
	if pos.PeakPrice != 33 {
		t.Fatalf("NO peak not tracked: got %d want 33", pos.PeakPrice)
	}
	// YesAsk drops to 62 -> NO marketable = 38 (+8) -> take profit.
	exit, reason, _ := r.CheckExit(context.TODO(), pos, &PossessionResponse{YesBid: 60, YesAsk: 62}, 2, cfg)
	if !exit || reason != "TAKE_PROFIT" {
		t.Fatalf("expected TAKE_PROFIT, got (exit=%v reason=%q)", exit, reason)
	}
}
