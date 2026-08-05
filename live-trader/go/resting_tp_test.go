package main

import (
	"context"
	"testing"
)

// TestRestingTPPriceClamping covers the [1,99] clamp and the basic add path.
func TestRestingTPPriceClamping(t *testing.T) {
	cases := []struct {
		entry, tp, want int
	}{
		{30, 5, 35},   // typical entry
		{70, 5, 75},   // upper band
		{1, 5, 6},     // floor — nothing special
		{96, 5, 99},   // clamp ceiling
		{99, 5, 99},   // already at ceiling
		{0, 1, 1},     // would yield 1, clamped at 1
		{-3, 1, 1},    // pathological — still clamped
	}
	for _, tc := range cases {
		if got := restingTPPrice(tc.entry, tc.tp); got != tc.want {
			t.Errorf("restingTPPrice(%d,%d)=%d want %d", tc.entry, tc.tp, got, tc.want)
		}
	}
}

// TestPlaceRestingTPPaperMode verifies the paper-mode shortcut sets a target
// price and a fake order_id without hitting the network.
func TestPlaceRestingTPPaperMode(t *testing.T) {
	r := NewRouter("", true) // paper mode
	pos := &PaperPosition{
		Direction:   "NO",
		EntryPrice:  42, // NO bid
		Size:        10,
		EntryTicker: "KXNBASPREAD-26JUN05NYKSAS-SAS6",
	}

	if err := r.PlaceRestingTP(context.TODO(), pos, 5); err != nil {
		t.Fatalf("paper PlaceRestingTP returned err: %v", err)
	}
	if pos.RestingTPPrice != 47 {
		t.Errorf("RestingTPPrice = %d, want 47", pos.RestingTPPrice)
	}
	if pos.RestingTPStatus != "open" {
		t.Errorf("RestingTPStatus = %q, want 'open'", pos.RestingTPStatus)
	}
	if pos.RestingTPOrderID == "" {
		t.Error("RestingTPOrderID should be set in paper mode")
	}

	// Idempotency: calling twice doesn't change state.
	firstID := pos.RestingTPOrderID
	_ = r.PlaceRestingTP(context.TODO(), pos, 5)
	if pos.RestingTPOrderID != firstID {
		t.Errorf("PlaceRestingTP not idempotent: order_id changed %q -> %q", firstID, pos.RestingTPOrderID)
	}
}

// TestCancelRestingTPPaperMode verifies the paper-mode cancel just flips the
// state — no network round trip, no race condition to worry about.
func TestCancelRestingTPPaperMode(t *testing.T) {
	r := NewRouter("", true)
	pos := &PaperPosition{
		Direction:        "YES",
		EntryPrice:       45,
		Size:             10,
		RestingTPOrderID: "PAPER-deadbeef",
		RestingTPStatus:  "open",
		RestingTPPrice:   50,
	}
	status, err := r.CancelRestingTP(context.TODO(), pos)
	if err != nil {
		t.Fatalf("paper cancel returned err: %v", err)
	}
	if status != "canceled" {
		t.Errorf("status after cancel = %q, want 'canceled'", status)
	}
	if pos.RestingTPStatus != "canceled" {
		t.Errorf("pos.RestingTPStatus = %q, want 'canceled'", pos.RestingTPStatus)
	}

	// Cancel-on-canceled is a no-op (idempotent).
	status2, _ := r.CancelRestingTP(context.TODO(), pos)
	if status2 != "canceled" {
		t.Errorf("second cancel status = %q, want 'canceled'", status2)
	}
}

// TestCheckExitWithRestingTPExecuted: when the resting TP has filled (paper or
// live equivalent), CheckExit returns TAKE_PROFIT at RestingTPPrice — NOT at
// currentPrice. This is the whole point: maker fills at the limit, not through.
func TestCheckExitWithRestingTPExecuted(t *testing.T) {
	r := NewRouter("", true)
	cfg := exitCfg(5, 3, 20, 5, 0)
	pos := &PaperPosition{
		Direction:        "YES",
		EntryPrice:       45,
		Size:             10,
		PeakPrice:        45,
		RestingTPOrderID: "ord-123",
		RestingTPStatus:  "executed",
		RestingTPPrice:   50, // limit price target
	}

	// Even with a runaway price (yesBid=70), the TP closes at the limit (50),
	// not the through price.
	exit, reason, pnl := r.CheckExit(context.TODO(), pos, exitResp(70), 1, cfg)
	if !exit || reason != "TAKE_PROFIT" {
		t.Fatalf("got (exit=%v reason=%q) want (true, TAKE_PROFIT)", exit, reason)
	}
	// At the limit price (50), not the through price (70), and net of maker fees
	// on both legs since a resting TP fills as a maker.
	wantPnL := float64(50-45)*10/100.0 -
		kalshiFee(10, 45, makerFeeRate) - kalshiFee(10, 50, makerFeeRate)
	if pnl != wantPnL {
		t.Errorf("pnl = %v, want %v (at RestingTPPrice not yesBid, net of fees)", pnl, wantPnL)
	}
}

// TestPaperModeTPSimulatesRestingFill: in paper mode (where Kalshi can't fill
// the resting order for us), when price reaches the TP target, CheckExit
// simulates the fill — same return as if Kalshi had filled it, and it sets
// RestingTPStatus="executed" so subsequent calls don't double-fire.
func TestPaperModeTPSimulatesRestingFill(t *testing.T) {
	r := NewRouter("", true)
	cfg := exitCfg(5, 3, 20, 5, 0)
	pos := &PaperPosition{
		Direction:  "YES",
		EntryPrice: 40,
		Size:       10,
		PeakPrice:  40,
	}

	exit, reason, pnl := r.CheckExit(context.TODO(), pos, exitResp(50), 1, cfg)
	if !exit || reason != "TAKE_PROFIT" {
		t.Fatalf("paper TP not triggered: (exit=%v reason=%q)", exit, reason)
	}
	if pos.RestingTPStatus != "executed" {
		t.Errorf("RestingTPStatus should be 'executed' after paper TP, got %q", pos.RestingTPStatus)
	}
	// pnl reflects fill at TP target (45), not the runaway bid (50), and is NET
	// of both legs' fees. Entry and a resting TP both fill as makers:
	//   gross = 5¢ × 10 / 100                     = $0.50
	//   fee   = kalshiFee(10, 40, maker) at entry = $0.05
	//         + kalshiFee(10, 45, maker) at exit  = $0.05
	//   net                                       = $0.40
	wantGross := float64(5*10) / 100.0
	wantFees := kalshiFee(10, 40, makerFeeRate) + kalshiFee(10, 45, makerFeeRate)
	if pnl != wantGross-wantFees {
		t.Errorf("paper TP pnl = %v, want %v (gross %v less fees %v)",
			pnl, wantGross-wantFees, wantGross, wantFees)
	}
	if wantFees == 0 {
		t.Error("fees are zero — calcNetPnL has regressed to gross P&L")
	}
}
