package main

import (
	"math"
	"testing"
)

// TestAggregateTrajParity exercises the four aggregator modes against the same
// fixed inputs used in the Python sweep harness. If this test passes, the Go
// helper is in lockstep with backtesting/mmoe_backtest.py:aggregate_traj — the
// invariant that lets us trust live decisions to match backtest results.
func TestAggregateTrajParity(t *testing.T) {
	// Same vector as in the Python helper smoke test (rising then falling).
	traj := [10]float32{0.02, 0.05, 0.08, 0.10, 0.12, 0.14, 0.15, 0.13, 0.10, 0.07}

	cases := []struct {
		mode string
		want float32
	}{
		{"final", 0.07},
		{"mean", 0.0960},
		{"mean_3_to_9", 0.1157143},
		{"max_abs", 0.15},
		// Unknown mode falls back to "final" — defensive but documented.
		{"banana", 0.07},
	}

	for _, tc := range cases {
		got := aggregateTraj(traj, tc.mode)
		if math.Abs(float64(got-tc.want)) > 1e-4 {
			t.Errorf("aggregateTraj(traj, %q) = %.4f, want %.4f", tc.mode, got, tc.want)
		}
	}
}

// TestAggregateTrajSignPreservation: for a trajectory whose magnitude peaks at
// a negative element, max_abs must return the SIGNED value (not the absolute
// magnitude). Otherwise downstream direction inference (sign of return) would
// be wrong.
func TestAggregateTrajSignPreservation(t *testing.T) {
	neg := [10]float32{-0.05, -0.15, -0.20, -0.18, -0.10, -0.05, 0.02, 0.05, 0.03, 0.01}
	got := aggregateTraj(neg, "max_abs")
	if got != -0.20 {
		t.Errorf("max_abs sign preservation: got %.4f, want -0.20", got)
	}
}

// TestKellyContracts validates the doubled-sizing formula. Anchor = 0.08 (entry
// threshold for the `mean` aggregator), slope = 200, min = 10, cap = 80.
// Numbers in this table must match the docstring on kellyContracts in orders.go.
func TestKellyContracts(t *testing.T) {
	cases := []struct {
		trajUsed float32
		want     int
	}{
		{0.05, 10}, // below anchor → floor
		{0.08, 10}, // at anchor → floor
		{0.12, 10}, // (0.12-0.08)*200 = 8, floored to 10
		{0.15, 14}, // (0.15-0.08)*200 = 14
		{0.20, 24},
		{0.30, 44},
		// 0.48 sits at the boundary — float32 (0.48-0.08)*200 rounds slightly under
		// 80, so int() yields 79. The cap kicks in at 0.49.
		{0.50, 80}, // (0.50-0.08)*200 = 84, capped to 80
		{0.60, 80}, // capped
		{-0.20, 24}, // negative input → absolute value used
	}
	for _, tc := range cases {
		got := kellyContracts(tc.trajUsed, 80, 0.08, 200, 10)
		if got != tc.want {
			t.Errorf("kellyContracts(%.2f) = %d, want %d", tc.trajUsed, got, tc.want)
		}
	}
}
