// Behavior tests for the backtest-aligned Bandit.Decide.
//
// On 2026-05-10 the bandit's entry gates were realigned to match the
// validated backtest config (Phase 8). The old Phase 3 equivalence test
// compared Decide against a frozen legacy implementation — which is no
// longer the desired behavior, so that test was replaced with this
// behavior-table suite that asserts specific (input → action, blocker)
// outcomes for every gate's pass/fail path.
//
// Adding new gates / changing thresholds: extend the `cases` slice. Each
// case is one row, easy to scan, easy to diff in code review.
package main

import "testing"

func newBacktestBandit() *Bandit {
	cfg := Config{}
	cfg.Agent.MinYesBid = 30
	cfg.Agent.MaxYesBid = 70
	cfg.Agent.MinAbsTrajEntry = 0.08
	cfg.Agent.MinRunLengthEntry = 2
	return NewBandit(&cfg)
}

// resp builds a PossessionResponse with overridable knobs. Default is a
// "trading-eligible context" (bid in band, not garbage/blowout) so each
// case can flip one or two fields without restating everything.
func resp(opts ...func(*PossessionResponse)) *PossessionResponse {
	r := &PossessionResponse{
		YesBid:        50,
		YesAsk:        51,
		IsGarbageTime: false,
		IsBlowout:     false,
		Features:      map[string]float32{"current_run_length": 0},
	}
	for _, o := range opts {
		o(r)
	}
	return r
}

func withBid(v int) func(*PossessionResponse)        { return func(r *PossessionResponse) { r.YesBid = v } }
func withTraj9(v float32) func(*PossessionResponse)  { return func(r *PossessionResponse) { r.Trajectory[9] = v } }
func withHaz4(v float32) func(*PossessionResponse)   { return func(r *PossessionResponse) { r.Hazard[4] = v } }
func withGarbage() func(*PossessionResponse)         { return func(r *PossessionResponse) { r.IsGarbageTime = true } }
func withBlowout() func(*PossessionResponse)         { return func(r *PossessionResponse) { r.IsBlowout = true } }
func withRunLen(v float32) func(*PossessionResponse) {
	return func(r *PossessionResponse) {
		if r.Features == nil {
			r.Features = map[string]float32{}
		}
		r.Features["current_run_length"] = v
	}
}

func TestDecideBacktestConfig(t *testing.T) {
	b := newBacktestBandit()

	cases := []struct {
		name        string
		resp        *PossessionResponse
		hasPosition bool
		wantAction  Action
		wantBlocker string // empty when wantAction != Wait
	}{
		// ── Guard rails: highest-precedence gates ────────────────────────
		{"garbage_time_blocks_entry", resp(withGarbage(), withTraj9(0.5), withRunLen(5)), false, Wait, "is_garbage_time"},
		{"garbage_time_blocks_held_position", resp(withGarbage(), withHaz4(0.9)), true, Wait, "is_garbage_time"},
		{"blowout_blocks_entry", resp(withBlowout(), withTraj9(0.5), withRunLen(5)), false, Wait, "is_blowout"},
		{"garbage_AND_blowout_attributes_to_garbage", resp(withGarbage(), withBlowout()), false, Wait, "is_garbage_time"},

		// ── Price band ───────────────────────────────────────────────────
		{"below_band", resp(withBid(29), withTraj9(0.5), withRunLen(5)), false, Wait, "in_price_band"},
		{"above_band", resp(withBid(71), withTraj9(0.5), withRunLen(5)), false, Wait, "in_price_band"},
		{"at_lower_band_eligible", resp(withBid(30), withTraj9(0.5), withRunLen(5)), false, BuyYes, ""},
		{"at_upper_band_eligible", resp(withBid(70), withTraj9(-0.5), withRunLen(5)), false, BuyNo, ""},
		{"band_takes_precedence_over_entry_signal", resp(withBid(80), withTraj9(0.5), withRunLen(5)), false, Wait, "in_price_band"},

		// ── Has-position branch (hazard exit — UNCHANGED behavior) ────────
		{"position_high_hazard_exits", resp(withHaz4(0.8)), true, Exit, ""},
		{"position_low_hazard_waits", resp(withHaz4(0.5)), true, Wait, "hazard_exit_pass"},
		{"position_at_hazard_threshold_waits", resp(withHaz4(0.75)), true, Wait, "hazard_exit_pass"},
		{"position_just_above_threshold_exits", resp(withHaz4(0.7501)), true, Exit, ""},
		{"position_max_hazard_exits", resp(withHaz4(1.0)), true, Exit, ""},
		{"position_zero_hazard_waits", resp(), true, Wait, "hazard_exit_pass"},

		// ── Entry path: trajectory magnitude gate ────────────────────────
		{"traj_zero_blocks", resp(withTraj9(0.0), withRunLen(5)), false, Wait, "traj_magnitude_pass"},
		{"traj_below_threshold_pos_blocks", resp(withTraj9(0.079), withRunLen(5)), false, Wait, "traj_magnitude_pass"},
		{"traj_below_threshold_neg_blocks", resp(withTraj9(-0.079), withRunLen(5)), false, Wait, "traj_magnitude_pass"},
		{"traj_at_threshold_pos_passes_magnitude", resp(withTraj9(0.08), withRunLen(5)), false, BuyYes, ""},
		{"traj_at_threshold_neg_passes_magnitude", resp(withTraj9(-0.08), withRunLen(5)), false, BuyNo, ""},
		{"traj_strong_positive", resp(withTraj9(0.5), withRunLen(5)), false, BuyYes, ""},
		{"traj_strong_negative", resp(withTraj9(-1.2), withRunLen(5)), false, BuyNo, ""},

		// ── Entry path: run_length gate ──────────────────────────────────
		{"strong_traj_zero_run_length_blocks", resp(withTraj9(0.5), withRunLen(0)), false, Wait, "run_length_pass"},
		{"strong_traj_run_length_1_blocks", resp(withTraj9(0.5), withRunLen(1)), false, Wait, "run_length_pass"},
		{"strong_traj_run_length_at_threshold_passes", resp(withTraj9(0.5), withRunLen(2)), false, BuyYes, ""},
		{"strong_traj_run_length_3_passes", resp(withTraj9(-0.5), withRunLen(3)), false, BuyNo, ""},
		{"missing_run_length_feature_defaults_to_zero_blocks", &PossessionResponse{
			YesBid: 50, Trajectory: [10]float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5},
			Features: map[string]float32{}, // explicitly empty
		}, false, Wait, "run_length_pass"},

		// ── Gate ordering: ensure traj_magnitude is checked BEFORE run_length ───
		{"weak_traj_and_zero_runlen_attrib_to_traj", resp(withTraj9(0.05), withRunLen(0)), false, Wait, "traj_magnitude_pass"},

		// ── Realistic scenarios from the 2026-05-09 OKC@LAL post-mortem ──
		// The actual losing trade: traj=-0.021, run_length=0 — should NOT trade now.
		{"yesterdays_losing_trade_now_blocked", resp(withTraj9(-0.021), withRunLen(0)), false, Wait, "traj_magnitude_pass"},
		// The 65¢→72¢ jump we missed: traj=-1.19, run_length≥2 — should now BUY NO.
		{"yesterdays_missed_jump_now_traded", resp(withBid(65), withTraj9(-1.19), withRunLen(4)), false, BuyNo, ""},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			action, gates := b.Decide(tc.resp, tc.hasPosition)
			if action != tc.wantAction {
				t.Errorf("action: got %v, want %v\n  case=%q resp=%+v",
					action, tc.wantAction, tc.name, tc.resp)
			}
			if action == Wait && gates.FirstBlocking != tc.wantBlocker {
				t.Errorf("first_blocking: got %q, want %q\n  case=%q",
					gates.FirstBlocking, tc.wantBlocker, tc.name)
			}
			if action != Wait && gates.FirstBlocking != "" {
				t.Errorf("first_blocking should be empty for action=%v, got %q\n  case=%q",
					action, gates.FirstBlocking, tc.name)
			}
		})
	}
}

// TestGateResultInvariants checks the structural properties that the
// GateResult contract guarantees, regardless of which fixture caused them:
//   - HazardExitPass is set iff HasPosition
//   - TrajMagnitudePass / RunLengthPass are only set in the no-position branch
//   - When FirstBlocking is set, action == Wait
func TestGateResultInvariants(t *testing.T) {
	b := newBacktestBandit()

	withPositionCases := []*PossessionResponse{
		resp(withHaz4(0.9)),
		resp(withHaz4(0.5)),
	}
	for _, r := range withPositionCases {
		_, g := b.Decide(r, true)
		if g.HazardExitPass == nil {
			t.Errorf("hazard_exit_pass should be set when has_position=true (resp=%+v)", r)
		}
		if g.TrajMagnitudePass != nil || g.RunLengthPass != nil {
			t.Errorf("entry gates should be nil when has_position=true (got traj=%v run_len=%v)",
				g.TrajMagnitudePass, g.RunLengthPass)
		}
	}

	noPositionCases := []*PossessionResponse{
		resp(withTraj9(0.05), withRunLen(0)),
		resp(withTraj9(0.5), withRunLen(0)),
		resp(withTraj9(0.5), withRunLen(5)),
	}
	for _, r := range noPositionCases {
		_, g := b.Decide(r, false)
		if g.HazardExitPass != nil {
			t.Errorf("hazard_exit_pass should be nil when has_position=false (got %v)", g.HazardExitPass)
		}
		if g.TrajMagnitudePass == nil {
			t.Errorf("traj_magnitude_pass should be set when has_position=false")
		}
	}
}
