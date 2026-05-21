// Behavior tests for the Bandit.Decide entry path, realigned with the
// validated MMoE backtest config on 2026-05-11 (CLAUDE.md: +$9,977 net,
// 41.1% win rate, 112 trades on Apr 7 → May 10 val set).
//
// Two regression fixtures at the bottom encode the lessons from our paper-
// trade losses:
//   - yesterdays_OKC_losing_trade_now_blocked_by_run_prob
//   - last_nights_SAS_marginal_trade_blocked_by_run_prob
package main

import "testing"

func newBacktestBandit() *Bandit {
	cfg := Config{}
	cfg.Agent.MinYesBid = 30
	cfg.Agent.MaxYesBid = 70
	cfg.Agent.MinRunProbEntry = 0.15
	cfg.Agent.MinAbsTrajEntry = 0.08
	cfg.Agent.MinRunLengthEntry = 2
	return NewBandit(&cfg)
}

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

func withBid(v int) func(*PossessionResponse)      { return func(r *PossessionResponse) { r.YesBid = v } }
func withRunProb(v float32) func(*PossessionResponse) { return func(r *PossessionResponse) { r.RunProb = v } }
func withTraj9(v float32) func(*PossessionResponse) { return func(r *PossessionResponse) { r.Trajectory[9] = v } }
func withHaz4(v float32) func(*PossessionResponse)  { return func(r *PossessionResponse) { r.Hazard[4] = v } }
func withGarbage() func(*PossessionResponse)        { return func(r *PossessionResponse) { r.IsGarbageTime = true } }
func withBlowout() func(*PossessionResponse)        { return func(r *PossessionResponse) { r.IsBlowout = true } }
func withRunLen(v float32) func(*PossessionResponse) {
	return func(r *PossessionResponse) {
		if r.Features == nil {
			r.Features = map[string]float32{}
		}
		r.Features["current_run_length"] = v
	}
}
func withPeriod(v float32) func(*PossessionResponse) {
	return func(r *PossessionResponse) {
		if r.Features == nil {
			r.Features = map[string]float32{}
		}
		r.Features["period"] = v
	}
}

// strongEntrySignal is a "should fire BUY_YES" combo, used as a template
// to flip one knob at a time in the test cases below.
func strongEntrySignal() *PossessionResponse {
	return resp(withBid(50), withRunProb(0.30), withTraj9(0.5), withRunLen(5))
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
		// ── Guard rails ──────────────────────────────────────────────────
		{"garbage_time_blocks_entry", resp(withGarbage(), withRunProb(0.5), withTraj9(0.5), withRunLen(5)), false, Wait, "is_garbage_time"},
		{"garbage_time_blocks_held_position", resp(withGarbage(), withHaz4(0.9)), true, Wait, "is_garbage_time"},
		{"blowout_blocks_entry", resp(withBlowout(), withRunProb(0.5), withTraj9(0.5), withRunLen(5)), false, Wait, "is_blowout"},
		{"garbage_and_blowout_attributes_to_garbage", resp(withGarbage(), withBlowout()), false, Wait, "is_garbage_time"},

		// ── Overtime skip rule (added 2026-05-18) ────────────────────────
		// period == 4 OK (regulation); period == 5 OT1, period == 6 OT2 = block.
		// Model has zero training rows in the OT regime and the market scanner
		// thrashes in OT — see notes in agent.go.
		{"regulation_q4_not_overtime", resp(withPeriod(4), withRunProb(0.5), withTraj9(0.5), withRunLen(5)), false, BuyYes, ""},
		{"ot1_blocks_entry", resp(withPeriod(5), withRunProb(0.5), withTraj9(0.5), withRunLen(5)), false, Wait, "is_overtime"},
		{"ot2_blocks_entry", resp(withPeriod(6), withRunProb(0.5), withTraj9(0.5), withRunLen(5)), false, Wait, "is_overtime"},
		{"ot_blocks_held_position_too", resp(withPeriod(5), withHaz4(0.99)), true, Wait, "is_overtime"},
		{"ot_takes_precedence_over_garbage_time", resp(withPeriod(5), withGarbage()), false, Wait, "is_overtime"},

		// ── Price band ───────────────────────────────────────────────────
		{"below_band", resp(withBid(29), withRunProb(0.5), withTraj9(0.5), withRunLen(5)), false, Wait, "in_price_band"},
		{"above_band", resp(withBid(71), withRunProb(0.5), withTraj9(0.5), withRunLen(5)), false, Wait, "in_price_band"},
		{"at_lower_band_eligible", resp(withBid(30), withRunProb(0.5), withTraj9(0.5), withRunLen(5)), false, BuyYes, ""},
		{"at_upper_band_eligible", resp(withBid(70), withRunProb(0.5), withTraj9(-0.5), withRunLen(5)), false, BuyNo, ""},
		{"band_takes_precedence_over_strong_signal", resp(withBid(80), withRunProb(0.5), withTraj9(0.5), withRunLen(5)), false, Wait, "in_price_band"},

		// ── Has-position branch — bandit always waits, router handles exits ──
		// (Hazard exit is REMOVED to match the backtest. The router's
		//  CheckExit gates on TP/SL/TIME_STOP only.)
		{"position_high_hazard_no_longer_exits", resp(withHaz4(0.99)), true, Wait, "holding_position"},
		{"position_low_hazard_waits", resp(withHaz4(0.5)), true, Wait, "holding_position"},
		{"position_zero_hazard_waits", resp(), true, Wait, "holding_position"},
		{"position_max_hazard_waits", resp(withHaz4(1.0)), true, Wait, "holding_position"},

		// ── Entry path: run_prob gate (PRIMARY, restored 2026-05-11) ─────
		{"run_prob_zero_blocks", resp(withRunProb(0.0), withTraj9(0.5), withRunLen(5)), false, Wait, "run_prob_pass"},
		{"run_prob_just_below_threshold_blocks", resp(withRunProb(0.149), withTraj9(0.5), withRunLen(5)), false, Wait, "run_prob_pass"},
		{"run_prob_at_threshold_passes", resp(withRunProb(0.15), withTraj9(0.5), withRunLen(5)), false, BuyYes, ""},
		{"run_prob_well_above_threshold_passes", resp(withRunProb(0.3), withTraj9(0.5), withRunLen(5)), false, BuyYes, ""},

		// ── Entry path: trajectory magnitude gate ────────────────────────
		{"strong_run_prob_zero_traj_blocks", resp(withRunProb(0.3), withTraj9(0.0), withRunLen(5)), false, Wait, "traj_magnitude_pass"},
		{"strong_run_prob_weak_pos_traj_blocks", resp(withRunProb(0.3), withTraj9(0.079), withRunLen(5)), false, Wait, "traj_magnitude_pass"},
		{"strong_run_prob_weak_neg_traj_blocks", resp(withRunProb(0.3), withTraj9(-0.079), withRunLen(5)), false, Wait, "traj_magnitude_pass"},
		{"traj_at_threshold_pos_passes", resp(withRunProb(0.3), withTraj9(0.08), withRunLen(5)), false, BuyYes, ""},
		{"traj_at_threshold_neg_passes", resp(withRunProb(0.3), withTraj9(-0.08), withRunLen(5)), false, BuyNo, ""},

		// ── Entry path: run_length gate ──────────────────────────────────
		{"run_length_zero_blocks", resp(withRunProb(0.3), withTraj9(0.5), withRunLen(0)), false, Wait, "run_length_pass"},
		{"run_length_one_blocks", resp(withRunProb(0.3), withTraj9(0.5), withRunLen(1)), false, Wait, "run_length_pass"},
		{"run_length_at_threshold_passes", resp(withRunProb(0.3), withTraj9(0.5), withRunLen(2)), false, BuyYes, ""},

		// ── Gate ordering: run_prob is checked BEFORE traj before run_length ──
		{"weak_run_prob_and_weak_traj_blames_run_prob", resp(withRunProb(0.1), withTraj9(0.05), withRunLen(5)), false, Wait, "run_prob_pass"},
		{"strong_run_prob_weak_traj_and_zero_runlen_blames_traj", resp(withRunProb(0.3), withTraj9(0.05), withRunLen(0)), false, Wait, "traj_magnitude_pass"},

		// ── Missing-feature defaults ─────────────────────────────────────
		{"missing_run_length_feature_treated_as_zero_blocks", &PossessionResponse{
			YesBid: 50, RunProb: 0.3, Trajectory: [10]float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5},
			Features: map[string]float32{}, // explicitly empty
		}, false, Wait, "run_length_pass"},

		// ── 🔴 Named regression fixtures from live paper-trade losses ────
		// 2026-05-09 OKC@LAL: BUY_NO @ 57¢, lost $4.93 via immediate HAZARD_EXIT
		// Snapshot at entry: run_prob=0.107, traj_final=-0.021, hazard5=0.947, run_length=0
		// With restored run_prob gate, this trade is blocked at run_prob (0.107 < 0.15).
		{"yesterdays_OKC_losing_trade_now_blocked_by_run_prob",
			resp(withBid(57), withRunProb(0.107), withTraj9(-0.021), withHaz4(0.947), withRunLen(0)),
			false, Wait, "run_prob_pass"},

		// 2026-05-10 SAS@MIN T1: BUY_NO @ 43¢, lost $1.74 via HAZARD_EXIT (1 poss held)
		// Snapshot: run_prob=0.086, traj_final=-0.082, hazard5=0.897, run_length unknown but
		// since live had no run_prob gate, this fired. With gate restored, blocked at run_prob.
		{"last_nights_SAS_marginal_trade_blocked_by_run_prob",
			resp(withBid(43), withRunProb(0.086), withTraj9(-0.082), withHaz4(0.897), withRunLen(3)),
			false, Wait, "run_prob_pass"},

		// 🟢 The backtest WINNERS from SAS@MIN — all had run_prob ≥ 0.169.
		// These should now fire correctly under restored config.
		{"backtest_SAS_win_T1_now_fires", resp(withBid(37), withRunProb(0.179), withTraj9(-0.104), withRunLen(3)), false, BuyNo, ""},
		{"backtest_SAS_win_T2_now_fires", resp(withBid(60), withRunProb(0.188), withTraj9(-0.095), withRunLen(4)), false, BuyNo, ""},
		{"backtest_SAS_win_T3_now_fires", resp(withBid(39), withRunProb(0.169), withTraj9(-0.093), withRunLen(3)), false, BuyNo, ""},
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

// TestGateResultInvariants covers structural properties of GateResult that
// must hold across any input — independent of the specific trade decision.
func TestGateResultInvariants(t *testing.T) {
	b := newBacktestBandit()

	// has_position=true → bandit must always return Wait with no entry-side
	// gates populated. The router owns exit decisions, not the bandit.
	for _, r := range []*PossessionResponse{
		resp(withHaz4(0.9)),
		resp(withHaz4(0.5)),
		resp(withRunProb(0.5), withTraj9(0.5), withRunLen(5)), // even strong entry signal
	} {
		action, g := b.Decide(r, true)
		if action != Wait {
			t.Errorf("has_position=true must return Wait, got %v (resp=%+v)", action, r)
		}
		if g.RunProbPass != nil || g.TrajMagnitudePass != nil || g.RunLengthPass != nil {
			t.Errorf("entry gates should be nil when has_position=true (got rp=%v traj=%v rl=%v)",
				g.RunProbPass, g.TrajMagnitudePass, g.RunLengthPass)
		}
	}

	// has_position=false → bandit either returns Wait (with named blocker) or
	// trades. RunProbPass should ALWAYS be populated since it's the first
	// entry gate.
	for _, r := range []*PossessionResponse{
		resp(withRunProb(0.05)),                                // blocked at run_prob
		strongEntrySignal(),                                    // passes everything → trades
		resp(withRunProb(0.3), withTraj9(0.05), withRunLen(5)), // blocked at traj
	} {
		_, g := b.Decide(r, false)
		if g.RunProbPass == nil {
			t.Errorf("run_prob_pass must be populated when has_position=false (resp=%+v)", r)
		}
	}
}
