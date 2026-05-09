// Equivalence test for the Phase 3 Bandit.Decide refactor.
//
// The new Decide returns (Action, GateResult) instead of Action alone. The
// Action returned MUST be bit-identical to the legacy implementation across
// every possible (PossessionResponse, hasPosition) input — otherwise we have
// silently changed live trading behavior, which is the single highest-risk
// thing in the entire logging refactor.
//
// decideLegacy is a verbatim copy of pre-refactor Decide kept solely as the
// equivalence reference. Do not modify it.
package main

import "testing"

// decideLegacy is the exact pre-Phase-3 implementation. Frozen in time.
func decideLegacy(b *Bandit, resp *PossessionResponse, hasPosition bool) Action {
	if resp.IsGarbageTime || resp.IsBlowout {
		return Wait
	}
	if resp.YesBid < b.minYesBid || resp.YesBid > b.maxYesBid {
		return Wait
	}
	if hasPosition {
		if resp.Hazard[4] > b.maxHazardForHold {
			return Exit
		}
		return Wait
	}
	if resp.RunProb >= b.minRunProbEntry {
		if resp.Trajectory[9] > 0 {
			return BuyYes
		} else if resp.Trajectory[9] < 0 {
			return BuyNo
		}
	}
	return Wait
}

// makeResp builds a PossessionResponse with overridable knobs. Defaults
// represent a "trading-eligible" possession: bid in band, no garbage time,
// no blowout, no run-prob entry signal.
func makeResp(opts ...func(*PossessionResponse)) *PossessionResponse {
	r := &PossessionResponse{
		YesBid:        50,
		YesAsk:        51,
		RunProb:       0.05,
		IsGarbageTime: false,
		IsBlowout:     false,
	}
	for _, o := range opts {
		o(r)
	}
	return r
}

func withYesBid(v int) func(*PossessionResponse)  { return func(r *PossessionResponse) { r.YesBid = v } }
func withRunProb(v float32) func(*PossessionResponse) { return func(r *PossessionResponse) { r.RunProb = v } }
func withTraj9(v float32) func(*PossessionResponse) {
	return func(r *PossessionResponse) { r.Trajectory[9] = v }
}
func withHazard4(v float32) func(*PossessionResponse) {
	return func(r *PossessionResponse) { r.Hazard[4] = v }
}
func withGarbageTime() func(*PossessionResponse) {
	return func(r *PossessionResponse) { r.IsGarbageTime = true }
}
func withBlowout() func(*PossessionResponse) {
	return func(r *PossessionResponse) { r.IsBlowout = true }
}

func newTestBandit() *Bandit {
	cfg := Config{}
	cfg.Agent.MinYesBid = 30
	cfg.Agent.MaxYesBid = 70
	cfg.Agent.MinRunProbEntry = 0.10
	return NewBandit(&cfg)
}

func TestDecideEquivalence(t *testing.T) {
	b := newTestBandit()

	cases := []struct {
		name        string
		resp        *PossessionResponse
		hasPosition bool
	}{
		// ── Garbage time / blowout (highest precedence) ────────────────
		{"garbage_time_no_position", makeResp(withGarbageTime()), false},
		{"garbage_time_has_position", makeResp(withGarbageTime()), true},
		{"blowout_no_position", makeResp(withBlowout()), false},
		{"blowout_has_position", makeResp(withBlowout()), true},
		{"garbage_time_and_blowout", makeResp(withGarbageTime(), withBlowout()), false},
		{"garbage_time_with_high_run_prob",
			makeResp(withGarbageTime(), withRunProb(0.5), withTraj9(0.3)), false},

		// ── Price band ──────────────────────────────────────────────────
		{"below_band", makeResp(withYesBid(29)), false},
		{"above_band", makeResp(withYesBid(71)), false},
		{"at_lower_bound", makeResp(withYesBid(30), withRunProb(0.5), withTraj9(0.1)), false},
		{"at_upper_bound", makeResp(withYesBid(70), withRunProb(0.5), withTraj9(0.1)), false},
		{"out_of_band_with_signal",
			makeResp(withYesBid(80), withRunProb(0.5), withTraj9(0.3)), false},
		{"out_of_band_with_position",
			makeResp(withYesBid(20), withHazard4(0.9)), true},

		// ── Has-position branch (hazard exit) ──────────────────────────
		{"position_high_hazard", makeResp(withHazard4(0.8)), true},
		{"position_low_hazard", makeResp(withHazard4(0.5)), true},
		{"position_at_hazard_threshold", makeResp(withHazard4(0.75)), true}, // > 0.75 fires; ==0.75 does not
		{"position_just_above_threshold", makeResp(withHazard4(0.7501)), true},
		{"position_zero_hazard", makeResp(), true},
		{"position_max_hazard", makeResp(withHazard4(1.0)), true},

		// ── No-position branch (entry) ─────────────────────────────────
		{"low_run_prob_no_traj", makeResp(withRunProb(0.05)), false},
		{"low_run_prob_pos_traj", makeResp(withRunProb(0.05), withTraj9(0.5)), false},
		{"low_run_prob_neg_traj", makeResp(withRunProb(0.05), withTraj9(-0.5)), false},
		{"high_run_prob_pos_traj", makeResp(withRunProb(0.5), withTraj9(0.3)), false},
		{"high_run_prob_neg_traj", makeResp(withRunProb(0.5), withTraj9(-0.3)), false},
		{"high_run_prob_zero_traj", makeResp(withRunProb(0.5), withTraj9(0.0)), false},
		{"at_run_prob_threshold", makeResp(withRunProb(0.10), withTraj9(0.1)), false},
		{"just_below_run_prob_threshold",
			makeResp(withRunProb(0.0999), withTraj9(0.1)), false},

		// ── Combinations / corner cases ────────────────────────────────
		{"strong_buy_yes_signal",
			makeResp(withYesBid(45), withRunProb(0.4), withTraj9(0.5)), false},
		{"strong_buy_no_signal",
			makeResp(withYesBid(55), withRunProb(0.4), withTraj9(-0.5)), false},
		{"all_zero_features", makeResp(withYesBid(50)), false},
		{"all_zero_features_with_position", makeResp(withYesBid(50)), true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			expected := decideLegacy(b, tc.resp, tc.hasPosition)
			actual, gates := b.Decide(tc.resp, tc.hasPosition)
			if actual != expected {
				t.Errorf("ACTION MISMATCH: case=%q hasPos=%v\n  legacy=%v  new=%v\n  resp=%+v",
					tc.name, tc.hasPosition, expected, actual, tc.resp)
			}
			// Sanity: when action is Wait, we must have a FirstBlocking name.
			// When action is anything else, FirstBlocking should be empty.
			if actual == Wait && gates.FirstBlocking == "" {
				t.Errorf("case=%q: action=Wait but FirstBlocking is empty", tc.name)
			}
			if actual != Wait && gates.FirstBlocking != "" {
				t.Errorf("case=%q: action=%v but FirstBlocking=%q (should be empty)",
					tc.name, actual, gates.FirstBlocking)
			}
		})
	}
}
