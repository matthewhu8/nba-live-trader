// Thompson Sampling contextual bandit.
// Sits on top of the MMoE outputs and decides: BUY_YES / BUY_NO / EXIT / WAIT.
//
// The agent is NOT responsible for feature computation or model inference —
// that is the Python service's job. The agent receives already-computed
// MMoE outputs and uses them to make a binary enter/wait/exit decision.
//
// Context buckets: (quarter, score_diff_bucket, run_length_bucket)
// Each (context, arm) pair has Beta distribution parameters (α, β).
// Sample from Beta → choose highest sample → that's the action.
//
// Reward: realized PnL after maker fees, updated by OrderRouter on position close.
// Start simple: fixed threshold rules first, bandit layer on top once validated.
package main

type Action string

const (
	Wait   Action = "WAIT"
	BuyYes Action = "BUY_YES"
	BuyNo  Action = "BUY_NO"
	Exit   Action = "EXIT"
)

type ContextKey struct {
	Quarter         int
	ScoreDiffBucket int // bucketed: ≤5, 6-12, 13-20
	RunLengthBucket int // bucketed: 0, 1-3, 4-6, 7+
}

type BetaParams struct {
	Alpha float64
	Beta  float64
}

type Bandit struct {
	minYesBid         int
	maxYesBid         int
	minAbsTrajEntry   float32 // |traj_final| ≥ this to enter (backtest: 0.08)
	minRunLengthEntry float32 // current_run_length ≥ this to enter (backtest: 2)
	maxHazardForHold  float32 // hazard5 > this triggers exit while holding
	params            map[ContextKey][4]BetaParams // reserved for future bandit
}

func NewBandit(cfg *Config) *Bandit {
	return &Bandit{
		minYesBid:         cfg.Agent.MinYesBid,
		maxYesBid:         cfg.Agent.MaxYesBid,
		minAbsTrajEntry:   cfg.Agent.MinAbsTrajEntry,
		minRunLengthEntry: float32(cfg.Agent.MinRunLengthEntry),
		maxHazardForHold:  0.75,
		params:            make(map[ContextKey][4]BetaParams),
	}
}

// GateResult records the outcome of every gate evaluated during Decide.
// Emitted alongside each possession so post-mortems can answer
// "which gate blocked entry on possession N?" without re-running anything.
//
// Conditional gates use *bool so JSON null distinguishes "not reached" from
// "reached and false":
//   - HazardExitPass:     nil unless HasPosition is true
//   - TrajMagnitudePass:  nil unless HasPosition is false
//   - RunLengthPass:      nil unless HasPosition is false AND TrajMagnitudePass is true
//
// Entry gates were aligned to the validated backtest config on 2026-05-10
// (post-mortem of 2026-05-09 OKC@LAL game) — Head A's gate collapsed in
// production so we route entry decisions through Head B (trajectory) + the
// raw `current_run_length` feature instead of the broken run-prob signal.
type GateResult struct {
	IsGarbageTime     bool    `json:"is_garbage_time"`
	IsBlowout         bool    `json:"is_blowout"`
	InPriceBand       bool    `json:"in_price_band"`
	YesBid            int     `json:"yes_bid"`
	HasPosition       bool    `json:"has_position"`
	TrajMagnitudePass *bool   `json:"traj_magnitude_pass"`
	TrajFinal         float32 `json:"traj_final"`
	RunLengthPass     *bool   `json:"run_length_pass"`
	CurrentRunLength  float32 `json:"current_run_length"`
	TrajectorySign    string  `json:"trajectory_sign"` // "pos" | "neg" | "zero"
	HazardExitPass    *bool   `json:"hazard_exit_pass"`
	FirstBlocking     string  `json:"first_blocking,omitempty"`
}

// Decide returns the recommended action AND a structured record of every gate
// that was evaluated.
//
// Gate ordering:
//  1. is_garbage_time   → Wait
//  2. is_blowout        → Wait
//  3. in_price_band     → Wait if outside 30-70¢
//  4. has_position branch (EXIT path — unchanged):
//       hazard_exit_pass → Exit if hazard5 > maxHazardForHold, else Wait
//  5. no-position branch (ENTRY path — backtest-aligned):
//       traj_magnitude_pass → Wait if |traj_final| < min_abs_traj_entry
//       run_length_pass     → Wait if current_run_length < min_run_length_entry
//       trajectory_sign     → BuyYes if pos, BuyNo if neg
//                             (sign cannot be zero here because magnitude gate
//                              already required |traj_final| ≥ 0.08)
func (b *Bandit) Decide(resp *PossessionResponse, hasPosition bool) (Action, GateResult) {
	trajFinal := resp.Trajectory[9]
	trajSign := "zero"
	if trajFinal > 0 {
		trajSign = "pos"
	} else if trajFinal < 0 {
		trajSign = "neg"
	}

	// current_run_length comes from the Python feature pipeline. Missing
	// from the map → defaults to 0, which fails the run_length gate (safe).
	currentRunLength := resp.Features["current_run_length"]

	inBand := resp.YesBid >= b.minYesBid && resp.YesBid <= b.maxYesBid

	g := GateResult{
		IsGarbageTime:    resp.IsGarbageTime,
		IsBlowout:        resp.IsBlowout,
		InPriceBand:      inBand,
		YesBid:           resp.YesBid,
		HasPosition:      hasPosition,
		TrajFinal:        trajFinal,
		CurrentRunLength: currentRunLength,
		TrajectorySign:   trajSign,
	}

	if resp.IsGarbageTime {
		g.FirstBlocking = "is_garbage_time"
		return Wait, g
	}
	if resp.IsBlowout {
		g.FirstBlocking = "is_blowout"
		return Wait, g
	}
	if !inBand {
		g.FirstBlocking = "in_price_band"
		return Wait, g
	}

	if hasPosition {
		hazardExit := resp.Hazard[4] > b.maxHazardForHold
		g.HazardExitPass = &hazardExit
		if hazardExit {
			return Exit, g
		}
		g.FirstBlocking = "hazard_exit_pass"
		return Wait, g
	}

	// Entry path (backtest-aligned): trajectory magnitude → run length → sign.
	trajMagPass := absF32(trajFinal) >= b.minAbsTrajEntry
	g.TrajMagnitudePass = &trajMagPass
	if !trajMagPass {
		g.FirstBlocking = "traj_magnitude_pass"
		return Wait, g
	}

	runLengthPass := currentRunLength >= b.minRunLengthEntry
	g.RunLengthPass = &runLengthPass
	if !runLengthPass {
		g.FirstBlocking = "run_length_pass"
		return Wait, g
	}

	// Both magnitude and length passed — sign of traj_final picks direction.
	// (Magnitude ≥ 0.08 guarantees non-zero, so we always have a direction.)
	if trajFinal > 0 {
		return BuyYes, g
	}
	return BuyNo, g
}

func absF32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// Update adjusts Beta parameters after a trade closes.
// reward > 0: win (increment alpha), reward <= 0: loss (increment beta).
// No-op until bandit training is enabled.
func (b *Bandit) Update(ctx ContextKey, armIdx int, reward float64) {}
