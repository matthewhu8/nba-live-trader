// Bandit — sits on top of the MMoE outputs and decides BUY_YES / BUY_NO / WAIT.
//
// On 2026-05-11 the entry path and exit path were both realigned to exactly
// match the validated backtest config (CLAUDE.md: 112 trades, 41% win rate,
// +$9,977 net on Apr 7 → May 10 val set). The realignment reverses two
// divergences that had been silently hurting live performance:
//
//   1. The run_prob entry gate was removed yesterday (Phase 8). Today's
//      backtest re-confirmed that gate is what makes the strategy profitable —
//      Head A's "rare but real" elevated outputs ARE the trades that win.
//      Restored: `run_prob >= min_run_prob_entry` (default 0.15).
//
//   2. The hazard-based exit was present in live but NOT in the validated
//      backtest. Every backtest exit is TP / SL / momentum_flip / time_gate
//      — no hazard exit anywhere. Live hazard exit was firing on 38% of
//      all possessions (hazard5 > 0.75) and forcing premature exits at
//      a net loss across 3 paper-trade games. Removed: bandit no longer
//      returns Exit. The router's CheckExit (TP/SL/TIME_STOP) is now the
//      sole exit mechanism, exactly like the backtest.
//
// Entry gates (matched to backtest, in order of evaluation):
//   1. is_garbage_time / is_blowout  → Wait
//   2. in_price_band  [30..70]       → Wait if outside
//   3. run_prob ≥ min_run_prob_entry → Wait if below
//   4. |traj_final| ≥ min_abs_traj   → Wait if below
//   5. current_run_length ≥ min     → Wait if below
//   6. trajectory sign               → BuyYes (>0) or BuyNo (<0)
//
// Has-position branch: always Wait. Exit logic is owned by router.CheckExit.
package main

type Action string

const (
	Wait   Action = "WAIT"
	BuyYes Action = "BUY_YES"
	BuyNo  Action = "BUY_NO"
	Exit   Action = "EXIT" // retained for future use; bandit no longer returns this
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
	minRunProbEntry   float32 // Head A gate — restored 2026-05-11
	minAbsTrajEntry   float32 // Head B confidence (backtest: 0.08)
	minRunLengthEntry float32 // momentum filter (backtest: 2)
	params            map[ContextKey][4]BetaParams // reserved for future bandit
}

func NewBandit(cfg *Config) *Bandit {
	return &Bandit{
		minYesBid:         cfg.Agent.MinYesBid,
		maxYesBid:         cfg.Agent.MaxYesBid,
		minRunProbEntry:   cfg.Agent.MinRunProbEntry,
		minAbsTrajEntry:   cfg.Agent.MinAbsTrajEntry,
		minRunLengthEntry: float32(cfg.Agent.MinRunLengthEntry),
		params:            make(map[ContextKey][4]BetaParams),
	}
}

// GateResult records the outcome of every gate evaluated during Decide.
// Emitted alongside each possession JSONL record so post-mortems can answer
// "which gate blocked entry on possession N?" without re-running anything.
//
// Conditional gates use *bool so JSON null distinguishes "not reached" from
// "reached and false":
//   - RunProbPass / TrajMagnitudePass / RunLengthPass — only set in the
//     no-position branch, in evaluation order. First failing gate is named
//     in FirstBlocking.
type GateResult struct {
	IsGarbageTime     bool    `json:"is_garbage_time"`
	IsBlowout         bool    `json:"is_blowout"`
	InPriceBand       bool    `json:"in_price_band"`
	YesBid            int     `json:"yes_bid"`
	HasPosition       bool    `json:"has_position"`
	RunProb           float32 `json:"run_prob"`
	RunProbPass       *bool   `json:"run_prob_pass"`
	TrajFinal         float32 `json:"traj_final"`
	TrajMagnitudePass *bool   `json:"traj_magnitude_pass"`
	CurrentRunLength  float32 `json:"current_run_length"`
	RunLengthPass     *bool   `json:"run_length_pass"`
	TrajectorySign    string  `json:"trajectory_sign"` // "pos" | "neg" | "zero"
	FirstBlocking     string  `json:"first_blocking,omitempty"`
}

func (b *Bandit) Decide(resp *PossessionResponse, hasPosition bool) (Action, GateResult) {
	trajFinal := resp.Trajectory[9]
	trajSign := "zero"
	if trajFinal > 0 {
		trajSign = "pos"
	} else if trajFinal < 0 {
		trajSign = "neg"
	}

	currentRunLength := resp.Features["current_run_length"]
	inBand := resp.YesBid >= b.minYesBid && resp.YesBid <= b.maxYesBid

	g := GateResult{
		IsGarbageTime:    resp.IsGarbageTime,
		IsBlowout:        resp.IsBlowout,
		InPriceBand:      inBand,
		YesBid:           resp.YesBid,
		HasPosition:      hasPosition,
		RunProb:          resp.RunProb,
		TrajFinal:        trajFinal,
		CurrentRunLength: currentRunLength,
		TrajectorySign:   trajSign,
	}

	// Highest-precedence gates apply equally to entry and to held positions.
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

	// Has-position branch: bandit does nothing. The router (TP/SL/TIME_STOP)
	// is solely responsible for exits — this matches the validated backtest.
	if hasPosition {
		g.FirstBlocking = "holding_position"
		return Wait, g
	}

	// Entry path — backtest-aligned gate order:
	//   run_prob → |traj| → run_length → sign
	runProbPass := resp.RunProb >= b.minRunProbEntry
	g.RunProbPass = &runProbPass
	if !runProbPass {
		g.FirstBlocking = "run_prob_pass"
		return Wait, g
	}

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

	// Magnitude ≥ 0.08 guarantees non-zero, so sign always picks a direction.
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

// Update is a no-op stub for the future Thompson Sampling bandit. Kept so
// existing call sites compile; remove when the RL agent ships.
func (b *Bandit) Update(ctx ContextKey, armIdx int, reward float64) {}
