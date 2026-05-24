// Bandit — sits on top of the MMoE outputs and decides BUY_YES / BUY_NO / EXIT / WAIT.
//
// Entry gates (in order of evaluation):
//   1. is_overtime (period >= 5)     → Wait
//   2. is_garbage_time / is_blowout  → Wait
//   3. in_price_band  [30..70]       → Wait if outside
//   4. run_prob ≥ min_run_prob_entry → Wait if below
//   5. |traj_final| ≥ min_abs_traj   → Wait if below
//   6. current_run_length ≥ min      → Wait if below
//   7. trajectory sign               → BuyYes (>0) or BuyNo (<0)
//
// Has-position branch: always returns Wait — Router.CheckExit owns TP / SL / TIME_STOP.
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
	minRunProbEntry   float32 // Head A gate (threshold=0.0 in live config = effectively off)
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
// "reached and false". RunProbPass / TrajMagnitudePass / RunLengthPass are
// only set in the no-position branch.
type GateResult struct {
	IsOvertime        bool    `json:"is_overtime"`
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

	// Overtime detection (period >= 5): the model has zero training rows in
	// the OT regime, and on 2026-05-13 OT triggered scanner thrash (6 market
	// swaps in 6 minutes) and a 50¢ scanner-vs-WebSocket price disagreement.
	// Treat OT as a hard skip — same effect as garbage time, distinct
	// telemetry label so post-game inspection can attribute correctly.
	isOvertime := resp.Features["period"] >= 5

	g := GateResult{
		IsOvertime:       isOvertime,
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
	if isOvertime {
		g.FirstBlocking = "is_overtime"
		return Wait, g
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

	// Has-position branch: Router.CheckExit owns TP / SL / TIME_STOP.
	// Bandit always returns Wait here.
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
