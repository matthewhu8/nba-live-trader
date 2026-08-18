// Bandit reads the MMoE outputs and decides BUY_YES / BUY_NO / WAIT.
//
// Entry gates, in evaluation order and matched to the backtest:
//  1. is_overtime / is_garbage_time / is_blowout
//  2. yes_bid inside [min_yes_bid, max_yes_bid]
//  3. run_prob >= min_run_prob_entry
//  4. |traj_used| >= min_abs_traj_entry
//  5. current_run_length >= min_run_length_entry
//  6. sign(traj_used) picks BuyYes or BuyNo
//
// While a position is open this always returns Wait. Router.CheckExit owns exits.
package main

type Action string

const (
	Wait   Action = "WAIT"
	BuyYes Action = "BUY_YES"
	BuyNo  Action = "BUY_NO"
	Exit   Action = "EXIT" // unused; the bandit never returns this
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
	minRunProbEntry   float32                      // Head A gate; 0.0 in the live config turns it off
	minAbsTrajEntry   float32                      // Head B confidence, applied to traj_used
	minRunLengthEntry float32                      // momentum filter
	trajAggregator    string                       // "final" | "mean" | "mean_3_to_9" | "max_abs"
	params            map[ContextKey][4]BetaParams // reserved for the future bandit
}

func NewBandit(cfg *Config) *Bandit {
	agg := cfg.Agent.TrajAggregator
	if agg == "" {
		agg = "final"
	}
	return &Bandit{
		minYesBid:         cfg.Agent.MinYesBid,
		maxYesBid:         cfg.Agent.MaxYesBid,
		minRunProbEntry:   cfg.Agent.MinRunProbEntry,
		minAbsTrajEntry:   cfg.Agent.MinAbsTrajEntry,
		minRunLengthEntry: float32(cfg.Agent.MinRunLengthEntry),
		trajAggregator:    agg,
		params:            make(map[ContextKey][4]BetaParams),
	}
}

// aggregateTraj reduces the 10-element Head B trajectory to one signed scalar for
// entry gating and sizing. Must match `aggregate_traj` in backtesting/mmoe_backtest.py.
//
//   - "final"       trajectory[9], a single horizon
//   - "mean"        mean of all 10 horizons, the lowest-variance option
//   - "mean_3_to_9" mean of horizons 3-9, skipping the noisy short ones
//   - "max_abs"     the largest element by magnitude, sign preserved
//
// The sign of the result picks the trade direction, so every mode must preserve it.
func aggregateTraj(traj [10]float32, mode string) float32 {
	switch mode {
	case "final":
		return traj[9]
	case "mean":
		var sum float32
		for _, v := range traj {
			sum += v
		}
		return sum / 10.0
	case "mean_3_to_9":
		var sum float32
		for i := 3; i < 10; i++ {
			sum += traj[i]
		}
		return sum / 7.0
	case "max_abs":
		maxIdx := 0
		maxAbs := absF32(traj[0])
		for i := 1; i < 10; i++ {
			if a := absF32(traj[i]); a > maxAbs {
				maxAbs = a
				maxIdx = i
			}
		}
		return traj[maxIdx]
	default:
		return traj[9]
	}
}

// GateResult records the outcome of every gate Decide evaluated. It rides along
// with each possession record so a post-mortem can tell which gate blocked entry.
//
// Conditional gates are *bool so JSON null distinguishes "not reached" from
// "reached and false". Those are only set on the no-position path.
type GateResult struct {
	IsOvertime        bool    `json:"is_overtime"`
	IsGarbageTime     bool    `json:"is_garbage_time"`
	IsBlowout         bool    `json:"is_blowout"`
	InPriceBand       bool    `json:"in_price_band"`
	YesBid            int     `json:"yes_bid"`
	HasPosition       bool    `json:"has_position"`
	RunProb           float32 `json:"run_prob"`
	RunProbPass       *bool   `json:"run_prob_pass"`
	TrajFinal         float32 `json:"traj_final"`      // raw Trajectory[9]
	TrajUsed          float32 `json:"traj_used"`       // aggregated, drives the decision
	Aggregator        string  `json:"traj_aggregator"` // which mode produced TrajUsed
	TrajMagnitudePass *bool   `json:"traj_magnitude_pass"`
	CurrentRunLength  float32 `json:"current_run_length"`
	RunLengthPass     *bool   `json:"run_length_pass"`
	TrajectorySign    string  `json:"trajectory_sign"` // sign of TrajUsed: "pos" | "neg" | "zero"
	FirstBlocking     string  `json:"first_blocking,omitempty"`
}

func (b *Bandit) Decide(resp *PossessionResponse, hasPosition bool) (Action, GateResult) {
	trajFinal := resp.Trajectory[9]
	trajUsed := aggregateTraj(resp.Trajectory, b.trajAggregator)
	trajSign := "zero"
	if trajUsed > 0 {
		trajSign = "pos"
	} else if trajUsed < 0 {
		trajSign = "neg"
	}

	// Gate inputs are named response fields, never Features lookups. Go returns
	// zero for a missing map key with no error, so reading a gate out of the
	// feature map lets a feature rename silently disable it.
	currentRunLength := resp.CurrentRunLength
	inBand := resp.YesBid >= b.minYesBid && resp.YesBid <= b.maxYesBid

	// Overtime is a hard skip: training excludes OT rows, so the model was never
	// fit on the regime. Same effect as garbage time, separate telemetry label.
	isOvertime := resp.IsOvertime

	g := GateResult{
		IsOvertime:       isOvertime,
		IsGarbageTime:    resp.IsGarbageTime,
		IsBlowout:        resp.IsBlowout,
		InPriceBand:      inBand,
		YesBid:           resp.YesBid,
		HasPosition:      hasPosition,
		RunProb:          resp.RunProb,
		TrajFinal:        trajFinal,
		TrajUsed:         trajUsed,
		Aggregator:       b.trajAggregator,
		CurrentRunLength: currentRunLength,
		TrajectorySign:   trajSign,
	}

	// These gates apply equally to entries and to held positions.
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

	// Router.CheckExit owns TP, SL and the time stop, so holding always waits here.
	if hasPosition {
		g.FirstBlocking = "holding_position"
		return Wait, g
	}

	runProbPass := resp.RunProb >= b.minRunProbEntry
	g.RunProbPass = &runProbPass
	if !runProbPass {
		g.FirstBlocking = "run_prob_pass"
		return Wait, g
	}

	trajMagPass := absF32(trajUsed) >= b.minAbsTrajEntry
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

	// Clearing the magnitude gate guarantees a non-zero value, so the sign always
	// picks a side. Head B's sign drives it, independent of which team is on a run.
	if trajUsed > 0 {
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

// Update is a stub for the future Thompson Sampling bandit.
func (b *Bandit) Update(ctx ContextKey, armIdx int, reward float64) {}
