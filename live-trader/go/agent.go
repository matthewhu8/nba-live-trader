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
	minYesBid        int
	maxYesBid        int
	minRunProbEntry  float32
	maxHazardForHold float32
	params           map[ContextKey][4]BetaParams // 4 arms: Wait, BuyYes, BuyNo, Exit — reserved for future bandit
}

func NewBandit(cfg *Config) *Bandit {
	return &Bandit{
		minYesBid:        cfg.Agent.MinYesBid,
		maxYesBid:        cfg.Agent.MaxYesBid,
		minRunProbEntry:  cfg.Agent.MinRunProbEntry,
		maxHazardForHold: 0.75,
		params:           make(map[ContextKey][4]BetaParams),
	}
}

// GateResult records the outcome of every gate evaluated during Decide.
// Emitted alongside each possession so post-mortems can answer
// "which gate blocked entry on possession N?" without re-running anything.
//
// Conditional gates use *bool so JSON null distinguishes "not reached" from
// "reached and false":
//   - HazardExitPass: nil unless HasPosition is true
//   - RunProbPass:    nil unless HasPosition is false
type GateResult struct {
	IsGarbageTime  bool    `json:"is_garbage_time"`
	IsBlowout      bool    `json:"is_blowout"`
	InPriceBand    bool    `json:"in_price_band"`
	YesBid         int     `json:"yes_bid"`
	HasPosition    bool    `json:"has_position"`
	RunProbPass    *bool   `json:"run_prob_pass"`
	TrajectorySign string  `json:"trajectory_sign"` // "pos" | "neg" | "zero"
	HazardExitPass *bool   `json:"hazard_exit_pass"`
	FirstBlocking  string  `json:"first_blocking,omitempty"`
}

// Decide returns the recommended action AND a structured record of every gate
// that was evaluated. Decision logic is unchanged from the pre-refactor
// implementation — see TestDecideEquivalence for the proof.
//
// Gate ordering (must remain identical to legacy Decide):
//  1. is_garbage_time → Wait
//  2. is_blowout      → Wait
//  3. in_price_band   → Wait if outside
//  4. has_position branch:
//       hazard_exit_pass → Exit if true, else Wait
//  5. no-position branch:
//       run_prob_pass    → Wait if false
//       trajectory_sign  → BuyYes if pos, BuyNo if neg, Wait if zero
func (b *Bandit) Decide(resp *PossessionResponse, hasPosition bool) (Action, GateResult) {
	trajFinal := resp.Trajectory[9]
	trajSign := "zero"
	if trajFinal > 0 {
		trajSign = "pos"
	} else if trajFinal < 0 {
		trajSign = "neg"
	}

	inBand := resp.YesBid >= b.minYesBid && resp.YesBid <= b.maxYesBid

	g := GateResult{
		IsGarbageTime:  resp.IsGarbageTime,
		IsBlowout:      resp.IsBlowout,
		InPriceBand:    inBand,
		YesBid:         resp.YesBid,
		HasPosition:    hasPosition,
		TrajectorySign: trajSign,
	}

	// Match legacy short-circuit ordering exactly: garbage_time before blowout.
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

	runProbPass := resp.RunProb >= b.minRunProbEntry
	g.RunProbPass = &runProbPass
	if !runProbPass {
		g.FirstBlocking = "run_prob_pass"
		return Wait, g
	}

	if trajFinal > 0 {
		return BuyYes, g
	}
	if trajFinal < 0 {
		return BuyNo, g
	}
	g.FirstBlocking = "trajectory_sign"
	return Wait, g
}

// Update adjusts Beta parameters after a trade closes.
// reward > 0: win (increment alpha), reward <= 0: loss (increment beta).
// No-op until bandit training is enabled.
func (b *Bandit) Update(ctx ContextKey, armIdx int, reward float64) {}
