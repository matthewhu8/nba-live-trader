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

// Decide returns the recommended action given the MMoE output and current game state.
// Checks hard filters first, then uses threshold rules to enter or exit.
func (b *Bandit) Decide(resp *PossessionResponse, hasPosition bool) Action {
	if resp.IsGarbageTime || resp.IsBlowout {
		return Wait
	}
	if resp.YesBid < b.minYesBid || resp.YesBid > b.maxYesBid {
		return Wait
	}

	if hasPosition {
		// Head C: survival hazard at horizon 4 (mid-hold check)
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

// Update adjusts Beta parameters after a trade closes.
// reward > 0: win (increment alpha), reward <= 0: loss (increment beta).
// No-op until bandit training is enabled.
func (b *Bandit) Update(ctx ContextKey, armIdx int, reward float64) {}
