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

// Hard entry filters — checked before bandit fires.
// If any fail, return WAIT immediately (no bandit sampling needed).
const (
	minYesBid       = 30 // cents — below this, certainty too high for our signal
	maxYesBid       = 70 // cents
	minRunProbEntry = 0.15 // ~2x base rate of 7.6%
)

type ContextKey struct {
	Quarter       int
	ScoreDiffBucket int // bucketed: ≤5, 6-12, 13-20
	RunLengthBucket int // bucketed: 0, 1-3, 4-6, 7+
}

type BetaParams struct {
	Alpha float64
	Beta  float64
}

type Bandit struct {
	params map[ContextKey][4]BetaParams // 4 arms: Wait, BuyYes, BuyNo, Exit
}

func NewBandit() *Bandit {
	return &Bandit{params: make(map[ContextKey][4]BetaParams)}
}

// Decide returns the recommended action given the MMoE output and current game state.
func (b *Bandit) Decide(resp *PossessionResponse, hasOpenPosition bool) Action {
	// Hard filters first
	if resp.IsGarbageTime || resp.IsBlowout {
		return Wait
	}
	if resp.YesBid < minYesBid || resp.YesBid > maxYesBid {
		return Wait
	}
	if resp.RunProb < minRunProbEntry && !hasOpenPosition {
		return Wait
	}

	// TODO: extract context key from resp
	// TODO: sample from Beta distributions for each arm
	// TODO: return arm with highest sample
	return Wait
}

// Update adjusts Beta parameters after a trade closes.
// reward > 0: win (increment alpha), reward <= 0: loss (increment beta)
func (b *Bandit) Update(ctx ContextKey, armIdx int, reward float64) {
	// TODO
}
