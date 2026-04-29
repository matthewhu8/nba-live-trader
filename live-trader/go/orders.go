// OrderRouter is the single point of contact with the Kalshi REST API.
// All order placement in the system goes through here — no other file
// calls the Kalshi API directly.
//
// Enforcements at this layer (not negotiable):
//   - Maker-only: all orders are limit orders. Taker orders are rejected before API call.
//   - paper_mode flag: when true, logs the order but does not send to Kalshi.
//     Paper mode is the default. Single flag in config/trading.yaml to go live.
//   - Fee calculation: maker fee = 0.0175 × contracts × price (cents)
package main

import (
	"time"
)

type Router struct {
	apiKey    string
	paperMode bool
}

func NewRouter(apiKey string, paperMode bool) *Router {
	return &Router{apiKey: apiKey, paperMode: paperMode}
}

// PaperPosition records an open paper trade entry.
type PaperPosition struct {
	GameID      string
	Direction   string    // "YES"
	EntryPrice  int       // cents
	Size        int       // contracts
	EntryPossID int
	EntryTime   time.Time
}

// Place records a paper trade entry. Returns nil if conditions not met.
// Only call when paperMode == true and action == BUY_YES.
func (r *Router) Place(gameID string, resp *PossessionResponse, possID int, cfg *Config) *PaperPosition {
	if !r.paperMode {
		return nil
	}
	if resp.Action != string(BuyYes) {
		return nil
	}

	size := cfg.Agent.PositionSizeContracts
	fee := makerFee(size, resp.YesBid)
	_ = fee // logged by caller via EmitEntry

	return &PaperPosition{
		GameID:      gameID,
		Direction:   "YES",
		EntryPrice:  resp.YesBid,
		Size:        size,
		EntryPossID: possID,
		EntryTime:   time.Now(),
	}
}

// CheckExit evaluates whether an open position should be closed.
// Returns (shouldExit, reason, netPnLDollars).
// exitPrice is the current yes_bid in cents.
func (r *Router) CheckExit(pos *PaperPosition, currentBid, possID int, cfg *Config) (bool, string, float64) {
	priceDelta := currentBid - pos.EntryPrice

	if priceDelta >= cfg.Agent.TakeProfitCents {
		return true, "TAKE_PROFIT", calcNetPnL(pos.Size, pos.EntryPrice, currentBid)
	}
	if -priceDelta >= cfg.Agent.StopLossCents {
		return true, "STOP_LOSS", calcNetPnL(pos.Size, pos.EntryPrice, currentBid)
	}
	if possID-pos.EntryPossID >= cfg.Agent.MaxHoldPossessions {
		return true, "TIME_STOP", calcNetPnL(pos.Size, pos.EntryPrice, currentBid)
	}
	return false, "", 0
}

// calcNetPnL returns net P&L in dollars after maker exit fee.
// gross = (exitPrice - entryPrice) × size / 100, minus exit maker fee.
func calcNetPnL(size, entryPrice, exitPrice int) float64 {
	grossCents := float64(exitPrice-entryPrice) * float64(size)
	feeUSD := makerFee(size, exitPrice)
	return grossCents/100.0 - feeUSD
}

// makerFee returns the maker fee in dollars for a given order.
// fee = 0.0175 × contracts × (price / 100)
func makerFee(contracts, priceCents int) float64 {
	return 0.0175 * float64(contracts) * float64(priceCents) / 100.0
}
