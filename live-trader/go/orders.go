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
	"context"

)

type Router struct {
	apiKey    string
	paperMode bool
}

func NewRouter(apiKey string, paperMode bool) *Router {
	return &Router{apiKey: apiKey, paperMode: paperMode}
}

// Place sends a limit order to Kalshi (or logs it in paper mode).
// resp contains the action and price determined by the inference service + agent.
func (r *Router) Place(ctx context.Context, gameID string, resp *PossessionResponse) {
	if resp.Action == "WAIT" || resp.Action == "EXIT" {
		return
	}

	// TODO: determine side (YES/NO) and price from resp.Action + resp.YesBid
	// TODO: calculate size from Kelly / fixed fraction
	// TODO: calculate expected maker fee
	// TODO: if paper_mode: log order details and return
	// TODO: POST to Kalshi REST API: /trade-api/v2/portfolio/orders
	// TODO: record fill in RiskLedger
}

// makerFee returns the maker fee in cents for a given order.
// fee = 0.0175 × contracts × (price / 100)
func makerFee(contracts, priceCents int) float64 {
	return 0.0175 * float64(contracts) * float64(priceCents) / 100.0
}
