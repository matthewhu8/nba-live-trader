// OrderRouter is the single point of contact with the Kalshi REST API.
// All order placement in the system goes through here — no other file
// calls the Kalshi API directly.
//
// Order policy at this layer:
//   - Maker ENTRIES: entry orders are limit orders with post_only=true. This is
//     where the edge lives — cheap maker fills ahead of slow retail repricing.
//   - Maker TAKE-PROFITS: when an entry fills, a resting opposite-side SELL is
//     placed immediately at `entry + TP` (post_only=true). The model says price
//     will move favorably; we let the limit sit until the book reaches it.
//     PollOrderStatus is consulted each possession to detect the fill. This
//     captures the maker discount on winners — 0.0175 × price/contract vs.
//     the 0.07 taker fee we used to pay on crossing TPs (~4× cheaper).
//   - Crossing STOP-LOSSES: stop-loss exits are marketable LIMIT orders
//     (post_only=false) priced a bounded number of cents through the best bid.
//     A post-only stop cannot fill when the market is running away from a losing
//     position (2026-05-28 OKC@SAS post-mortem: 175 rejected "post only cross"
//     exits, a 3¢ stop realized as -12¢). Stops are still LIMIT orders (never
//     market); slippage bounded by ExitSlippageBudgetCents. Stops pay taker.
//   - When an SL fires while a resting TP is open, the SL path FIRST cancels
//     the TP via DELETE, then places the crossing exit. The cancel response is
//     authoritative: if Kalshi says the TP was already filled, the position is
//     already closed at TP and no SL is needed.
//   - paper_mode flag: when true, logs the order but does not send to Kalshi.
//     Paper mode is the default. Single flag in config/trading.yaml to go live.
//   - Fee accounting: calcNetPnL returns gross P&L with no fee deduction.
package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type Router struct {
	apiKey    string
	paperMode bool
}

func NewRouter(apiKey string, paperMode bool) *Router {
	return &Router{apiKey: apiKey, paperMode: paperMode}
}

// PaperPosition records an open trade entry (paper or live).
type PaperPosition struct {
	GameID      string
	EntryTicker string    // Kalshi market ticker at entry — used for exits (active market may swap mid-game)
	Direction   string    // "YES" or "NO"
	EntryPrice  int       // cents
	Size        int       // contracts
	EntryPossID int
	EntryTime   time.Time
	OrderID     string    // Kalshi order_id (empty in paper mode)
	PeakPrice    int      // highest marketable price seen since entry — drives the trailing take-profit
	ExitAttempts int      // consecutive failed exit attempts — each one widens the crossing budget so we still get out

	// Resting maker take-profit. Placed immediately after a successful entry as
	// an opposite-side post_only=true sell at `EntryPrice + TakeProfitCents`.
	// Polled via PollRestingTPStatus each possession; when filled, the position
	// is closed at RestingTPPrice. When an SL fires, the resting TP must be
	// canceled first; if Kalshi reports it already executed, the SL is skipped
	// and the position is treated as TP-closed.
	RestingTPOrderID string // empty when no resting TP exists (paper mode, placement failed, or already cleared)
	RestingTPPrice   int    // target sell price in cents (entry + TP, clamped to [1,99])
	RestingTPStatus  string // "" (none) | "open" | "executed" | "canceled" | "rejected"
}

// kalshiOrderRequest is the JSON body for POST /portfolio/orders.
type kalshiOrderRequest struct {
	Ticker        string `json:"ticker"`
	Action        string `json:"action"`                   // "buy" or "sell"
	Side          string `json:"side"`                     // "yes" or "no"
	Type          string `json:"type"`                     // always "limit"
	Count         int    `json:"count"`
	YesPrice      int    `json:"yes_price,omitempty"`
	NoPrice       int    `json:"no_price,omitempty"`
	ClientOrderID string `json:"client_order_id"`
	PostOnly      bool   `json:"post_only"`                // enforces maker-only
}

type kalshiOrderResponse struct {
	Order struct {
		OrderID     string `json:"order_id"`
		Status      string `json:"status"`
		FilledCount int    `json:"filled_count"`
		YesPrice    int    `json:"yes_price"`
		NoPrice     int    `json:"no_price"`
	} `json:"order"`
}

// placeKalshiOrder sends a single limit order to the Kalshi REST API.
// Reuses the same HTTP + auth pattern as market_scanner.go:scanWithCurrent.
func placeKalshiOrder(ctx context.Context, req kalshiOrderRequest) (*kalshiOrderResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal order: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST",
		kalshiRESTURL("/portfolio/orders"), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}

	headers, err := GetKalshiAuthHeaders("POST", "/portfolio/orders")
	if err != nil {
		return nil, fmt.Errorf("auth headers: %w", err)
	}
	for k, v := range headers {
		httpReq.Header[k] = v
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("kalshi orders API %d: %s", resp.StatusCode, b)
	}

	var out kalshiOrderResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return &out, nil
}

// newUUID returns a random 32-hex-char string for use as client_order_id.
func newUUID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// Place records a paper trade or sends a live limit order to Kalshi.
// Returns nil only on live-mode API failure (logged); callers treat nil as a missed signal.
func (r *Router) Place(
	ctx context.Context,
	ticker string,
	gameID string,
	resp *PossessionResponse,
	possID int,
	size int,
	direction string,
) *PaperPosition {
	side := "yes"
	entryPrice := resp.YesBid
	if direction == "NO" {
		side = "no"
		entryPrice = 100 - resp.YesAsk
		if resp.YesAsk == 0 {
			entryPrice = 100 - resp.YesBid
		}
	}

	if r.paperMode {
		return &PaperPosition{
			GameID:      gameID,
			Direction:   direction,
			EntryPrice:  entryPrice,
			Size:        size,
			EntryPossID: possID,
			EntryTime:   time.Now(),
		}
	}

	// Live path: POST limit buy to Kalshi with post_only to enforce maker-only.
	req := kalshiOrderRequest{
		Ticker:        ticker,
		Action:        "buy",
		Side:          side,
		Type:          "limit",
		Count:         size,
		ClientOrderID: newUUID(),
		PostOnly:      true,
	}
	if side == "yes" {
		req.YesPrice = entryPrice
	} else {
		req.NoPrice = entryPrice
	}

	apiResp, err := placeKalshiOrder(ctx, req)
	if err != nil {
		zlog.Error().Err(err).
			Str("ticker", ticker).
			Str("direction", direction).
			Int("price", entryPrice).
			Int("size", size).
			Msg("live entry order failed")
		return nil
	}

	return &PaperPosition{
		GameID:      gameID,
		Direction:   direction,
		EntryPrice:  entryPrice,
		Size:        size,
		EntryPossID: possID,
		EntryTime:   time.Now(),
		OrderID:     apiResp.Order.OrderID,
	}
}

// exitLimitPrice returns the limit price for a crossing exit sell: the current
// best price for our side minus a slippage budget, clamped to Kalshi's [1,99]
// range. A sell limit at-or-below the best bid is marketable and fills at the
// best available price; the budget bounds the worst-case fill when the book is
// thin or moving. Note: a tighter limit (smaller budget) does NOT mean a worse
// fill — limit orders fill at the best available price, so the budget only sets
// how far we are willing to chase, not where we actually trade.
func exitLimitPrice(marketablePrice, slippageBudget int) int {
	limit := marketablePrice - slippageBudget
	if limit < 1 {
		limit = 1
	}
	if limit > 99 {
		limit = 99
	}
	return limit
}

// PlaceExit sends a live limit sell order to close an open position. The order
// CROSSES the book (post_only=false) so it actually fills when the market is
// moving against us — a post-only exit cannot. marketablePrice is the current
// best price for our side; slippageBudget caps how far through the bid we cross.
// Always uses pos.EntryTicker — the market where the position was opened — not
// the current active market, which may have swapped since entry.
// Returns true if the order was sent (paper always returns true).
func (r *Router) PlaceExit(ctx context.Context, pos *PaperPosition, marketablePrice, slippageBudget int) bool {
	if r.paperMode {
		return true
	}

	side := "yes"
	if pos.Direction == "NO" {
		side = "no"
	}

	limit := exitLimitPrice(marketablePrice, slippageBudget)
	req := kalshiOrderRequest{
		Ticker:        pos.EntryTicker,
		Action:        "sell",
		Side:          side,
		Type:          "limit",
		Count:         pos.Size,
		ClientOrderID: newUUID(),
		PostOnly:      false, // exits cross the book; a maker stop-loss is un-fillable when price runs away
	}
	if side == "yes" {
		req.YesPrice = limit
	} else {
		req.NoPrice = limit
	}

	if _, err := placeKalshiOrder(ctx, req); err != nil {
		zlog.Error().Err(err).
			Str("ticker", pos.EntryTicker).
			Str("entry_order_id", pos.OrderID).
			Str("direction", pos.Direction).
			Int("exit_limit", limit).
			Int("size", pos.Size).
			Msg("live exit order failed")
		return false
	}
	return true
}

// CheckExit evaluates whether an open position should be closed.
// Returns (shouldExit, reason, netPnLDollars).
//
// TP handling is asymmetric vs. SL by design:
//   - TP is owned by a resting maker order placed at entry. CheckExit consults
//     its Kalshi status each possession; if executed, the position is closed at
//     RestingTPPrice and the caller cancels nothing (the order self-cleared).
//     If the resting TP failed to place earlier, CheckExit retries placement
//     idempotently before checking status.
//   - SL is still a local price-trigger that returns "STOP_LOSS"; the caller
//     in game.go must cancel the resting TP before placing the crossing SL.
//   - Trailing TP and time stop are unchanged and still go through the
//     crossing PlaceExit path; converting those to maker is a future change.
func (r *Router) CheckExit(ctx context.Context, pos *PaperPosition, resp *PossessionResponse, possID int, cfg *Config) (bool, string, float64) {
	// Resting maker TP: place if missing, poll if open. Errors are logged
	// inside the helpers and never block the SL path below.
	if pos.RestingTPStatus == "" || pos.RestingTPStatus == "rejected" {
		_ = r.PlaceRestingTP(ctx, pos, cfg.Agent.TakeProfitCents)
	} else if pos.RestingTPStatus == "open" {
		if _, err := r.PollRestingTPStatus(ctx, pos); err != nil {
			zlog.Warn().Err(err).
				Str("order_id", pos.RestingTPOrderID).
				Msg("resting TP status poll failed — will retry next possession")
		}
	}
	if pos.RestingTPStatus == "executed" {
		return true, "TAKE_PROFIT", calcNetPnL(pos.Size, pos.EntryPrice, pos.RestingTPPrice)
	}

	currentPrice := resp.YesBid
	if pos.Direction == "NO" {
		currentPrice = 100 - resp.YesAsk
		if resp.YesAsk == 0 {
			currentPrice = 100 - resp.YesBid
		}
	}

	if currentPrice > pos.PeakPrice {
		pos.PeakPrice = currentPrice
	}

	priceDelta := currentPrice - pos.EntryPrice

	// Paper-mode resting-TP simulator. Paper never reaches Kalshi, so
	// PollRestingTPStatus can't observe the fill. When the marketable price
	// crosses the TP target, fake the fill at the target price (not at
	// currentPrice — the resting order would have filled at the limit price,
	// not at the through price). Disabled when trailing is on so trailing logic
	// can take over for a strong run. No-op in live mode.
	if r.paperMode && cfg.Agent.TrailGivebackCents <= 0 && priceDelta >= cfg.Agent.TakeProfitCents {
		pos.RestingTPStatus = "executed"
		if pos.RestingTPPrice == 0 {
			pos.RestingTPPrice = restingTPPrice(pos.EntryPrice, cfg.Agent.TakeProfitCents)
		}
		return true, "TAKE_PROFIT", calcNetPnL(pos.Size, pos.EntryPrice, pos.RestingTPPrice)
	}

	// Hard stop-loss — taker by design; loss-cutting cannot wait for maker fills.
	if -priceDelta >= cfg.Agent.StopLossCents {
		return true, "STOP_LOSS", calcNetPnL(pos.Size, pos.EntryPrice, currentPrice)
	}

	// Trailing take-profit: once the position has been at least TrailActivateCents
	// in profit, exit when it retraces TrailGivebackCents from its peak. Lets a
	// strong move run (e.g. the +15¢ excursion on 2026-05-28 that the broken
	// maker exit gave back to +3¢) while still locking in most of the gain.
	if cfg.Agent.TrailGivebackCents > 0 {
		peakDelta := pos.PeakPrice - pos.EntryPrice
		if peakDelta >= cfg.Agent.TrailActivateCents && (pos.PeakPrice-currentPrice) >= cfg.Agent.TrailGivebackCents {
			return true, "TRAIL_STOP", calcNetPnL(pos.Size, pos.EntryPrice, currentPrice)
		}
	}

	if possID-pos.EntryPossID >= cfg.Agent.MaxHoldPossessions {
		return true, "TIME_STOP", calcNetPnL(pos.Size, pos.EntryPrice, currentPrice)
	}
	return false, "", 0
}

// calcNetPnL returns gross P&L in dollars.
// Kalshi maker fees are $0 on standard markets, so gross = net.
func calcNetPnL(size, entryPrice, exitPrice int) float64 {
	return float64(exitPrice-entryPrice) * float64(size) / 100.0
}

// kellyContracts scales position size linearly with conviction above the anchor.
//
//	contracts = clamp(KellyMinContracts, (|traj_used| - anchor) × slope, maxContracts)
//
// The anchor MUST be set to the entry threshold (or just below it) so a trade
// that barely clears the gate gets KellyMinContracts, not an inflated count.
//
// Phase 1 winner (aggregator=mean, threshold=0.08) doubled config (2026-06-03):
//
//	anchor=0.08, slope=200, KellyMinContracts=10, maxContracts=80
//	|traj_used|=0.08 → 10  (floor)
//	|traj_used|=0.12 → 10  (floor, since (0.12-0.08)*200 = 8 < 10)
//	|traj_used|=0.15 → 14
//	|traj_used|=0.20 → 24
//	|traj_used|=0.30 → 44
//	|traj_used|=0.48 → 80  (capped)
//
// Old single-sized config used anchor=0.10, slope=100, min=5 — slope doubled to
// double per-conviction sizing; anchor moved to entry threshold to match the
// aggregator's empirical floor.
func kellyContracts(trajUsed float32, maxContracts int, anchor, slope float32, minContracts int) int {
	abs := trajUsed
	if abs < 0 {
		abs = -abs
	}
	n := int((abs - anchor) * slope)
	if n < minContracts {
		n = minContracts
	}
	if n > maxContracts {
		n = maxContracts
	}
	return n
}

// ─── Resting maker take-profit ─────────────────────────────────────────────
//
// kalshiGetOrderResponse is the shape returned by GET /portfolio/orders/{id}.
// Only fields we consult are decoded; the rest are ignored.
type kalshiGetOrderResponse struct {
	Order struct {
		OrderID         string `json:"order_id"`
		Status          string `json:"status"`           // "resting" | "executed" | "canceled" (Kalshi spec)
		FilledCount     int    `json:"filled_count"`
		RemainingCount  int    `json:"remaining_count"`
		YesPrice        int    `json:"yes_price"`
		NoPrice         int    `json:"no_price"`
	} `json:"order"`
}

// restingTPPrice returns the take-profit limit price for a given entry. Both
// directions sell back AT the position-side bid; "favorable" is always
// EntryPrice + TP regardless of side. Clamped to Kalshi's [1, 99] range so a
// theoretical TP at 100+ stays representable.
func restingTPPrice(entryPrice, tpCents int) int {
	target := entryPrice + tpCents
	if target < 1 {
		target = 1
	}
	if target > 99 {
		target = 99
	}
	return target
}

// PlaceRestingTP submits a post-only opposite-side limit at entry+TP. Idempotent
// on PaperPosition: returns immediately if a resting TP is already on the books
// for this position. Paper mode just records the target price.
//
// Returns nil on success. Non-fatal: any error is logged and pos.RestingTPStatus
// is set to "rejected" so callers can fall back to the crossing-exit path.
func (r *Router) PlaceRestingTP(ctx context.Context, pos *PaperPosition, tpCents int) error {
	if pos.RestingTPOrderID != "" {
		return nil // already placed
	}
	target := restingTPPrice(pos.EntryPrice, tpCents)
	pos.RestingTPPrice = target

	if r.paperMode {
		pos.RestingTPOrderID = "PAPER-" + newUUID()[:8]
		pos.RestingTPStatus = "open"
		return nil
	}

	side := "yes"
	if pos.Direction == "NO" {
		side = "no"
	}
	req := kalshiOrderRequest{
		Ticker:        pos.EntryTicker,
		Action:        "sell",
		Side:          side,
		Type:          "limit",
		Count:         pos.Size,
		ClientOrderID: newUUID(),
		PostOnly:      true, // maker — this is the whole point
	}
	if side == "yes" {
		req.YesPrice = target
	} else {
		req.NoPrice = target
	}

	apiResp, err := placeKalshiOrder(ctx, req)
	if err != nil {
		zlog.Warn().Err(err).
			Str("ticker", pos.EntryTicker).
			Str("entry_order_id", pos.OrderID).
			Int("tp_price", target).
			Msg("resting TP placement failed — will fall back to crossing exit on TP trigger")
		pos.RestingTPStatus = "rejected"
		return err
	}
	pos.RestingTPOrderID = apiResp.Order.OrderID
	pos.RestingTPStatus = "open"
	return nil
}

// PollRestingTPStatus updates pos.RestingTPStatus from Kalshi. Cheap to call
// every possession — single GET, no body. Returns the latest status string;
// any non-"open" terminal status indicates the position is closed.
//
// Paper mode is a no-op (status stays "open" forever); the simulator never
// fills resting TPs locally.
func (r *Router) PollRestingTPStatus(ctx context.Context, pos *PaperPosition) (string, error) {
	if pos.RestingTPOrderID == "" || pos.RestingTPStatus != "open" {
		return pos.RestingTPStatus, nil
	}
	if r.paperMode {
		return pos.RestingTPStatus, nil
	}

	endpoint := "/portfolio/orders/" + pos.RestingTPOrderID
	httpReq, err := http.NewRequestWithContext(ctx, "GET", kalshiRESTURL(endpoint), nil)
	if err != nil {
		return pos.RestingTPStatus, fmt.Errorf("build request: %w", err)
	}
	headers, err := GetKalshiAuthHeaders("GET", endpoint)
	if err != nil {
		return pos.RestingTPStatus, fmt.Errorf("auth headers: %w", err)
	}
	for k, v := range headers {
		httpReq.Header[k] = v
	}

	httpResp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return pos.RestingTPStatus, fmt.Errorf("http: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(httpResp.Body)
		return pos.RestingTPStatus, fmt.Errorf("get order %d: %s", httpResp.StatusCode, b)
	}
	var out kalshiGetOrderResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&out); err != nil {
		return pos.RestingTPStatus, fmt.Errorf("decode: %w", err)
	}

	switch out.Order.Status {
	case "executed", "filled":
		pos.RestingTPStatus = "executed"
	case "canceled", "cancelled":
		pos.RestingTPStatus = "canceled"
	case "resting", "pending", "open":
		// still on the book — leave status as "open"
	default:
		// Unknown status — log so we learn the API's full vocabulary, but don't change state.
		zlog.Warn().
			Str("status", out.Order.Status).
			Str("order_id", pos.RestingTPOrderID).
			Msg("unknown Kalshi order status — treating as still open")
	}
	return pos.RestingTPStatus, nil
}

// CancelRestingTP cancels the resting TP order via DELETE. Returns the
// post-cancel status. If Kalshi reports the order already executed, the
// position is already closed at TP and the caller should NOT place an SL —
// check the returned status before placing any follow-up exit.
//
// Paper mode just flips the local state to "canceled".
func (r *Router) CancelRestingTP(ctx context.Context, pos *PaperPosition) (string, error) {
	if pos.RestingTPOrderID == "" || pos.RestingTPStatus != "open" {
		return pos.RestingTPStatus, nil
	}
	if r.paperMode {
		pos.RestingTPStatus = "canceled"
		return pos.RestingTPStatus, nil
	}

	endpoint := "/portfolio/orders/" + pos.RestingTPOrderID
	httpReq, err := http.NewRequestWithContext(ctx, "DELETE", kalshiRESTURL(endpoint), nil)
	if err != nil {
		return pos.RestingTPStatus, fmt.Errorf("build request: %w", err)
	}
	headers, err := GetKalshiAuthHeaders("DELETE", endpoint)
	if err != nil {
		return pos.RestingTPStatus, fmt.Errorf("auth headers: %w", err)
	}
	for k, v := range headers {
		httpReq.Header[k] = v
	}

	httpResp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return pos.RestingTPStatus, fmt.Errorf("http: %w", err)
	}
	defer httpResp.Body.Close()

	// Kalshi returns the order body on cancel; the status tells us whether the
	// cancel succeeded or the order had already filled (race).
	if httpResp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(httpResp.Body)
		// 404 → order doesn't exist (already canceled or gone). Treat as canceled.
		if httpResp.StatusCode == http.StatusNotFound {
			pos.RestingTPStatus = "canceled"
			return pos.RestingTPStatus, nil
		}
		return pos.RestingTPStatus, fmt.Errorf("cancel order %d: %s", httpResp.StatusCode, b)
	}
	var out kalshiGetOrderResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&out); err != nil {
		// We got 200 but couldn't decode — assume cancel worked.
		pos.RestingTPStatus = "canceled"
		return pos.RestingTPStatus, nil
	}
	switch out.Order.Status {
	case "executed", "filled":
		// Race: order filled between our local exit decision and our cancel.
		// Position is closed at TP price; caller must NOT place a follow-up SL.
		pos.RestingTPStatus = "executed"
	default:
		pos.RestingTPStatus = "canceled"
	}
	return pos.RestingTPStatus, nil
}
