// Router is the only place in the system that calls the Kalshi order API.
//
// Order policy:
//   - Entries are post_only limit buys. Maker fills are where the edge lives.
//   - Take-profits rest as post_only sells at entry+TP and are polled for fills.
//   - Stop-losses cross the book. A post-only stop cannot fill when price is
//     running away from a losing position.
//   - paper_mode logs the order without sending it.
package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
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
	GameID       string
	EntryTicker  string // market at entry; exits use this, not the active market
	Direction    string // "YES" or "NO"
	EntryPrice   int    // cents
	Size         int    // contracts
	EntryPossID  int
	EntryTime    time.Time
	OrderID      string // Kalshi order_id (empty in paper mode)
	PeakPrice    int    // highest marketable price since entry, drives the trailing take-profit
	ExitAttempts int    // failed exit attempts; each one widens the crossing budget

	// Resting maker take-profit, placed right after the entry fills.
	RestingTPOrderID string // empty when no resting TP exists
	RestingTPPrice   int    // entry + TP, clamped to [1,99]
	RestingTPStatus  string // "" | "open" | "executed" | "canceled" | "rejected"
}

// kalshiOrderRequest is the JSON body for POST /portfolio/orders.
type kalshiOrderRequest struct {
	Ticker        string `json:"ticker"`
	Action        string `json:"action"` // "buy" or "sell"
	Side          string `json:"side"`   // "yes" or "no"
	Type          string `json:"type"`   // always "limit"
	Count         int    `json:"count"`
	YesPrice      int    `json:"yes_price,omitempty"`
	NoPrice       int    `json:"no_price,omitempty"`
	ClientOrderID string `json:"client_order_id"`
	PostOnly      bool   `json:"post_only"` // maker-only
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

// Place records a paper trade or sends a live limit buy to Kalshi.
// Returns nil on live-mode API failure; callers treat nil as a missed signal.
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

// exitLimitPrice returns the limit for a crossing exit sell: the best price for
// our side minus a slippage budget, clamped to Kalshi's [1,99] range. The budget
// sets how far we are willing to chase, not where we actually fill.
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

// PlaceExit sends a limit sell that crosses the book to close an open position.
// It always sells on pos.EntryTicker, not the active market, which may have
// swapped since entry. Returns true if the order was sent; paper always returns true.
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
		PostOnly:      false, // a maker stop-loss is un-fillable when price runs away
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
// TP is owned by the resting maker order, so this only reads its Kalshi status.
// SL, trail and time stop are local price triggers whose exits cross the book;
// the caller must cancel the resting TP before placing any of those.
func (r *Router) CheckExit(ctx context.Context, pos *PaperPosition, resp *PossessionResponse, possID int, cfg *Config) (bool, string, float64) {
	// Place the resting TP if missing, poll it if open. Errors never block the SL path.
	if pos.RestingTPStatus == "" || pos.RestingTPStatus == "rejected" {
		_ = r.PlaceRestingTP(ctx, pos, cfg.Agent.TakeProfitCents)
	} else if pos.RestingTPStatus == "open" {
		if _, err := r.PollRestingTPStatus(ctx, pos); err != nil {
			zlog.Warn().Err(err).
				Str("order_id", pos.RestingTPOrderID).
				Msg("resting TP status poll failed, retrying next possession")
		}
	}
	if pos.RestingTPStatus == "executed" {
		return true, "TAKE_PROFIT", calcNetPnL(pos.Size, pos.EntryPrice, pos.RestingTPPrice, makerFeeRate)
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

	// Paper never reaches Kalshi, so simulate the resting TP fill here. The fill is
	// at the target price, not the through price, because the order was a limit.
	if r.paperMode && cfg.Agent.TrailGivebackCents <= 0 && priceDelta >= cfg.Agent.TakeProfitCents {
		pos.RestingTPStatus = "executed"
		if pos.RestingTPPrice == 0 {
			pos.RestingTPPrice = restingTPPrice(pos.EntryPrice, cfg.Agent.TakeProfitCents)
		}
		return true, "TAKE_PROFIT", calcNetPnL(pos.Size, pos.EntryPrice, pos.RestingTPPrice, makerFeeRate)
	}

	// Stop-loss is taker by design. Cutting a loss cannot wait for a maker fill.
	if -priceDelta >= cfg.Agent.StopLossCents {
		return true, "STOP_LOSS", calcNetPnL(pos.Size, pos.EntryPrice, currentPrice, takerFeeRate)
	}

	// Trailing take-profit: once up TrailActivateCents, exit on a TrailGivebackCents
	// retrace from the peak. Lets a strong move run past the flat TP.
	if cfg.Agent.TrailGivebackCents > 0 {
		peakDelta := pos.PeakPrice - pos.EntryPrice
		if peakDelta >= cfg.Agent.TrailActivateCents && (pos.PeakPrice-currentPrice) >= cfg.Agent.TrailGivebackCents {
			return true, "TRAIL_STOP", calcNetPnL(pos.Size, pos.EntryPrice, currentPrice, takerFeeRate)
		}
	}

	if possID-pos.EntryPossID >= cfg.Agent.MaxHoldPossessions {
		return true, "TIME_STOP", calcNetPnL(pos.Size, pos.EntryPrice, currentPrice, takerFeeRate)
	}
	return false, "", 0
}

// Kalshi fee rates. Maker is a quarter of taker, which is why the strategy is maker-first.
const (
	makerFeeRate = 0.0175
	takerFeeRate = 0.07
)

// kalshiFee returns the fee in DOLLARS for one leg of a trade:
//
//	fee = ceil(rate * contracts * P * (1-P) * 100) / 100      where P = price/100
//
// The P*(1-P) term is required. Fees peak at 50c and fall toward both ends of the book.
func kalshiFee(contracts, priceCents int, rate float64) float64 {
	p := float64(priceCents) / 100.0
	return math.Ceil(rate*float64(contracts)*p*(1.0-p)*100.0) / 100.0
}

// calcNetPnL returns P&L in dollars net of both legs' fees. The entry leg is always
// maker; exitRate is maker for a resting TP and taker for any crossing exit.
func calcNetPnL(size, entryPrice, exitPrice int, exitRate float64) float64 {
	gross := float64(exitPrice-entryPrice) * float64(size) / 100.0
	fees := kalshiFee(size, entryPrice, makerFeeRate) + kalshiFee(size, exitPrice, exitRate)
	return gross - fees
}

// calcGrossPnL returns P&L in dollars before fees, for telemetry that separates
// the market move from its cost.
func calcGrossPnL(size, entryPrice, exitPrice int) float64 {
	return float64(exitPrice-entryPrice) * float64(size) / 100.0
}

// kellyContracts scales position size linearly with conviction above the anchor:
//
//	contracts = clamp(minContracts, (|traj_used| - anchor) * slope, maxContracts)
//
// Set anchor at (or just below) the entry threshold so a trade that barely clears
// the gate sizes at the floor rather than an inflated count.
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

// kalshiGetOrderResponse is the shape returned by GET /portfolio/orders/{id}.
type kalshiGetOrderResponse struct {
	Order struct {
		OrderID        string `json:"order_id"`
		Status         string `json:"status"` // "resting" | "executed" | "canceled"
		FilledCount    int    `json:"filled_count"`
		RemainingCount int    `json:"remaining_count"`
		YesPrice       int    `json:"yes_price"`
		NoPrice        int    `json:"no_price"`
	} `json:"order"`
}

// restingTPPrice returns the take-profit limit for an entry. Both directions sell
// back at the position-side bid, so the target is EntryPrice + TP either way.
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

// PlaceRestingTP submits a post-only opposite-side limit at entry+TP. Idempotent:
// returns immediately if this position already has one on the book. On failure it
// sets RestingTPStatus to "rejected" so callers fall back to a crossing exit.
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
		PostOnly:      true,
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
			Msg("resting TP placement failed, falling back to a crossing exit")
		pos.RestingTPStatus = "rejected"
		return err
	}
	pos.RestingTPOrderID = apiResp.Order.OrderID
	pos.RestingTPStatus = "open"
	return nil
}

// PollRestingTPStatus refreshes pos.RestingTPStatus from Kalshi with a single GET.
// Any terminal status other than "open" means the position is closed. Paper mode
// is a no-op; CheckExit simulates the fill instead.
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
		// still on the book, leave status as "open"
	default:
		zlog.Warn().
			Str("status", out.Order.Status).
			Str("order_id", pos.RestingTPOrderID).
			Msg("unknown Kalshi order status, treating as still open")
	}
	return pos.RestingTPStatus, nil
}

// CancelRestingTP cancels the resting TP and returns its post-cancel status.
// A returned "executed" means the order filled first and the position is already
// closed at TP, so the caller must not place a follow-up exit.
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

	if httpResp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(httpResp.Body)
		// 404 means the order is already gone. Treat it as canceled.
		if httpResp.StatusCode == http.StatusNotFound {
			pos.RestingTPStatus = "canceled"
			return pos.RestingTPStatus, nil
		}
		return pos.RestingTPStatus, fmt.Errorf("cancel order %d: %s", httpResp.StatusCode, b)
	}
	var out kalshiGetOrderResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&out); err != nil {
		// 200 but undecodable. Assume the cancel worked.
		pos.RestingTPStatus = "canceled"
		return pos.RestingTPStatus, nil
	}
	switch out.Order.Status {
	case "executed", "filled":
		// The TP filled between our exit decision and this cancel.
		pos.RestingTPStatus = "executed"
	default:
		pos.RestingTPStatus = "canceled"
	}
	return pos.RestingTPStatus, nil
}
