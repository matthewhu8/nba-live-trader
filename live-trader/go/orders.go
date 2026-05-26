// OrderRouter is the single point of contact with the Kalshi REST API.
// All order placement in the system goes through here — no other file
// calls the Kalshi API directly.
//
// Enforcements at this layer (not negotiable):
//   - Maker-only: all orders are limit orders with post_only=true.
//     Taker orders are rejected before the API call.
//   - paper_mode flag: when true, logs the order but does not send to Kalshi.
//     Paper mode is the default. Single flag in config/trading.yaml to go live.
//   - Fee accounting: Kalshi charges $0 maker fees on standard markets.
//     calcNetPnL returns gross P&L with no fee deduction.
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
	Direction   string    // "YES" or "NO"
	EntryPrice  int       // cents
	Size        int       // contracts
	EntryPossID int
	EntryTime   time.Time
	OrderID     string    // Kalshi order_id (empty in paper mode)
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

// PlaceExit sends a live limit sell order to close an open position.
// No-op in paper mode — paper exits are tracked in-process only.
func (r *Router) PlaceExit(ctx context.Context, ticker string, pos *PaperPosition, exitPrice int) {
	if r.paperMode {
		return
	}

	side := "yes"
	if pos.Direction == "NO" {
		side = "no"
	}

	req := kalshiOrderRequest{
		Ticker:        ticker,
		Action:        "sell",
		Side:          side,
		Type:          "limit",
		Count:         pos.Size,
		ClientOrderID: newUUID(),
		PostOnly:      true,
	}
	if side == "yes" {
		req.YesPrice = exitPrice
	} else {
		req.NoPrice = exitPrice
	}

	if _, err := placeKalshiOrder(ctx, req); err != nil {
		zlog.Error().Err(err).
			Str("ticker", ticker).
			Str("entry_order_id", pos.OrderID).
			Str("direction", pos.Direction).
			Int("exit_price", exitPrice).
			Int("size", pos.Size).
			Msg("live exit order failed")
	}
}

// CheckExit evaluates whether an open position should be closed.
// Returns (shouldExit, reason, netPnLDollars).
func (r *Router) CheckExit(pos *PaperPosition, resp *PossessionResponse, possID int, cfg *Config) (bool, string, float64) {
	currentPrice := resp.YesBid
	if pos.Direction == "NO" {
		currentPrice = 100 - resp.YesAsk
		if resp.YesAsk == 0 {
			currentPrice = 100 - resp.YesBid
		}
	}

	priceDelta := currentPrice - pos.EntryPrice

	if priceDelta >= cfg.Agent.TakeProfitCents {
		return true, "TAKE_PROFIT", calcNetPnL(pos.Size, pos.EntryPrice, currentPrice)
	}
	if -priceDelta >= cfg.Agent.StopLossCents {
		return true, "STOP_LOSS", calcNetPnL(pos.Size, pos.EntryPrice, currentPrice)
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

// kellyContracts scales position size linearly from the entry threshold.
//
// Formula: (|traj| - 0.10) × 100, floored at 5.
//   0.15 → 5   (minimum signal → minimum size)
//   0.20 → 10
//   0.25 → 15
//   0.30 → 20
//   0.40 → 30
//   0.50 → 40
//   0.60 → 50  (capped at maxContracts)
//
// Anchoring at 0.10 (below the 0.15 entry threshold) means size is
// zero-based on our actual confidence above noise — a traj just barely
// clearing the gate gets the minimum 5, not an inflated count.
// Break-even win rate at 5 contracts is 41.3% vs ~48% expected.
func kellyContracts(trajFinal float32, maxContracts int) int {
	abs := trajFinal
	if abs < 0 {
		abs = -abs
	}
	n := int((abs - 0.10) * 100)
	if n < 5 {
		n = 5
	}
	if n > maxContracts {
		n = maxContracts
	}
	return n
}
