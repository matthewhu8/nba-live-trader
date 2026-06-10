// InferenceClient is the Go side of the Go↔Python boundary.
// Sends (raw NBA event + Kalshi market snapshot) to the Python inference service.
// Receives back the agent action, MMoE outputs, and assembled feature dict.
//
// HTTP POST to localhost:8001/game/{gameID}/possession
// Keep-alive connection reused across calls (<1ms overhead on localhost).
// Timeout: 500ms. On timeout/error: log and return error (caller skips possession).
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

const startGameTimeout = 30 * time.Second

const inferenceTimeout = 500 * time.Millisecond

// PossessionRequest is sent to the Python inference service.
//
// The garbage/blowout threshold fields are forwarded from trading.yaml so the
// agent GATE flags (is_garbage_time / is_blowout) are config-driven rather than
// hardcoded in Python. omitempty keeps the wire backwards-compatible: an older
// Go binary that doesn't send them lets Python fall back to its 30/4/360 defaults.
type PossessionRequest struct {
	RawEvent       NBAEvent    `json:"raw_event"`
	KalshiSnapshot [14]float32 `json:"kalshi_snapshot"` // pre-computed by RingBuffer
	WallClockTS    time.Time   `json:"wall_clock_ts"`

	BlowoutMarginPts     int `json:"blowout_margin_pts,omitempty"`
	GarbageTimePeriod    int `json:"garbage_time_period,omitempty"`
	GarbageTimeClockSecs int `json:"garbage_time_clock_secs,omitempty"`
}

// InferenceConfig holds the trading.yaml-derived values the Go engine forwards
// to the Python inference service. Per-possession thresholds ride on
// PossessionRequest; the once-per-game values ride on the /game/start body.
type InferenceConfig struct {
	// Garbage/blowout GATE (per-possession).
	BlowoutMarginPts     int
	GarbageTimePeriod    int
	GarbageTimeClockSecs int
	// Model artifact paths (once per game, on /game/start).
	ModelPath  string
	ScalerPath string
	// Dashboard gate thresholds (once per game) — let the dashboard green-light
	// mirror the real agent thresholds instead of stale hardcoded literals.
	MinAbsTraj   float32
	MinYesBid    int
	MaxYesBid    int
	MinRunLength int
}

// PossessionResponse is returned by the Python inference service.
// Python has already assembled the full 83-feature vector and run the model.
type PossessionResponse struct {
	Action      string    `json:"action"`       // "BUY_YES" | "BUY_NO" | "EXIT" | "WAIT"
	RunProb     float32   `json:"run_prob"`
	Trajectory  [10]float32 `json:"trajectory"` // log-odds delta checkpoints
	Hazard      [10]float32 `json:"hazard"`     // survival hazard per horizon
	YesBid      int       `json:"yes_bid"`
	YesAsk      int       `json:"yes_ask"`
	IsGarbageTime bool    `json:"is_garbage_time"`
	IsBlowout   bool      `json:"is_blowout"`
	Features    map[string]float32 `json:"features"` // full 83-feature dict for logging
	PipelineMS  int64     `json:"pipeline_ms"`
}

type InferenceClient struct {
	baseURL       string
	cfg           InferenceConfig
	httpClient    *http.Client // 500ms timeout — used for ProcessPossession
	slowClient    *http.Client // no client-level timeout — used for StartGame (ctx controls deadline)
}

func NewInferenceClient(baseURL string, cfg InferenceConfig) *InferenceClient {
	transport := &http.Transport{
		MaxIdleConns:    10,
		IdleConnTimeout: 90 * time.Second,
	}
	return &InferenceClient{
		baseURL: baseURL,
		cfg:     cfg,
		httpClient: &http.Client{
			Timeout:   inferenceTimeout,
			Transport: transport,
		},
		slowClient: &http.Client{
			Transport: transport,
		},
	}
}

// StartGame initializes a game on the Python inference service.
// Must be called before any ProcessPossession calls for this game.
// Uses a 30s timeout — Python loads pregame context from MotherDuck on this call.
//
// runID and logDir activate Python-side JSONL logging for this run. Both
// are optional during the rolling upgrade: a Go binary that doesn't yet
// send them works against an old Python service, and a new Python service
// that doesn't receive them simply skips structured logging for that run.
func (c *InferenceClient) StartGame(ctx context.Context, gameID, ticker string, homeID, awayID int64, runID, logDir string) error {
	body, err := json.Marshal(struct {
		MarketTicker string `json:"market_ticker"`
		HomeTeamID   int64  `json:"home_team_id"`
		AwayTeamID   int64  `json:"away_team_id"`
		RunID        string `json:"run_id,omitempty"`
		LogDir       string `json:"log_dir,omitempty"`
		// Forwarded config — model paths (Python reloads only on change) and the
		// dashboard gate thresholds (so the dashboard tracks the live agent).
		ModelPath    string  `json:"model_path,omitempty"`
		ScalerPath   string  `json:"scaler_path,omitempty"`
		MinAbsTraj   float32 `json:"min_abs_traj,omitempty"`
		MinYesBid    int     `json:"min_yes_bid,omitempty"`
		MaxYesBid    int     `json:"max_yes_bid,omitempty"`
		MinRunLength int     `json:"min_run_length,omitempty"`
	}{
		MarketTicker: ticker,
		HomeTeamID:   homeID,
		AwayTeamID:   awayID,
		RunID:        runID,
		LogDir:       logDir,
		ModelPath:    c.cfg.ModelPath,
		ScalerPath:   c.cfg.ScalerPath,
		MinAbsTraj:   c.cfg.MinAbsTraj,
		MinYesBid:    c.cfg.MinYesBid,
		MaxYesBid:    c.cfg.MaxYesBid,
		MinRunLength: c.cfg.MinRunLength,
	})
	if err != nil {
		return fmt.Errorf("marshal start game request: %w", err)
	}

	reqCtx, cancel := context.WithTimeout(ctx, startGameTimeout)
	defer cancel()

	url := fmt.Sprintf("%s/game/%s/start", c.baseURL, gameID)
	httpReq, err := http.NewRequestWithContext(reqCtx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("build start game request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := c.slowClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("start game call failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		return fmt.Errorf("start game returned status %d for game %s", httpResp.StatusCode, gameID)
	}
	return nil
}

// EndGame cleans up game state on the Python inference service.
// Best-effort — errors are logged by the caller but not fatal.
func (c *InferenceClient) EndGame(ctx context.Context, gameID string) error {
	body := []byte("{}")

	url := fmt.Sprintf("%s/game/%s/end", c.baseURL, gameID)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("build end game request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("end game call failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		return fmt.Errorf("end game returned status %d for game %s", httpResp.StatusCode, gameID)
	}
	return nil
}

func (c *InferenceClient) ProcessPossession(
	ctx context.Context,
	gameID string,
	event NBAEvent,
	snap MarketSnapshot,
) (*PossessionResponse, error) {
	req := PossessionRequest{
		RawEvent:             event,
		KalshiSnapshot:       snap.Features,
		WallClockTS:          time.Now(),
		BlowoutMarginPts:     c.cfg.BlowoutMarginPts,
		GarbageTimePeriod:    c.cfg.GarbageTimePeriod,
		GarbageTimeClockSecs: c.cfg.GarbageTimeClockSecs,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal possession request: %w", err)
	}

	url := fmt.Sprintf("%s/game/%s/possession", c.baseURL, gameID)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("inference call failed: %w", err)
	}
	defer httpResp.Body.Close()

	var resp PossessionResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("decode inference response: %w", err)
	}
	return &resp, nil
}

// TradePayload is sent back to Python when the Go engine executes a paper trade.
// MarketTicker records the Kalshi market actually subscribed at order time —
// the dashboard parses it to label trades with the real team being backed
// (e.g., "BUY SAS" instead of "BUY YES"). Set this from a snapshot taken at
// the moment of order placement, never from a possession message, so a
// late-arriving market swap can't relabel a trade in flight.
type TradePayload struct {
	Action       string  `json:"action"`
	Direction    string  `json:"direction"`
	Price        int     `json:"price"`       // exit price on EXIT, entry price on ENTRY
	EntryPrice   int     `json:"entry_price"` // entry price on EXIT (for fee calculation); 0 on ENTRY
	Size         int     `json:"size"`
	PnL          float64 `json:"pnl"`
	Reason       string  `json:"reason"`
	MarketTicker string  `json:"market_ticker,omitempty"`
}

// ReportTrade sends a fire-and-forget HTTP request to the Python dashboard endpoint.
func (c *InferenceClient) ReportTrade(gameID string, payload TradePayload) {
	url := fmt.Sprintf("%s/game/%s/trade", c.baseURL, gameID)
	data, err := json.Marshal(payload)
	if err != nil {
		return
	}
	
	// Create a new context with a short timeout so we don't block
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(data))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")
	
	resp, err := c.slowClient.Do(req)
	if err == nil {
		resp.Body.Close()
	}
}
