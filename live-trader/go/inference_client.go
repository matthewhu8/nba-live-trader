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
type PossessionRequest struct {
	RawEvent       NBAEvent  `json:"raw_event"`
	KalshiSnapshot [14]float32    `json:"kalshi_snapshot"` // pre-computed by RingBuffer
	WallClockTS    time.Time      `json:"wall_clock_ts"`
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
	httpClient    *http.Client // 500ms timeout — used for ProcessPossession
	slowClient    *http.Client // no client-level timeout — used for StartGame (ctx controls deadline)
}

func NewInferenceClient(baseURL string) *InferenceClient {
	transport := &http.Transport{
		MaxIdleConns:    10,
		IdleConnTimeout: 90 * time.Second,
	}
	return &InferenceClient{
		baseURL: baseURL,
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
func (c *InferenceClient) StartGame(ctx context.Context, gameID, ticker string, homeID, awayID int64) error {
	body, err := json.Marshal(struct {
		MarketTicker string `json:"market_ticker"`
		HomeTeamID   int64  `json:"home_team_id"`
		AwayTeamID   int64  `json:"away_team_id"`
	}{
		MarketTicker: ticker,
		HomeTeamID:   homeID,
		AwayTeamID:   awayID,
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
		RawEvent:       event,
		KalshiSnapshot: snap.Features,
		WallClockTS:    time.Now(),
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
