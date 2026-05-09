// NBAFeed polls the NBA Stats CDN for live play-by-play events.
//
// Endpoint: cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gameID}.json
// Poll interval: 3s (CDN cache TTL during live games — polling faster is pointless)
// Lag behind real events: ~15-20s (CDN delay, not network)
//
// Emits only NEW events (actionNumber > last seen) to avoid reprocessing.
// Never panics — logs errors and skips the poll cycle on failure.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

// NBAEvent is one action from the NBA CDN play-by-play response.
// Field names match the CDN JSON schema directly.
type NBAEvent struct {
	ActionNumber      int     `json:"actionNumber"`
	ActionType        string  `json:"actionType"` // "2pt", "3pt", "rebound", "substitution", "foul", "timeout"
	Period            int     `json:"period"`
	Clock             string  `json:"clock"` // "PT06M23.00S"
	TeamID            int64   `json:"teamId"`
	PersonID          int64   `json:"personId"`
	ShotResult        string  `json:"shotResult"` // "Made" | "Missed" | ""
	ShotDistance      float64 `json:"shotDistance"`
	ShotArea          string  `json:"area"`
	IsFieldGoal       int     `json:"isFieldGoal"`
	ScoreHome         string  `json:"scoreHome"`
	ScoreAway         string  `json:"scoreAway"`
	FoulPersonalTotal int     `json:"foulPersonalTotal"`
	SubType           string  `json:"subType"` // "offensive" | "defensive" for rebounds
	Description       string  `json:"description"`
	IsBackfill        bool    `json:"-"`
}

// cdnResponse mirrors the top-level JSON structure from the NBA CDN.
type cdnResponse struct {
	Game struct {
		Actions []NBAEvent `json:"actions"`
	} `json:"game"`
}

// NBAFeed polls the CDN for a single game and emits new events.
type NBAFeed struct {
	gameID        string
	lastActionNum int
	bootstrapped  bool
	client        *http.Client
}

const (
	nbaPollInterval = 3 * time.Second
	nbaCDNURL       = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_%s.json"
	nbaHTTPTimeout  = 5 * time.Second
)

func NewNBAFeed(gameID string) *NBAFeed {
	return &NBAFeed{
		gameID: gameID,
		client: &http.Client{
			Timeout: nbaHTTPTimeout,
			Transport: &http.Transport{
				MaxIdleConns:    5,
				IdleConnTimeout: 30 * time.Second,
			},
		},
	}
}

// Run polls until ctx is cancelled. New events are sent to out.
func (f *NBAFeed) Run(ctx context.Context, out chan<- NBAEvent) {
	ticker := time.NewTicker(nbaPollInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			f.poll(ctx, out)
		case <-ctx.Done():
			return
		}
	}
}

func (f *NBAFeed) poll(ctx context.Context, out chan<- NBAEvent) {
	url := fmt.Sprintf(nbaCDNURL, f.gameID)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		log.Printf("[nba_feed] build request error game=%s: %v", f.gameID, err)
		return
	}
	// Mimic a browser — CDN blocks obvious bot user-agents
	req.Header.Set("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
	req.Header.Set("Accept", "application/json, text/plain, */*")
	req.Header.Set("Referer", "https://www.nba.com/")
	req.Header.Set("Origin", "https://www.nba.com")

	resp, err := f.client.Do(req)
	if err != nil {
		log.Printf("[nba_feed] GET error game=%s: %v", f.gameID, err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusTooManyRequests {
		log.Printf("[nba_feed] RATE LIMITED (429) game=%s — backing off", f.gameID)
		time.Sleep(10 * time.Second)
		return
	}
	if resp.StatusCode != http.StatusOK {
		log.Printf("[nba_feed] unexpected status %d game=%s", resp.StatusCode, f.gameID)
		return
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("[nba_feed] read body error game=%s: %v", f.gameID, err)
		return
	}

	var payload cdnResponse
	if err := json.Unmarshal(body, &payload); err != nil {
		log.Printf("[nba_feed] JSON parse error game=%s: %v", f.gameID, err)
		return
	}

	if !f.bootstrapped {
		f.emitStartupBackfill(out, payload.Game.Actions)
		f.bootstrapped = true
		return
	}

	newCount := 0
	for _, event := range payload.Game.Actions {
		if event.ActionNumber <= f.lastActionNum {
			continue
		}
		// Non-blocking send — drop if consumer is behind (shouldn't happen at 1 event/45s)
		select {
		case out <- event:
			newCount++
		default:
			log.Printf("[nba_feed] channel full, dropping event %d game=%s", event.ActionNumber, f.gameID)
		}
		if event.ActionNumber > f.lastActionNum {
			f.lastActionNum = event.ActionNumber
		}
	}

	if newCount > 0 {
		log.Printf("[nba_feed] game=%s polled %d new events (last_action=%d)", f.gameID, newCount, f.lastActionNum)
	}
}

func (f *NBAFeed) emitStartupBackfill(out chan<- NBAEvent, events []NBAEvent) {
	replayed := 0
	for _, event := range events {
		if event.ActionNumber > f.lastActionNum {
			f.lastActionNum = event.ActionNumber
		}
		event.IsBackfill = true
		select {
		case out <- event:
			replayed++
		default:
			log.Printf("[nba_feed] channel full, dropping backfill event %d game=%s", event.ActionNumber, f.gameID)
		}
	}

	log.Printf(
		"[nba_feed] game=%s startup backfill replayed %d events through action=%d; trading starts on next live poll",
		f.gameID,
		replayed,
		f.lastActionNum,
	)
}
