package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

type Coordinator struct {
	cfg        Config
	ledger     *Ledger
	killSwitch *KillSwitch
	engines    map[string]context.CancelFunc
	mu         sync.Mutex
}

func NewCoordinator(cfg Config, ledger *Ledger, ks *KillSwitch) *Coordinator {
	return &Coordinator{
		cfg:        cfg,
		ledger:     ledger,
		killSwitch: ks,
		engines:    make(map[string]context.CancelFunc),
	}
}

// Run blocks until ctx is cancelled. Polls for active games and manages engine lifecycle.
func (c *Coordinator) Run(ctx context.Context) error {
	log.Println("[COORDINATOR] Started NBA scoreboard poller (30s interval)")
	
	// Initial poll
	c.pollScoreboard(ctx)
	
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			c.mu.Lock()
			for gameID, cancel := range c.engines {
				log.Printf("[COORDINATOR] Shutting down engine for game %s", gameID)
				cancel()
			}
			c.mu.Unlock()
			return nil
		case <-ticker.C:
			c.pollScoreboard(ctx)
		}
	}
}

type nbaScoreboard struct {
	Scoreboard struct {
		Games []struct {
			GameID      string `json:"gameId"`
			GameStatus  int    `json:"gameStatus"`
			GameTimeUTC string `json:"gameTimeUTC"`
			AwayTeam    struct {
				TeamTricode string `json:"teamTricode"`
			} `json:"awayTeam"`
			HomeTeam struct {
				TeamTricode string `json:"teamTricode"`
			} `json:"homeTeam"`
		} `json:"games"`
	} `json:"scoreboard"`
}

func (c *Coordinator) pollScoreboard(ctx context.Context) {
	url := "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
	req.Header.Set("Accept", "application/json, text/plain, */*")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		zlog.Warn().Err(err).Msg("coordinator failed to fetch scoreboard")
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		zlog.Warn().Int("status", resp.StatusCode).Msg("coordinator scoreboard non-200")
		return
	}

	var data nbaScoreboard
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		zlog.Warn().Err(err).Msg("coordinator failed to decode scoreboard")
		return
	}

	now := time.Now()

	c.mu.Lock()
	defer c.mu.Unlock()

	for _, g := range data.Scoreboard.Games {
		_, exists := c.engines[g.GameID]

		// Final game - shutdown engine if it exists
		if g.GameStatus == 3 {
			if exists {
				log.Printf("[COORDINATOR] Game %s reached FINAL. Terminating engine.", g.GameID)
				c.engines[g.GameID]() // call cancel
				delete(c.engines, g.GameID)
			}
			continue
		}

		// Check if we should start it
		shouldStart := false
		if g.GameStatus == 2 {
			shouldStart = true
		} else if g.GameStatus == 1 {
			// Pre-game. Check if within 5 minutes of start time.
			t, err := time.Parse(time.RFC3339, g.GameTimeUTC)
			if err == nil {
				timeUntilTip := t.Sub(now)
				if timeUntilTip <= 5*time.Minute && timeUntilTip >= -2*time.Hour {
					shouldStart = true
				}
			}
		}

		if shouldStart && !exists {
			log.Printf("[COORDINATOR] Game %s (%s @ %s) is Live/Approaching! Spawning engine.", g.GameID, g.AwayTeam.TeamTricode, g.HomeTeam.TeamTricode)
			
			eventTicker := buildKalshiEventTicker(g.GameTimeUTC, g.AwayTeam.TeamTricode, g.HomeTeam.TeamTricode)
			
			engineCtx, cancel := context.WithCancel(ctx)
			c.engines[g.GameID] = cancel
			
			go c.spawnEngine(engineCtx, g.GameID, eventTicker)
		}
	}
}

func buildKalshiEventTicker(gameTimeUTC, away, home string) string {
	loc, _ := time.LoadLocation("America/New_York")
	t, err := time.Parse(time.RFC3339, gameTimeUTC)
	if err != nil {
		t = time.Now()
	}
	tEST := t.In(loc)
	dateStr := strings.ToUpper(tEST.Format("06Jan02")) // e.g. 26MAY06
	return fmt.Sprintf("KXNBASPREAD-%s%s%s", dateStr, away, home)
}

func (c *Coordinator) spawnEngine(ctx context.Context, gameID, eventTicker string) {
	engine := NewGameEngine(gameID, eventTicker, c.cfg, c.ledger, c.killSwitch)
	engine.Run(ctx)
}
