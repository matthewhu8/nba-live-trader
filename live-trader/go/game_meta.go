package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// fetchTeamIDs fetches home and away team integer IDs from the NBA CDN boxscore.
// Retries up to 3× with 2s delay on failure.
func fetchTeamIDs(ctx context.Context, gameID string) (homeID, awayID int64, err error) {
	url := fmt.Sprintf("https://cdn.nba.com/static/json/liveData/boxscore/boxscore_%s.json", gameID)
	client := &http.Client{Timeout: 10 * time.Second}

	for attempt := 1; attempt <= 3; attempt++ {
		homeID, awayID, err = doFetchTeamIDs(ctx, client, url)
		if err == nil {
			return
		}
		if attempt < 3 {
			select {
			case <-time.After(2 * time.Second):
			case <-ctx.Done():
				return 0, 0, ctx.Err()
			}
		}
	}
	return
}

// doFetchTeamIDs performs a single fetch attempt.
// JSON path: .game.homeTeam.teamId and .game.awayTeam.teamId
func doFetchTeamIDs(ctx context.Context, client *http.Client, url string) (homeID, awayID int64, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, 0, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
	req.Header.Set("Accept", "application/json, text/plain, */*")
	req.Header.Set("Referer", "https://www.nba.com/")
	req.Header.Set("Origin", "https://www.nba.com")

	resp, err := client.Do(req)
	if err != nil {
		return 0, 0, fmt.Errorf("GET boxscore: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, 0, fmt.Errorf("boxscore CDN returned status %d for URL %s", resp.StatusCode, url)
	}

	var payload struct {
		Game struct {
			HomeTeam struct {
				TeamID int64 `json:"teamId"`
			} `json:"homeTeam"`
			AwayTeam struct {
				TeamID int64 `json:"teamId"`
			} `json:"awayTeam"`
		} `json:"game"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return 0, 0, fmt.Errorf("decode boxscore JSON: %w", err)
	}

	homeID = payload.Game.HomeTeam.TeamID
	awayID = payload.Game.AwayTeam.TeamID

	if homeID == 0 || awayID == 0 {
		return 0, 0, fmt.Errorf("boxscore returned zero team IDs (game may not have started yet)")
	}

	return homeID, awayID, nil
}
