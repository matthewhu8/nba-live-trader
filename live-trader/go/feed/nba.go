// NBAFeed polls the NBA Stats CDN for live play-by-play events.
//
// Endpoint: cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gameID}.json
// Poll interval: 3s (CDN TTL during live games)
// Lag behind real events: ~15-20s
//
// Emits new NBAEvent structs to the provided channel whenever the action
// number advances since the last poll. Handles reconnects and empty responses
// gracefully — never panics, logs warnings instead.
package feed

import (
	"context"
	"time"
)

const nbaPollInterval = 3 * time.Second

// NBAEvent is one action from the NBA CDN play-by-play response.
// Field names match the CDN JSON schema directly.
type NBAEvent struct {
	ActionNumber int    `json:"actionNumber"`
	ActionType   string `json:"actionType"` // "2pt", "3pt", "rebound", "substitution", "foul", "timeout", ...
	Period       int    `json:"period"`
	Clock        string `json:"clock"` // "PT06M23.00S"
	TeamID       int64  `json:"teamId"`
	PersonID     int64  `json:"personId"`
	ShotResult   string `json:"shotResult"`  // "Made" | "Missed" | ""
	ShotDistance int    `json:"shotDistance"` // feet
	ShotArea     string `json:"area"`        // "Left Side(L)", "In The Paint (Non-RA)", ...
	IsFieldGoal  int    `json:"isFieldGoal"` // 1 if field goal attempt
	ScoreHome    string `json:"scoreHome"`
	ScoreAway    string `json:"scoreAway"`
	FoulPersonalTotal int `json:"foulPersonalTotal"` // cumulative fouls for this player
	Description  string `json:"description"`
}

// NBAFeed polls the CDN for a single game and emits new events.
type NBAFeed struct {
	gameID        string
	lastActionNum int
}

func NewNBAFeed(gameID string) *NBAFeed {
	return &NBAFeed{gameID: gameID}
}

// Run polls until ctx is cancelled. New events are sent to out.
func (f *NBAFeed) Run(ctx context.Context, out chan<- NBAEvent) {
	ticker := time.NewTicker(nbaPollInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			f.poll(out)
		case <-ctx.Done():
			return
		}
	}
}

func (f *NBAFeed) poll(out chan<- NBAEvent) {
	// TODO: GET cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gameID}.json
	// TODO: parse actions array
	// TODO: filter to actionNumber > f.lastActionNum
	// TODO: emit each new event to out (non-blocking, drop if channel full)
	// TODO: update f.lastActionNum
}
