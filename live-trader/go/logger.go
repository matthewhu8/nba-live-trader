// Structured logging and metrics emission.
// Every possession emits one canonical JSON log line via zerolog.
// Log lines are also pushed to a Redis Stream (non-blocking) for the
// dashboard consumer and Prometheus metrics writer.
//
// Log line schema (matches plan/architecture doc):
//   ts, event, game_id, possession_id, quarter, clock, score_diff,
//   run_prob, traj_final, hazard_10, action, yes_bid, yes_ask,
//   risk_approved, paper_mode, pipeline_ms
package main

import (
	"context"
	"os"
	"time"

	"github.com/rs/zerolog"

)

var zlog = zerolog.New(os.Stdout).With().Timestamp().Logger()

type PossessionLogEntry struct {
	GameID       string  `json:"game_id"`
	PossessionID int     `json:"possession_id"`
	Quarter      int     `json:"quarter"`
	Clock        string  `json:"clock"`
	ScoreDiff    int     `json:"score_diff"`
	RunProb      float32 `json:"run_prob"`
	TrajFinal    float32 `json:"traj_final"`
	Hazard10     float32 `json:"hazard_10"`
	Action       string  `json:"action"`
	YesBid       int     `json:"yes_bid"`
	YesAsk       int     `json:"yes_ask"`
	RiskApproved bool    `json:"risk_approved"`
	PaperMode    bool    `json:"paper_mode"`
	PipelineMS   int64   `json:"pipeline_ms"`
}

type Logger struct {
	paperMode    bool
	redisStream  string // Redis stream key, e.g. "possession_events"
	// TODO: redis client
}

func NewLogger(paperMode bool, redisStream string) *Logger {
	return &Logger{paperMode: paperMode, redisStream: redisStream}
}

func (l *Logger) EmitPossession(
	gameID string,
	event NBAEvent,
	resp *PossessionResponse,
	riskApproved bool,
	start time.Time,
) {
	entry := PossessionLogEntry{
		GameID:       gameID,
		Quarter:      event.Period,
		Clock:        event.Clock,
		RunProb:      resp.RunProb,
		TrajFinal:    resp.Trajectory[9],
		Hazard10:     resp.Hazard[9],
		Action:       resp.Action,
		YesBid:       resp.YesBid,
		YesAsk:       resp.YesAsk,
		RiskApproved: riskApproved,
		PaperMode:    l.paperMode,
		PipelineMS:   time.Since(start).Milliseconds(),
	}

	// Structured log to stdout (zerolog)
	zlog.Info().
		Str("event", "possession_decision").
		Str("game_id", entry.GameID).
		Int("quarter", entry.Quarter).
		Str("clock", entry.Clock).
		Float32("run_prob", entry.RunProb).
		Float32("traj_final", entry.TrajFinal).
		Str("action", entry.Action).
		Int("yes_bid", entry.YesBid).
		Bool("risk_approved", entry.RiskApproved).
		Bool("paper_mode", entry.PaperMode).
		Int64("pipeline_ms", entry.PipelineMS).
		Send()

	// Non-blocking push to Redis Stream for dashboard consumer
	go l.pushToStream(context.Background(), entry)
}

func (l *Logger) pushToStream(_ context.Context, entry PossessionLogEntry) {
	// TODO: XADD to Redis stream (non-blocking — if Redis is down, log and return)
}
