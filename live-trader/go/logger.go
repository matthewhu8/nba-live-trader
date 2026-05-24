// Human-readable console logger for the paper trading system.
//
// Every meaningful event produces one self-contained line to stdout so anyone
// watching the terminal can follow exactly what the system is doing. Trade
// events (Entry, Hold, Exit) are also written to a per-game log file under
// logs/paper_trades/ for post-game review.
//
// EmitPossession replaces the old zerolog JSON line with a plain formatted
// text line. The zerolog global (zlog) is kept only for debug/warn/error
// messages that don't need human-readable formatting.
package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/rs/zerolog"
)

// zlog is for structured debug/warn/error messages only — not for the
// human-readable trade lines, which go through writeLine / fmt.Fprintln.
var zlog = zerolog.New(os.Stderr).With().Timestamp().Logger()

// GameSummary holds end-of-game stats passed to EmitGameSummary.
type GameSummary struct {
	GameID          string
	PossCount       int
	AvgPipelineMS   float64
	SignalCount     int
	PositionsOpened int
	PositionsClosed int
	NetPnLDollars   float64
	WinRate         float64 // 0.0–1.0
}

// Logger writes human-readable event lines to stdout and, for trade events,
// to a per-game file under logs/paper_trades/.
type Logger struct {
	paperMode   bool
	redisStream string
	logFile     *os.File
}

func NewLogger(paperMode bool, redisStream string) *Logger {
	return &Logger{paperMode: paperMode, redisStream: redisStream}
}

// OpenTradeLog creates (or appends to) logs/paper_trades/{gameID}_{date}.log.
// Call once at game start. On failure, logs a warning and continues — the
// process never crashes because of a missing log directory.
//
// runID may be empty; if set, a session banner line is written to mark this
// session boundary. Banners make it possible to tell apart multiple runs that
// share a single date-anchored .log file (the issue we saw last night where
// one file contained 4 GAME SUMMARY blocks glued together).
func (l *Logger) OpenTradeLog(gameID, runID string) {
	dir := "logs/paper_trades"
	if err := os.MkdirAll(dir, 0o755); err != nil {
		zlog.Warn().Str("dir", dir).Err(err).Msg("could not create paper_trades directory — file logging disabled")
		return
	}

	date := time.Now().UTC().Format("2006-01-02")
	path := filepath.Join(dir, fmt.Sprintf("%s_%s.log", gameID, date))

	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		zlog.Warn().Str("path", path).Err(err).Msg("could not open trade log — file logging disabled")
		return
	}

	l.logFile = f
	zlog.Info().Str("path", path).Msg("trade log opened")

	if runID != "" {
		fmt.Fprintf(l.logFile,
			"─── SESSION START %s  game=%s  run_id=%s  paper_mode=%v ───\n",
			ts(), gameID, runID, l.paperMode,
		)
	}
}

// writeLine writes msg to stdout and, if a log file is open, to the file too.
// All trade events (Entry, Hold, Exit) go through here.
func (l *Logger) writeLine(msg string) {
	fmt.Fprintln(os.Stdout, msg)
	if l.logFile != nil {
		fmt.Fprintln(l.logFile, msg)
	}
}

// ts returns the current UTC timestamp in the standard log format.
func ts() string {
	return time.Now().UTC().Format("2006-01-02T15:04:05Z")
}

// EmitPossession prints one line per possession showing model output and the
// action the system intends to take. Not written to the trade log file —
// stdout only, because every possession would bloat the file.
//
// Example:
//
//	2026-04-29T20:15:30Z  Q2 07:32  WAIT     run=0.12  traj=-0.03  bid=47¢  ask=49¢  game=0042500121  pipeline=87ms
func (l *Logger) EmitPossession(
	gameID string,
	event NBAEvent,
	resp *PossessionResponse,
	riskApproved bool,
	start time.Time,
) {
	trajFinal := resp.Trajectory[9]
	trajSign := ""
	if trajFinal >= 0 {
		trajSign = "+"
	}

	actionLabel := resp.Action
	if event.IsBackfill {
		actionLabel = "BACKFILL"
	}

	line := fmt.Sprintf(
		"%s  Q%d %s  %-8s  run=%.2f  traj=%s%.2f  bid=%d¢  ask=%d¢  game=%s  pipeline=%dms",
		ts(),
		event.Period,
		formatClock(event.Clock),
		actionLabel,
		resp.RunProb,
		trajSign, trajFinal,
		resp.YesBid,
		resp.YesAsk,
		gameID,
		time.Since(start).Milliseconds(),
	)

	fmt.Fprintln(os.Stdout, line)

	go l.pushToStream(context.Background(), gameID, resp)
}

// EmitEntry prints and files a BUY entry event.
//
// Example:
//
//	2026-04-29T20:15:45Z  ★ BUY_YES  run=0.23  traj=+0.09  bid=52¢  3×  game=0042500121
func (l *Logger) EmitEntry(gameID string, pos *PaperPosition, resp *PossessionResponse) {
	trajFinal := resp.Trajectory[9]
	trajSign := ""
	if trajFinal >= 0 {
		trajSign = "+"
	}

	line := fmt.Sprintf(
		"%s  ★ BUY_%s  run=%.2f  traj=%s%.2f  bid=%d¢  %d×  game=%s",
		ts(),
		pos.Direction,
		resp.RunProb,
		trajSign, trajFinal,
		pos.EntryPrice,
		pos.Size,
		gameID,
	)

	l.writeLine(line)
}

// EmitHold prints a HOLD update for an open position. Called every possession
// while a position is live.
//
// Example:
//
//	2026-04-29T20:15:52Z  HOLD  bid=54¢  +2¢  unrealized=+$1.17  poss=3  hazard5=0.31  game=0042500121
func (l *Logger) EmitHold(gameID string, pos *PaperPosition, resp *PossessionResponse, possHeld int) {
	priceDelta := resp.YesBid - pos.EntryPrice
	deltaSign := ""
	if priceDelta >= 0 {
		deltaSign = "+"
	}

	unrealized := float64(priceDelta) * float64(pos.Size) / 100.0
	unrealizedSign := ""
	if unrealized >= 0 {
		unrealizedSign = "+"
	}

	line := fmt.Sprintf(
		"%s  HOLD  bid=%d¢  %s%d¢  unrealized=%s$%.2f  poss=%d  hazard5=%.2f  game=%s",
		ts(),
		resp.YesBid,
		deltaSign, priceDelta,
		unrealizedSign, unrealized,
		possHeld,
		resp.Hazard[4],
		gameID,
	)

	l.writeLine(line)
}

// EmitExit prints and files a position close event.
// reason is the raw string from CheckExit: "TAKE_PROFIT", "STOP_LOSS", "TIME_STOP".
//
// Example:
//
//	2026-04-29T20:16:10Z  TP_EXIT    entry=52¢  exit=60¢  P&L=+$5.90  poss=3  game=0042500121
func (l *Logger) EmitExit(gameID, reason string, pos *PaperPosition, exitPrice int, netPnL float64, possHeld int) {
	pnlSign := ""
	if netPnL >= 0 {
		pnlSign = "+"
	}

	line := fmt.Sprintf(
		"%s  %-9s  entry=%d¢  exit=%d¢  P&L=%s$%.2f  poss=%d  game=%s",
		ts(),
		exitReasonLabel(reason),
		pos.EntryPrice,
		exitPrice,
		pnlSign, netPnL,
		possHeld,
		gameID,
	)

	l.writeLine(line)
}

// EmitGarbageTime prints a one-time notice when garbage time is detected.
// game.go is responsible for calling this only once per detection event —
// not on every subsequent possession in garbage time.
//
// Example:
//
//	2026-04-29T20:44:00Z  GARBAGE_TIME  score_diff=24  game=0042500121
func (l *Logger) EmitGarbageTime(gameID string, resp *PossessionResponse) {
	scoreDiff := 0
	if v, ok := resp.Features["score_diff"]; ok {
		scoreDiff = int(v)
	}

	fmt.Fprintf(os.Stdout, "%s  GARBAGE_TIME  score_diff=%d  game=%s\n",
		ts(), scoreDiff, gameID)
}

// EmitMarketSwap prints a notice when the engine switches to a new Kalshi market.
//
// Example:
//
//	2026-04-29T20:15:30Z  [MARKET SWAP] KXNBASPREAD-26MAY06MINSAS -> KXNBASPREAD-26MAY06SASMIN (@ 55¢)
func (l *Logger) EmitMarketSwap(gameID, oldTicker, newTicker string, bid int) {
	fmt.Fprintf(os.Stdout, "%s  [MARKET SWAP] %s -> %s (@ %d¢)  game=%s\n",
		ts(), oldTicker, newTicker, bid, gameID)
}

// EmitStale prints a warning when the Kalshi market feed has gone quiet.
// The market snapshot has been zeroed out for this possession.
//
// Example:
//
//	2026-04-29T20:15:00Z  KALSHI_STALE  last_tick=35s ago  game=0042500121  market_features=zeroed
func (l *Logger) EmitStale(gameID string, secsSince int) {
	fmt.Fprintf(os.Stdout, "%s  KALSHI_STALE  last_tick=%ds ago  game=%s  market_features=zeroed\n",
		ts(), secsSince, gameID)
}

// EmitGameSummary prints a boxed end-of-game summary and closes the log file.
//
// Example:
//
//	──────────────────────────────────────────────────────
//	GAME SUMMARY  0042500121
//	  possessions: 142  avg_pipeline: 91ms
//	  signals: 8 BUY_YES   positions: 3 opened, 3 closed
//	  net P&L: +$8.73      win_rate: 2/3 (66.7%)
//	──────────────────────────────────────────────────────
func (l *Logger) EmitGameSummary(s GameSummary) {
	border := "──────────────────────────────────────────────────────"

	pnlSign := ""
	if s.NetPnLDollars >= 0 {
		pnlSign = "+"
	}

	wins := 0
	winPct := 0.0
	if s.PositionsClosed > 0 {
		wins = int(s.WinRate * float64(s.PositionsClosed))
		winPct = s.WinRate * 100.0
	}

	block := fmt.Sprintf(
		"%s\nGAME SUMMARY  %s\n  possessions: %d  avg_pipeline: %.0fms\n  signals: %d BUY_YES   positions: %d opened, %d closed\n  net P&L: %s$%.2f      win_rate: %d/%d (%.1f%%)\n%s",
		border,
		s.GameID,
		s.PossCount, s.AvgPipelineMS,
		s.SignalCount, s.PositionsOpened, s.PositionsClosed,
		pnlSign, s.NetPnLDollars,
		wins, s.PositionsClosed, winPct,
		border,
	)

	fmt.Fprintln(os.Stdout, block)

	if l.logFile != nil {
		fmt.Fprintln(l.logFile, block)
		l.logFile.Close()
		l.logFile = nil
	}
}

// pushToStream is a no-op stub. Redis integration is deferred.
func (l *Logger) pushToStream(_ context.Context, _ string, _ *PossessionResponse) {}

// formatClock converts the NBA CDN clock string "PT06M23.00S" to "06:23".
// Falls back to a truncated raw string if parsing fails — never panics.
func formatClock(raw string) string {
	var minutes int
	var secondsFloat float64
	_, err := fmt.Sscanf(raw, "PT%dM%fS", &minutes, &secondsFloat)
	if err != nil {
		if len(raw) > 8 {
			return raw[:8]
		}
		return raw
	}
	return fmt.Sprintf("%02d:%02d", minutes, int(secondsFloat))
}

// exitReasonLabel maps internal CheckExit reason codes to display labels.
func exitReasonLabel(reason string) string {
	switch reason {
	case "TAKE_PROFIT":
		return "TP_EXIT"
	case "STOP_LOSS":
		return "SL_EXIT"
	case "TIME_STOP":
		return "TIME_EXIT"
	default:
		return reason
	}
}
