// Human-readable console logger. Every meaningful event prints one self-contained
// line to stdout so anyone watching the terminal can follow what the system is
// doing. Trade events also go to a per-game file under logs/paper_trades/.
package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/rs/zerolog"
)

// zlog carries structured debug and error messages. Human-readable trade lines go
// through writeLine instead.
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

type Logger struct {
	paperMode      bool
	redisStream    string
	trajAggregator string // matches the agent's, so console output tracks the decision
	logFile        *os.File
}

func NewLogger(paperMode bool, redisStream, trajAggregator string) *Logger {
	if trajAggregator == "" {
		trajAggregator = "final"
	}
	return &Logger{paperMode: paperMode, redisStream: redisStream, trajAggregator: trajAggregator}
}

// OpenTradeLog opens logs/paper_trades/{gameID}_{date}.log, creating or appending.
// Call it once at game start; a missing log directory warns rather than crashing.
// A non-empty runID writes a banner so runs sharing one date-anchored file stay
// distinguishable.
func (l *Logger) OpenTradeLog(gameID, runID string) {
	dir := "logs/paper_trades"
	if err := os.MkdirAll(dir, 0o755); err != nil {
		zlog.Warn().Str("dir", dir).Err(err).Msg("could not create paper_trades directory, file logging disabled")
		return
	}

	date := time.Now().UTC().Format("2006-01-02")
	path := filepath.Join(dir, fmt.Sprintf("%s_%s.log", gameID, date))

	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		zlog.Warn().Str("path", path).Err(err).Msg("could not open trade log, file logging disabled")
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

// writeLine writes msg to stdout and, when one is open, to the trade log file.
func (l *Logger) writeLine(msg string) {
	fmt.Fprintln(os.Stdout, msg)
	if l.logFile != nil {
		fmt.Fprintln(l.logFile, msg)
	}
}

// ts returns the current UTC timestamp in the log format.
func ts() string {
	return time.Now().UTC().Format("2006-01-02T15:04:05Z")
}

// EmitPossession prints one line per possession with the model output and the
// intended action. Stdout only; writing every possession would bloat the trade log.
//
//	2026-04-29T20:15:30Z  Q2 07:32  WAIT  run=0.12  traj=-0.03  bid=47¢  ask=49¢  game=0042500121  pipeline=87ms
func (l *Logger) EmitPossession(
	gameID string,
	event NBAEvent,
	resp *PossessionResponse,
	riskApproved bool,
	start time.Time,
) {
	trajUsed := aggregateTraj(resp.Trajectory, l.trajAggregator)
	trajSign := ""
	if trajUsed >= 0 {
		trajSign = "+"
	}

	// Skipped events carry no model output, so printing them would look like a
	// WAIT on a zero signal.
	if resp.Action == "SKIP" {
		return
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
		trajSign, trajUsed,
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
//	2026-04-29T20:15:45Z  ★ BUY_YES  run=0.23  traj=+0.09  bid=52¢  3×  game=0042500121
func (l *Logger) EmitEntry(gameID string, pos *PaperPosition, resp *PossessionResponse) {
	trajUsed := aggregateTraj(resp.Trajectory, l.trajAggregator)
	trajSign := ""
	if trajUsed >= 0 {
		trajSign = "+"
	}

	line := fmt.Sprintf(
		"%s  ★ BUY_%s  run=%.2f  traj=%s%.2f  bid=%d¢  %d×  game=%s",
		ts(),
		pos.Direction,
		resp.RunProb,
		trajSign, trajUsed,
		pos.EntryPrice,
		pos.Size,
		gameID,
	)

	l.writeLine(line)
}

// EmitHold prints a HOLD update once per possession while a position is live.
//
// The printed bid is the price of the side we own, mirroring Router.CheckExit, so
// the operator sees the price moving for or against the position. The raw yes_bid
// would invert for NO holds.
//
//	2026-04-29T20:15:52Z  HOLD  bid=54¢  +2¢  unrealized=+$1.17  poss=3  hazard5=0.31  game=0042500121
func (l *Logger) EmitHold(gameID string, pos *PaperPosition, resp *PossessionResponse, possHeld int) {
	currentPrice := resp.YesBid
	if pos.Direction == "NO" {
		currentPrice = 100 - resp.YesAsk
		if resp.YesAsk == 0 {
			currentPrice = 100 - resp.YesBid
		}
	}

	priceDelta := currentPrice - pos.EntryPrice
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
		currentPrice,
		deltaSign, priceDelta,
		unrealizedSign, unrealized,
		possHeld,
		resp.Hazard[4],
		gameID,
	)

	l.writeLine(line)
}

// EmitExit prints and files a position close. reason is the raw string from
// CheckExit, e.g. "TAKE_PROFIT" or "STOP_LOSS".
//
//	2026-04-29T20:16:10Z  TP_EXIT  entry=52¢  exit=60¢  P&L=+$5.90  poss=3  game=0042500121
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

// EmitGarbageTime prints a notice when garbage time is detected.
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

// EmitMarketSwap prints a notice when the engine switches Kalshi markets.
//
//	2026-04-29T20:15:30Z  [MARKET SWAP] KXNBASPREAD-26MAY06MINSAS -> KXNBASPREAD-26MAY06SASMIN (@ 55¢)
func (l *Logger) EmitMarketSwap(gameID, oldTicker, newTicker string, bid int) {
	fmt.Fprintf(os.Stdout, "%s  [MARKET SWAP] %s -> %s (@ %d¢)  game=%s\n",
		ts(), oldTicker, newTicker, bid, gameID)
}

// EmitStale warns that the Kalshi feed has gone quiet and the market snapshot for
// this possession was zeroed out.
//
//	2026-04-29T20:15:00Z  KALSHI_STALE  last_tick=35s ago  game=0042500121  market_features=zeroed
func (l *Logger) EmitStale(gameID string, secsSince int) {
	fmt.Fprintf(os.Stdout, "%s  KALSHI_STALE  last_tick=%ds ago  game=%s  market_features=zeroed\n",
		ts(), secsSince, gameID)
}

// EmitGameSummary prints a boxed end-of-game summary and closes the log file.
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

// pushToStream is a stub. Redis integration is deferred.
func (l *Logger) pushToStream(_ context.Context, _ string, _ *PossessionResponse) {}

// formatClock converts the CDN clock string "PT06M23.00S" to "06:23", falling back
// to a truncated raw string when parsing fails.
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
	case "TRAIL_STOP":
		return "TRAIL_EXIT"
	default:
		return reason
	}
}
