// GameEngine orchestrates a single live game.
// Runs 3 concurrent goroutines and a main decision loop:
//
//   goroutine 1 (NBAFeed):     polls CDN every 3s, emits NBAEvent to possessionCh
//   goroutine 2 (KalshiFeed):  WebSocket, emits KalshiTick to tickCh
//   goroutine 3 (RingBuffer):  consumes tickCh, maintains rolling market windows
//
//   main select loop:
//     on NBAEvent → call Python inference service → agent decision → risk check → order
//
// The Python inference service receives (raw_event + market_snapshot) and returns
// (action, run_prob, trajectory[10], hazard[10], features[83]).
// Go never computes physics features — that is entirely Python's responsibility.
package main

import (
	"context"
	"time"
)

type GameEngine struct {
	gameID          string
	marketTicker    string
	inferenceClient *InferenceClient
	ringBuffer      *RingBuffer
	orderRouter     *Router
	riskLedger      *Ledger
	killSwitch      *KillSwitch
	bandit          *Bandit
	logger          *Logger
	cfg             *Config
	// runtime state
	position        *PaperPosition
	possCount       int
	totalPipelineMS int64
	signalCount     int
	positionsOpened int
	positionsClosed int
	netPnLDollars   float64
	wins            int
}

func NewGameEngine(
	gameID, marketTicker string,
	inferenceClient *InferenceClient,
	orderRouter *Router,
	ledger *Ledger,
	ks *KillSwitch,
	bandit *Bandit,
	logger *Logger,
	cfg *Config,
) *GameEngine {
	return &GameEngine{
		gameID:          gameID,
		marketTicker:    marketTicker,
		inferenceClient: inferenceClient,
		ringBuffer:      NewRingBuffer(),
		orderRouter:     orderRouter,
		riskLedger:      ledger,
		killSwitch:      ks,
		bandit:          bandit,
		logger:          logger,
		cfg:             cfg,
	}
}

// Run blocks until ctx is cancelled.
func (g *GameEngine) Run(ctx context.Context) {
	defer g.emitSummary()

	possessionCh := make(chan NBAEvent, 10)
	tickCh := make(chan KalshiTick, 100)

	nbaFeed := NewNBAFeed(g.gameID)
	kalshiFeed := NewKalshiFeed(g.marketTicker, "")

	go nbaFeed.Run(ctx, possessionCh)
	go kalshiFeed.Run(ctx, tickCh)
	go g.runRingBuffer(ctx, tickCh)

	for {
		select {
		case event := <-possessionCh:
			g.onEvent(ctx, event)
		case <-ctx.Done():
			return
		}
	}
}

func (g *GameEngine) runRingBuffer(ctx context.Context, tickCh <-chan KalshiTick) {
	for {
		select {
		case tick := <-tickCh:
			g.ringBuffer.Update(tick)
		case <-ctx.Done():
			return
		}
	}
}

func (g *GameEngine) onEvent(ctx context.Context, event NBAEvent) {
	if g.killSwitch.IsSet() {
		return
	}

	start := time.Now()
	marketSnap := g.ringBuffer.Snapshot()

	resp, err := g.inferenceClient.ProcessPossession(ctx, g.gameID, event, marketSnap)
	if err != nil {
		zlog.Error().Str("game_id", g.gameID).Err(err).Msg("inference error")
		return
	}

	g.possCount++
	g.totalPipelineMS += resp.PipelineMS

	g.logger.EmitPossession(g.gameID, event, resp, false, start)

	// Check exit before considering new entries.
	if g.position != nil {
		if resp.IsGarbageTime || resp.IsBlowout {
			pnl := calcNetPnL(g.position.Size, g.position.EntryPrice, resp.YesBid)
			g.logger.EmitExit(g.gameID, "GARBAGE_TIME", g.position, resp.YesBid, pnl, g.possCount-g.position.EntryPossID)
			g.updatePnL(pnl)
			g.position = nil
			g.logger.EmitGarbageTime(g.gameID, resp)
			return
		}

		possHeld := g.possCount - g.position.EntryPossID
		g.logger.EmitHold(g.gameID, g.position, resp, possHeld)

		shouldExit, reason, pnl := g.orderRouter.CheckExit(g.position, resp.YesBid, g.possCount, g.cfg)
		if shouldExit {
			g.logger.EmitExit(g.gameID, reason, g.position, resp.YesBid, pnl, possHeld)
			g.updatePnL(pnl)
			g.position = nil
			return
		}
		return // still holding — don't consider new entries
	}

	if resp.IsGarbageTime || resp.IsBlowout {
		g.logger.EmitGarbageTime(g.gameID, resp)
		return
	}

	action := g.bandit.Decide(resp, false)
	if action != BuyYes {
		return
	}

	approved, _ := g.riskLedger.Check(string(action), g.gameID, resp.YesBid)
	if !approved {
		return
	}

	pos := g.orderRouter.Place(g.gameID, resp, g.possCount, g.cfg)
	if pos == nil {
		return
	}

	g.position = pos
	g.positionsOpened++
	g.signalCount++
	g.logger.EmitEntry(g.gameID, pos, resp)
}

func (g *GameEngine) updatePnL(pnl float64) {
	g.netPnLDollars += pnl
	g.positionsClosed++
	if pnl > 0 {
		g.wins++
	}
}

func (g *GameEngine) emitSummary() {
	var avgPipeline float64
	if g.possCount > 0 {
		avgPipeline = float64(g.totalPipelineMS) / float64(g.possCount)
	}
	var winRate float64
	if g.positionsClosed > 0 {
		winRate = float64(g.wins) / float64(g.positionsClosed)
	}
	g.logger.EmitGameSummary(GameSummary{
		GameID:          g.gameID,
		PossCount:       g.possCount,
		AvgPipelineMS:   avgPipeline,
		SignalCount:     g.signalCount,
		PositionsOpened: g.positionsOpened,
		PositionsClosed: g.positionsClosed,
		NetPnLDollars:   g.netPnLDollars,
		WinRate:         winRate,
	})
}
