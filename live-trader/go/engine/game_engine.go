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
package engine

import (
	"context"

	"live-trader/go/feed"
	"live-trader/go/orders"
	"live-trader/go/risk"
)

type GameEngine struct {
	gameID          string
	marketTicker    string
	inferenceClient *InferenceClient
	ringBuffer      *RingBuffer
	orderRouter     *orders.Router
	riskLedger      *risk.Ledger
	killSwitch      *risk.KillSwitch
}

func New(
	gameID, marketTicker string,
	inferenceClient *InferenceClient,
	orderRouter *orders.Router,
	ledger *risk.Ledger,
	ks *risk.KillSwitch,
) *GameEngine {
	return &GameEngine{
		gameID:          gameID,
		marketTicker:    marketTicker,
		inferenceClient: inferenceClient,
		ringBuffer:      NewRingBuffer(),
		orderRouter:     orderRouter,
		riskLedger:      ledger,
		killSwitch:      ks,
	}
}

// Run blocks until ctx is cancelled.
func (g *GameEngine) Run(ctx context.Context) {
	possessionCh := make(chan feed.NBAEvent, 10)
	tickCh := make(chan feed.KalshiTick, 100)

	nbaFeed := feed.NewNBAFeed(g.gameID)
	kalshiFeed := feed.NewKalshiFeed(g.marketTicker, "")

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

func (g *GameEngine) runRingBuffer(ctx context.Context, tickCh <-chan feed.KalshiTick) {
	for {
		select {
		case tick := <-tickCh:
			g.ringBuffer.Update(tick)
		case <-ctx.Done():
			return
		}
	}
}

func (g *GameEngine) onEvent(ctx context.Context, event feed.NBAEvent) {
	if g.killSwitch.IsSet() {
		return
	}

	marketSnap := g.ringBuffer.Snapshot()

	// Python service receives raw event + market snapshot.
	// Returns action + full MMoE output + assembled feature dict (for logging).
	resp, err := g.inferenceClient.ProcessPossession(ctx, g.gameID, event, marketSnap)
	if err != nil {
		// Log and skip this possession — don't crash the engine
		return
	}

	if resp.IsGarbageTime || resp.IsBlowout {
		return
	}

	approved := g.riskLedger.Check(resp.Action, g.gameID, resp.YesBid)
	if !approved {
		return
	}

	g.orderRouter.Place(ctx, g.gameID, resp)
}
