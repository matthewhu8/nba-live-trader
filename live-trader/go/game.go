package main

import (
	"context"
	"log"
	"os"
	"time"
)

type GameEngine struct {
	gameID      string
	eventTicker string
	cfg         Config
	ledger      *Ledger
	killSwitch  *KillSwitch
}

func NewGameEngine(
	gameID, eventTicker string,
	cfg Config,
	ledger *Ledger,
	ks *KillSwitch,
) *GameEngine {
	return &GameEngine{
		gameID:      gameID,
		eventTicker: eventTicker,
		cfg:         cfg,
		ledger:      ledger,
		killSwitch:  ks,
	}
}

// Run blocks until ctx is cancelled.
func (g *GameEngine) Run(ctx context.Context) {
	log.Printf("[INIT] fetching team IDs for game %s from NBA CDN...", g.gameID)
	homeID, awayID, err := fetchTeamIDs(ctx, g.gameID)
	if err != nil {
		log.Printf("[WARN] could not fetch team IDs (game may not have started): %v", err)
	} else {
		log.Printf("[INIT] homeTeamID=%d  awayTeamID=%d", homeID, awayID)
	}

	inferenceURL := g.cfg.Inference.BaseURL
	if inferenceURL == "" {
		inferenceURL = "http://localhost:8001"
	}
	inference := NewInferenceClient(inferenceURL)

	log.Printf("[INIT] calling Python /game/%s/start ...", g.gameID)
	if err := inference.StartGame(ctx, g.gameID, g.eventTicker, homeID, awayID); err != nil {
		log.Printf("[WARN] StartGame failed: %v — inference will return zeros", err)
	} else {
		log.Printf("[INIT] Python game session started")
	}

	ringBuffer := NewRingBuffer()
	bandit := NewBandit(&g.cfg)
	router := NewRouter(os.Getenv("KALSHI_KEY_ID"), g.cfg.Trading.PaperMode)
	logger := NewLogger(g.cfg.Trading.PaperMode, g.cfg.Observability.RedisStream)
	logger.OpenTradeLog(g.gameID)

	possessionCh := make(chan NBAEvent, 500)
	tickCh := make(chan KalshiTick, 50000)
	marketCh := make(chan string, 10)

	nbaFeed := NewNBAFeed(g.gameID)
	go nbaFeed.Run(ctx, possessionCh)

	var initialTicker string

	if g.eventTicker != "" {
		scanner := NewMarketScanner(g.eventTicker, g.cfg.Agent.MinYesBid, g.cfg.Agent.MaxYesBid)
		best, _, err := scanner.scan(ctx)
		if err == nil && best != "" {
			initialTicker = best
		} else {
			initialTicker = g.eventTicker
		}

		go scanner.Run(ctx, initialTicker, marketCh)

		kalshiFeed := NewKalshiFeed(initialTicker)
		go kalshiFeed.Run(ctx, marketCh, tickCh)
		go func() {
			for {
				select {
				case tick := <-tickCh:
					ringBuffer.Update(tick)
				case <-ctx.Done():
					return
				}
			}
		}()
		log.Printf("[FEED] Kalshi WebSocket started with initial market %s", initialTicker)
	} else {
		log.Printf("[FEED] No event ticker — running without Kalshi data")
	}
	log.Printf("[FEED] NBA CDN poller started for game %s (3s interval)", g.gameID)

	var openPosition *PaperPosition
	var possCount int
	var signalCount int
	var positionsOpened int
	var positionsClosed int
	var netPnL float64
	var wins int
	var totalPipelineMS int64

	log.Println("──────────────────────────────────────────────────────")
	log.Printf("  PAPER TRADING ENGINE STARTED FOR %s", g.gameID)
	log.Println("──────────────────────────────────────────────────────")

	for {
		if g.killSwitch.IsSet() {
			return // Panic shut down
		}

		select {
		case event := <-possessionCh:
			start := time.Now()
			possCount++

			snap := ringBuffer.Snapshot()

			resp, err := inference.ProcessPossession(ctx, g.gameID, event, snap)
			if err != nil {
				log.Printf("[ERR] inference failed poss=%d: %v", possCount, err)
				continue
			}

			totalPipelineMS += resp.PipelineMS

			riskOK := true
			logger.EmitPossession(g.gameID, event, resp, riskOK, start)

			if event.IsBackfill {
				continue
			}

			hasPosition := openPosition != nil
			action := bandit.Decide(resp, hasPosition)

			if openPosition != nil {
				shouldExit, reason, pnl := router.CheckExit(openPosition, resp, possCount, &g.cfg)

				currentPrice := resp.YesBid
				if openPosition.Direction == "NO" {
					currentPrice = 100 - resp.YesAsk
					if resp.YesAsk == 0 {
						currentPrice = 100 - resp.YesBid
					}
				}

				if action == Exit {
					shouldExit = true
					reason = "HAZARD_EXIT"
					pnl = calcNetPnL(openPosition.Size, openPosition.EntryPrice, currentPrice)
				}

				if shouldExit {
					possHeld := possCount - openPosition.EntryPossID
					logger.EmitExit(g.gameID, reason, openPosition, currentPrice, pnl, possHeld)
					go inference.ReportTrade(g.gameID, TradePayload{
						Action:    "EXIT",
						Direction: openPosition.Direction,
						Price:     currentPrice,
						Size:      openPosition.Size,
						PnL:       pnl,
						Reason:    reason,
					})
					netPnL += pnl
					positionsClosed++
					if pnl > 0 {
						wins++
					}
					openPosition = nil
				} else {
					possHeld := possCount - openPosition.EntryPossID
					logger.EmitHold(g.gameID, openPosition, resp, possHeld)
				}
			} else if action == BuyYes || action == BuyNo {
				direction := "YES"
				if action == BuyNo {
					direction = "NO"
				}
				pos := router.Place(g.gameID, resp, possCount, &g.cfg, direction)
				if pos != nil {
					openPosition = pos
					signalCount++
					positionsOpened++
					logger.EmitEntry(g.gameID, pos, resp)
					go inference.ReportTrade(g.gameID, TradePayload{
						Action:    "ENTRY",
						Direction: pos.Direction,
						Price:     pos.EntryPrice,
						Size:      pos.Size,
						PnL:       0,
						Reason:    "SIGNAL",
					})
				}
			}

			if resp.IsGarbageTime {
				logger.EmitGarbageTime(g.gameID, resp)
			}

		case <-ctx.Done():
			if openPosition != nil {
				positionsClosed++
			}

			avgPipeline := float64(0)
			if possCount > 0 {
				avgPipeline = float64(totalPipelineMS) / float64(possCount)
			}
			winRate := float64(0)
			if positionsClosed > 0 {
				winRate = float64(wins) / float64(positionsClosed)
			}

			logger.EmitGameSummary(GameSummary{
				GameID:          g.gameID,
				PossCount:       possCount,
				AvgPipelineMS:   avgPipeline,
				SignalCount:     signalCount,
				PositionsOpened: positionsOpened,
				PositionsClosed: positionsClosed,
				NetPnLDollars:   netPnL,
				WinRate:         winRate,
			})

			endCtx, endCancel := context.WithTimeout(context.Background(), 5*time.Second)
			_ = inference.EndGame(endCtx, g.gameID)
			endCancel()

			return
		}
	}
}
