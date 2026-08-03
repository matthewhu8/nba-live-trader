package main

import (
	"context"
	"log"
	"os"
	"sync/atomic"
	"time"
)

type GameEngine struct {
	gameID      string
	eventTicker string
	cfg         Config
	ledger      *Ledger
	killSwitch  *KillSwitch
	run         *Run        // process-level run identity (Phase 1+); nil-safe
	jsonLog     *JSONLogger // shared structured-log writer (Phase 1+); nil-safe
}

func NewGameEngine(
	gameID, eventTicker string,
	cfg Config,
	ledger *Ledger,
	ks *KillSwitch,
	run *Run,
	jsonLog *JSONLogger,
) *GameEngine {
	return &GameEngine{
		gameID:      gameID,
		eventTicker: eventTicker,
		cfg:         cfg,
		ledger:      ledger,
		killSwitch:  ks,
		run:         run,
		jsonLog:     jsonLog,
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
	inference := NewInferenceClient(inferenceURL, InferenceConfig{
		BlowoutMarginPts:     g.cfg.Agent.BlowoutMarginPts,
		GarbageTimePeriod:    g.cfg.Agent.GarbageTimePeriod,
		GarbageTimeClockSecs: g.cfg.Agent.GarbageTimeClockSecs,
		ModelPath:            g.cfg.Inference.ModelPath,
		ScalerPath:           g.cfg.Inference.ScalerPath,
		MinAbsTraj:           g.cfg.Agent.MinAbsTrajEntry,
		MinYesBid:            g.cfg.Agent.MinYesBid,
		MaxYesBid:            g.cfg.Agent.MaxYesBid,
		MinRunLength:         g.cfg.Agent.MinRunLengthEntry,
	})

	log.Printf("[INIT] calling Python /game/%s/start ...", g.gameID)
	runIDForPy := ""
	logDirForPy := ""
	if g.run != nil {
		runIDForPy = g.run.ID
		logDirForPy = g.run.LogDir
	}
	const startRetries = 3
	var startErr error
	for attempt := 1; attempt <= startRetries; attempt++ {
		startErr = inference.StartGame(ctx, g.gameID, g.eventTicker, homeID, awayID, runIDForPy, logDirForPy)
		if startErr == nil {
			log.Printf("[INIT] Python game session started")
			break
		}
		log.Printf("[WARN] StartGame attempt %d/%d failed: %v", attempt, startRetries, startErr)
		if attempt < startRetries {
			select {
			case <-time.After(3 * time.Second):
			case <-ctx.Done():
				return
			}
		}
	}
	if startErr != nil {
		log.Printf("[ERROR] StartGame failed after %d attempts — aborting game %s: %v", startRetries, g.gameID, startErr)
		g.jsonLog.Emit("error", g.gameID, map[string]interface{}{
			"where":   "start_game",
			"message": startErr.Error(),
			"fatal":   true,
		})
		return
	}

	ringBuffer := NewRingBuffer()
	bandit := NewBandit(&g.cfg)
	router := NewRouter(os.Getenv("KALSHI_KEY_ID"), g.cfg.Trading.PaperMode)
	logger := NewLogger(g.cfg.Trading.PaperMode, g.cfg.Observability.RedisStream, g.cfg.Agent.TrajAggregator)
	runID := ""
	if g.run != nil {
		runID = g.run.ID
	}
	logger.OpenTradeLog(g.gameID, runID)

	possessionCh := make(chan NBAEvent, 500)
	tickCh := make(chan KalshiTick, 50000)
	marketCh := make(chan string, 10)

	nbaFeed := NewNBAFeed(g.gameID)
	go nbaFeed.Run(ctx, possessionCh)

	var initialTicker string

	// activeMarketTicker tracks the currently-subscribed Kalshi market. Stamped
	// onto every TradePayload so the dashboard can label trades with the team
	// actually backed, race-free against in-flight scanner swaps. Declared at
	// the outer scope so the main event loop (below) can read it; only the
	// has-event-ticker branch actually populates it via the fanout goroutine.
	var activeMarketTicker atomic.Value
	activeMarketTicker.Store("")

	// positionOpen lets the market scanner defer swaps while we hold a position.
	// Exits price off the active market's order book, so swapping mid-position
	// would compute the exit price from a different strike than the one we
	// actually hold (2026-05-28 post-mortem latent hazard). The engine sets
	// this on entry/exit; the scanner reads it before every swap.
	var positionOpen atomic.Bool

	if g.eventTicker != "" {
		scanner := NewMarketScanner(g.eventTicker, g.cfg.Agent.MarketDriftLowBid, g.cfg.Agent.MarketDriftHighBid, g.jsonLog, g.gameID)
		best, _, err := scanner.scan(ctx)
		if err == nil && best != "" {
			initialTicker = best
		} else {
			initialTicker = g.eventTicker
		}
		activeMarketTicker.Store(initialTicker)

		go scanner.Run(ctx, initialTicker, marketCh, &positionOpen)

		// Fanout: the scanner emits ticker swaps on marketCh; kalshi_feed needs
		// them to re-subscribe, and trade reporting needs them so it can tag
		// each ENTRY/EXIT payload with the market actually subscribed at order
		// time. A single broadcast goroutine forwards both, with a per-consumer
		// non-blocking send so a slow reader can't stall the scanner.
		kalshiMarketCh := make(chan string, 10)
		go func() {
			for {
				select {
				case <-ctx.Done():
					return
				case t, ok := <-marketCh:
					if !ok {
						return
					}
					activeMarketTicker.Store(t)
					select {
					case kalshiMarketCh <- t:
					default:
					}
				}
			}
		}()

		kalshiFeed := NewKalshiFeed(initialTicker)
		go kalshiFeed.Run(ctx, kalshiMarketCh, tickCh)
		go func() {
			for {
				select {
				case tick := <-tickCh:
					// Only accept ticks from the currently-subscribed market.
					// After a swap, Kalshi may deliver a few more ticks from
					// the old market before the unsubscribe is acknowledged —
					// those would corrupt the ring buffer with stale prices.
					cur, _ := activeMarketTicker.Load().(string)
					if tick.MarketTicker == cur || cur == "" {
						ringBuffer.Update(tick)
					}
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

	mode := "PAPER"
	if !g.cfg.Trading.PaperMode {
		mode = "LIVE"
	}
	log.Println("──────────────────────────────────────────────────────")
	log.Printf("  %s TRADING ENGINE STARTED FOR %s", mode, g.gameID)
	log.Println("──────────────────────────────────────────────────────")

	// game_start records the engine going live for this game. Captures the
	// initial market context and team IDs (which may be 0 if the CDN fetch
	// failed pre-tip). One record per game per Run.
	g.jsonLog.Emit("game_start", g.gameID, map[string]interface{}{
		"event_ticker":          g.eventTicker,
		"home_team_id":          homeID,
		"away_team_id":          awayID,
		"initial_market_ticker": initialTicker,
		"paper_mode":            g.cfg.Trading.PaperMode,
	})

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
				g.jsonLog.Emit("error", g.gameID, map[string]interface{}{
					"where":         "inference",
					"message":       err.Error(),
					"possession_id": possCount,
					"is_backfill":   event.IsBackfill,
				})
				continue
			}

			totalPipelineMS += resp.PipelineMS

			// Compute the aggregated traj once per possession — used for telemetry,
			// the entry gate (via bandit.Decide), and Kelly sizing. Keeps every
			// downstream consumer reading the same number for the same possession.
			trajAggregator := g.cfg.Agent.TrajAggregator
			if trajAggregator == "" {
				trajAggregator = "final"
			}
			trajUsed := aggregateTraj(resp.Trajectory, trajAggregator)

			riskOK := !g.killSwitch.IsSet()
			logger.EmitPossession(g.gameID, event, resp, riskOK, start)

			// possessionFields is the structured per-possession record.
			// action_chosen is filled in below — set to BACKFILL for replay
			// events, otherwise to the bandit's decision once known. Built
			// once so we emit exactly one possession record per possession.
			possessionFields := map[string]interface{}{
				"possession_id":      possCount,
				"is_backfill":        event.IsBackfill,
				"has_position":       openPosition != nil,
				"period":             event.Period,
				"clock":              event.Clock,
				"score_home":         event.ScoreHome,
				"score_away":         event.ScoreAway,
				"action_type":        event.ActionType,
				"yes_bid":            resp.YesBid,
				"yes_ask":            resp.YesAsk,
				"run_prob":           resp.RunProb,
				"trajectory":         resp.Trajectory,
				"traj_final":         resp.Trajectory[9], // raw single horizon — legacy
				"traj_used":          trajUsed,           // aggregated value driving decisions
				"traj_aggregator":    trajAggregator,
				"hazard":             resp.Hazard,
				"hazard5":            resp.Hazard[4],
				"is_garbage_time":    resp.IsGarbageTime,
				"is_blowout":         resp.IsBlowout,
				"server_pipeline_ms": resp.PipelineMS,
				"client_total_ms":    time.Since(start).Milliseconds(),
				"features":           resp.Features,
			}

			if event.IsBackfill {
				possessionFields["action_chosen"] = "BACKFILL"
				g.jsonLog.Emit("possession", g.gameID, possessionFields)
				continue
			}

			hasPosition := openPosition != nil
			action, gates := bandit.Decide(resp, hasPosition)
			possessionFields["action_chosen"] = string(action)
			possessionFields["gates"] = gates
			g.jsonLog.Emit("possession", g.gameID, possessionFields)

			if openPosition != nil {
				shouldExit, reason, pnl := router.CheckExit(ctx, openPosition, resp, possCount, &g.cfg)

				// For TP_EXIT, the resting maker order filled at RestingTPPrice;
				// for everything else (SL/TIME/momentum), the close price is the
				// direction-aware current price the crossing exit will reach.
				currentPrice := resp.YesBid
				if openPosition.Direction == "NO" {
					currentPrice = 100 - resp.YesAsk
					if resp.YesAsk == 0 {
						currentPrice = 100 - resp.YesBid
					}
				}
				closePrice := currentPrice
				if reason == "TAKE_PROFIT" && openPosition.RestingTPStatus == "executed" {
					closePrice = openPosition.RestingTPPrice
				}

				if shouldExit {
					// Resting TP fill: position is ALREADY closed on Kalshi — no
					// further order needed. Just record the trade and free state.
					tpFilled := reason == "TAKE_PROFIT" && openPosition.RestingTPStatus == "executed"

					ok := tpFilled
					if !tpFilled {
						// Non-TP exit (SL, TIME, momentum). If a resting TP is still
						// open, cancel it FIRST so we don't double-close. The cancel
						// response is authoritative — if Kalshi says the TP just
						// executed, treat the position as TP-closed and skip the SL.
						if openPosition.RestingTPStatus == "open" {
							status, err := router.CancelRestingTP(ctx, openPosition)
							if err != nil {
								zlog.Warn().Err(err).
									Str("order_id", openPosition.RestingTPOrderID).
									Msg("resting TP cancel failed — proceeding with crossing exit anyway")
							}
							if status == "executed" {
								// Cancel-vs-fill race: TP won. Re-route as TP exit.
								reason = "TAKE_PROFIT"
								closePrice = openPosition.RestingTPPrice
								pnl = calcNetPnL(openPosition.Size, openPosition.EntryPrice, closePrice)
								tpFilled = true
								ok = true
							}
						}

						if !tpFilled {
							// Crossing exit (taker) — same path as before.
							// Each failed attempt widens the crossing budget by 1¢ so a
							// thin/fast book can't trap us in a position — we cross a
							// little deeper next possession until we're out.
							budget := g.cfg.Agent.ExitSlippageBudgetCents + openPosition.ExitAttempts
							ok = router.PlaceExit(ctx, openPosition, currentPrice, budget)
							if !ok {
								openPosition.ExitAttempts++
								g.jsonLog.Emit("exit_retry", g.gameID, map[string]interface{}{
									"possession_id": possCount,
									"reason":        reason,
									"entry_ticker":  openPosition.EntryTicker,
									"exit_price":    currentPrice,
									"attempts":      openPosition.ExitAttempts,
									"budget_cents":  budget,
								})
							}
						}
					}

					if ok {
						g.ledger.RecordExit(g.gameID, openPosition.Size, openPosition.EntryPrice, closePrice)
						possHeld := possCount - openPosition.EntryPossID
						logger.EmitExit(g.gameID, reason, openPosition, closePrice, pnl, possHeld)
						g.jsonLog.Emit("exit", g.gameID, map[string]interface{}{
							"possession_id":      possCount,
							"reason":             reason,
							"direction":          openPosition.Direction,
							"entry_price":        openPosition.EntryPrice,
							"exit_price":         closePrice,
							"size":               openPosition.Size,
							"net_pnl_dollars":    pnl,
							"possessions_held":   possHeld,
							"run_prob":           resp.RunProb,
							"traj_final":         resp.Trajectory[9],
							"traj_used":          trajUsed,
							"hazard5":            resp.Hazard[4],
							"resting_tp_order":   openPosition.RestingTPOrderID,
							"resting_tp_status":  openPosition.RestingTPStatus,
							"resting_tp_filled":  tpFilled,
						})
						go inference.ReportTrade(g.gameID, TradePayload{
							Action:       "EXIT",
							Direction:    openPosition.Direction,
							Price:        closePrice,
							EntryPrice:   openPosition.EntryPrice,
							Size:         openPosition.Size,
							PnL:          pnl,
							Reason:       reason,
							MarketTicker: openPosition.EntryTicker,
						})
						netPnL += pnl
						positionsClosed++
						if pnl > 0 {
							wins++
						}
						openPosition = nil
						positionOpen.Store(false) // let the scanner resume swapping
					}
				} else {
					possHeld := possCount - openPosition.EntryPossID
					logger.EmitHold(g.gameID, openPosition, resp, possHeld)
					priceDelta := currentPrice - openPosition.EntryPrice
					unrealized := float64(priceDelta) * float64(openPosition.Size) / 100.0
					g.jsonLog.Emit("hold", g.gameID, map[string]interface{}{
						"possession_id":      possCount,
						"direction":          openPosition.Direction,
						"entry_price":        openPosition.EntryPrice,
						"current_price":      currentPrice,
						"price_delta_cents":  priceDelta,
						"unrealized_dollars": unrealized,
						"possessions_held":   possHeld,
						"hazard5":            resp.Hazard[4],
						"run_prob":           resp.RunProb,
						"traj_final":         resp.Trajectory[9],
						"traj_used":          trajUsed,
					})
				}
			} else if action == BuyYes || action == BuyNo {
				direction := "YES"
				if action == BuyNo {
					direction = "NO"
				}
				size := kellyContracts(
					trajUsed,
					g.cfg.Risk.MaxContractsPerOrder,
					g.cfg.Agent.KellyAnchorTraj,
					g.cfg.Agent.KellySlope,
					g.cfg.Agent.KellyMinContracts,
				)
				approved, blockReason := g.ledger.Check(
					string(action), g.gameID, resp.YesBid, size,
				)
				if !approved {
					log.Printf("[RISK] order blocked: %s", blockReason)
					g.jsonLog.Emit("risk_block", g.gameID, map[string]interface{}{
						"possession_id": possCount,
						"action":        string(action),
						"reason":        blockReason,
					})
				} else {
					entryTicker, _ := activeMarketTicker.Load().(string)
					pos := router.Place(ctx, entryTicker, g.gameID, resp, possCount, size, direction)
					if pos != nil {
						pos.EntryTicker = entryTicker
						pos.PeakPrice = pos.EntryPrice // trailing take-profit baseline
						g.ledger.RecordFill(g.gameID, pos.Size, pos.EntryPrice)
						openPosition = pos
						positionOpen.Store(true) // freeze market swaps while we hold this position
						signalCount++
						positionsOpened++

						// Immediately place the resting maker TP. Failure is non-fatal
						// (CheckExit will retry next possession); errors are logged inside.
						if err := router.PlaceRestingTP(ctx, pos, g.cfg.Agent.TakeProfitCents); err != nil {
							zlog.Warn().Err(err).
								Str("ticker", entryTicker).
								Int("tp_price", pos.RestingTPPrice).
								Msg("initial resting TP placement failed — CheckExit will retry")
						}

						logger.EmitEntry(g.gameID, pos, resp)
						g.jsonLog.Emit("entry", g.gameID, map[string]interface{}{
							"possession_id":     possCount,
							"direction":         pos.Direction,
							"entry_price":       pos.EntryPrice,
							"size":              pos.Size,
							"kelly_size":        size,
							"traj_final":        resp.Trajectory[9],
							"traj_used":         trajUsed,
							"traj_aggregator":   trajAggregator,
							"yes_bid":           resp.YesBid,
							"yes_ask":           resp.YesAsk,
							"run_prob":          resp.RunProb,
							"hazard5":           resp.Hazard[4],
							"resting_tp_price":  pos.RestingTPPrice,
							"resting_tp_order":  pos.RestingTPOrderID,
							"resting_tp_status": pos.RestingTPStatus,
						})
						go inference.ReportTrade(g.gameID, TradePayload{
							Action:       "ENTRY",
							Direction:    pos.Direction,
							Price:        pos.EntryPrice,
							Size:         pos.Size,
							PnL:          0,
							Reason:       "SIGNAL",
							MarketTicker: entryTicker,
						})
					}
				}
			}

			if resp.IsGarbageTime {
				logger.EmitGarbageTime(g.gameID, resp)
				gtFields := map[string]interface{}{
					"possession_id": possCount,
				}
				if scoreDiff, ok := resp.Features["score_diff"]; ok {
					gtFields["score_diff"] = scoreDiff
				}
				g.jsonLog.Emit("garbage_time", g.gameID, gtFields)
			}

		case <-ctx.Done():
			if openPosition != nil {
				positionsClosed++
				if !g.cfg.Trading.PaperMode {
					// Best-effort emergency close — use a fresh context since engineCtx is cancelled.
					exitCtx, exitCancel := context.WithTimeout(context.Background(), 5*time.Second)

					// Cancel the resting maker TP FIRST. Otherwise shutdown leaves an
					// orphan sell on the book that can later fill into an unmanaged
					// short with nothing watching it. If the cancel reports the TP
					// already executed (race), the position is already flat at TP
					// price — skip the crossing exit so we don't double-close.
					tpStatus, tpErr := router.CancelRestingTP(exitCtx, openPosition)
					if tpErr != nil {
						zlog.Warn().Err(tpErr).
							Str("order_id", openPosition.RestingTPOrderID).
							Msg("emergency resting TP cancel failed — placing crossing exit anyway")
					}

					if tpStatus == "executed" {
						exitCancel()
						g.jsonLog.Emit("emergency_exit", g.gameID, map[string]interface{}{
							"direction":    openPosition.Direction,
							"entry_ticker": openPosition.EntryTicker,
							"exit_price":   openPosition.RestingTPPrice,
							"sent":         true,
							"reason":       "tp_filled_during_cancel",
						})
					} else {
						// MarketSnapshot.Features[0]=YesBid, Features[1]=YesAsk (cents as float32).
						snap := ringBuffer.Snapshot()
						yesBid := int(snap.Features[0])
						yesAsk := int(snap.Features[1])
						emergencyPrice := yesBid
						if openPosition.Direction == "NO" {
							emergencyPrice = 100 - yesAsk
							if yesAsk == 0 {
								emergencyPrice = 100 - yesBid
							}
						}
						// Shutdown: cross more aggressively to guarantee we're flat.
						emergencyBudget := g.cfg.Agent.ExitSlippageBudgetCents + 3
						ok := router.PlaceExit(exitCtx, openPosition, emergencyPrice, emergencyBudget)
						exitCancel()
						g.jsonLog.Emit("emergency_exit", g.gameID, map[string]interface{}{
							"direction":    openPosition.Direction,
							"entry_ticker": openPosition.EntryTicker,
							"exit_price":   emergencyPrice,
							"sent":         ok,
						})
					}
				}
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

			g.jsonLog.Emit("game_end", g.gameID, map[string]interface{}{
				"possession_count":  possCount,
				"signal_count":      signalCount,
				"positions_opened":  positionsOpened,
				"positions_closed":  positionsClosed,
				"wins":              wins,
				"net_pnl_dollars":   netPnL,
				"win_rate":          winRate,
				"avg_pipeline_ms":   avgPipeline,
				"total_pipeline_ms": totalPipelineMS,
			})

			endCtx, endCancel := context.WithTimeout(context.Background(), 5*time.Second)
			_ = inference.EndGame(endCtx, g.gameID)
			endCancel()

			return
		}
	}
}
