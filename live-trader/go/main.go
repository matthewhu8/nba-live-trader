// Entry point for the live trading engine.
//
// Wires up the full pipeline: NBA feed → Python inference → Agent → Paper trades.
// Runs as a single-game engine. Ctrl+C for graceful shutdown + game summary.
//
// Usage:
//
//	go run . --game 0022501234 --market KXNBAGAME-26MAY06PHINYK
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
)

func main() {
	gameID := flag.String("game", "", "NBA game ID (e.g. 0042500212)")
	marketTicker := flag.String("market", "", "Kalshi market ticker (e.g. KXNBAGAME-26MAY06PHINYK)")
	configPath := flag.String("config", "", "Path to trading.yaml (default: auto-detect)")
	flag.Parse()

	if *gameID == "" {
		log.Fatal("usage: go run . --game <game_id> --market <market_ticker>")
	}

	// ── Load config ──────────────────────────────────────────────────────
	cfgFile := *configPath
	if cfgFile == "" {
		// Try relative to the go directory, then the project root
		candidates := []string{
			"../config/trading.yaml",
			"../../live-trader/config/trading.yaml",
		}
		for _, c := range candidates {
			if _, err := os.Stat(c); err == nil {
				cfgFile = c
				break
			}
		}
	}

	var cfg Config
	if cfgFile != "" {
		loaded, err := LoadConfig(cfgFile)
		if err != nil {
			log.Printf("[WARN] could not load config %s: %v — using defaults", cfgFile, err)
			cfg = defaultConfig()
		} else {
			cfg = *loaded
			log.Printf("[CONFIG] loaded from %s (paper_mode=%v)", cfgFile, cfg.Trading.PaperMode)
		}
	} else {
		cfg = defaultConfig()
		log.Printf("[CONFIG] no config file found — using defaults (paper_mode=true)")
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	// ── Fetch team IDs from NBA CDN ─────────────────────────────────────
	log.Printf("[INIT] fetching team IDs for game %s from NBA CDN...", *gameID)
	homeID, awayID, err := fetchTeamIDs(ctx, *gameID)
	if err != nil {
		log.Printf("[WARN] could not fetch team IDs (game may not have started): %v", err)
		log.Printf("[WARN] continuing without team IDs — pregame fallback will handle it")
	} else {
		log.Printf("[INIT] homeTeamID=%d  awayTeamID=%d", homeID, awayID)
	}

	// ── Create inference client and start game ──────────────────────────
	inferenceURL := cfg.Inference.BaseURL
	if inferenceURL == "" {
		inferenceURL = "http://localhost:8001"
	}
	inference := NewInferenceClient(inferenceURL)

	log.Printf("[INIT] calling Python /game/%s/start ...", *gameID)
	if err := inference.StartGame(ctx, *gameID, *marketTicker, homeID, awayID); err != nil {
		log.Printf("[WARN] StartGame failed: %v — inference will return zeros", err)
	} else {
		log.Printf("[INIT] Python game session started")
	}

	// ── Create components ───────────────────────────────────────────────
	ringBuffer := NewRingBuffer()
	bandit := NewBandit(&cfg)
	router := NewRouter(os.Getenv("KALSHI_KEY_ID"), cfg.Trading.PaperMode)
	logger := NewLogger(cfg.Trading.PaperMode, cfg.Observability.RedisStream)
	logger.OpenTradeLog(*gameID)

	// ── Start feeds ─────────────────────────────────────────────────────
	possessionCh := make(chan NBAEvent, 500)
	tickCh := make(chan KalshiTick, 50000)

	nbaFeed := NewNBAFeed(*gameID)
	go nbaFeed.Run(ctx, possessionCh)

	if *marketTicker != "" {
		kalshiFeed := NewKalshiFeed(*marketTicker, "")
		go kalshiFeed.Run(ctx, tickCh)
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
		log.Printf("[FEED] Kalshi WebSocket started for %s", *marketTicker)
	} else {
		log.Printf("[FEED] No market ticker — running without Kalshi data")
	}
	log.Printf("[FEED] NBA CDN poller started for game %s (3s interval)", *gameID)

	// ── Paper trading state ─────────────────────────────────────────────
	var openPosition *PaperPosition
	var possCount int
	var signalCount int
	var positionsOpened int
	var positionsClosed int
	var netPnL float64
	var wins int
	var totalPipelineMS int64

	// ── Main event loop ─────────────────────────────────────────────────
	log.Println("──────────────────────────────────────────────────────")
	log.Println("  PAPER TRADING ENGINE RUNNING")
	log.Printf("  Game: %s  Market: %s", *gameID, *marketTicker)
	log.Printf("  Paper mode: %v  Entry threshold: %.2f", cfg.Trading.PaperMode, cfg.Agent.MinRunProbEntry)
	log.Println("  Press Ctrl+C to stop and see game summary")
	log.Println("──────────────────────────────────────────────────────")

	for {
		select {
		case event := <-possessionCh:
			start := time.Now()
			possCount++

			// Get market snapshot
			snap := ringBuffer.Snapshot()

			// Call Python inference
			resp, err := inference.ProcessPossession(ctx, *gameID, event, snap)
			if err != nil {
				log.Printf("[ERR] inference failed poss=%d: %v", possCount, err)
				continue
			}

			totalPipelineMS += resp.PipelineMS

			// Agent decision
			hasPosition := openPosition != nil
			action := bandit.Decide(resp, hasPosition)

			// Override action with model's action if it's more specific
			if resp.Action != "WAIT" && action == Wait {
				// Python said something, but our thresholds disagree — log it
			}

			// Log every possession
			riskOK := true // simplified — risk ledger not enforced yet
			logger.EmitPossession(*gameID, event, resp, riskOK, start)

			// ── Paper trade logic ─────────────────────────────
			if openPosition != nil {
				// Check exit conditions
				shouldExit, reason, pnl := router.CheckExit(openPosition, resp, possCount, &cfg)

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
					logger.EmitExit(*gameID, reason, openPosition, currentPrice, pnl, possHeld)
					go inference.ReportTrade(*gameID, TradePayload{
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
					logger.EmitHold(*gameID, openPosition, resp, possHeld)
				}
			} else if action == BuyYes || action == BuyNo {
				// Enter new position
				direction := "YES"
				if action == BuyNo {
					direction = "NO"
				}
				pos := router.Place(*gameID, resp, possCount, &cfg, direction)
				if pos != nil {
					openPosition = pos
					signalCount++
					positionsOpened++
					logger.EmitEntry(*gameID, pos, resp)
					go inference.ReportTrade(*gameID, TradePayload{
						Action:    "ENTRY",
						Direction: pos.Direction,
						Price:     pos.EntryPrice,
						Size:      pos.Size,
						PnL:       0,
						Reason:    "SIGNAL",
					})
				}
			}

			// Garbage time detection (log once)
			if resp.IsGarbageTime {
				logger.EmitGarbageTime(*gameID, resp)
			}

		case <-ctx.Done():
			// ── Shutdown + Game Summary ──────────────────────
			// Close any open position at last known price
			if openPosition != nil {
				positionsClosed++
				// No exit price available — mark as scratch
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
				GameID:          *gameID,
				PossCount:       possCount,
				AvgPipelineMS:   avgPipeline,
				SignalCount:     signalCount,
				PositionsOpened: positionsOpened,
				PositionsClosed: positionsClosed,
				NetPnLDollars:   netPnL,
				WinRate:         winRate,
			})

			// Best-effort game end
			endCtx, endCancel := context.WithTimeout(context.Background(), 5*time.Second)
			_ = inference.EndGame(endCtx, *gameID)
			endCancel()

			return
		}
	}
}

func defaultConfig() Config {
	return Config{
		Trading: struct {
			PaperMode bool `yaml:"paper_mode"`
		}{PaperMode: true},
		Inference: struct {
			BaseURL   string `yaml:"base_url"`
			TimeoutMS int    `yaml:"timeout_ms"`
		}{BaseURL: "http://localhost:8001", TimeoutMS: 500},
		Agent: struct {
			MinYesBid             int     `yaml:"min_yes_bid"`
			MaxYesBid             int     `yaml:"max_yes_bid"`
			MinRunProbEntry       float32 `yaml:"min_run_prob_entry"`
			TakeProfitCents       int     `yaml:"take_profit_cents"`
			StopLossCents         int     `yaml:"stop_loss_cents"`
			MaxHoldPossessions    int     `yaml:"max_hold_possessions"`
			PositionSizeContracts int     `yaml:"position_size_contracts"`
		}{
			MinYesBid: 30, MaxYesBid: 70, MinRunProbEntry: 0.15,
			TakeProfitCents: 8, StopLossCents: 5, MaxHoldPossessions: 6,
			PositionSizeContracts: 100,
		},
		Feeds: struct {
			NBAPollIntervalMS      int `yaml:"nba_poll_interval_ms"`
			KalshiStaleThresholdMS int `yaml:"kalshi_stale_threshold_ms"`
		}{NBAPollIntervalMS: 3000, KalshiStaleThresholdMS: 30000},
	}
}

// logDir returns the directory for paper trade logs, creating it if needed.
func logDir() string {
	dir := filepath.Join("logs", "paper_trades")
	os.MkdirAll(dir, 0o755)
	return dir
}
