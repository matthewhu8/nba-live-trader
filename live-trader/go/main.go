package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	gameID := flag.String("game", "", "Optional: Specific NBA game ID to trade (e.g. 0042500212)")
	eventTicker := flag.String("event", "", "Optional: Kalshi base event ticker (e.g. KXNBASPREAD-26MAY06MINSAS)")
	configPath := flag.String("config", "", "Path to trading.yaml (default: auto-detect)")
	flag.Parse()

	// ── Load .env ────────────────────────────────────────────────────────
	if loadedEnv, err := LoadEnvCandidates(".env", "../.env", "../../.env"); err != nil {
		log.Printf("[WARN] could not load .env from project root: %v", err)
	} else {
		log.Printf("[ENV] loaded %s", loadedEnv)
	}

	// ── Load config ──────────────────────────────────────────────────────
	cfgFile := *configPath
	if cfgFile == "" {
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

	// ── Run identity ─────────────────────────────────────────────────────
	// Every process invocation gets a Run with its own logs/runs/{date}/{id}/
	// directory. Manifest is written immediately; live-trader.jsonl will
	// receive run_start / run_end here in Phase 1, and richer events in
	// later phases. All failures here are non-fatal — the trading engine
	// runs even if structured logging cannot start.
	run, runErr := NewRun(cfg.Trading.PaperMode)
	if runErr != nil {
		log.Printf("[WARN] could not create run dir: %v — continuing without structured logging", runErr)
	} else {
		// Capture zerolog warnings/errors into stderr.log in the run dir.
		// MultiWriter keeps terminal output unchanged. Best-effort.
		if stderrFile := run.CaptureStderr(); stderrFile != nil {
			defer stderrFile.Close()
		}
		run.WriteManifest(cfg)
		log.Printf("[RUN] id=%s dir=%s", run.ID, run.LogDir)
	}

	var jsonLog *JSONLogger
	if run != nil {
		var jlErr error
		jsonLog, jlErr = NewJSONLogger(run)
		if jlErr != nil {
			log.Printf("[WARN] could not open jsonl: %v — continuing without structured logging", jlErr)
		}
	}

	// Defers run LIFO. Order matters: emit run_end + finalize manifest
	// FIRST (registered last so it runs first), then close the JSONL file.
	defer jsonLog.Close()
	defer func() {
		const endReason = "ctx_cancel"
		jsonLog.Emit("run_end", "", map[string]interface{}{"end_reason": endReason})
		run.FinalizeManifest(endReason)
	}()

	jsonLog.Emit("run_start", "", map[string]interface{}{
		"paper_mode": cfg.Trading.PaperMode,
		"git_sha":    readGitSHA(),
		"pid":        os.Getpid(),
	})

	ks := NewKillSwitch()
	ledger := NewLedger(RiskConfig{
		MaxTotalExposureCents:   cfg.Risk.MaxTotalExposureCents,
		MaxPerGameExposureCents: cfg.Risk.MaxPerGameExposureCents,
		MaxDailyLossCents:       cfg.Risk.MaxDailyLossCents,
		MaxContractsPerOrder:    cfg.Risk.MaxContractsPerOrder,
	}, ks)

	if *gameID != "" && *eventTicker != "" {
		// Single game mode
		log.Printf("[MAIN] Starting in Single-Game mode for %s (%s)", *gameID, *eventTicker)
		engine := NewGameEngine(*gameID, *eventTicker, cfg, ledger, ks, run, jsonLog)
		engine.Run(ctx)
	} else {
		// Coordinator mode
		log.Printf("[MAIN] Starting Coordinator mode. Will auto-detect and manage today's games.")
		coordinator := NewCoordinator(cfg, ledger, ks, run, jsonLog)
		if err := coordinator.Run(ctx); err != nil {
			log.Fatalf("Coordinator error: %v", err)
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
			MinAbsTrajEntry       float32 `yaml:"min_abs_traj_entry"`
			MinRunLengthEntry     int     `yaml:"min_run_length_entry"`
			TakeProfitCents       int     `yaml:"take_profit_cents"`
			StopLossCents         int     `yaml:"stop_loss_cents"`
			MaxHoldPossessions    int     `yaml:"max_hold_possessions"`
			PositionSizeContracts int     `yaml:"position_size_contracts"`
			MarketDriftLowBid     int     `yaml:"market_drift_low_bid"`
			MarketDriftHighBid    int     `yaml:"market_drift_high_bid"`
		}{
			MinYesBid: 30, MaxYesBid: 70, MinRunProbEntry: 0.0,
			MinAbsTrajEntry: 0.08, MinRunLengthEntry: 2,
			TakeProfitCents: 5, StopLossCents: 3, MaxHoldPossessions: 6,
			PositionSizeContracts: 100,
			MarketDriftLowBid: 20, MarketDriftHighBid: 80,
		},
		Feeds: struct {
			NBAPollIntervalMS      int `yaml:"nba_poll_interval_ms"`
			KalshiStaleThresholdMS int `yaml:"kalshi_stale_threshold_ms"`
		}{NBAPollIntervalMS: 3000, KalshiStaleThresholdMS: 30000},
	}
}
