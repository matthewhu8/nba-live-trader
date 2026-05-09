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
		engine := NewGameEngine(*gameID, *eventTicker, cfg, ledger, ks)
		engine.Run(ctx)
	} else {
		// Coordinator mode
		log.Printf("[MAIN] Starting Coordinator mode. Will auto-detect and manage today's games.")
		coordinator := NewCoordinator(cfg, ledger, ks)
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
			TakeProfitCents       int     `yaml:"take_profit_cents"`
			StopLossCents         int     `yaml:"stop_loss_cents"`
			MaxHoldPossessions    int     `yaml:"max_hold_possessions"`
			PositionSizeContracts int     `yaml:"position_size_contracts"`
		}{
			MinYesBid: 30, MaxYesBid: 70, MinRunProbEntry: 0.10,
			TakeProfitCents: 8, StopLossCents: 5, MaxHoldPossessions: 6,
			PositionSizeContracts: 100,
		},
		Feeds: struct {
			NBAPollIntervalMS      int `yaml:"nba_poll_interval_ms"`
			KalshiStaleThresholdMS int `yaml:"kalshi_stale_threshold_ms"`
		}{NBAPollIntervalMS: 3000, KalshiStaleThresholdMS: 30000},
	}
}
