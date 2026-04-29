package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	gameID := flag.String("game", "", "NBA game ID (e.g. 0042500121)")
	ticker := flag.String("ticker", "", "Kalshi market ticker (e.g. KXNBASPREAD-...)")
	testMode := flag.Bool("test", false, "run for 30s then exit")
	flag.Parse()

	if *gameID == "" {
		log.Fatal("--game is required (e.g. --game 0042500121)")
	}
	if *ticker == "" {
		log.Fatal("--ticker is required (e.g. --ticker KXNBASPREAD-...)")
	}

	cfg, err := LoadConfig("config/trading.yaml")
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	if *testMode {
		var tc context.CancelFunc
		ctx, tc = context.WithTimeout(ctx, 30*time.Second)
		defer tc()
	}

	client := NewInferenceClient(cfg.Inference.BaseURL)
	router := NewRouter("", cfg.Trading.PaperMode)
	ks := NewKillSwitch()
	ledger := NewLedger(RiskConfig{
		MaxTotalExposureCents:   cfg.Risk.MaxTotalExposureCents,
		MaxPerGameExposureCents: cfg.Risk.MaxPerGameExposureCents,
		MaxDailyLossCents:       cfg.Risk.MaxDailyLossCents,
		MaxContractsPerOrder:    cfg.Risk.MaxContractsPerOrder,
	}, ks)
	logger := NewLogger(cfg.Trading.PaperMode, cfg.Observability.RedisStream)
	bandit := NewBandit(cfg)

	zlog.Info().
		Str("event", "startup").
		Str("game", *gameID).
		Str("ticker", *ticker).
		Bool("paper_mode", cfg.Trading.PaperMode).
		Msg("[STARTUP]")

	homeID, awayID, err := fetchTeamIDs(ctx, *gameID)
	if err != nil {
		log.Fatalf("failed to fetch team IDs for game %s: %v", *gameID, err)
	}

	zlog.Info().
		Str("event", "startup").
		Int64("home_team_id", homeID).
		Int64("away_team_id", awayID).
		Msg("[STARTUP]")

	if err := client.StartGame(ctx, *gameID, *ticker, homeID, awayID); err != nil {
		log.Fatalf("failed to start game on inference service: %v", err)
	}

	zlog.Info().
		Str("event", "startup").
		Str("inference_url", cfg.Inference.BaseURL).
		Msg("[STARTUP] inference service connected")

	engine := NewGameEngine(*gameID, *ticker, client, router, ledger, ks, bandit, logger, cfg)
	engine.Run(ctx)

	if err := client.EndGame(context.Background(), *gameID); err != nil {
		zlog.Warn().Str("game", *gameID).Err(err).Msg("end game cleanup failed (non-fatal)")
	}

	zlog.Info().Msg("shutdown complete")
}
