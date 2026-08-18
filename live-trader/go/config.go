package main

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Trading struct {
		PaperMode bool `yaml:"paper_mode"`
	} `yaml:"trading"`
	Inference struct {
		BaseURL   string `yaml:"base_url"`
		TimeoutMS int    `yaml:"timeout_ms"`
		// Model artifacts, relative to the repo root. Python reloads only when
		// these change; empty leaves Python on its default.
		ModelPath  string `yaml:"model_path"`
		ScalerPath string `yaml:"scaler_path"`
	} `yaml:"inference"`
	Risk struct {
		MaxTotalExposureCents   int `yaml:"max_total_exposure_cents"`
		MaxPerGameExposureCents int `yaml:"max_per_game_exposure_cents"`
		MaxDailyLossCents       int `yaml:"max_daily_loss_cents"`
		MaxContractsPerOrder    int `yaml:"max_contracts_per_order"`
	} `yaml:"risk"`
	Agent struct {
		MinYesBid         int     `yaml:"min_yes_bid"`
		MaxYesBid         int     `yaml:"max_yes_bid"`
		MinRunProbEntry   float32 `yaml:"min_run_prob_entry"`   // Head A gate; 0.0 turns it off
		MinAbsTrajEntry   float32 `yaml:"min_abs_traj_entry"`   // |traj_used| >= this
		MinRunLengthEntry int     `yaml:"min_run_length_entry"` // current_run_length >= this
		// TrajAggregator reduces the Head B trajectory to one signed scalar for
		// gating and sizing: "final", "mean", "mean_3_to_9" or "max_abs". Must
		// match `aggregate_traj` in backtesting/mmoe_backtest.py.
		TrajAggregator        string `yaml:"traj_aggregator"`
		TakeProfitCents       int    `yaml:"take_profit_cents"`
		StopLossCents         int    `yaml:"stop_loss_cents"`
		MaxHoldPossessions    int    `yaml:"max_hold_possessions"`
		PositionSizeContracts int    `yaml:"position_size_contracts"`
		// Kelly sizing, capped at Risk.MaxContractsPerOrder:
		// contracts = max(KellyMinContracts, (|traj_used| - KellyAnchorTraj) * KellySlope)
		KellyAnchorTraj   float32 `yaml:"kelly_anchor_traj"`
		KellySlope        float32 `yaml:"kelly_slope"`
		KellyMinContracts int     `yaml:"kelly_min_contracts"`
		// Band the pinned market's bid must stay inside. Deliberately wider than
		// the entry band: holding the same logical bet at a lopsided price beats
		// swapping to a sibling contract and flipping what the trajectory sign means.
		MarketDriftLowBid  int `yaml:"market_drift_low_bid"`
		MarketDriftHighBid int `yaml:"market_drift_high_bid"`
		// How many cents through the best bid a crossing exit may reach. Exits cross
		// because a post-only stop cannot fill when price is running away.
		ExitSlippageBudgetCents int `yaml:"exit_slippage_budget_cents"`
		// Trailing take-profit. Disabled when TrailGivebackCents <= 0, which leaves
		// the flat TakeProfitCents in charge. When enabled, a position that reaches
		// TrailActivateCents of profit rides until it gives back TrailGivebackCents
		// from its peak. Validate in paper before enabling live.
		TrailActivateCents int `yaml:"trail_activate_cents"`
		TrailGivebackCents int `yaml:"trail_giveback_cents"`
		// Thresholds for the skip-trading gate, forwarded to Python each possession.
		// Distinct from the `garbage_time_risk` model feature, which is frozen at its
		// training values in features.py. Tuning these moves the gate, not the model.
		BlowoutMarginPts     int `yaml:"blowout_margin_pts"`      // |score_diff| > this
		GarbageTimePeriod    int `yaml:"garbage_time_period"`     // this period or later
		GarbageTimeClockSecs int `yaml:"garbage_time_clock_secs"` // and game_clock_secs below this
	} `yaml:"agent"`
	Feeds struct {
		NBAPollIntervalMS      int `yaml:"nba_poll_interval_ms"`
		KalshiStaleThresholdMS int `yaml:"kalshi_stale_threshold_ms"`
	} `yaml:"feeds"`
	Observability struct {
		RedisStream string `yaml:"redis_stream"`
		LogLevel    string `yaml:"log_level"`
	} `yaml:"observability"`
}

func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config %q: %w", path, err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config %q: %w", path, err)
	}
	return &cfg, nil
}
