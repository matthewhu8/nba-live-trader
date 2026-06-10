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
		// Model artifact paths (relative to repo root). Forwarded to the Python
		// inference service on /game/start; Python reloads only if these differ
		// from the currently-loaded predictor. Empty = Python keeps its default.
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
		MinYesBid             int     `yaml:"min_yes_bid"`
		MaxYesBid             int     `yaml:"max_yes_bid"`
		// MinRunProbEntry is retained for backwards compatibility with older
		// YAML files but is no longer consulted by the bandit — the live
		// entry config was aligned to the validated backtest, which gates on
		// trajectory magnitude + run length instead of run probability (see
		// MMoEPredictor gate-collapse finding, 2026-05-09 post-mortem).
		MinRunProbEntry       float32 `yaml:"min_run_prob_entry"`
		MinAbsTrajEntry       float32 `yaml:"min_abs_traj_entry"`   // |traj_used| ≥ this  (Phase 1 winner: 0.08 with `mean` aggregator)
		MinRunLengthEntry     int     `yaml:"min_run_length_entry"` // current_run_length ≥ this  (backtest: 2)
		// TrajAggregator reduces the 10-element Head B trajectory to one signed scalar
		// for entry gating + sizing. Must match the Python helper in
		// backtesting/mmoe_backtest.py (`aggregate_traj`). Valid: "final", "mean",
		// "mean_3_to_9", "max_abs". Phase 1 sweep winner: "mean" — best net P&L
		// (+$1,500 vs raw trajectory[9]) and higher win rate (+6.5pt).
		TrajAggregator string `yaml:"traj_aggregator"`
		TakeProfitCents       int     `yaml:"take_profit_cents"`
		StopLossCents         int     `yaml:"stop_loss_cents"`
		MaxHoldPossessions    int     `yaml:"max_hold_possessions"`
		PositionSizeContracts int     `yaml:"position_size_contracts"`
		// Kelly sizing — contracts = max(KellyMinContracts, (|traj_used| - KellyAnchorTraj) * KellySlope),
		// capped at Risk.MaxContractsPerOrder. Anchor is the empirical floor of the
		// chosen aggregator's distribution (median |traj_used| from Phase 1 winner CSV).
		KellyAnchorTraj  float32 `yaml:"kelly_anchor_traj"`
		KellySlope       float32 `yaml:"kelly_slope"`
		KellyMinContracts int    `yaml:"kelly_min_contracts"`
		// 2026-05-19 pin-the-market: the scanner locks on to the initial closest-
		// to-50¢ market and only swaps if the LOCKED market's bid drifts outside
		// [MarketDriftLowBid, MarketDriftHighBid]. Wider than the entry band on
		// purpose — we want to keep the same logical bet (and thus the same
		// trajectory-sign meaning) even when the price drifts to lopsided.
		MarketDriftLowBid  int `yaml:"market_drift_low_bid"`
		MarketDriftHighBid int `yaml:"market_drift_high_bid"`
		// Exit execution. Exits CROSS the book (post_only=false) so a stop-loss
		// can actually fill when price is running away — a post-only exit is
		// structurally un-fillable in exactly that case (2026-05-28 OKC@SAS
		// post-mortem: 175 rejected post-only exits turned a 3¢ stop into -12¢).
		// ExitSlippageBudgetCents bounds how many cents through the best bid we
		// cross; the realized fill is at the best available price up to that.
		ExitSlippageBudgetCents int `yaml:"exit_slippage_budget_cents"`
		// Trailing take-profit (dynamic exit). Disabled when TrailGivebackCents<=0,
		// in which case the validated flat TakeProfitCents governs. When enabled,
		// once profit reaches TrailActivateCents the position rides until it gives
		// back TrailGivebackCents from its peak — letting a strong move run instead
		// of capping at a fixed +TakeProfitCents. Validate in paper before live.
		TrailActivateCents int `yaml:"trail_activate_cents"`
		TrailGivebackCents int `yaml:"trail_giveback_cents"`
		// Garbage-time / blowout GATE thresholds (skip-trading). Forwarded to
		// Python per-possession; they drive the is_garbage_time / is_blowout
		// flags the agent uses to refuse trades. These are DISTINCT from the
		// frozen `garbage_time_risk` MODEL feature (fixed at training values of
		// 30/4/360 in features.py) — tuning these moves the trade gate, not the
		// model input. Zero = Python falls back to its 30/4/360 defaults.
		BlowoutMarginPts     int `yaml:"blowout_margin_pts"`     // |score_diff| > this → is_blowout
		GarbageTimePeriod    int `yaml:"garbage_time_period"`    // garbage time only in this period (or later)
		GarbageTimeClockSecs int `yaml:"garbage_time_clock_secs"` // ...and game_clock_secs < this
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
