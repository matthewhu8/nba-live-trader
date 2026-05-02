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
		MinRunProbEntry       float32 `yaml:"min_run_prob_entry"`
		TakeProfitCents       int     `yaml:"take_profit_cents"`
		StopLossCents         int     `yaml:"stop_loss_cents"`
		MaxHoldPossessions    int     `yaml:"max_hold_possessions"`
		PositionSizeContracts int     `yaml:"position_size_contracts"`
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
