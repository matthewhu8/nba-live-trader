"""
MMoE Feature Configuration.

Single source of truth for all feature lists used across dataset, model, and predictor.
Total: 83 features = 58 physics + 11 pregame (10 table cols + 1 flag) + 14 market.
"""

# --- X_physics: 58 features (from run_predictor.py FEATURE_COLS) ---
PHYSICS_COLS: list[str] = [
    # Score × time
    "score_diff",
    "period",
    "minutes_into_game",
    "trailing_team_urgency",
    "comeback_probability_proxy",
    "q4_close_game",
    "garbage_time_risk",
    # Shot info
    "shot_value",
    "shot_distance",
    # Momentum
    "home_points_last_5_poss",
    "away_points_last_5_poss",
    "home_points_last_10_poss",
    "away_points_last_10_poss",
    "current_run_team_encoded",
    "current_run_length",
    "current_run_points",
    "current_run_3pt_pct",
    "current_run_paint_pct",
    # Pace
    "pace_last_10_possessions",
    "pace_season_baseline",
    # Shot quality
    "home_scoring_sustainable",
    "away_scoring_sustainable",
    "home_xPPP_last_5",
    "away_xPPP_last_5",
    "home_actual_vs_expected_PPP",
    "away_actual_vs_expected_PPP",
    "home_shot_quality_trend",
    "away_shot_quality_trend",
    # Foul state
    "home_team_fouls_q",
    "away_team_fouls_q",
    "home_cum_fouls",
    "away_cum_fouls",
    "home_in_bonus",
    "away_in_bonus",
    "home_fouls_until_bonus",
    "away_fouls_until_bonus",
    "home_star_in_foul_trouble",
    "away_star_in_foul_trouble",
    "home_star_on_court",
    "away_star_on_court",
    # Event context
    "was_foul",
    "was_sub",
    "had_shooting_foul",
    "had_personal_foul",
    "home_sub_count",
    "away_sub_count",
    # Timeout signals
    "possessions_since_last_timeout",
    "home_called_timeout_in_last_3_poss",
    "away_called_timeout_in_last_3_poss",
    "home_full_timeouts_remaining",
    "away_full_timeouts_remaining",
    # Lineup signal
    "home_lineup_net_rating",
    "away_lineup_net_rating",
    "lineup_net_rating_delta",
    "home_lineup_sample_size",
    "away_lineup_sample_size",
    "home_lineup_just_changed",
    "away_lineup_just_changed",
]

# --- X_pregame: 10 features (from features.pregame, joined by game_id) ---
# Coverage: 2025-26 season only (1,065 games). 2024-25 rows filled with 0.
# has_pregame_data=0 signals to the model that pregame features are absent.
PREGAME_COLS: list[str] = [
    "team_net_rating_delta",     # home net rating minus away (std≈5.2)
    "home_off_rating",           # EWMA offensive rating entering game (std≈2.2)
    "away_off_rating",
    "home_def_rating",           # EWMA defensive rating entering game
    "away_def_rating",
    "roster_rapm_gap",           # RAPM-based home-minus-away quality gap (std≈0.39)
    "missing_rapm_impact",       # impact of missing/resting players (std≈13.9)
    "rest_advantage",            # rest-days differential, home minus away (std≈1.18)
    "expected_pace",             # expected pace in secs/possession (std≈0.18)
    "form_delta",                # recent form differential (std≈9.9)
    "has_pregame_data",          # binary: 1 for 2025-26 rows, 0 for 2024-25
]

# --- X_market: 14 features (computed from main.kalshi_ticks) ---
# Coverage: 148 games (Mar 23 – Apr 12, 2026). All other rows filled with 0.
# has_market_data=0 signals to the model that market features are absent.
MARKET_COLS: list[str] = [
    # Raw LOB state (5)
    "yes_bid",
    "yes_ask",
    "spread",                    # yes_ask - yes_bid
    "yes_last",                  # last traded price
    "open_interest",
    # Liquidity regime (3)
    "trade_volume_60s",          # rolling 60s sum of volume
    "time_since_last_trade_ms",  # ms since last volume > 0 tick
    "open_interest_change_60s",  # delta in open_interest over last 60s
    # Price velocity (4)
    "d_yes_bid",                 # change from previous possession's tick
    "d_spread",                  # change in spread from previous possession's tick
    "bid_velocity_30s",          # (current_bid - bid_30s_ago) / 30
    "bid_acceleration_30s",      # velocity_now - velocity_30s_ago
    # Market-vs-physics divergence (2)
    "bid_vs_last_divergence",    # yes_bid - yes_last (stale order detection)
    "has_market_data",           # binary: 1 if tick data present
]

ALL_FEATURE_COLS: list[str] = PHYSICS_COLS + PREGAME_COLS + MARKET_COLS

assert len(PHYSICS_COLS) == 58,       f"Expected 58 physics cols, got {len(PHYSICS_COLS)}"
assert len(PREGAME_COLS) == 11,       f"Expected 11 pregame cols, got {len(PREGAME_COLS)}"
assert len(MARKET_COLS)  == 14,       f"Expected 14 market cols, got {len(MARKET_COLS)}"
assert len(ALL_FEATURE_COLS) == 83,   f"Expected 83 total cols, got {len(ALL_FEATURE_COLS)}"
