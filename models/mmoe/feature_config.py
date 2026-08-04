"""
MMoE Feature Configuration.

Single source of truth for all feature lists used across dataset, model, and predictor.
Total: 58 features = 33 physics + 11 pregame (10 table cols + 1 flag) + 14 market.

The physics block was consolidated from 58 columns to 33. Three structural problems
drove it, all of which cost the model capacity it does not have at 24K joint rows:

  1. Direction and magnitude were split across separate columns in five places
     (score_diff vs comeback_probability_proxy, current_run_team_encoded vs
     current_run_points, and three home/away level pairs), forcing the MLP to learn
     products it learns badly. Each is now one signed quantity.
  2. Seven columns were exact algebraic functions of other columns
     (lineup_net_rating_delta's own operands, in_bonus == fouls_until_bonus == 0,
     was_foul == the OR of two present columns).
  3. Two columns were destroyed by their own encoding: a 999 sentinel that
     compressed the real 0-70 range into a 0.24-wide z band under StandardScaler,
     and np.sign() quantization that discarded shot-quality magnitude.

Every derived value below is computed by exactly one function in
models/features/transforms.py, shared by the offline builder and live inference.
"""

# --- X_physics: 33 features ---
PHYSICS_COLS: list[str] = [
    # Score x time (4) — was 7
    # lead_z = score_diff / sqrt(minutes_remaining) replaces trailing_team_urgency
    # and comeback_probability_proxy, both of which discarded the sign.
    # time_leverage is a sqrt-scaled clock, replacing linear minutes_into_game and
    # the q4_close_game threshold.
    "score_diff",
    "lead_z",
    "time_leverage",
    "garbage_time_risk",
    # Shot info (2) — unchanged
    "shot_value",
    "shot_distance",
    # Runs (3) — was 5
    "run_signed_points",
    "run_efficiency",
    "run_fragility",
    # Recent scoring (2) — was 4
    "swing_5",
    "swing_accel",
    # Pace (2) — restructured, shrinks toward the pregame expected_pace prior
    "pace_ref",
    "pace_surprise",
    # Shot quality (3) — was 8
    "xppp_edge",
    "luck_edge",
    "quality_trend_edge",
    # Foul state (5) — was 12
    # fouls_until_bonus stays unfolded: the gate at 0 is nonlinear and asymmetric,
    # since being in the bonus helps the opponent.
    "team_foul_edge",
    "home_fouls_until_bonus",
    "away_fouls_until_bonus",
    "star_trouble_edge",
    "star_on_court_edge",
    # Event context (4) — was 6
    "was_sub",
    "had_shooting_foul",
    "had_personal_foul",
    "sub_count_edge",
    # Timeouts (4) — was 5; the 999 sentinel is split into a bounded count + flag
    "poss_since_timeout",
    "no_timeout_yet",
    "timeout_called_edge",
    "timeouts_remaining_edge",
    # Lineup (4) — was 7
    "lineup_net_rating_delta",
    "lineup_confidence",
    "lineup_changed_edge",
    "lineup_changed_any",
]

# --- X_pregame: 11 features (from features.pregame, joined by game_id) ---
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
# These stay unfolded and are compressed by a learned nn.Linear(14, 4) encoder in
# MMoEModel rather than by hand: unlike the basketball physics, we have no strong
# prior about how LOB microstructure should combine.
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

# Index range of the market block inside ALL_FEATURE_COLS. The model slices here to
# route the market features through their encoder, and dataset.py uses it to refit
# the market scaler on joint rows only.
MARKET_START: int = len(PHYSICS_COLS) + len(PREGAME_COLS)
MARKET_END: int = MARKET_START + len(MARKET_COLS)

assert len(PHYSICS_COLS) == 33,       f"Expected 33 physics cols, got {len(PHYSICS_COLS)}"
assert len(PREGAME_COLS) == 11,       f"Expected 11 pregame cols, got {len(PREGAME_COLS)}"
assert len(MARKET_COLS)  == 14,       f"Expected 14 market cols, got {len(MARKET_COLS)}"
assert len(ALL_FEATURE_COLS) == 58,   f"Expected 58 total cols, got {len(ALL_FEATURE_COLS)}"
assert MARKET_END == len(ALL_FEATURE_COLS), "Market block must be the tail of the vector"
