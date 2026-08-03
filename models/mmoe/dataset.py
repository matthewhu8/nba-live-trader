"""
MMoE Unified Dataset Builder.

Loads and prepares the two training datasets:
  Dataset A (basketball-only): ~390K rows — possession_flat train split
    Features: X_physics + X_pregame (0-filled for 2024-25). No X_market.
    Targets:  target_run (Head A), hazard_0..9 (Head C). No trajectory targets.

  Dataset B (joint): ~24K rows — possessions with Kalshi tick data
    Features: X_physics + X_pregame + X_market (all 83 features)
    Targets:  target_run, hazard_0..9, traj_0..9 (all three heads)

Train/val splits:
  Basketball: Train = 2024-25 full + 2025-26 Oct-Jan,  Val = 2025-26 Feb-Mar5
  Head B:     Train = Mar 23 – Apr 6 2026 (~80%),       Val = Apr 7+ 2026 (~20%)

Usage:
    from models.mmoe.dataset import build_dataloaders
    train_loader, val_loader = build_dataloaders()
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.mmoe.feature_config import ALL_FEATURE_COLS, MARKET_COLS, PHYSICS_COLS, PREGAME_COLS
from models.targets.exit_simulator import build_trajectory_targets
from models.targets.kalshi_targets import add_hazard_targets

logger = logging.getLogger(__name__)

# ── Split boundaries ────────────────────────────────────────────────────────

# Basketball splits (keyed on game_date from dim_games)
BBALL_TRAIN_END  = pd.Timestamp("2026-01-31")
BBALL_VAL_END    = pd.Timestamp("2026-03-05")
# Test: Mar 6+ — sacred, untouched during training

# Head B (joint market) splits: 80/20 within tick-data window
HEADB_SPLIT_DATE = pd.Timestamp("2026-04-07")  # ~80/20 within 148-game window

# Feed delay: time from IRL game event to when we can act on it.
# NBA CDN polling:  ~17s CDN delay + ~1.5s avg poll wait = 20s
# Sportradar WS:    ~2-5s broadcast delay only           =  5s
# Used to shift the market feature lookup so training reflects live entry conditions.
FEED_DELAY_SECONDS_NBA        = 20
FEED_DELAY_SECONDS_SPORTRADAR =  5

# ── Column lists ────────────────────────────────────────────────────────────

TRAJ_COLS  = [f"traj_{i}" for i in range(10)]
HAZ_COLS   = [f"haz_{i}"  for i in range(10)]
TARGET_RUN = "target_meaningful_run_5_scoring"

PREGAME_TABLE_COLS = [c for c in PREGAME_COLS if c != "has_pregame_data"]

BOOL_COLS = [
    "home_scoring_sustainable", "away_scoring_sustainable",
    "q4_close_game",
    "home_in_bonus", "away_in_bonus",
    "home_star_in_foul_trouble", "away_star_in_foul_trouble",
    "home_star_on_court", "away_star_on_court",
    "was_foul", "was_sub",
    "had_shooting_foul", "had_personal_foul",
    "home_called_timeout_in_last_3_poss", "away_called_timeout_in_last_3_poss",
    "home_lineup_just_changed", "away_lineup_just_changed",
]

_MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def _connect_motherduck() -> duckdb.DuckDBPyConnection:
    load_dotenv()
    token = os.environ.get("MOTHERDUCK_TOKEN", "")
    if not token:
        raise RuntimeError("MOTHERDUCK_TOKEN not set in environment")
    return duckdb.connect(f"md:kalshi_trading?motherduck_token={token}")


def _load_possession_flat(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load all possessions joined with game_date for split logic."""
    df = conn.execute("""
        SELECT
            pf.*,
            dg.game_date,
            dg.home_team,
            dg.away_team
        FROM kalshi_trading.features.possession_flat pf
        JOIN kalshi_trading.main.dim_games dg ON pf.game_id = dg.game_id
        ORDER BY pf.game_id, pf.event_id
    """).df()
    logger.info("Loaded %d possession rows", len(df))
    return df


def _load_pregame(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load pre-computed pregame context (2025-26 season only)."""
    df = conn.execute("""
        SELECT game_id, team_net_rating_delta,
               home_off_rating, away_off_rating,
               home_def_rating, away_def_rating,
               roster_rapm_gap, missing_rapm_impact,
               rest_advantage, expected_pace, form_delta
        FROM kalshi_trading.features.pregame
    """).df()
    logger.info("Loaded %d pregame rows", len(df))
    return df


def _load_kalshi_ticks(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load all Kalshi ticks with full column set."""
    df = conn.execute("""
        SELECT market_ticker, ts, yes_bid, yes_ask, yes_last, volume, open_interest
        FROM kalshi_trading.main.kalshi_ticks
        ORDER BY ts
    """).df()
    logger.info("Loaded %d Kalshi tick rows", len(df))
    return df


# ── Derived features ──────────────────────────────────────────────────────────

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features derived from raw possession_flat columns."""
    out = df.copy()

    elapsed_in_period = (720 - out["game_clock_secs"].clip(0, 720)) / 60
    out["minutes_into_game"] = (out["period"] - 1) * 12 + elapsed_in_period

    for col in ("home_team_fouls_q", "away_team_fouls_q", "home_timeouts_used", "away_timeouts_used"):
        out[col] = out[col].fillna(0)

    out["home_in_bonus"]          = (out["home_team_fouls_q"] >= 5).astype(int)
    out["away_in_bonus"]          = (out["away_team_fouls_q"] >= 5).astype(int)
    out["home_fouls_until_bonus"] = (5 - out["home_team_fouls_q"]).clip(lower=0)
    out["away_fouls_until_bonus"] = (5 - out["away_team_fouls_q"]).clip(lower=0)

    out["home_full_timeouts_remaining"] = (4 - out["home_timeouts_used"]).clip(lower=0)
    out["away_full_timeouts_remaining"] = (4 - out["away_timeouts_used"]).clip(lower=0)

    out["current_run_team_encoded"] = (
        out["current_run_team"].map({"home": 1, "away": -1}).fillna(0)
    )
    out["home_star_in_foul_trouble"] = (out["home_trouble_star_tier"] > 0).astype(float)
    out["away_star_in_foul_trouble"] = (out["away_trouble_star_tier"] > 0).astype(float)

    return out


def _join_pregame(possessions: pd.DataFrame, pregame: pd.DataFrame) -> pd.DataFrame:
    """
    LEFT JOIN pregame features onto possessions by game_id.
    Rows without a pregame entry (2024-25 season) get 0-filled features
    and has_pregame_data=0.
    """
    merged = possessions.merge(pregame, on="game_id", how="left")

    for col in PREGAME_TABLE_COLS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
        else:
            merged[col] = 0.0

    merged["has_pregame_data"] = merged["team_net_rating_delta"].notna().astype(float)
    # Re-fill after marking (team_net_rating_delta was already filled above)
    merged["has_pregame_data"] = (
        possessions["game_id"].isin(pregame["game_id"]).astype(float).values
    )

    logger.info(
        "Pregame join: %d rows have pregame data, %d do not",
        merged["has_pregame_data"].sum(),
        (merged["has_pregame_data"] == 0).sum(),
    )
    return merged


# ── Ticker parsing + tick selection ──────────────────────────────────────────

def _parse_ticker(ticker: str) -> dict:
    parts = ticker.split("-")
    if len(parts) != 3:
        return {}
    middle, suffix = parts[1], parts[2]
    year  = "20" + middle[:2]
    month = _MONTH_MAP.get(middle[2:5], "01")
    day   = middle[5:7]
    teams = middle[7:]
    if len(teams) < 6 or not suffix[-1].isdigit():
        return {}
    return {
        "game_date_str": f"{year}-{month}-{day}",
        "away_team":     teams[:3],
        "home_team":     teams[3:6],
        "contract_team": suffix[:-1],
        "spread_val":    int(suffix[-1]),
    }


def _select_home_best_contract(
    ticks: pd.DataFrame,
    possessions: pd.DataFrame,
) -> pd.DataFrame:
    """Select the lowest-spread home-team contract per game and assign game_id."""
    parsed = ticks["market_ticker"].apply(_parse_ticker).apply(pd.Series)
    ticks = pd.concat([ticks, parsed], axis=1)
    ticks = ticks.dropna(subset=["home_team", "contract_team", "spread_val"])
    ticks = ticks[ticks["contract_team"] == ticks["home_team"]].copy()

    game_lookup = (
        possessions[["game_id", "game_date", "home_team", "away_team"]]
        .drop_duplicates("game_id")
        .assign(game_date=lambda d: pd.to_datetime(d["game_date"]).dt.strftime("%Y-%m-%d"))
    )
    ticks = ticks.merge(
        game_lookup,
        left_on=["game_date_str", "home_team", "away_team"],
        right_on=["game_date", "home_team", "away_team"],
        how="left",
    ).drop(columns=["game_date", "game_date_str"], errors="ignore")

    unmatched = ticks["game_id"].isna().sum()
    if unmatched > 0:
        logger.warning("%d ticks could not be matched to a game_id", unmatched)
    ticks = ticks.dropna(subset=["game_id"])

    min_spread = ticks.groupby("game_id")["spread_val"].min()
    ticks = ticks[ticks.apply(lambda r: r["spread_val"] == min_spread[r["game_id"]], axis=1)]

    logger.info("Selected %d home-contract ticks across %d games", len(ticks), ticks["game_id"].nunique())
    return ticks


# ── Enriched market features ──────────────────────────────────────────────────

def _compute_market_features_for_game(game_ticks: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all X_market rolling/velocity features over the tick time series
    for a single game. Operates on a time-indexed DataFrame.

    Returns the same rows with market feature columns added.
    """
    out = game_ticks.set_index("ts").sort_index().copy()

    # Spread
    out["spread"] = out["yes_ask"] - out["yes_bid"]

    # Trade volume 60s
    out["trade_volume_60s"] = out["volume"].rolling("60s").sum().fillna(0)

    # Time since last trade (ms)
    trade_mask = out["volume"] > 0
    last_trade_ts = out.index.to_series().where(trade_mask).ffill()
    last_trade_ts = last_trade_ts.fillna(out.index[0])
    out["time_since_last_trade_ms"] = (out.index - last_trade_ts).dt.total_seconds() * 1000

    # Open interest change over last 60s
    oi_60s_ago = (
        out["open_interest"]
        .rolling("60s", min_periods=1)
        .apply(lambda x: x.iloc[0], raw=False)
    )
    out["open_interest_change_60s"] = out["open_interest"] - oi_60s_ago

    # Bid velocity: (current - 30s_ago) / 30
    bid_30s_ago = (
        out["yes_bid"]
        .rolling("30s", min_periods=1)
        .apply(lambda x: x.iloc[0], raw=False)
    )
    out["bid_velocity_30s"] = (out["yes_bid"] - bid_30s_ago) / 30.0

    # Bid acceleration: velocity_now - velocity_30s_ago
    vel_30s_ago = (
        out["bid_velocity_30s"]
        .rolling("30s", min_periods=1)
        .apply(lambda x: x.iloc[0], raw=False)
    )
    out["bid_acceleration_30s"] = out["bid_velocity_30s"] - vel_30s_ago

    # Bid vs last divergence (stale order detection)
    yes_last_filled = out["yes_last"].ffill().fillna(out["yes_bid"])
    out["bid_vs_last_divergence"] = out["yes_bid"] - yes_last_filled

    out["has_market_data"] = 1.0

    return out.reset_index()


def validate_possession_tick_overlap(
    possessions: pd.DataFrame,
    ticks: pd.DataFrame,
    tolerance_minutes: float = 5.0,
    min_coverage_pct: float = 15.0,
    raise_on_fail: bool = False,
) -> pd.DataFrame:
    """
    Assert that each game's possession times actually overlap its Kalshi ticks.

    Why this exists: pd.merge_asof(direction="backward") NEVER fails. If a game's
    wall_clock_ts is wrong — e.g. off by a day — every possession silently matches
    the LAST recorded tick, which for a finished market is the settled price
    (bid=1 or 99). Those rows look populated, carry has_market_data=1, and then get
    quietly discarded downstream by the 30-70c band, so the corruption is invisible
    in aggregate metrics while roughly halving the effective sample.

    That is exactly what happened: 36 of 71 games had wall_clock_ts one day late
    (games tipping after 20:00 ET, i.e. crossing 00:00 UTC), and it went unnoticed
    through a model retrain and a full backtest sweep.

    The invariant checked is that the FIRST possession falls inside the recording
    window: tick_start - tolerance <= poss_start <= tick_end.

    Deliberately not a lead-time window and not full containment — both were tried
    against real data and both produced false failures:

      * "first possession within +-60 min of first tick" fails whenever Kalshi opens
        a market early, which it routinely does (a Finals market opened 3 days ahead;
        many open the prior evening). 10 of 85 games failed this way.
      * "possession window fully inside tick window" fails whenever the recorder
        stops before the final buzzer, which is normal once a market settles — 56 of
        83 games ran 5-57 min past their last tick.

    Anchoring on poss_start still catches both real corruptions, because both move
    the start out of the window: a game written one day late starts long after the
    ticks end, and a game whose later periods lost a day starts before they begin.

    A start-of-game check alone is not sufficient, so `tick_coverage_pct` (the share
    of the possession window the recorder actually covered) is a second criterion.
    When only the LATER periods of a game carry a wrong date, the minimum timestamp
    can still land inside the tick window and slip past a start-only check, while
    most of the game sits a day away. Measured on the real corrupt data the split is
    unambiguous: corrupt games score 0.0-2.7% coverage, legitimate games 26.6-100%
    (median 90.4%), so the 15% default separates them with roughly 10x margin.

    Returns a per-game report; callers should log it and act on `ok`.
    """
    poss = possessions.dropna(subset=["wall_clock_ts"]).copy()
    poss["wall_clock_ts"] = pd.to_datetime(poss["wall_clock_ts"], utc=True)
    tk = ticks.copy()
    tk["ts"] = pd.to_datetime(tk["ts"], utc=True)

    p_agg = poss.groupby("game_id")["wall_clock_ts"].agg(poss_start="min", poss_end="max")
    t_agg = tk.groupby("game_id")["ts"].agg(tick_start="min", tick_end="max")
    rep = p_agg.join(t_agg, how="inner").reset_index()
    if rep.empty:
        return rep

    tol = pd.Timedelta(minutes=tolerance_minutes)
    rep["lead_minutes"] = (rep["poss_start"] - rep["tick_start"]).dt.total_seconds() / 60.0

    # Minutes the first possession sits outside the recording window, either side.
    before = ((rep["tick_start"] - tol) - rep["poss_start"]).dt.total_seconds() / 60.0
    after = (rep["poss_start"] - rep["tick_end"]).dt.total_seconds() / 60.0
    # No leading underscore: DataFrame.itertuples() renames such columns.
    rep["start_outside_min"] = pd.concat([before, after], axis=1).max(axis=1).clip(lower=0)

    # Informational: how much of the game the recorder actually covered.
    overlap = (
        rep[["poss_end", "tick_end"]].min(axis=1) - rep[["poss_start", "tick_start"]].max(axis=1)
    ).dt.total_seconds() / 60.0
    span = (rep["poss_end"] - rep["poss_start"]).dt.total_seconds() / 60.0
    rep["tick_coverage_pct"] = (overlap.clip(lower=0) / span.where(span > 0)) * 100.0

    rep["start_ok"] = (
        (rep["poss_start"] >= rep["tick_start"] - tol)
        & (rep["poss_start"] <= rep["tick_end"])
    )
    rep["coverage_ok"] = rep["tick_coverage_pct"] >= min_coverage_pct
    rep["ok"] = rep["start_ok"] & rep["coverage_ok"]

    n_bad = int((~rep["ok"]).sum())
    if n_bad:
        worst = rep.loc[~rep["ok"]].nsmallest(5, "tick_coverage_pct", keep="all").head(5)
        n_start = int((~rep["start_ok"]).sum())
        n_cov = int((~rep["coverage_ok"]).sum())
        msg = (
            f"{n_bad} of {len(rep)} games do not line up with their tick recording "
            f"window — the asof join will silently return settled end-of-game prices "
            f"for these ({n_start} first possession outside the window, "
            f"{n_cov} below {min_coverage_pct:.0f}% tick coverage). "
            f"Worst (game_id, coverage): "
            + ", ".join(f"({r.game_id}, {r.tick_coverage_pct:.1f}%)" for r in worst.itertuples())
        )
        if raise_on_fail:
            raise ValueError(msg)
        logger.error(msg)
    else:
        logger.info(
            "Tick window OK for all %d games (first possession %.0f..%.0f min after "
            "first tick — wide is normal, markets open early; median tick coverage %.0f%%)",
            len(rep), rep["lead_minutes"].min(), rep["lead_minutes"].max(),
            rep["tick_coverage_pct"].median(),
        )
    return rep


def _join_ticks_to_possessions(
    possessions: pd.DataFrame,
    ticks: pd.DataFrame,
    feed_delay_seconds: int = FEED_DELAY_SECONDS_NBA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backward ASOF join: for each possession, attach the most recent Kalshi tick
    at or before (wall_clock_ts + feed_delay_seconds). The delay simulates the
    real-time lag between when a game event occurs and when we can act on it,
    ensuring training uses the same market snapshot we'd observe in live trading.

    Returns:
        (joint_df, basketball_only_df)
        joint_df has yes_bid populated; basketball_only_df does not.
    """
    possessions = possessions.copy()
    possessions["wall_clock_ts"] = pd.to_datetime(possessions["wall_clock_ts"], utc=True)
    ticks["ts"] = pd.to_datetime(ticks["ts"], utc=True)

    # Shift lookup window forward by feed delay to match live entry conditions
    delay = pd.Timedelta(seconds=feed_delay_seconds)
    possessions["_join_ts"] = possessions["wall_clock_ts"] + delay

    # Fail loudly on wrong-day / non-overlapping timestamps before the asof join
    # silently substitutes settled prices for them.
    validate_possession_tick_overlap(possessions, ticks)

    joint_parts: list[pd.DataFrame] = []
    games_with_ticks = set(ticks["game_id"].unique())

    for game_id, poss_game in possessions.groupby("game_id"):
        poss_game = poss_game.sort_values("_join_ts").copy()

        if game_id not in games_with_ticks:
            continue

        tick_game = ticks[ticks["game_id"] == game_id].sort_values("ts")
        tick_game_feat = _compute_market_features_for_game(tick_game)
        tick_game_feat["ts"] = pd.to_datetime(tick_game_feat["ts"], utc=True)

        merged = pd.merge_asof(
            poss_game,
            tick_game_feat[[
                "ts", "yes_bid", "yes_ask", "yes_last", "open_interest",
                "spread", "trade_volume_60s", "time_since_last_trade_ms",
                "open_interest_change_60s", "bid_velocity_30s",
                "bid_acceleration_30s", "bid_vs_last_divergence", "has_market_data",
            ]],
            left_on="_join_ts",
            right_on="ts",
            direction="backward",
        ).drop(columns=["ts", "_join_ts"], errors="ignore")

        # Possession-level d_yes_bid and d_spread (within game)
        merged = merged.sort_values("event_id").copy()
        merged["d_yes_bid"] = merged["yes_bid"].diff().fillna(0)
        merged["d_spread"]  = merged["spread"].diff().fillna(0)

        # Only keep possessions that have a tick (yes_bid populated)
        merged = merged.dropna(subset=["yes_bid"])
        if not merged.empty:
            joint_parts.append(merged)

    basketball_only = possessions[~possessions["game_id"].isin(games_with_ticks)].copy()

    joint_df = pd.concat(joint_parts, ignore_index=True) if joint_parts else pd.DataFrame()
    logger.info(
        "Joint rows: %d across %d games; basketball-only rows: %d",
        len(joint_df), joint_df["game_id"].nunique() if not joint_df.empty else 0,
        len(basketball_only),
    )
    return joint_df, basketball_only


# ── Target generation ─────────────────────────────────────────────────────────

def _add_run_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add target_meaningful_run_5_scoring from targets.py, per game."""
    from models.features.targets import add_targets

    parts = []
    for _, game_df in df.groupby("game_id", sort=False):
        parts.append(add_targets(game_df.reset_index(drop=True)))
    return pd.concat(parts, ignore_index=True)


def _add_hazard_targets_all_games(df: pd.DataFrame) -> pd.DataFrame:
    """Add haz_0..9 columns for all rows, per game."""
    parts = []
    for _, game_df in df.groupby("game_id", sort=False):
        parts.append(add_hazard_targets(game_df.sort_values("event_id").reset_index(drop=True)))
    return pd.concat(parts, ignore_index=True)


# ── Feature matrix preparation ───────────────────────────────────────────────

def _prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Cast booleans, fill missing, return numeric matrix aligned to ALL_FEATURE_COLS."""
    out = df.copy()

    for col in BOOL_COLS:
        if col in out.columns:
            out[col] = out[col].astype(float)

    available = [c for c in ALL_FEATURE_COLS if c in out.columns]
    missing   = [c for c in ALL_FEATURE_COLS if c not in out.columns]
    if missing:
        logger.warning("Features missing from data (filling with 0): %s", missing)

    result = out[available].copy()
    for col in missing:
        result[col] = 0.0

    return result[ALL_FEATURE_COLS].fillna(0).astype(float)


# ── Train/val split logic ─────────────────────────────────────────────────────

def _split_basketball(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split basketball-only rows into train/val by game_date.
    Train: up to Jan 31 2026 (includes full 2024-25 season)
    Val:   Feb 1 – Mar 5 2026
    Test:  Mar 6+ — not returned (excluded entirely)
    """
    game_date = pd.to_datetime(df["game_date"])
    train_mask = game_date <= BBALL_TRAIN_END
    val_mask   = (game_date > BBALL_TRAIN_END) & (game_date <= BBALL_VAL_END)
    return df[train_mask].copy(), df[val_mask].copy()


def _split_joint(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split joint rows 80/20 chronologically within the tick-data window.
    Train: Mar 23 – Apr 6 2026
    Val:   Apr 7+ 2026
    The test set for Head B is future tick data not yet recorded.
    """
    game_date = pd.to_datetime(df["game_date"])
    train_mask = game_date < HEADB_SPLIT_DATE
    val_mask   = game_date >= HEADB_SPLIT_DATE
    return df[train_mask].copy(), df[val_mask].copy()


# ── DataLoader construction ───────────────────────────────────────────────────

@dataclass
class MMoEBatch:
    """Container for target tensors with masks."""
    X:                torch.Tensor  # (N, 83) float32
    target_run:       torch.Tensor  # (N,)    float32  — binary
    target_trajectory: torch.Tensor # (N, 10) float32  — logit deltas (NaN for non-joint)
    target_hazard:    torch.Tensor  # (N, 10) float32  — binary survival
    has_market_data:  torch.Tensor  # (N,)    bool     — mask for Head B loss
    has_run_target:   torch.Tensor  # (N,)    bool     — mask for Heads A/C loss


def _build_tensor_dataset(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False,
) -> tuple[TensorDataset, Optional[StandardScaler]]:
    """
    Convert a merged (basketball + joint) DataFrame into a TensorDataset.
    If fit_scaler=True, fits a new StandardScaler on this data.
    """
    X = _prepare_feature_matrix(df)

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values)

        # Re-fit market feature statistics using only joint rows (has_market_data == 1).
        #
        # The global fit is distorted: 94% of training rows have market features
        # zero-filled (no Kalshi data). This pulls yes_bid's mean to ~3¢ and std to
        # ~12, so at inference (always has_market_data=1) a typical yes_bid of 50¢
        # produces a z-score of +3.9 — consistently out-of-distribution for the model.
        #
        # Fix: recompute mean/std for the 13 price/liquidity market features from
        # joint rows only, so that live inference values are centered near z=0.
        # has_market_data itself (index 82, last column) is intentionally excluded:
        # its global stats (mean≈0.06, std≈0.24) let the model distinguish live rows
        # (z≈+4) from basketball-only rows (z≈-0.25), which is a useful signal.
        market_start = len(PHYSICS_COLS) + len(PREGAME_COLS)          # 69
        market_end   = market_start + len(MARKET_COLS) - 1            # 81 (excl. has_market_data)

        joint_mask = X["has_market_data"].values > 0
        n_joint = int(joint_mask.sum())
        if n_joint >= 100:
            old_bid_mean  = float(scaler.mean_[market_start])
            old_bid_scale = float(scaler.scale_[market_start])

            aux = StandardScaler().fit(X.values[joint_mask, market_start:market_end])
            scaler.mean_[market_start:market_end]  = aux.mean_
            scaler.scale_[market_start:market_end] = aux.scale_

            # Re-transform the already-scaled market columns with corrected stats.
            X_scaled[:, market_start:market_end] = (
                (X.values[:, market_start:market_end] - aux.mean_) / aux.scale_
            )
            logger.info(
                "Market scaler refitted on %d joint rows. "
                "yes_bid: mean %.1f→%.1f¢, std %.1f→%.1f",
                n_joint,
                old_bid_mean, float(aux.mean_[0]),
                old_bid_scale, float(aux.scale_[0]),
            )
        else:
            logger.warning(
                "Too few joint rows (%d) to refit market scaler — using global stats",
                n_joint,
            )
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit_scaler=False")
        X_scaled = scaler.transform(X.values)

    X_t = torch.tensor(X_scaled, dtype=torch.float32)

    # Target A: run prediction
    run_vals = df[TARGET_RUN].values if TARGET_RUN in df.columns else np.full(len(df), np.nan)
    has_run = ~np.isnan(run_vals.astype(float))
    run_vals = np.where(np.isnan(run_vals.astype(float)), 0.0, run_vals.astype(float))
    target_run_t = torch.tensor(run_vals, dtype=torch.float32)
    has_run_t    = torch.tensor(has_run,  dtype=torch.bool)

    # Target B: trajectory (NaN for basketball-only rows)
    traj_arr = np.full((len(df), 10), np.nan)
    for i, col in enumerate(TRAJ_COLS):
        if col in df.columns:
            traj_arr[:, i] = df[col].values.astype(float)
    target_traj_t = torch.tensor(np.nan_to_num(traj_arr, nan=0.0), dtype=torch.float32)

    # has_market_data mask (whether trajectory targets are valid)
    has_market = (df["has_market_data"].fillna(0).values > 0) if "has_market_data" in df.columns else np.zeros(len(df), dtype=bool)
    has_market_t = torch.tensor(has_market, dtype=torch.bool)

    # Target C: hazard (10 horizons)
    haz_arr = np.zeros((len(df), 10), dtype=float)
    for i, col in enumerate(HAZ_COLS):
        if col in df.columns:
            haz_arr[:, i] = df[col].fillna(1.0).values.astype(float)
        else:
            haz_arr[:, i] = 1.0  # Default: run broken (safe/conservative)
    target_haz_t = torch.tensor(haz_arr, dtype=torch.float32)

    dataset = TensorDataset(
        X_t, target_run_t, target_traj_t, target_haz_t,
        has_market_t.float(), has_run_t.float(),
    )
    return dataset, scaler


def build_dataloaders(
    batch_size: int = 512,
    tp: float = 5.0,
    sl: float = 3.0,
    num_workers: int = 0,
    feed_delay_seconds: int = FEED_DELAY_SECONDS_NBA,
) -> tuple[DataLoader, DataLoader, StandardScaler]:
    """
    Full data pipeline: load → feature engineer → targets → split → DataLoaders.

    Returns:
        train_loader: combined basketball + joint train data
        val_loader:   combined basketball + joint val data
        scaler:       fitted StandardScaler (save alongside model weights)
    """
    conn = _connect_motherduck()

    possessions = _load_possession_flat(conn)
    pregame     = _load_pregame(conn)
    ticks_raw   = _load_kalshi_ticks(conn)
    conn.close()

    # Basketball feature engineering
    possessions = _add_derived_features(possessions)
    possessions = _join_pregame(possessions, pregame)

    # Tick parsing + home-contract selection
    ticks = _select_home_best_contract(ticks_raw, possessions)

    logger.info("Feed delay: %ds (market features look up tick at wall_clock_ts + %ds)", feed_delay_seconds, feed_delay_seconds)
    # Split possessions into joint (has ticks) and basketball-only
    joint_df, bball_only = _join_ticks_to_possessions(possessions, ticks, feed_delay_seconds=feed_delay_seconds)

    # --- Targets ---
    # Run target + hazard for all rows (both datasets)
    logger.info("Computing run targets and hazard targets for basketball-only rows...")
    bball_only = _add_run_target(bball_only)
    bball_only = _add_hazard_targets_all_games(bball_only)

    if not joint_df.empty:
        logger.info("Computing run targets and hazard targets for joint rows...")
        joint_df = _add_run_target(joint_df)
        joint_df = _add_hazard_targets_all_games(joint_df)

        logger.info("Computing trajectory targets (exit simulator) for joint rows...")
        # For exit simulator we also need the full possession history per game
        # (to detect momentum flips and garbage time)
        joint_df = build_trajectory_targets(
            entry_rows=joint_df,
            all_ticks=ticks,
            all_possessions=possessions,
            tp=tp,
            sl=sl,
        )

        # Ensure market feature columns exist on bball_only (filled with 0)
        for col in MARKET_COLS:
            if col not in bball_only.columns:
                bball_only[col] = 0.0
        bball_only["has_market_data"] = 0.0

        for col in TRAJ_COLS:
            if col not in bball_only.columns:
                bball_only[col] = np.nan

    # --- Train/val splits ---
    bball_train, bball_val = _split_basketball(bball_only)
    joint_train, joint_val = _split_joint(joint_df) if not joint_df.empty else (pd.DataFrame(), pd.DataFrame())

    train_df = pd.concat([bball_train, joint_train], ignore_index=True) if not joint_train.empty else bball_train
    val_df   = pd.concat([bball_val,   joint_val],   ignore_index=True) if not joint_val.empty   else bball_val

    logger.info(
        "Train: %d rows (%d basketball, %d joint) | Val: %d rows (%d basketball, %d joint)",
        len(train_df), len(bball_train), len(joint_train),
        len(val_df),   len(bball_val),   len(joint_val),
    )

    # --- DataLoaders ---
    train_dataset, scaler = _build_tensor_dataset(train_df, fit_scaler=True)
    val_dataset,   _      = _build_tensor_dataset(val_df,   scaler=scaler, fit_scaler=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, scaler
