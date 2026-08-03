"""
MMoE Backtest — Validate Edge on Head B Val Set (Apr 7–12, 2026).

Replays each val game possession-by-possession, applies the MMoE signal filter,
simulates entries/exits using the same TP/SL/momentum logic as training, and
computes net PnL after maker fees.

Key question: does total_net_pnl > 0 across val games?

Usage:
    python -m backtesting.mmoe_backtest
    python -m backtesting.mmoe_backtest --threshold 0.12 --tp 6 --sl 3
"""

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mmoe.dataset import (
    FEED_DELAY_SECONDS_NBA,
    HEADB_SPLIT_DATE,
    BOOL_COLS,
    _add_derived_features,
    _connect_motherduck,
    _join_pregame,
    _load_kalshi_ticks,
    _load_possession_flat,
    _load_pregame,
    _compute_market_features_for_game,
    _select_home_best_contract,
)
from models.mmoe.feature_config import ALL_FEATURE_COLS, PHYSICS_COLS, PREGAME_COLS, MARKET_COLS
from models.mmoe.predictor import MMoEPredictor
from models.targets.exit_simulator import simulate_exit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

_CACHE_DIR     = Path("data/feature_store")
_POSS_CACHE    = _CACHE_DIR / "possession_flat.parquet"
_TICKS_CACHE   = _CACHE_DIR / "kalshi_ticks.parquet"
_PREGAME_CACHE = _CACHE_DIR / "pregame.parquet"


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three source tables, preferring the local parquet cache.

    A full MotherDuck scan of possession_flat + kalshi_ticks exhausted the
    free-tier daily compute limit, so the cache is the default path. Rebuild it
    with data/ingestion/export_backtest_cache.py.
    """
    if _POSS_CACHE.exists() and _TICKS_CACHE.exists() and _PREGAME_CACHE.exists():
        logger.info("Loading from local parquet cache...")
        poss    = pd.read_parquet(_POSS_CACHE)
        ticks   = pd.read_parquet(_TICKS_CACHE)
        pregame = pd.read_parquet(_PREGAME_CACHE)
        logger.info("  possession_flat: %d rows / %d games | ticks: %d rows | pregame: %d rows",
                    len(poss), poss["game_id"].nunique(), len(ticks), len(pregame))
        return poss, ticks, pregame
    logger.info("Local cache not found — connecting to MotherDuck...")
    conn    = _connect_motherduck()
    poss    = _load_possession_flat(conn)
    ticks   = _load_kalshi_ticks(conn)
    pregame = _load_pregame(conn)
    conn.close()
    return poss, ticks, pregame


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ClosedPosition:
    game_id:       str
    possession_id: int
    wall_clock_ts: pd.Timestamp
    entry_price:   float
    exit_price:    float
    exit_reason:   str
    entry_side:    int          # +1 BUY_YES | -1 BUY_NO
    hold_time_s:   float
    gross_pnl:     float        # entry_side * (exit_price - entry_price)
    net_pnl:       float        # gross_pnl * contracts - fees
    run_prob:        float
    traj_final:      float      # trajectory[-1] at entry (legacy column, kept for backward compat)
    traj_used:       float      # aggregated trajectory actually consulted for entry gate
    traj_aggregator: str        # which aggregator produced traj_used ("final"/"mean"/"mean_3_to_9"/"max_abs")
    hazard_final:    float      # hazard[-1] at entry
    quarter:         int
    score_diff:      int
    run_length:      int
    fees:            float = 0.0   # round-trip fees actually charged (maker entry + reason-based exit)
    entry_tick_age_s: float = 0.0  # how stale the entry tick was vs wct + feed_delay_s


TRAJ_AGGREGATORS = ("final", "mean", "mean_3_to_9", "max_abs")


def aggregate_traj(trajectory: list[float], mode: str) -> float:
    """
    Reduce the 10-element Head B trajectory to a single signed scalar for entry gating.

    Modes:
      - "final"        : trajectory[-1]                      (current production behavior)
      - "mean"         : mean of all 10 horizons             (lowest variance, smooths Huber noise)
      - "mean_3_to_9"  : mean of horizons 3-9                (drops the very-short-term noisy heads)
      - "max_abs"      : the element with largest |.|        (peak conviction across horizons)

    For "max_abs" the sign is preserved from the source element so downstream
    direction inference (sign(traj_final)) still works.
    """
    if mode == "final":
        return float(trajectory[-1])
    if mode == "mean":
        return float(sum(trajectory) / len(trajectory))
    if mode == "mean_3_to_9":
        window = trajectory[3:]
        return float(sum(window) / len(window))
    if mode == "max_abs":
        idx = max(range(len(trajectory)), key=lambda i: abs(trajectory[i]))
        return float(trajectory[idx])
    raise ValueError(f"unknown traj aggregator mode: {mode!r} (valid: {TRAJ_AGGREGATORS})")


@dataclass
class BacktestSummary:
    positions:       list[ClosedPosition]
    total_gross_pnl: float
    total_net_pnl:   float
    n_trades:        int
    win_rate:        float
    avg_hold_time_s: float
    exit_reasons:    dict[str, int]
    by_quarter:      dict[int, dict]
    by_score_bucket: dict[str, dict]
    by_run_length:   dict[str, dict]


# ── Market feature helpers ───────────────────────────────────────────────────

def _tick_at_delay(
    enriched_game_ticks: pd.DataFrame,
    wall_clock_ts: pd.Timestamp,
    delay_s: int,
) -> tuple[Optional[pd.Series], pd.Timestamp]:
    """The tick a trader could actually act on for a possession ending at wall_clock_ts.

    Returns (tick, entry_anchor_ts) where entry_anchor_ts is wall_clock_ts + delay_s —
    the earliest moment the position can exist. The tick itself is the most recent one
    at or before that anchor (a backward asof, matching the training-time join), so it
    may be older than the anchor; `tick["ts"]` vs the anchor is the entry staleness.

    Single source of truth for the delay: the exit window must start at the anchor this
    returns, or the simulation can exit before it entered.
    """
    entry_anchor_ts = wall_clock_ts + pd.Timedelta(seconds=delay_s)
    candidates = enriched_game_ticks[enriched_game_ticks["ts"] <= entry_anchor_ts]
    if candidates.empty:
        return None, entry_anchor_ts
    return candidates.iloc[-1], entry_anchor_ts


def _get_market_features_at_delay(
    enriched_game_ticks: pd.DataFrame,
    wall_clock_ts: pd.Timestamp,
    prev_yes_bid: float,
    prev_spread: float,
    delay_s: int = FEED_DELAY_SECONDS_NBA,
) -> dict[str, float]:
    """
    Replicates the backward asof join from dataset._join_ticks_to_possessions()
    for a single possession.

    enriched_game_ticks must already have market feature columns computed
    (output of _compute_market_features_for_game).

    prev_yes_bid and prev_spread are the values from the previous possession's tick,
    used to compute d_yes_bid and d_spread — matching the diff() logic in training.
    """
    tick, _ = _tick_at_delay(enriched_game_ticks, wall_clock_ts, delay_s)
    if tick is None:
        return {col: 0.0 for col in MARKET_COLS}

    current_bid    = float(tick["yes_bid"])
    current_spread = float(tick["spread"])

    return {
        "yes_bid":                   current_bid,
        "yes_ask":                   float(tick["yes_ask"]),
        "spread":                    current_spread,
        "yes_last":                  float(tick["yes_last"]) if pd.notna(tick["yes_last"]) else current_bid,
        "open_interest":             float(tick["open_interest"]) if pd.notna(tick["open_interest"]) else 0.0,
        "trade_volume_60s":          float(tick["trade_volume_60s"]),
        "time_since_last_trade_ms":  float(tick["time_since_last_trade_ms"]),
        "open_interest_change_60s":  float(tick["open_interest_change_60s"]),
        "d_yes_bid":                 current_bid    - prev_yes_bid,
        "d_spread":                  current_spread - prev_spread,
        "bid_velocity_30s":          float(tick["bid_velocity_30s"]),
        "bid_acceleration_30s":      float(tick["bid_acceleration_30s"]),
        "bid_vs_last_divergence":    float(tick["bid_vs_last_divergence"]),
        "has_market_data":           1.0,
    }


def _build_feature_dict(
    poss_row: pd.Series,
    market_features: dict[str, float],
) -> dict[str, float]:
    """
    Assemble 83-dim feature dict from a possession_flat row + market feature dict.
    Physics (58) + Pregame (11) come from poss_row; Market (14) from market_features.
    """
    fd: dict[str, float] = {}

    for col in PHYSICS_COLS + PREGAME_COLS:
        val = poss_row.get(col, 0.0)
        if pd.isna(val):
            val = 0.0
        fd[col] = float(val)

    fd.update(market_features)
    return fd


MAKER_FEE_RATE = 0.0175
TAKER_FEE_RATE = 0.07

# Exits that rest a limit order and therefore pay the maker rate. Every other exit
# reason crosses the book to get out now, so it pays taker — see CLAUDE.md Phase 6:
# resting maker take-profits (PR #50), stops cross the book.
_MAKER_EXIT_REASONS = frozenset({"take_profit"})


def _fee_one_leg(rate: float, price_cents: float, contracts: int) -> float:
    """Kalshi fee for a single leg: rate x C x P x (1-P), rounded up to the cent.

    The P(1-P) term is part of Kalshi's published formula; CLAUDE.md's fee table
    omits it in the expression but its worked example ("~$0.44 per 100 at 50c")
    assumes it. This matches live-trader/inference/dashboard.py.
    """
    p = price_cents / 100.0
    return math.ceil(rate * contracts * p * (1.0 - p) * 100.0) / 100.0


def _compute_fees(
    entry_price: float,
    exit_price: float,
    contracts: int,
    exit_reason: str,
) -> float:
    """Round-trip fees. Entry is always a resting limit (maker); the exit leg's rate
    depends on how we got out.

    Previously returned 0.0 unconditionally, which silently zeroed the taker cost on
    every stop-out — i.e. on the losers.
    """
    entry_fee = _fee_one_leg(MAKER_FEE_RATE, entry_price, contracts)
    exit_rate = MAKER_FEE_RATE if exit_reason in _MAKER_EXIT_REASONS else TAKER_FEE_RATE
    exit_fee  = _fee_one_leg(exit_rate, exit_price, contracts)
    return entry_fee + exit_fee


# ── Per-game replay ──────────────────────────────────────────────────────────

def _run_game(
    game_id: str,
    game_poss: pd.DataFrame,
    game_ticks: pd.DataFrame,
    predictor: MMoEPredictor,
    run_prob_threshold: float,
    tp: float,
    sl: float,
    contracts: int,
    feed_delay_s: int,
    use_traj_for_side: bool = False,
    min_abs_traj: float = 0.0,
    min_run_length: int = 1,
    hold_seconds: int = 120,
    traj_aggregator: str = "final",
) -> list[ClosedPosition]:
    """Replay one game and return all closed positions."""
    if game_ticks.empty:
        logger.warning("Game %s: no tick data, skipping", game_id)
        return []

    # Pre-compute all market features for the game once
    enriched_ticks = _compute_market_features_for_game(game_ticks.copy())
    enriched_ticks["ts"] = pd.to_datetime(enriched_ticks["ts"], utc=True)
    enriched_ticks = enriched_ticks.sort_values("ts").reset_index(drop=True)

    game_poss = game_poss.sort_values("event_id").reset_index(drop=True)
    game_poss["wall_clock_ts"] = pd.to_datetime(game_poss["wall_clock_ts"], utc=True)

    positions: list[ClosedPosition] = []
    prev_yes_bid: float = 50.0
    prev_spread:  float = 1.0
    position_exit_ts: Optional[pd.Timestamp] = None

    for _, row in game_poss.iterrows():
        wct: pd.Timestamp = row["wall_clock_ts"]

        # Skip garbage time
        if row.get("is_blowout", False) or row.get("is_garbage_time", False):
            continue

        # Get market features at wall_clock_ts + feed_delay_s (every possession)
        market_feats = _get_market_features_at_delay(
            enriched_game_ticks=enriched_ticks,
            wall_clock_ts=wct,
            prev_yes_bid=prev_yes_bid,
            prev_spread=prev_spread,
            delay_s=feed_delay_s,
        )
        yes_bid = market_feats["yes_bid"]
        if yes_bid > 0:
            prev_yes_bid = yes_bid
            prev_spread  = market_feats["spread"]

        # Skip if no market data or still inside a prior position's hold window
        if market_feats["has_market_data"] == 0.0:
            continue
        if position_exit_ts is not None and wct < position_exit_ts:
            continue

        # Run length pre-filter (before paying inference cost)
        if int(row.get("current_run_length", 0)) < min_run_length:
            continue

        # Build 83-dim feature vector and run inference
        fd = _build_feature_dict(row, market_feats)
        output = predictor.predict(fd)

        # Entry filter: Head A gate
        if output.run_prob < run_prob_threshold:
            continue
        if not (30 <= yes_bid <= 70):
            continue

        # `traj_final` (legacy column name) preserves trajectory[-1] for backward-compatible CSVs.
        # `traj_used` is the aggregated value the entry gate and direction logic actually consult.
        traj_final = float(output.trajectory[-1])
        traj_used = aggregate_traj(output.trajectory, traj_aggregator)

        # Head B directional confidence filter — applied to the aggregated value
        if abs(traj_used) < min_abs_traj:
            continue

        # Determine trade direction.
        # use_traj_for_side: let Head B prediction set direction (positive→BUY_YES, negative→BUY_NO)
        # Default: use basketball run_team_encoded (home run→BUY_YES, away run→BUY_NO)
        if use_traj_for_side:
            entry_side = 1 if traj_used >= 0 else -1
        else:
            run_team_encoded = fd.get("current_run_team_encoded", 0.0)
            entry_side = 1 if run_team_encoded >= 0 else -1

        # Simulate exit from the moment the position can actually exist.
        #
        # The entry price was read at wct + feed_delay_s, so the exit search must start
        # there too. Anchoring it at wct let a position exit up to feed_delay_s BEFORE
        # it entered, harvesting price movement that had already happened — which was
        # the entire measured "edge" (see CLAUDE.md).
        entry_tick, entry_anchor_ts = _tick_at_delay(enriched_ticks, wct, feed_delay_s)
        entry_tick_age_s = (
            (entry_anchor_ts - entry_tick["ts"]).total_seconds() if entry_tick is not None else 0.0
        )

        future_ticks = enriched_ticks[enriched_ticks["ts"] > entry_anchor_ts]
        future_poss  = game_poss[game_poss["wall_clock_ts"] > entry_anchor_ts]

        sim = simulate_exit(
            entry_wall_clock=entry_anchor_ts,
            entry_yes_bid=yes_bid,
            entry_run_team=row.get("current_run_team", None),
            future_ticks=future_ticks,
            future_possessions=future_poss,
            tp=tp,
            sl=sl,
            entry_side=entry_side,
            max_seconds=hold_seconds,
        )

        # Take-profits rest a maker limit at entry + TP (PR #50), so the fill cannot be
        # better than that limit. simulate_exit reports the price of the tick that
        # breached the threshold, which overshot the limit by up to 19c on this sample
        # and booked the overshoot as profit we could never have collected.
        exit_price = sim.exit_price
        if sim.exit_reason == "take_profit":
            exit_price = yes_bid + entry_side * tp

        # Invariant: a position cannot exit before it existed. Stated against wct (not
        # entry_anchor_ts) so that re-anchoring the exit window to wct fails loudly here
        # instead of silently reinflating results.
        exit_abs_ts = entry_anchor_ts + pd.Timedelta(seconds=sim.exit_time_offset_s)
        if sim.exit_time_offset_s < 0 or exit_abs_ts < wct + pd.Timedelta(seconds=feed_delay_s):
            raise RuntimeError(
                f"exit-before-entry in game {game_id} @ {wct}: "
                f"hold={sim.exit_time_offset_s}s, exit_ts={exit_abs_ts}, "
                f"entry_anchor={entry_anchor_ts} (feed_delay={feed_delay_s}s)"
            )

        gross = entry_side * (exit_price - yes_bid)
        fees  = _compute_fees(yes_bid, exit_price, contracts, sim.exit_reason)
        net   = gross * contracts - fees

        positions.append(ClosedPosition(
            game_id       = game_id,
            possession_id = int(row.get("possession_id", row.get("event_id", 0))),
            wall_clock_ts = wct,
            entry_price   = yes_bid,
            exit_price    = exit_price,
            exit_reason   = sim.exit_reason,
            entry_side    = entry_side,
            hold_time_s   = sim.exit_time_offset_s,
            gross_pnl     = gross,
            net_pnl       = net,
            run_prob        = output.run_prob,
            traj_final      = traj_final,
            traj_used       = traj_used,
            traj_aggregator = traj_aggregator,
            hazard_final    = output.hazard[-1],
            quarter         = int(row.get("period", 0)),
            score_diff      = int(row.get("score_diff", 0)),
            run_length      = int(row.get("current_run_length", 0)),
            fees            = fees,
            entry_tick_age_s = entry_tick_age_s,
        ))

        # Block new entries until exit time. Measured from the entry anchor, not wct —
        # otherwise the guard clears feed_delay_s early and lets the next position open
        # while this one is still open.
        position_exit_ts = exit_abs_ts

    logger.info("Game %s: %d trades", game_id, len(positions))
    return positions


# ── Context breakdown helpers ────────────────────────────────────────────────

def _bucket_stats(positions: list[ClosedPosition], key_fn) -> dict[str, dict]:
    """Group positions by key_fn(pos) and compute per-bucket stats."""
    buckets: dict[str, list[ClosedPosition]] = {}
    for pos in positions:
        k = key_fn(pos)
        buckets.setdefault(k, []).append(pos)

    result = {}
    for k, group in sorted(buckets.items()):
        net_pnls = [p.net_pnl for p in group]
        result[str(k)] = {
            "n_trades":      len(group),
            "total_net_pnl": round(sum(net_pnls), 2),
            "win_rate":      round(sum(1 for p in net_pnls if p > 0) / len(net_pnls), 3),
        }
    return result


def _score_bucket(pos: ClosedPosition) -> str:
    abs_diff = abs(pos.score_diff)
    if abs_diff <= 5:
        return "±0-5"
    if abs_diff <= 12:
        return "±6-12"
    if abs_diff <= 20:
        return "±13-20"
    return "±21+"


def _run_length_bucket(pos: ClosedPosition) -> str:
    if pos.run_length <= 3:
        return "1-3"
    if pos.run_length <= 6:
        return "4-6"
    return "7+"


# ── Main entry point ─────────────────────────────────────────────────────────

def run_backtest(
    run_prob_threshold: float = 0.15,
    tp: float = 5.0,
    sl: float = 3.0,
    contracts: int = 100,
    feed_delay_s: int = FEED_DELAY_SECONDS_NBA,
    model_path: Path = Path("models/saved/mmoe_delay20.pt"),
    scaler_path: Path = Path("models/saved/mmoe_scaler_delay20.pkl"),
    use_traj_for_side: bool = False,
    min_abs_traj: float = 0.0,
    min_run_length: int = 1,
    hold_seconds: int = 120,
    only_game: str | None = None,
    traj_aggregator: str = "final",
) -> BacktestSummary:
    logger.info("Loading data...")
    all_poss, all_ticks, pregame = _load_data()

    # Derive features and join pregame
    all_poss = _add_derived_features(all_poss)
    all_poss = _join_pregame(all_poss, pregame)

    # Assign game_id to ticks via ticker → game lookup
    all_ticks = _select_home_best_contract(all_ticks, all_poss)

    # Filter to Head B val set: game_date >= HEADB_SPLIT_DATE
    all_poss["game_date"] = pd.to_datetime(all_poss["game_date"])
    val_mask  = all_poss["game_date"] >= HEADB_SPLIT_DATE
    val_games = set(all_poss.loc[val_mask, "game_id"].unique())

    # Only keep val games that have tick data
    tick_games   = set(all_ticks["game_id"].unique())
    tradeable    = val_games & tick_games
    # Single-game filter: lets us replay one specific game (e.g. last night's
    # paper-trade run) under the validated single-stable-market assumption,
    # to estimate "what would pin-the-market have looked like for this game?"
    if only_game is not None:
        if only_game not in tradeable:
            logger.error(
                "Game %s not in tradeable set (in val window: %s, has ticks: %s)",
                only_game, only_game in val_games, only_game in tick_games,
            )
        tradeable = tradeable & {only_game}
    logger.info(
        "Val games: %d | games with ticks: %d | tradeable: %d",
        len(val_games), len(tick_games), len(tradeable),
    )

    val_poss  = all_poss[all_poss["game_id"].isin(tradeable)].copy()
    val_ticks = all_ticks[all_ticks["game_id"].isin(tradeable)].copy()

    val_poss["wall_clock_ts"]  = pd.to_datetime(val_poss["wall_clock_ts"], utc=True)
    val_ticks["ts"]            = pd.to_datetime(val_ticks["ts"], utc=True)

    # Cast bool columns
    for col in BOOL_COLS:
        if col in val_poss.columns:
            val_poss[col] = val_poss[col].astype(float)

    # Load predictor
    logger.info("Loading MMoE predictor from %s", model_path)
    predictor = MMoEPredictor.load(model_path, scaler_path)

    # Run per-game backtest
    all_positions: list[ClosedPosition] = []
    game_ids = sorted(tradeable)

    for game_id in game_ids:
        game_poss  = val_poss[val_poss["game_id"] == game_id].copy()
        game_ticks = val_ticks[val_ticks["game_id"] == game_id].copy()

        positions = _run_game(
            game_id=game_id,
            game_poss=game_poss,
            game_ticks=game_ticks,
            predictor=predictor,
            run_prob_threshold=run_prob_threshold,
            tp=tp,
            sl=sl,
            contracts=contracts,
            feed_delay_s=feed_delay_s,
            use_traj_for_side=use_traj_for_side,
            min_abs_traj=min_abs_traj,
            min_run_length=min_run_length,
            hold_seconds=hold_seconds,
            traj_aggregator=traj_aggregator,
        )
        all_positions.extend(positions)

    # Aggregate
    if not all_positions:
        logger.warning("No trades fired across all val games. Check threshold and data coverage.")
        return BacktestSummary(
            positions=[], total_gross_pnl=0.0, total_net_pnl=0.0,
            n_trades=0, win_rate=0.0, avg_hold_time_s=0.0,
            exit_reasons={}, by_quarter={}, by_score_bucket={}, by_run_length={},
        )

    total_gross = sum(p.gross_pnl * contracts for p in all_positions)
    total_net   = sum(p.net_pnl for p in all_positions)
    win_rate    = sum(1 for p in all_positions if p.net_pnl > 0) / len(all_positions)
    avg_hold    = np.mean([p.hold_time_s for p in all_positions])

    exit_reasons: dict[str, int] = {}
    for p in all_positions:
        exit_reasons[p.exit_reason] = exit_reasons.get(p.exit_reason, 0) + 1

    return BacktestSummary(
        positions       = all_positions,
        total_gross_pnl = round(total_gross, 2),
        total_net_pnl   = round(total_net, 2),
        n_trades        = len(all_positions),
        win_rate        = round(win_rate, 3),
        avg_hold_time_s = round(float(avg_hold), 1),
        exit_reasons    = exit_reasons,
        by_quarter      = _bucket_stats(all_positions, lambda p: p.quarter),
        by_score_bucket = _bucket_stats(all_positions, _score_bucket),
        by_run_length   = _bucket_stats(all_positions, _run_length_bucket),
    )


def _print_summary(
    summary: BacktestSummary,
    contracts: int,
    label: str = "",
) -> None:
    print("\n" + "=" * 60)
    print(f"MMoE BACKTEST RESULTS{' — ' + label if label else ''}")
    print("=" * 60)
    print(f"  Trades:          {summary.n_trades}")
    print(f"  Win rate:        {summary.win_rate:.1%}")
    print(f"  Avg hold time:   {summary.avg_hold_time_s:.1f}s")
    print(f"  Total gross PnL: ${summary.total_gross_pnl:+.2f}  (per {contracts} contracts)")
    print(f"  Total fees:      ${sum(p.fees for p in summary.positions):.2f}  "
          f"(maker entry + maker TP / taker stop)")
    print(f"  Total net PnL:   ${summary.total_net_pnl:+.2f}  (after fees)")
    if summary.positions:
        ages = [p.entry_tick_age_s for p in summary.positions]
        print(f"  Entry tick age:  median {float(np.median(ages)):.1f}s | "
              f"p90 {float(np.percentile(ages, 90)):.1f}s | max {max(ages):.1f}s")
        print(f"  Min hold time:   {min(p.hold_time_s for p in summary.positions):.2f}s "
              f"(must be >= 0; exit-before-entry raises)")
    print(f"\n  Exit reasons:")
    for reason, count in sorted(summary.exit_reasons.items(), key=lambda x: -x[1]):
        pct = count / summary.n_trades * 100
        print(f"    {reason:<20} {count:4d}  ({pct:.1f}%)")

    print(f"\n  By quarter:")
    for q, stats in summary.by_quarter.items():
        print(f"    Q{q}: {stats['n_trades']:3d} trades | net {stats['total_net_pnl']:+8.2f} | wr {stats['win_rate']:.1%}")

    print(f"\n  By score diff:")
    for bucket, stats in summary.by_score_bucket.items():
        print(f"    {bucket:<10}: {stats['n_trades']:3d} trades | net {stats['total_net_pnl']:+8.2f} | wr {stats['win_rate']:.1%}")

    print(f"\n  By run length at entry:")
    for bucket, stats in summary.by_run_length.items():
        print(f"    {bucket:<5}: {stats['n_trades']:3d} trades | net {stats['total_net_pnl']:+8.2f} | wr {stats['win_rate']:.1%}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMoE backtest on Head B val set")
    parser.add_argument("--threshold",        type=float, default=0.15,   help="run_prob entry threshold")
    parser.add_argument("--tp",               type=float, default=5.0,    help="take-profit in cents")
    parser.add_argument("--sl",               type=float, default=3.0,    help="stop-loss in cents")
    parser.add_argument("--contracts",        type=int,   default=100,    help="contract size per trade")
    parser.add_argument("--delay",            type=int,   default=FEED_DELAY_SECONDS_NBA, help="feed delay in seconds")
    parser.add_argument("--use-traj-for-side",action="store_true",        help="use Head B traj_final sign to set BUY_YES vs BUY_NO (vs basketball run_team)")
    parser.add_argument("--min-abs-traj",     type=float, default=0.0,    help="minimum |traj_used| to enter (Head B confidence filter applied to aggregated value)")
    parser.add_argument("--traj-aggregator",  type=str,   default="final",
                        choices=list(TRAJ_AGGREGATORS),
                        help="how to reduce the 10-element Head B trajectory to a scalar for entry gating")
    parser.add_argument("--min-run-length",   type=int,   default=1,      help="minimum run_length at entry")
    parser.add_argument("--hold-seconds",     type=int,   default=120,    help="time gate override in seconds (default 120 matching training)")
    parser.add_argument(
        "--model", type=Path, default=Path("models/saved/mmoe_delay20.pt"),
        help="path to model checkpoint",
    )
    parser.add_argument(
        "--scaler", type=Path, default=Path("models/saved/mmoe_scaler_delay20.pkl"),
        help="path to scaler pickle",
    )
    parser.add_argument("--save-csv", action="store_true", help="save positions to CSV")
    parser.add_argument("--game",     type=str, default=None,
                        help="run backtest on a single game_id only (e.g. 0042500311)")
    args = parser.parse_args()

    summary = run_backtest(
        run_prob_threshold = args.threshold,
        tp                 = args.tp,
        sl                 = args.sl,
        contracts          = args.contracts,
        feed_delay_s       = args.delay,
        model_path         = args.model,
        scaler_path        = args.scaler,
        use_traj_for_side  = args.use_traj_for_side,
        min_abs_traj       = args.min_abs_traj,
        min_run_length     = args.min_run_length,
        hold_seconds       = args.hold_seconds,
        only_game          = args.game,
        traj_aggregator    = args.traj_aggregator,
    )

    label = (
        f"thr={args.threshold} tp={args.tp} sl={args.sl} "
        f"hold={args.hold_seconds}s rl>={args.min_run_length} "
        f"agg={args.traj_aggregator} "
        f"{'traj-side ' if args.use_traj_for_side else ''}"
        f"min|traj|={args.min_abs_traj}"
    )
    _print_summary(summary, args.contracts, label=label)

    if args.save_csv or True:  # always save
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("backtesting/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"mmoe_backtest_{ts_str}.csv"

        rows = [vars(p) for p in summary.positions]
        pd.DataFrame(rows).to_csv(out_path, index=False)
        logger.info("Positions saved to %s", out_path)

    # Exit with non-zero if edge test fails
    if summary.n_trades == 0 or summary.total_net_pnl <= 0:
        logger.warning(
            "Edge test FAILED: net_pnl=%.2f across %d trades. "
            "Consider adjusting threshold, TP/SL, or retraining.",
            summary.total_net_pnl, summary.n_trades,
        )
        sys.exit(1)
    else:
        logger.info(
            "Edge test PASSED: net_pnl=+%.2f across %d trades.",
            summary.total_net_pnl, summary.n_trades,
        )
