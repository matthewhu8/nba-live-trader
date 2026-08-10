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
from decimal import Decimal
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mmoe.dataset import (
    FEED_DELAY_SECONDS_NBA,
    HEADB_SPLIT_DATE,
    MARKET_STALENESS_TOLERANCE_SECONDS,
    BOOL_COLS,
    _add_derived_features,
    _connect_motherduck,
    _filter_to_traded_regime,
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
    gross_pnl:     float        # CENTS per contract: entry_side * (exit_price - entry_price)
    net_pnl:       float        # DOLLARS: pnl_dollars(gross_pnl, contracts) - fees
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

    The lookback is bounded by MARKET_STALENESS_TOLERANCE_SECONDS (PR #54), matching the
    training join and the live staleness gate in ring_buffer.go. Unbounded, a possession
    binds to a tick of any age and backtests as a real trade; returning None instead sets
    has_market_data=0, the honest signal that we could not have priced it live. The bound
    lives here rather than in the caller so the feature lookup and the exit anchor cannot
    drift apart — that divergence is what produced the exit-window lookahead.
    """
    entry_anchor_ts = wall_clock_ts + pd.Timedelta(seconds=delay_s)
    earliest_ts = entry_anchor_ts - pd.Timedelta(seconds=MARKET_STALENESS_TOLERANCE_SECONDS)
    candidates = enriched_game_ticks[
        (enriched_game_ticks["ts"] <= entry_anchor_ts)
        & (enriched_game_ticks["ts"] >= earliest_ts)
    ]
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
    # _tick_at_delay applies the MARKET_STALENESS_TOLERANCE_SECONDS bound (PR #54).
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


# Features that production does NOT actually deliver to the model, as measured from
# 3,131 recorded possessions across 27 sessions in live-trader/go/logs/runs/ (2026-08-03).
# The backtest reads these from the feature store, so it trades a strategy production
# cannot run. `--prod-features` zeroes them to measure what live would really have done.
#
# Sources of each gap:
#   11 pregame  — features.pregame has no row at tip-off (3 AM ET prefill runs after the
#                 day's games), so live-trader/inference/pregame.py:137 zero-fills. 0.0 in
#                 100% of recorded prod possessions.
#   2 lineup sample_size — hardcoded `0.0  # TODO` at inference/features.py:228-229.
#   3 lineup net_rating  — non-zero in only 1.8-8.8% of prod possessions vs 31-41% here,
#                 because game_state.py:129-131 never seeds starters so the lineup hash
#                 rarely matches. Zeroing slightly OVERSHOOTS the real gap.
_PROD_ZERO_FEATURES = tuple(PREGAME_COLS) + (
    "home_lineup_sample_size", "away_lineup_sample_size",
    "home_lineup_net_rating",  "away_lineup_net_rating", "lineup_net_rating_delta",
)
# Losing expected_pace also pins this constant in prod (inference/pregame.py:95).
_PROD_PINNED_FEATURES = {"pace_season_baseline": 14.0}

# NOT masked, deliberately: was_foul / was_sub are populated in prod (20.2% / 16.5%) but
# are all-NaN in the local cache, so this backtest already understates them. That skew
# runs the opposite way and cannot be corrected from the cache.


def _apply_prod_feature_gaps(fd: dict[str, float]) -> dict[str, float]:
    """Degrade a feature dict to what production actually supplies. See _PROD_ZERO_FEATURES."""
    out = dict(fd)
    for col in _PROD_ZERO_FEATURES:
        if col in out:
            out[col] = 0.0
    for col, val in _PROD_PINNED_FEATURES.items():
        if col in out:
            out[col] = val
    return out


def _build_feature_dict(
    poss_row: pd.Series,
    market_features: dict[str, float],
) -> dict[str, float]:
    """
    Assemble the model's feature dict from a possession_flat row + market feature dict.
    Physics + Pregame come from poss_row; Market from market_features. Sizes follow
    feature_config.py (58 = 33 + 11 + 14 as of PR #51); do not hardcode a count here.
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


def pnl_dollars(price_move_cents: float, contracts: int) -> float:
    """Convert a price move in cents into dollars for `contracts` contracts.

    A Kalshi contract settles at $1, quoted 1-99c, so one contract moving 1c is $0.01:
    100 contracts x 5c = $5.00, NOT $500. Fees are computed in dollars, so P&L must be
    too -- `gross_cents * contracts - fees_dollars` silently mixes the two and makes the
    fee look 100x smaller than it is. That mix is why the old backtests reported figures
    like "+$16,100" (really 16,100 cents = $161) while CLAUDE.md's break-even win rate of
    55-56% was derived with correct units.
    """
    return price_move_cents * contracts / 100.0

# Exits that rest a limit order and therefore pay the maker rate. Every other exit
# reason crosses the book to get out now, so it pays taker — see CLAUDE.md Phase 6:
# resting maker take-profits (PR #50), stops cross the book.
#
# `take_profit_widened` is emitted only by `dynamic_exit.simulate_exit_dynamic`, whose streak
# rule re-posts the limit further out — still a resting order, so still maker. It was missing
# here, so `tools/sweep_dynamic_exit.py` charged the 4x taker rate on precisely the exits the
# streak rule is designed to produce. Adding it cannot move the main backtest: `simulate_exit`
# only ever emits the five EXIT_REASONS, none of which is the widened variant.
_MAKER_EXIT_REASONS = frozenset({"take_profit", "take_profit_widened"})


def _fee_one_leg(rate: float, price_cents: float, contracts: int) -> float:
    """Kalshi fee for a single leg, in DOLLARS:

        fee = ceil(rate x C x P x (1-P) x 100) / 100      P = price/100

    The P(1-P) term is part of Kalshi's published formula; fees peak at 50c and fall
    toward both ends of the book. Dropping it overstates the fee ~2x at the midpoint.
    Worked example: 100 contracts at 50c maker, 0.0175 x 100 x 0.5 x 0.5 = 0.4375 ->
    ceil(43.75)/100 = $0.44.

    Computed in Decimal, NOT float. In float64 the product lands a few ulp above an
    exact cent and `ceil` then rounds a whole cent up, contradicting Kalshi's published
    table on three of five taker rows: 0.07*100*0.5*0.5*100 evaluates to
    175.00000000000003, so 100 contracts at 50c taker returned $1.76 instead of $1.75
    (likewise $0.10 -> $0.64 not $0.63, and $0.20 -> $1.13 not $1.12). Both this
    function and the `kalshi_fee` added by PR #54 had the defect, and `.claude/CLAUDE.md`
    on main documented the wrong $1.76 as if it were correct. See tests/test_fees.py.

    This matches live-trader/inference/dashboard.py::kalshiMakerFee.
    """
    p = Decimal(str(price_cents)) / Decimal(100)
    raw_cents = Decimal(str(rate)) * Decimal(contracts) * p * (Decimal(1) - p) * Decimal(100)
    return math.ceil(raw_cents) / 100.0


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

    Supersedes `_compute_maker_fees` from PR #54, which charged maker on both legs. Its
    own docstring conceded that stop-losses cross the book at 4x the rate and that it did
    not model them, so its output was a floor rather than a cost. Since stop_loss is the
    single most common exit reason, that floor sat well under the real number.
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
    prod_features: bool = False,
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

        # No garbage-time skip here. `_filter_to_traded_regime` in run_backtest already
        # excluded overtime and |score_diff| > blowout_margin_pts, using the same 30-pt
        # margin from trading.yaml that training and the live gate use.
        #
        # Filtering again on the stored `is_blowout` / `is_garbage_time` columns would be
        # STRICTER than training, not equivalent to it: those columns are computed with a
        # 20-pt margin, and per skills/feature-engineering.md they discard 28,627 rows the
        # system would really trade. Stacking both gates made the backtest refuse
        # possessions the live agent accepts, which understates the trade population.

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
        if prod_features:
            fd = _apply_prod_feature_gaps(fd)
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
            # run_signed_points carries the run direction in its sign (it is
            # encoded_team * points, and points is non-negative), so this is
            # equivalent to the old current_run_team_encoded test.
            run_signed = fd.get("run_signed_points", 0.0)
            entry_side = 1 if run_signed >= 0 else -1

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

        # Single source of truth for where the exit window opens. Everything below —
        # the tick/possession windows, the call, and the invariant — reads this one name.
        exit_search_start = entry_anchor_ts

        future_ticks = enriched_ticks[enriched_ticks["ts"] > exit_search_start]
        future_poss  = game_poss[game_poss["wall_clock_ts"] > exit_search_start]

        sim = simulate_exit(
            entry_wall_clock=exit_search_start,
            entry_yes_bid=yes_bid,
            entry_run_team=row.get("current_run_team", None),
            future_ticks=future_ticks,
            future_possessions=future_poss,
            tp=tp,
            sl=sl,
            entry_side=entry_side,
            max_seconds=hold_seconds,
        )

        # The take-profit clamp (PR #50 rests the limit at entry + TP, so the fill cannot
        # beat it) now lives inside simulate_exit, applied before the trajectory is built.
        # It was here until Level 2, which fixed the P&L but left the Head B labels
        # over-credited — the post-exit checkpoints froze at the unclamped price. Behaviour
        # at this call site is unchanged; the clamp is simply no longer duplicated.
        exit_price = sim.exit_price

        # Invariant: a position cannot exit before it existed.
        #
        # This is asserted on the INPUTS to simulate_exit, not on its output. The previous
        # form — `entry_anchor_ts + offset < wct + feed_delay_s` — was a tautology: the
        # anchor IS `wct + feed_delay_s`, so the test reduced to `offset < 0`, duplicating
        # the clause beside it. Measured 2026-08-05: with the original three-line bug
        # restored the backtest ran to completion, 178 trades at 30.9% with a 0.03s minimum
        # hold, and the guard never fired. `skills/backtesting.md` claimed it was "verified
        # to fire when the bug is reintroduced"; it was not.
        exit_abs_ts = exit_search_start + pd.Timedelta(seconds=sim.exit_time_offset_s)

        if sim.exit_time_offset_s < 0:
            raise RuntimeError(
                f"negative hold in game {game_id} @ {wct}: {sim.exit_time_offset_s}s"
            )

        # (a) The window must open no earlier than the entry anchor. Fires if
        #     exit_search_start is re-pointed at wct.
        if exit_search_start < wct + pd.Timedelta(seconds=feed_delay_s):
            raise RuntimeError(
                f"exit window opens before entry in game {game_id} @ {wct}: "
                f"exit_search_start={exit_search_start}, "
                f"anchor={wct + pd.Timedelta(seconds=feed_delay_s)} (feed_delay={feed_delay_s}s)"
            )

        # (b) Nothing reachable by the exit search may predate the anchor. Fires if the
        #     future_ticks / future_poss filters are widened back to wct — the actual
        #     historical defect, which let a position close on movement that had already
        #     happened before it opened.
        if not future_ticks.empty and future_ticks["ts"].min() <= exit_search_start:
            raise RuntimeError(
                f"exit search can see pre-entry ticks in game {game_id} @ {wct}: "
                f"earliest={future_ticks['ts'].min()}, anchor={exit_search_start}"
            )
        if not future_poss.empty and future_poss["wall_clock_ts"].min() <= exit_search_start:
            raise RuntimeError(
                f"exit search can see pre-entry possessions in game {game_id} @ {wct}: "
                f"earliest={future_poss['wall_clock_ts'].min()}, anchor={exit_search_start}"
            )

        # (c) A tick-driven exit must land on a real tick. simulate_exit measures its
        #     offset from whatever anchor it was handed, so if the call site is anchored
        #     at wct while the windows stay at the anchor, every implied exit timestamp
        #     is shifted by feed_delay_s and lands between ticks. time_gate exits resolve
        #     at the deadline rather than a tick, so they are exempt.
        if sim.exit_reason != "time_gate" and not future_ticks.empty:
            gap_s = (future_ticks["ts"] - exit_abs_ts).abs().min().total_seconds()
            if gap_s > 1e-3:
                raise RuntimeError(
                    f"exit does not land on a tick in game {game_id} @ {wct}: "
                    f"exit_ts={exit_abs_ts} is {gap_s:.3f}s from the nearest tick "
                    f"(reason={sim.exit_reason}) — the exit window and the entry anchor "
                    f"have drifted apart"
                )

        # gross is CENTS per contract; pnl_dollars converts. A 3c move on 100 contracts
        # is $3.00, not $300 — a 100-contract position cannot swing more than $100 total.
        # `exit_price` is `sim.exit_price` — the TP clamp lives inside simulate_exit now, so
        # do NOT re-add a clamp here. It would be a no-op in this file, but the same reasoning
        # applied to sweep_dynamic_exit.py (where the limit is `effective_tp`, not `tp`) would
        # under-credit every widened exit.
        gross = entry_side * (exit_price - yes_bid)
        fees  = _compute_fees(yes_bid, exit_price, contracts, sim.exit_reason)
        net   = pnl_dollars(gross, contracts) - fees

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
    prod_features: bool = False,
) -> BacktestSummary:
    logger.info("Loading data...")
    all_poss, all_ticks, pregame = _load_data()

    # Derive features and join pregame
    # Order matters: _add_derived_features needs `expected_pace` (for pace_ref), which
    # _join_pregame supplies. dataset.py:276 raises if called the other way round.
    all_poss = _join_pregame(all_poss, pregame)
    all_poss = _add_derived_features(all_poss)

    # Restrict to the regime the agent will actually trade — the same call, in the same
    # position, as build_dataloaders (dataset.py:926). Training drops overtime,
    # |score_diff| > blowout_margin_pts and NULL-pace rows; without this the backtest
    # scored the model on rows it never saw in training and the live gates would refuse.
    #
    # Runs after the derived features so the within-game rolling windows are still built
    # from the complete possession sequence, and before the tick join so dropped rows
    # never reach it. `_filter_to_traded_regime` logs its own before/after breakdown.
    n_before = len(all_poss)
    all_poss = _filter_to_traded_regime(all_poss)
    logger.info(
        "Traded-regime filter (backtest): %d → %d possession rows (%.1f%% dropped)",
        n_before, len(all_poss), 100.0 * (n_before - len(all_poss)) / n_before if n_before else 0.0,
    )

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
            prod_features=prod_features,
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

    # gross_pnl is cents/contract; pnl_dollars converts, matching net_pnl.
    total_gross = sum(pnl_dollars(p.gross_pnl, contracts) for p in all_positions)
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
    parser.add_argument("--prod-features", action="store_true",
                        help="zero the pregame/lineup features production does not actually "
                             "deliver, to measure what live would really have traded "
                             "(see _PROD_ZERO_FEATURES)")
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
        prod_features      = args.prod_features,
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
