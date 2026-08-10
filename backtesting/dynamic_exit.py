"""
Dynamic-exit simulator for Phase 2 backtests.

Unlike `models.targets.exit_simulator.simulate_exit` which runs inference only
at entry and then walks ticks with fixed TP/SL, this module re-runs the MMoE
model at every future possession during the hold and applies two
trajectory-driven rules:

  1. **Reversal exit** — if the aggregated trajectory at a later possession
     flips sign from the entry value AND has |traj_now| >= reversal_min_abs,
     exit immediately ("reversal_exit").

  2. **Conviction-streak TP widening** — if the trajectory stays same-sign
     as entry AND |traj_now| >= |traj_entry| for >= streak_widen_at consecutive
     possessions, widen the effective TP by `streak_widen_cents`. This lets
     strong winners run past the fixed 5¢ cap.

The streak/reversal comparison aggregator is INDEPENDENT of the entry-signal
aggregator. `max_abs` for entry → `final` or `mean_3_to_9` for comparison is
the intended pattern (per plan): entry wants peak conviction, mid-trade
comparison wants a lower-variance reference.

This file does not modify `models.targets.exit_simulator` — that module is
reused by training-time target generation and must stay free of inference
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.special import logit

from backtesting.mmoe_backtest import (
    FEED_DELAY_SECONDS_NBA,
    _build_feature_dict,
    _get_market_features_at_delay,
    aggregate_traj,
)
from models.mmoe.predictor import MMoEPredictor


# Same constants as the static simulator so trajectories remain comparable.
CHECKPOINT_SECONDS = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
N_CHECKPOINTS = len(CHECKPOINT_SECONDS)
DEFAULT_MAX_SECONDS = CHECKPOINT_SECONDS[-1]


@dataclass
class DynamicExitResult:
    exit_price:         float
    exit_reason:        str
    exit_time_offset_s: float
    simulated_pnl:      float
    # Telemetry — not used for P&L but key for sweep analysis
    reversal_fired:     bool = False
    max_streak:         int  = 0
    tp_widened:         bool = False
    n_reinference:      int  = 0       # how many in-hold possessions re-ran the model
    traj_history:       list = field(default_factory=list)  # (elapsed_s, traj_compare)


def _logit_delta(entry_price: float, checkpoint_price: float) -> float:
    p_entry = np.clip(entry_price / 100.0, 0.01, 0.99)
    p_check = np.clip(checkpoint_price / 100.0, 0.01, 0.99)
    return float(logit(p_check) - logit(p_entry))


def simulate_exit_dynamic(
    entry_wall_clock:        pd.Timestamp,
    entry_yes_bid:           float,
    entry_run_team:          Optional[str],
    future_ticks:            pd.DataFrame,            # ts > entry_wall_clock, sorted
    future_possessions:      pd.DataFrame,            # wall_clock_ts > entry, sorted, with full possession_flat columns
    enriched_ticks_for_game: pd.DataFrame,            # full enriched ticks dataframe (with market feature cols)
    *,
    predictor:               MMoEPredictor,
    entry_traj_for_compare:  float,                   # entry trajectory aggregated with streak_compare_aggregator
                                                      # (NOT entry_aggregator — see plan: streak comparison uses
                                                      # a separate, lower-variance aggregator)
    streak_compare_aggregator: str = "final",         # which aggregator to use for streak/reversal comparison
    tp:                      float = 5.0,
    sl:                      float = 3.0,
    entry_side:              int = 1,
    max_seconds:             Optional[int] = None,
    # Dynamic-rule knobs
    reversal_enabled:        bool = True,
    reversal_min_abs:        float = 0.05,
    streak_enabled:          bool = True,
    streak_widen_at:         int = 2,
    streak_widen_cents:      float = 3.0,
    # Plumbing for per-possession feature rebuild
    feed_delay_s:            int = FEED_DELAY_SECONDS_NBA,
    prev_yes_bid_init:       float = 50.0,
    prev_spread_init:        float = 1.0,
) -> DynamicExitResult:
    """
    Walk forward through ticks, re-running inference at each future possession
    boundary, and apply both the static (TP/SL/momentum_flip/garbage) rules
    AND the dynamic reversal+streak rules.

    All rules are checked in this order at each tick:
      1. garbage_time / blowout (highest priority)
      2. momentum_flip (existing static rule)
      3. reversal_exit (new dynamic rule)
      4. take_profit at effective_tp (possibly widened by streak)
      5. stop_loss at sl
      6. time_gate at deadline (default)
    """
    hold_limit = max_seconds if max_seconds is not None else DEFAULT_MAX_SECONDS
    deadline = entry_wall_clock + pd.Timedelta(seconds=hold_limit)

    window_ticks = future_ticks[future_ticks["ts"] <= deadline].copy()
    window_poss = future_possessions[
        future_possessions["wall_clock_ts"] <= deadline
    ].sort_values("wall_clock_ts").reset_index(drop=True)

    exit_price = entry_yes_bid
    exit_reason = "time_gate"
    exit_time_offset_s = hold_limit

    # Dynamic state
    effective_tp = tp
    streak = 0
    max_streak = 0
    reversal_fired = False
    tp_widened = False
    n_reinference = 0
    traj_history: list[tuple[float, float]] = []

    # Possession iteration state: track the next possession we haven't re-inferenced yet.
    next_poss_idx = 0
    prev_yes_bid = prev_yes_bid_init
    prev_spread = prev_spread_init
    # All comparisons use the streak_compare aggregator, including the entry-time baseline.
    # This decouples the streak rule from any inflation/distortion of the entry aggregator.
    entry_sign = 1.0 if entry_traj_for_compare >= 0 else -1.0
    entry_abs = abs(entry_traj_for_compare)

    for _, tick in window_ticks.iterrows():
        current_bid = float(tick["yes_bid"])
        elapsed_s = (tick["ts"] - entry_wall_clock).total_seconds()

        # Process any possessions whose wall_clock_ts has been reached by this tick.
        # Each crossed possession triggers one re-inference + dynamic-rule update.
        while (
            next_poss_idx < len(window_poss)
            and window_poss.iloc[next_poss_idx]["wall_clock_ts"] <= tick["ts"]
        ):
            poss_row = window_poss.iloc[next_poss_idx]
            next_poss_idx += 1

            # Skip garbage time possessions silently (the rule below will catch garbage_time exit).
            if poss_row.get("is_blowout", False) or poss_row.get("is_garbage_time", False):
                continue

            market_feats = _get_market_features_at_delay(
                enriched_game_ticks=enriched_ticks_for_game,
                wall_clock_ts=poss_row["wall_clock_ts"],
                prev_yes_bid=prev_yes_bid,
                prev_spread=prev_spread,
                delay_s=feed_delay_s,
            )
            if market_feats.get("has_market_data", 0.0) == 0.0:
                continue

            if market_feats["yes_bid"] > 0:
                prev_yes_bid = market_feats["yes_bid"]
                prev_spread = market_feats["spread"]

            fd = _build_feature_dict(poss_row, market_feats)
            output = predictor.predict(fd)
            n_reinference += 1

            traj_compare = aggregate_traj(output.trajectory, streak_compare_aggregator)
            traj_history.append((elapsed_s, traj_compare))

            # Reversal rule
            now_sign = 1.0 if traj_compare >= 0 else -1.0
            if (
                reversal_enabled
                and now_sign != entry_sign
                and abs(traj_compare) >= reversal_min_abs
            ):
                exit_price = current_bid
                exit_reason = "reversal_exit"
                exit_time_offset_s = elapsed_s
                reversal_fired = True
                break

            # Streak rule
            if (
                streak_enabled
                and now_sign == entry_sign
                and abs(traj_compare) >= entry_abs
            ):
                streak += 1
                if streak > max_streak:
                    max_streak = streak
                if streak >= streak_widen_at and not tp_widened:
                    effective_tp = tp + streak_widen_cents
                    tp_widened = True
            else:
                streak = 0

        if exit_reason == "reversal_exit":
            break

        # Static rules: garbage_time + momentum_flip (consult most-recent processed possession)
        if next_poss_idx > 0:
            last_processed = window_poss.iloc[next_poss_idx - 1]
            if last_processed.get("is_blowout", False) or last_processed.get("is_garbage_time", False):
                exit_price = current_bid
                exit_reason = "garbage_time"
                exit_time_offset_s = elapsed_s
                break
            current_run_team = last_processed.get("current_run_team", None)
            if entry_run_team is not None and current_run_team != entry_run_team:
                exit_price = current_bid
                exit_reason = "momentum_flip"
                exit_time_offset_s = elapsed_s
                break

        # TP / SL
        move = entry_side * (current_bid - entry_yes_bid)
        if move >= effective_tp:
            # Clamp to the resting limit, matching exit_simulator.simulate_exit. Booking
            # `current_bid` credited the overshoot past the limit as profit that PR #50's
            # resting order could never have collected — and because the streak rule widens
            # the target, the overshoot scaled with the very knob these sweeps exist to tune.
            #
            # `effective_tp`, not `tp`: once the streak rule widens the target the order is
            # re-posted at `entry + effective_tp`, so that is the price a fill can achieve.
            # Clamping to `tp` here would under-credit every widened exit instead.
            exit_price = entry_yes_bid + entry_side * effective_tp
            exit_reason = "take_profit_widened" if tp_widened else "take_profit"
            exit_time_offset_s = elapsed_s
            break
        if move <= -sl:
            exit_price = current_bid
            exit_reason = "stop_loss"
            exit_time_offset_s = elapsed_s
            break

    simulated_pnl = entry_side * (exit_price - entry_yes_bid)

    return DynamicExitResult(
        exit_price=exit_price,
        exit_reason=exit_reason,
        exit_time_offset_s=exit_time_offset_s,
        simulated_pnl=simulated_pnl,
        reversal_fired=reversal_fired,
        max_streak=max_streak,
        tp_widened=tp_widened,
        n_reinference=n_reinference,
        traj_history=traj_history,
    )
