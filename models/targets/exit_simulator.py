"""
Hybrid Exit Simulator for MMoE Target B (price trajectory).

For each entry row (possession with Kalshi tick data), simulates forward through
real tick + possession data and exits at whichever condition fires first:
  1. Take Profit (TP): yes_bid moves >= tp_cents in our favor
  2. Stop Loss (SL):   yes_bid moves >= sl_cents against us
  3. Momentum flip:    current_run_team changes to opposing team or neutral
  4. Time gate:        120 seconds elapsed (hard deadline)
  5. Garbage time:     is_blowout or is_garbage_time becomes True

Outputs per entry row:
  - 10-checkpoint price trajectory (logit delta units, every 12s)
  - exit_price, exit_reason, exit_time_offset_s, simulated_pnl

Price lookup rule: for each checkpoint time t, use the FIRST tick with ts >= t
(merge_asof direction="forward"). This ensures we see price after the event,
not before — no lookahead bias.

Entry anchor: every offset here is measured from `wall_clock_ts + feed_delay_seconds`,
never from `wall_clock_ts`. The entry price is read at that anchor by the backward asof
join in `dataset._join_ticks_to_possessions`, so the exit search, the checkpoint grid and
the time gate all have to start there as well. Anchoring at `wall_clock_ts` made the
labels anti-causal rather than merely early: TP and SL resolved against ticks that
preceded the price the position entered at.

Trade direction:
  - Home run prediction → BUY YES → favorable = price up → entry_side=+1
  - Away run prediction → BUY NO  → favorable = price down → entry_side=-1
  PnL = entry_side * (exit_price - entry_price)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import logit

logger = logging.getLogger(__name__)

# Head B always emits a fixed-length trajectory (model output dim). The horizon
# (how far out the last checkpoint sits) is configurable so we can train scalping
# (120s) and swing (360s/600s) variants without changing the model architecture.
N_CHECKPOINTS = 10
DEFAULT_HORIZON_SECONDS = 120


def make_checkpoints(horizon_seconds: int, n: int = N_CHECKPOINTS) -> list[int]:
    """N evenly-spaced checkpoint times from horizon/n up to horizon (inclusive).

    horizon=120 → [12,24,...,120]; horizon=360 → [36,...,360]; horizon=600 → [60,...,600].
    """
    step = horizon_seconds / n
    return [int(round(step * (k + 1))) for k in range(n)]


# Back-compat module-level defaults (120s scalping horizon).
CHECKPOINT_SECONDS = make_checkpoints(DEFAULT_HORIZON_SECONDS)
MAX_SECONDS = DEFAULT_HORIZON_SECONDS

EXIT_REASONS = ("take_profit", "stop_loss", "momentum_flip", "time_gate", "garbage_time")


@dataclass
class ExitResult:
    exit_price: float
    exit_reason: str
    exit_time_offset_s: float
    simulated_pnl: float
    trajectory: list[float]  # 10 values in logit-delta units


def _logit_delta(entry_price: float, checkpoint_price: float) -> float:
    """Convert raw cent prices to log-odds delta, clipping to avoid inf."""
    p_entry = np.clip(entry_price / 100.0, 0.01, 0.99)
    p_check = np.clip(checkpoint_price / 100.0, 0.01, 0.99)
    return float(logit(p_check) - logit(p_entry))


def _lookup_price_at_or_after(
    ticks: pd.DataFrame,
    target_ts: pd.Timestamp,
    fallback_price: float,
) -> float:
    """Return yes_bid from the first tick at or after target_ts."""
    future = ticks[ticks["ts"] >= target_ts]
    if future.empty:
        return fallback_price
    return float(future.iloc[0]["yes_bid"])


def simulate_exit(
    entry_wall_clock: pd.Timestamp,
    entry_yes_bid: float,
    entry_run_team: Optional[str],
    future_ticks: pd.DataFrame,
    future_possessions: pd.DataFrame,
    tp: float = 5.0,
    sl: float = 3.0,
    entry_side: int = 1,
    max_seconds: Optional[int] = None,
) -> ExitResult:
    """
    Simulate one trade exit from a single entry point.

    Args:
        entry_wall_clock: wall clock timestamp at entry possession end
        entry_yes_bid: yes_bid at entry (cents)
        entry_run_team: current_run_team value at entry ("home", "away", or None)
        future_ticks: all ticks for this game with ts > entry_wall_clock, sorted by ts
        future_possessions: all possessions for this game after entry, sorted by wall_clock_ts
        tp: take profit threshold in cents
        sl: stop loss threshold in cents
        entry_side: +1 for BUY YES (home run), -1 for BUY NO (away run)
        max_seconds: time gate override in seconds (defaults to MAX_SECONDS=120)
    """
    hold_limit = max_seconds if max_seconds is not None else MAX_SECONDS
    deadline = entry_wall_clock + pd.Timedelta(seconds=hold_limit)
    window_ticks = future_ticks[future_ticks["ts"] <= deadline].copy()
    window_possessions = future_possessions[
        future_possessions["wall_clock_ts"] <= deadline
    ].copy()

    exit_price = entry_yes_bid
    exit_reason = "time_gate"
    exit_time_offset_s = hold_limit

    # Walk forward through ticks to find first exit condition
    for _, tick in window_ticks.iterrows():
        current_bid = float(tick["yes_bid"])
        elapsed_s = (tick["ts"] - entry_wall_clock).total_seconds()

        # Check garbage time via possession state at this point in time
        poss_so_far = window_possessions[
            window_possessions["wall_clock_ts"] <= tick["ts"]
        ]
        if not poss_so_far.empty:
            last_poss = poss_so_far.iloc[-1]
            if last_poss.get("is_blowout", False) or last_poss.get("is_garbage_time", False):
                exit_price = current_bid
                exit_reason = "garbage_time"
                exit_time_offset_s = elapsed_s
                break

            # Momentum flip: run_team changed from entry state
            current_run_team = last_poss.get("current_run_team", None)
            if entry_run_team is not None and current_run_team != entry_run_team:
                exit_price = current_bid
                exit_reason = "momentum_flip"
                exit_time_offset_s = elapsed_s
                break

        # TP / SL (direction-adjusted)
        move = entry_side * (current_bid - entry_yes_bid)
        if move >= tp:
            exit_price = current_bid
            exit_reason = "take_profit"
            exit_time_offset_s = elapsed_s
            break
        if move <= -sl:
            exit_price = current_bid
            exit_reason = "stop_loss"
            exit_time_offset_s = elapsed_s
            break

    # Take-profits rest a maker limit at entry + TP (PR #50), so the fill cannot be better
    # than that limit. The loop above reports the price of the tick that *breached* the
    # threshold, which overshot by a median 6c (max 24c) on the 2026-08-03 sample.
    #
    # Clamped here rather than in the caller so that the trajectory below and
    # simulated_pnl both see the collectable price. `mmoe_backtest._run_game` used to
    # clamp its own copy after this function returned, which corrected its P&L but left
    # the Head B labels over-credited: the post-exit checkpoints freeze at `exit_price`,
    # so the uncollectable overshoot was being learned as the target.
    if exit_reason == "take_profit":
        exit_price = entry_yes_bid + entry_side * tp

    # Build 10-checkpoint trajectory with exit clipping. Checkpoints span the hold
    # window evenly, so a 120s scalp and a 600s swing both yield 10 comparable points.
    trajectory: list[float] = []
    checkpoints = make_checkpoints(hold_limit)
    exit_abs_ts = entry_wall_clock + pd.Timedelta(seconds=exit_time_offset_s)

    for checkpoint_s in checkpoints:
        checkpoint_ts = entry_wall_clock + pd.Timedelta(seconds=checkpoint_s)
        if checkpoint_ts <= exit_abs_ts:
            price = _lookup_price_at_or_after(future_ticks, checkpoint_ts, exit_price)
            # Clamping `exit_price` alone does not close the overshoot leak. Checkpoints at
            # or before the exit are looked up forward, so whenever no tick falls between a
            # checkpoint and the breach, the checkpoint resolves to the *breaching* tick and
            # carries its uncollectable price. Measured: entry 50c, ticks at +10s (50c) and
            # +30s (64c), tp=5 — exit_price clamps to 55 but traj_0 and traj_1 both reported
            # logit_delta(50, 64) = 0.5754 against the collectable 0.2007.
            #
            # On a take-profit path the position ceased to exist at the resting limit, so no
            # checkpoint can report a move beyond it. `exit_price` already *is* that limit
            # here, which makes it the cap. Direction-agnostic via entry_side, so BUY NO
            # (favourable = price down) is capped at entry - tp.
            if exit_reason == "take_profit" and entry_side * (price - entry_yes_bid) > tp:
                price = exit_price
        else:
            # After exit: freeze at exit price
            price = exit_price
        trajectory.append(_logit_delta(entry_yes_bid, price))

    simulated_pnl = entry_side * (exit_price - entry_yes_bid)

    return ExitResult(
        exit_price=exit_price,
        exit_reason=exit_reason,
        exit_time_offset_s=exit_time_offset_s,
        simulated_pnl=simulated_pnl,
        trajectory=trajectory,
    )


def build_trajectory_targets(
    entry_rows: pd.DataFrame,
    all_ticks: pd.DataFrame,
    all_possessions: pd.DataFrame,
    # Keyword-only from here. `feed_delay_seconds` as the 4th *positional* parameter meant a
    # call written against the old signature — build_trajectory_targets(rows, ticks, poss,
    # 5.0, 3.0) — silently bound feed_delay_seconds=5.0 and tp=3.0, and pd.Timedelta accepts
    # the float without complaint. Keyword-only makes that a TypeError, which is the point of
    # having no default in the first place.
    *,
    feed_delay_seconds: int,
    tp: float = 5.0,
    sl: float = 3.0,
    horizon_seconds: int = DEFAULT_HORIZON_SECONDS,
) -> pd.DataFrame:
    """
    Build trajectory targets for all entry rows with Kalshi tick data.

    Args:
        entry_rows: possession rows that have wall_clock_ts and yes_bid — one row per
                    potential trade entry. Must have columns:
                      game_id, wall_clock_ts, yes_bid, current_run_team,
                      current_run_team_encoded (1=home, -1=away, 0=none)
        all_ticks: all Kalshi ticks (main.kalshi_ticks), must have ts, game_id, yes_bid
        all_possessions: all possessions (features.possession_flat), must have
                         game_id, wall_clock_ts, current_run_team, is_blowout, is_garbage_time
        feed_delay_seconds: lag from game event to actionable, in seconds. Required, with
                         no default, deliberately: omitting it is precisely the defect this
                         parameter exists to prevent, so a new call site has to state it.
                         Must be the same value `dataset._join_ticks_to_possessions` used
                         to attach `yes_bid`, or the entry price and the exit window are
                         anchored at different moments.

    Returns:
        entry_rows with added columns:
          traj_0 .. traj_9  — 10 logit-delta checkpoints
          exit_price, exit_reason, exit_time_offset_s, simulated_pnl
    """
    if entry_rows.empty:
        return entry_rows

    ticks_sorted = all_ticks.sort_values("ts").copy()
    poss_sorted = all_possessions.sort_values("wall_clock_ts").copy()

    traj_cols = [f"traj_{i}" for i in range(N_CHECKPOINTS)]
    meta_cols = ["exit_price", "exit_reason", "exit_time_offset_s", "simulated_pnl"]

    # Reset index to ensure contiguous 0..N-1 labels — we write results back by label.
    out = entry_rows.copy().reset_index(drop=True)
    for col in traj_cols + ["exit_price", "exit_time_offset_s", "simulated_pnl"]:
        out[col] = np.nan
    out["exit_reason"] = None

    # Process per-game. Iterate over the reset-indexed `out` so idx is always valid.
    for game_id, game_entries in out.groupby("game_id"):
        game_ticks = ticks_sorted[ticks_sorted["game_id"] == game_id].copy()
        game_poss = poss_sorted[poss_sorted["game_id"] == game_id].copy()

        if game_ticks.empty:
            logger.warning("No ticks for game %s — skipping %d entries", game_id, len(game_entries))
            continue

        for idx, row in game_entries.iterrows():
            # The exit window opens when the position can first exist, not at the
            # possession's wall clock. `yes_bid` on this row was read at
            # `wall_clock_ts + feed_delay_seconds` by the backward asof join in
            # dataset._join_ticks_to_possessions, so anchoring here at `wall_clock_ts`
            # let TP/SL resolve against ticks that preceded the entry price — the labels
            # were anti-causal, not merely early, and Head B learned that as signal.
            #
            # This single name feeds all four downstream uses: both `future_*` filters,
            # the checkpoint grid, and the time gate (simulate_exit computes
            # `deadline = entry_wall_clock + horizon`, so the hold is a true
            # `horizon_seconds` measured from entry once the anchor is right — it used to
            # be `wall_clock_ts + horizon`, i.e. 100s of real hold at a 20s delay).
            #
            # No staleness bound is needed here, unlike the backtest's `_tick_at_delay`:
            # dataset.py has already dropped every row whose asof found no tick within
            # MARKET_STALENESS_TOLERANCE_SECONDS of the anchor, so each row reaching this
            # loop carries a bid that was genuinely observable at `entry_anchor_ts`.
            entry_anchor_ts: pd.Timestamp = pd.Timestamp(
                row["wall_clock_ts"]
            ) + pd.Timedelta(seconds=feed_delay_seconds)
            entry_bid = float(row["yes_bid"])
            entry_run_team = row.get("current_run_team", None)

            run_encoded = row.get("current_run_team_encoded", 0)
            entry_side = 1 if run_encoded >= 0 else -1

            future_ticks = game_ticks[game_ticks["ts"] > entry_anchor_ts]
            future_poss = game_poss[game_poss["wall_clock_ts"] > entry_anchor_ts]

            sim = simulate_exit(
                entry_wall_clock=entry_anchor_ts,
                entry_yes_bid=entry_bid,
                entry_run_team=entry_run_team,
                future_ticks=future_ticks,
                future_possessions=future_poss,
                tp=tp,
                sl=sl,
                entry_side=entry_side,
                max_seconds=horizon_seconds,
            )

            for i, val in enumerate(sim.trajectory):
                out.at[idx, f"traj_{i}"] = val
            out.at[idx, "exit_price"]        = sim.exit_price
            out.at[idx, "exit_reason"]        = sim.exit_reason
            out.at[idx, "exit_time_offset_s"] = sim.exit_time_offset_s
            out.at[idx, "simulated_pnl"]      = sim.simulated_pnl

    traj_filled = out[traj_cols].notna().all(axis=1).sum()
    logger.info("build_trajectory_targets: filled trajectories for %d/%d rows", traj_filled, len(out))
    return out
