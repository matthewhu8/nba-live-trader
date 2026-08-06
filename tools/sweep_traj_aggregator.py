"""
Phase 1 sweep — 4×5×2 grid over (traj_aggregator, min_abs_traj, use_traj_for_side).

Loads data once, runs all 40 configurations, writes a single summary CSV plus
the per-position CSV for the winner.

Goal: find the (aggregator, threshold, side_source) combination with the
highest net P&L on the current val game set, subject to n_trades >= 50 and
win_rate >= 0.45.

Run from project root:
    ./venv/bin/python -m tools.sweep_traj_aggregator
"""

from __future__ import annotations

import itertools
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.mmoe_backtest import (
    BOOL_COLS,
    FEED_DELAY_SECONDS_NBA,
    HEADB_SPLIT_DATE,
    TRAJ_AGGREGATORS,
    _add_derived_features,
    _connect_motherduck,
    _join_pregame,
    _load_kalshi_ticks,
    _load_possession_flat,
    _load_pregame,
    _run_game,
    _select_home_best_contract,
)
from models.mmoe.predictor import MMoEPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("sweep")


# Sweep dimensions
THRESHOLDS = (0.08, 0.10, 0.12, 0.15, 0.20)
SIDE_SOURCES = (True, False)  # True = use_traj_for_side


@dataclass
class SweepRow:
    aggregator:     str
    min_abs_traj:   float
    use_traj_side:  bool
    n_trades:       int
    n_wins:         int
    win_rate:       float
    gross_pnl_100c: float       # gross PnL scaled to 100 contracts
    net_pnl_100c:   float       # net PnL scaled to 100 contracts
    avg_hold_s:     float
    exit_tp:        int
    exit_sl:        int
    exit_momentum:  int
    exit_time:      int
    median_abs_traj_used: float


def _exit_count(positions: list, reason: str) -> int:
    return sum(1 for p in positions if p.exit_reason == reason)


def main() -> None:
    contracts = 100  # same as the validated baseline scaling

    # --- Load data once ---
    logger.info("Connecting to MotherDuck...")
    conn = _connect_motherduck()
    logger.info("Loading possessions / ticks / pregame...")
    all_poss  = _load_possession_flat(conn)
    all_ticks = _load_kalshi_ticks(conn)
    pregame   = _load_pregame(conn)
    conn.close()

    all_poss = _add_derived_features(all_poss)
    all_poss = _join_pregame(all_poss, pregame)
    all_ticks = _select_home_best_contract(all_ticks, all_poss)

    all_poss["game_date"] = pd.to_datetime(all_poss["game_date"])
    val_mask  = all_poss["game_date"] >= HEADB_SPLIT_DATE
    val_games = set(all_poss.loc[val_mask, "game_id"].unique())
    tick_games = set(all_ticks["game_id"].unique())
    tradeable = sorted(val_games & tick_games)
    logger.info("Tradeable val games: %d", len(tradeable))

    val_poss  = all_poss[all_poss["game_id"].isin(tradeable)].copy()
    val_ticks = all_ticks[all_ticks["game_id"].isin(tradeable)].copy()
    val_poss["wall_clock_ts"]  = pd.to_datetime(val_poss["wall_clock_ts"], utc=True)
    val_ticks["ts"]            = pd.to_datetime(val_ticks["ts"], utc=True)
    for col in BOOL_COLS:
        if col in val_poss.columns:
            val_poss[col] = val_poss[col].astype(float)

    # Pre-slice per-game (avoids 40 × N games of dataframe filtering)
    per_game_poss = {gid: val_poss[val_poss["game_id"] == gid].copy() for gid in tradeable}
    per_game_ticks = {gid: val_ticks[val_ticks["game_id"] == gid].copy() for gid in tradeable}

    logger.info("Loading MMoE predictor...")
    predictor = MMoEPredictor.load(
        Path("models/saved/mmoe_delay20.pt"),
        Path("models/saved/mmoe_scaler_delay20.pkl"),
    )

    # --- Sweep ---
    rows: list[SweepRow] = []
    total = len(TRAJ_AGGREGATORS) * len(THRESHOLDS) * len(SIDE_SOURCES)
    idx = 0
    winner_positions = None
    winner_label = None
    winner_net = -np.inf

    for aggregator, threshold, use_side in itertools.product(TRAJ_AGGREGATORS, THRESHOLDS, SIDE_SOURCES):
        idx += 1
        label = f"agg={aggregator} thr={threshold:.2f} traj_side={use_side}"
        logger.info("[%d/%d] %s", idx, total, label)

        all_positions = []
        for gid in tradeable:
            positions = _run_game(
                game_id=gid,
                game_poss=per_game_poss[gid],
                game_ticks=per_game_ticks[gid],
                predictor=predictor,
                run_prob_threshold=0.0,  # no Head A gate (memory: it hurts P&L)
                tp=5.0,
                sl=3.0,
                contracts=contracts,
                feed_delay_s=FEED_DELAY_SECONDS_NBA,
                use_traj_for_side=use_side,
                min_abs_traj=threshold,
                min_run_length=2,        # restored Phase 0 default
                hold_seconds=240,
                traj_aggregator=aggregator,
            )
            all_positions.extend(positions)

        n = len(all_positions)
        if n == 0:
            rows.append(SweepRow(
                aggregator=aggregator,
                min_abs_traj=threshold,
                use_traj_side=use_side,
                n_trades=0, n_wins=0, win_rate=0.0,
                gross_pnl_100c=0.0, net_pnl_100c=0.0, avg_hold_s=0.0,
                exit_tp=0, exit_sl=0, exit_momentum=0, exit_time=0,
                median_abs_traj_used=0.0,
            ))
            logger.info("   → 0 trades")
            continue

        n_wins = sum(1 for p in all_positions if p.net_pnl > 0)
        gross = sum(pnl_dollars(p.gross_pnl, contracts) for p in all_positions)
        net   = sum(p.net_pnl for p in all_positions)
        avg_hold = float(np.mean([p.hold_time_s for p in all_positions]))
        median_abs_traj = float(np.median([abs(p.traj_used) for p in all_positions]))

        rows.append(SweepRow(
            aggregator=aggregator,
            min_abs_traj=threshold,
            use_traj_side=use_side,
            n_trades=n,
            n_wins=n_wins,
            win_rate=n_wins / n,
            gross_pnl_100c=round(gross, 2),
            net_pnl_100c=round(net, 2),
            avg_hold_s=round(avg_hold, 1),
            exit_tp=_exit_count(all_positions, "take_profit"),
            exit_sl=_exit_count(all_positions, "stop_loss"),
            exit_momentum=_exit_count(all_positions, "momentum_flip"),
            exit_time=_exit_count(all_positions, "time_gate"),
            median_abs_traj_used=round(median_abs_traj, 4),
        ))
        logger.info(
            "   → %d trades  wr=%.1f%%  net=$%+.0f",
            n, 100.0 * n_wins / n, net,
        )

        # Track winner under success criterion (net P&L max with floor constraints)
        if n >= 50 and (n_wins / n) >= 0.45 and net > winner_net:
            winner_net = net
            winner_positions = all_positions
            winner_label = label

    # --- Write outputs ---
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("backtesting/results/sweeps")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / f"phase1_sweep_summary_{ts_str}.csv"
    pd.DataFrame([vars(r) for r in rows]).to_csv(summary_path, index=False)
    logger.info("Summary written to %s", summary_path)

    if winner_positions is not None:
        winner_path = out_dir / f"phase1_sweep_winner_{ts_str}.csv"
        pd.DataFrame([vars(p) for p in winner_positions]).to_csv(winner_path, index=False)
        logger.info("Winner: %s  net=$%+.0f  positions saved to %s",
                    winner_label, winner_net, winner_path)
    else:
        logger.warning(
            "No config met the success criterion (n>=50, win_rate>=0.45, max net P&L). "
            "Top 3 by net P&L regardless of floor:"
        )
        for r in sorted(rows, key=lambda r: -r.net_pnl_100c)[:3]:
            logger.warning(
                "   agg=%s thr=%.2f traj_side=%s  n=%d  wr=%.1f%%  net=$%+.0f",
                r.aggregator, r.min_abs_traj, r.use_traj_side,
                r.n_trades, 100 * r.win_rate, r.net_pnl_100c,
            )


if __name__ == "__main__":
    main()
