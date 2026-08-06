"""
Phase 2 sweep — dynamic-exit rule comparison.

Given a Phase 1 winning config (aggregator + threshold + side source), run four
variants:
    1. baseline      — reversal OFF, streak OFF (matches Phase 1 behavior)
    2. reversal_only — reversal ON,  streak OFF
    3. streak_only   — reversal OFF, streak ON
    4. both_on       — reversal ON,  streak ON

The streak/reversal comparison aggregator is configurable separately from the
entry aggregator; default is "final" because it has the lowest variance.

Run from project root, passing the Phase 1 winner config via CLI:
    ./venv/bin/python -m tools.sweep_dynamic_exit \\
        --aggregator max_abs --threshold 0.12 --use-traj-for-side \\
        --streak-compare final
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.dynamic_exit import simulate_exit_dynamic
from backtesting.mmoe_backtest import (
    BOOL_COLS,
    FEED_DELAY_SECONDS_NBA,
    HEADB_SPLIT_DATE,
    TRAJ_AGGREGATORS,
    _add_derived_features,
    _build_feature_dict,
    _compute_fees,
    pnl_dollars,
    _compute_market_features_for_game,
    _connect_motherduck,
    _get_market_features_at_delay,
    _join_pregame,
    _load_kalshi_ticks,
    _load_possession_flat,
    _load_pregame,
    _select_home_best_contract,
    aggregate_traj,
)
from models.mmoe.predictor import MMoEPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("sweep2")


@dataclass
class Trade:
    game_id:        str
    possession_id:  int
    wall_clock_ts:  pd.Timestamp
    entry_price:    float
    exit_price:     float
    exit_reason:    str
    entry_side:     int
    hold_time_s:    float
    gross_pnl:      float
    net_pnl:        float
    run_prob:       float
    traj_used:      float
    traj_aggregator: str
    streak_compare: str
    reversal_fired: bool
    max_streak:     int
    tp_widened:     bool
    n_reinference:  int
    quarter:        int
    score_diff:     int
    run_length:     int
    variant:        str


def _run_one_variant(
    *,
    variant_name:               str,
    tradeable:                  list[str],
    per_game_poss:              dict[str, pd.DataFrame],
    per_game_enriched_ticks:    dict[str, pd.DataFrame],
    predictor:                  MMoEPredictor,
    entry_aggregator:           str,
    streak_compare_aggregator:  str,
    min_abs_traj:               float,
    use_traj_for_side:          bool,
    reversal_enabled:           bool,
    streak_enabled:             bool,
    contracts:                  int,
    hold_seconds:               int,
    tp:                         float,
    sl:                         float,
) -> list[Trade]:
    trades: list[Trade] = []

    for game_id in tradeable:
        game_poss = per_game_poss[game_id]
        enriched_ticks = per_game_enriched_ticks[game_id]

        prev_yes_bid = 50.0
        prev_spread = 1.0
        position_exit_ts: pd.Timestamp | None = None

        for _, row in game_poss.iterrows():
            wct = row["wall_clock_ts"]

            if row.get("is_blowout", False) or row.get("is_garbage_time", False):
                continue

            market_feats = _get_market_features_at_delay(
                enriched_game_ticks=enriched_ticks,
                wall_clock_ts=wct,
                prev_yes_bid=prev_yes_bid,
                prev_spread=prev_spread,
                delay_s=FEED_DELAY_SECONDS_NBA,
            )
            yes_bid = market_feats["yes_bid"]
            if yes_bid > 0:
                prev_yes_bid = yes_bid
                prev_spread = market_feats["spread"]

            if market_feats["has_market_data"] == 0.0:
                continue
            if position_exit_ts is not None and wct < position_exit_ts:
                continue
            if int(row.get("current_run_length", 0)) < 2:
                continue

            fd = _build_feature_dict(row, market_feats)
            output = predictor.predict(fd)

            if not (30 <= yes_bid <= 70):
                continue

            traj_used = aggregate_traj(output.trajectory, entry_aggregator)
            if abs(traj_used) < min_abs_traj:
                continue

            if use_traj_for_side:
                entry_side = 1 if traj_used >= 0 else -1
            else:
                run_team_encoded = fd.get("current_run_team_encoded", 0.0)
                entry_side = 1 if run_team_encoded >= 0 else -1

            # Entry value re-aggregated using the streak-compare aggregator so the
            # streak/reversal rules compare apples-to-apples mid-trade.
            entry_traj_for_compare = aggregate_traj(output.trajectory, streak_compare_aggregator)

            future_ticks = enriched_ticks[enriched_ticks["ts"] > wct]
            future_poss = game_poss[game_poss["wall_clock_ts"] > wct]

            sim = simulate_exit_dynamic(
                entry_wall_clock=wct,
                entry_yes_bid=yes_bid,
                entry_run_team=row.get("current_run_team", None),
                future_ticks=future_ticks,
                future_possessions=future_poss,
                enriched_ticks_for_game=enriched_ticks,
                predictor=predictor,
                entry_traj_for_compare=entry_traj_for_compare,
                streak_compare_aggregator=streak_compare_aggregator,
                tp=tp,
                sl=sl,
                entry_side=entry_side,
                max_seconds=hold_seconds,
                reversal_enabled=reversal_enabled,
                reversal_min_abs=0.05,
                streak_enabled=streak_enabled,
                streak_widen_at=2,
                streak_widen_cents=3.0,
                feed_delay_s=FEED_DELAY_SECONDS_NBA,
                prev_yes_bid_init=prev_yes_bid,
                prev_spread_init=prev_spread,
            )

            gross = entry_side * (sim.exit_price - yes_bid)
            fees = _compute_fees(yes_bid, sim.exit_price, contracts, sim.exit_reason)
            net = pnl_dollars(gross, contracts) - fees

            trades.append(Trade(
                game_id=str(game_id),
                possession_id=int(row.get("possession_id", row.get("event_id", 0))),
                wall_clock_ts=wct,
                entry_price=float(yes_bid),
                exit_price=float(sim.exit_price),
                exit_reason=sim.exit_reason,
                entry_side=entry_side,
                hold_time_s=float(sim.exit_time_offset_s),
                gross_pnl=float(gross),
                net_pnl=float(net),
                run_prob=float(output.run_prob),
                traj_used=float(traj_used),
                traj_aggregator=entry_aggregator,
                streak_compare=streak_compare_aggregator,
                reversal_fired=sim.reversal_fired,
                max_streak=sim.max_streak,
                tp_widened=sim.tp_widened,
                n_reinference=sim.n_reinference,
                quarter=int(row.get("period", 0)),
                score_diff=int(row.get("score_diff", 0)),
                run_length=int(row.get("current_run_length", 0)),
                variant=variant_name,
            ))

            position_exit_ts = wct + pd.Timedelta(seconds=sim.exit_time_offset_s)

    return trades


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregator", choices=list(TRAJ_AGGREGATORS), required=True,
                        help="entry-signal aggregator (Phase 1 winner)")
    parser.add_argument("--threshold", type=float, required=True,
                        help="min |traj_used| at entry (Phase 1 winner)")
    parser.add_argument("--use-traj-for-side", action="store_true",
                        help="set BUY direction from traj sign (Phase 1 winner)")
    parser.add_argument("--streak-compare", choices=list(TRAJ_AGGREGATORS), default="final",
                        help="aggregator used for streak/reversal comparison (NOT entry aggregator)")
    parser.add_argument("--contracts", type=int, default=100)
    parser.add_argument("--hold-seconds", type=int, default=240)
    parser.add_argument("--tp", type=float, default=5.0)
    parser.add_argument("--sl", type=float, default=3.0)
    args = parser.parse_args()

    contracts = args.contracts

    logger.info("Connecting to MotherDuck...")
    conn = _connect_motherduck()
    logger.info("Loading data...")
    all_poss = _load_possession_flat(conn)
    all_ticks = _load_kalshi_ticks(conn)
    pregame = _load_pregame(conn)
    conn.close()

    all_poss = _add_derived_features(all_poss)
    all_poss = _join_pregame(all_poss, pregame)
    all_ticks = _select_home_best_contract(all_ticks, all_poss)

    all_poss["game_date"] = pd.to_datetime(all_poss["game_date"])
    val_mask = all_poss["game_date"] >= HEADB_SPLIT_DATE
    val_games = set(all_poss.loc[val_mask, "game_id"].unique())
    tick_games = set(all_ticks["game_id"].unique())
    tradeable = sorted(val_games & tick_games)
    logger.info("Tradeable val games: %d", len(tradeable))

    val_poss = all_poss[all_poss["game_id"].isin(tradeable)].copy()
    val_ticks = all_ticks[all_ticks["game_id"].isin(tradeable)].copy()
    val_poss["wall_clock_ts"] = pd.to_datetime(val_poss["wall_clock_ts"], utc=True)
    val_ticks["ts"] = pd.to_datetime(val_ticks["ts"], utc=True)
    for col in BOOL_COLS:
        if col in val_poss.columns:
            val_poss[col] = val_poss[col].astype(float)

    # Pre-compute per-game enriched ticks (with market feature columns)
    per_game_poss = {gid: val_poss[val_poss["game_id"] == gid].copy() for gid in tradeable}
    per_game_enriched: dict[str, pd.DataFrame] = {}
    for gid in tradeable:
        gt = val_ticks[val_ticks["game_id"] == gid].copy()
        if gt.empty:
            per_game_enriched[gid] = gt
            continue
        et = _compute_market_features_for_game(gt)
        et["ts"] = pd.to_datetime(et["ts"], utc=True)
        per_game_enriched[gid] = et.sort_values("ts").reset_index(drop=True)

    logger.info("Loading MMoE predictor...")
    predictor = MMoEPredictor.load(
        Path("models/saved/mmoe_delay20.pt"),
        Path("models/saved/mmoe_scaler_delay20.pkl"),
    )

    variants = [
        ("baseline",      False, False),
        ("reversal_only", True,  False),
        ("streak_only",   False, True),
        ("both_on",       True,  True),
    ]

    all_trades: list[Trade] = []
    for vname, rev, streak in variants:
        logger.info("--- Variant: %s (reversal=%s, streak=%s) ---", vname, rev, streak)
        trades = _run_one_variant(
            variant_name=vname,
            tradeable=tradeable,
            per_game_poss=per_game_poss,
            per_game_enriched_ticks=per_game_enriched,
            predictor=predictor,
            entry_aggregator=args.aggregator,
            streak_compare_aggregator=args.streak_compare,
            min_abs_traj=args.threshold,
            use_traj_for_side=args.use_traj_for_side,
            reversal_enabled=rev,
            streak_enabled=streak,
            contracts=contracts,
            hold_seconds=args.hold_seconds,
            tp=args.tp,
            sl=args.sl,
        )
        all_trades.extend(trades)

        if trades:
            n = len(trades)
            wins = sum(1 for t in trades if t.net_pnl > 0)
            net = sum(t.net_pnl for t in trades)
            reversal_count = sum(1 for t in trades if t.reversal_fired)
            tp_widened_count = sum(1 for t in trades if t.tp_widened)
            logger.info(
                "  → n=%d  wr=%.1f%%  net=$%+.0f  reversal=%d  tp_widened=%d",
                n, 100.0 * wins / n, net, reversal_count, tp_widened_count,
            )
        else:
            logger.info("  → 0 trades")

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("backtesting/results/sweeps")
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_path = out_dir / f"phase2_sweep_trades_{ts_str}.csv"
    pd.DataFrame([vars(t) for t in all_trades]).to_csv(trades_path, index=False)
    logger.info("Trades saved to %s", trades_path)

    # Summary table
    summary_rows = []
    for vname, _, _ in variants:
        v_trades = [t for t in all_trades if t.variant == vname]
        if not v_trades:
            summary_rows.append({"variant": vname, "n_trades": 0})
            continue
        wins = sum(1 for t in v_trades if t.net_pnl > 0)
        summary_rows.append({
            "variant": vname,
            "n_trades": len(v_trades),
            "win_rate": round(wins / len(v_trades), 3),
            "net_pnl_100c": round(sum(t.net_pnl for t in v_trades), 2),
            "avg_hold_s": round(float(np.mean([t.hold_time_s for t in v_trades])), 1),
            "reversal_count": sum(1 for t in v_trades if t.reversal_fired),
            "tp_widened_count": sum(1 for t in v_trades if t.tp_widened),
            "exit_tp": sum(1 for t in v_trades if t.exit_reason == "take_profit"),
            "exit_tp_widened": sum(1 for t in v_trades if t.exit_reason == "take_profit_widened"),
            "exit_sl": sum(1 for t in v_trades if t.exit_reason == "stop_loss"),
            "exit_momentum": sum(1 for t in v_trades if t.exit_reason == "momentum_flip"),
            "exit_reversal": sum(1 for t in v_trades if t.exit_reason == "reversal_exit"),
            "exit_time": sum(1 for t in v_trades if t.exit_reason == "time_gate"),
        })
    summary_path = out_dir / f"phase2_sweep_summary_{ts_str}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
