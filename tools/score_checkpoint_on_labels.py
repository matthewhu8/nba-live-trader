"""
Score a saved checkpoint against regenerated Head B labels. Inference only — no training.

Level 2 fixed the labels but not the checkpoint, so the question this answers is: how much of
Head B's reported skill was the anti-causal leakage? Same model, same features, same rows —
only the target changes. Old labels are reproduced by passing `feed_delay_seconds=0`, which
restores the anchor defect exactly (the exit search starts at `wall_clock_ts` while `yes_bid`
still comes from the `wct + 20s` join).

Runs off the local parquet cache, replicating `build_dataset`'s chain in order, so it needs no
MotherDuck scan (the free-tier daily compute limit is a real constraint). Every cache row
post-dates HEADB_SPLIT_DATE, so all of it is held-out val for `mmoe_delay20.pt`.

WHY THE GATED POPULATION IS THE ONE THAT MATTERS. Head B's raw output is wildly miscalibrated
over the full joint set — mean |traj| ~6.4 against labels of ~0.12, with values past -300 at
the extreme price wings, consistent with the scaler-era mismatch in `data-integrity.md` (the
cache's tick window sits entirely outside the scaler's fit window). Head A's gate filters
almost all of that out: only ~252 of 11,447 rows clear the full entry gate, and on those the
magnitudes are sane. Metrics over all joint rows are therefore dominated by rows the strategy
never trades. Report the gated population; keep the others only as context.

Usage:
    ./venv/bin/python -m tools.score_checkpoint_on_labels
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.mmoe_backtest import _build_feature_dict, aggregate_traj
from models.mmoe.dataset import (
    FEED_DELAY_SECONDS_NBA,
    MARKET_COLS,
    _add_derived_features,
    _filter_to_traded_regime,
    _join_pregame,
    _join_ticks_to_possessions,
    _select_home_best_contract,
)
from models.mmoe.predictor import MMoEPredictor
from models.targets.exit_simulator import build_trajectory_targets

CACHE = Path("data/feature_store")
TRAJ_COLS = [f"traj_{i}" for i in range(10)]


def _load_joint(delay: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    poss = pd.read_parquet(CACHE / "possession_flat.parquet")
    poss = _join_pregame(poss, pd.read_parquet(CACHE / "pregame.parquet"))
    # Order matters, same as build_dataset: pregame join before derived features (pace
    # shrinkage needs expected_pace), regime filter after (rolling windows need the full
    # possession sequence).
    poss = _add_derived_features(poss)
    poss = _filter_to_traded_regime(poss)
    ticks = _select_home_best_contract(pd.read_parquet(CACHE / "kalshi_ticks.parquet"), poss)
    joint, _ = _join_ticks_to_possessions(poss, ticks, feed_delay_seconds=delay)
    return joint, ticks, poss


def _labels(joint, ticks, poss, delay: int) -> np.ndarray:
    out = build_trajectory_targets(
        entry_rows=joint, all_ticks=ticks, all_possessions=poss,
        feed_delay_seconds=delay, tp=5.0, sl=3.0, horizon_seconds=120,
    )
    return out[TRAJ_COLS].to_numpy(dtype=float)


def _dir_acc(P: np.ndarray, Y: np.ndarray) -> tuple[float, float, int]:
    """Sign agreement on the final checkpoint, on rows with a meaningful move.

    Threshold 0.05 and the final-checkpoint choice both mirror `trainer.py`'s dir_acc_b, so
    the number is comparable to the 62.5% recorded for this checkpoint.
    """
    d = np.abs(Y[:, -1]) > 0.05
    n = int(d.sum())
    if n == 0:
        return float("nan"), float("nan"), 0
    acc = float((np.sign(P[d, -1]) == np.sign(Y[d, -1])).mean())
    return acc, float(np.sqrt(acc * (1 - acc) / n)), n


def _report(name: str, P: np.ndarray, Y: np.ndarray) -> None:
    ok = ~np.isnan(Y).any(axis=1)
    P, Y = P[ok], Y[ok]
    acc, se, n = _dir_acc(P, Y)
    rmse = float(np.sqrt(((P - Y) ** 2).mean()))
    # A model predicting 0.0 everywhere. Head B is only worth keeping if it beats this.
    zero = float(np.sqrt((Y ** 2).mean()))
    print(f"  {name:<22} dir acc {acc:.3f} [{acc - 1.96*se:.3f}, {acc + 1.96*se:.3f}] on {n:,} rows")
    print(f"  {'':<22} RMSE {rmse:.3f} vs zero-predictor {zero:.3f}  (ratio {rmse/zero:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delay", type=int, default=FEED_DELAY_SECONDS_NBA)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--min-abs-traj", type=float, default=0.08)
    args = ap.parse_args()

    joint, ticks, poss = _load_joint(args.delay)
    print(f"joint rows: {len(joint):,}")

    Y_new = _labels(joint, ticks, poss, args.delay)
    Y_old = _labels(joint, ticks, poss, 0)   # anchor defect restored

    pred = MMoEPredictor.load()
    outs = [pred.predict(_build_feature_dict(r, {c: float(r[c]) for c in MARKET_COLS}))
            for _, r in joint.iterrows()]
    P = np.array([o.trajectory for o in outs], dtype=float)
    run_prob = np.array([o.run_prob for o in outs], dtype=float)

    bid = joint["yes_bid"].to_numpy(dtype=float)
    run_len = joint["current_run_length"].fillna(0).to_numpy(dtype=float)
    traj_used = np.array([aggregate_traj(list(t), "mean") for t in P])
    gate = (
        (run_prob >= args.threshold) & (bid >= 30) & (bid <= 70)
        & (run_len >= 2) & (np.abs(traj_used) >= args.min_abs_traj)
    )

    print(f"\nALL joint rows (context only — mean |traj| {np.abs(P).mean():.2f}, "
          f"miscalibrated outside the gate):")
    _report("vs old labels", P, Y_old)
    _report("vs corrected labels", P, Y_new)

    print(f"\nGATED — the rows the strategy trades ({gate.sum():,} rows, mean |traj| "
          f"{np.abs(P[gate]).mean():.3f}):")
    _report("vs old labels", P[gate], Y_old[gate])
    _report("vs corrected labels", P[gate], Y_new[gate])


if __name__ == "__main__":
    main()
