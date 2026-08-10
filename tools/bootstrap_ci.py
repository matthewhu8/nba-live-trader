"""
Game-clustered bootstrap CI on per-trade edge.

Why this exists as a committed script rather than an ad-hoc snippet: the original
`[-$2.40, -$1.17]` was computed once, in a session, and never written down as code. It then
could not be carried across a code change, so every subsequent number has been quoted either
without a CI or with a stale one. A rerunnable artefact fixes that.

WHY GAME-CLUSTERED. Trades inside one game are correlated — same game state, often the same
run, overlapping tick data, correlated model errors. Resampling the 181 individual trades
would treat them as 181 independent draws and produce an interval that is too narrow. We
resample the ~44 GAMES with replacement instead, so the effective sample size is the number of
games, not the number of trades. Trade count varies between resamples because games differ in
trade count; that is correct, not a bug.

WHY PER-TRADE AND NOT TOTAL. With a negative per-trade edge, any change that trades less
improves total net P&L. A retrained model reporting a smaller loss may simply be trading less
at identical (bad) per-trade economics. Per-trade edge is the quantity that stays comparable
when the trade population changes — which it WILL after a retrain, since `run_prob` and
`traj_used` gate entry (`mmoe_backtest.py:447,458`).

Usage:
    ./venv/bin/python -m tools.bootstrap_ci backtesting/results/<run>.csv
    ./venv/bin/python -m tools.bootstrap_ci --compare <before>.csv <after>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_RESAMPLES = 10_000
SEED = 0
KEY = ["game_id", "possession_id"]


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"game_id", "possession_id", "net_pnl"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing required columns {sorted(missing)}")
    return df


def _cluster_bootstrap(
    values_by_game: list[np.ndarray],
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> np.ndarray:
    """Resample whole games with replacement; return the resampled means.

    `values_by_game[i]` is every per-trade value from game i. We draw len(games) game indices
    with replacement, pool their trades, and take the mean — repeated n_resamples times.
    """
    rng = np.random.default_rng(seed)
    n_games = len(values_by_game)
    means = np.empty(n_resamples)
    for b in range(n_resamples):
        picked = rng.integers(0, n_games, size=n_games)
        means[b] = np.concatenate([values_by_game[i] for i in picked]).mean()
    return means


def _report(label: str, values_by_game: list[np.ndarray], seed: int = SEED) -> None:
    flat = np.concatenate(values_by_game)
    means = _cluster_bootstrap(values_by_game, seed=seed)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # One-sided: how often does a resample show no loss? The strategy question is whether the
    # edge could be >= 0, not whether it differs from some other value.
    p_nonneg = float((means >= 0).mean())

    print(f"\n{label}")
    print(f"  trades {len(flat)} across {len(values_by_game)} games")
    print(f"  per-trade mean   {flat.mean():+.4f}")
    print(f"  95% CI           [{lo:+.4f}, {hi:+.4f}]   ({N_RESAMPLES:,} game-clustered resamples)")
    print(f"  P(edge >= 0)     {p_nonneg:.4f}" + ("  (< 1e-4)" if p_nonneg == 0 else ""))
    print(f"  total            {flat.sum():+.2f}")


def _by_game(df: pd.DataFrame, col: str) -> list[np.ndarray]:
    return [g[col].to_numpy(dtype=float) for _, g in df.groupby("game_id")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path, nargs="?", help="a backtest result CSV")
    ap.add_argument("--compare", type=Path, nargs=2, metavar=("BEFORE", "AFTER"),
                    help="paired delta between two runs over the same trades")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if args.compare:
        before, after = (_load(p) for p in args.compare)
        _report(f"BEFORE  {args.compare[0].name}", _by_game(before, "net_pnl"), args.seed)
        _report(f"AFTER   {args.compare[1].name}", _by_game(after, "net_pnl"), args.seed)

        # The paired delta is only meaningful while both runs trade the SAME positions. That
        # held for the possession-delay change by 2.6s of overlap-guard margin — contingent,
        # not structural — and it will NOT hold across a retrain, because Head A's run_prob and
        # Head B's traj_used gate entry. Without this check a plain merge would silently drop
        # non-matching rows and report a paired delta over whatever subset happened to align.
        sb = set(map(tuple, before[KEY].values))
        sa = set(map(tuple, after[KEY].values))
        if sb != sa:
            print(f"\nTrade sets differ (before-only {len(sb - sa)}, after-only {len(sa - sb)}, "
                  f"shared {len(sb & sa)}).")
            print("Refusing to pair — the two CIs above are the honest comparison. They are not")
            print("independent samples of the same population, so do NOT read overlap as")
            print("'no significant difference'.")
            return

        m = before.merge(after, on=KEY, suffixes=("_b", "_a"))
        m["delta"] = m["net_pnl_a"] - m["net_pnl_b"]
        _report("PAIRED DELTA  (after - before, same trades)", _by_game(m, "delta"), args.seed)
        changed = int((m["delta"].abs() > 1e-9).sum())
        print(f"  trades whose P&L moved: {changed} of {len(m)}")
    elif args.csv:
        _report(args.csv.name, _by_game(_load(args.csv), "net_pnl"), args.seed)
    else:
        ap.error("pass a CSV, or --compare BEFORE AFTER")


if __name__ == "__main__":
    main()
