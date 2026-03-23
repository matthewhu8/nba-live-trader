"""
Strategy sweep — runs all strategies across parameter variants and prints a ranked summary.

Usage:  python -m backtesting.strategy_sweep
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from backtesting.simulator import BacktestSimulator, SimulationResult
from strategies.mean_reversion import MeanReversionStrategy
from strategies.lineup_edge import LineupEdgeStrategy
from strategies.momentum import MomentumStrategy
from strategies.run_predictor_strategy import RunPredictorStrategy

logging.basicConfig(
    level=logging.WARNING,  # suppress per-game noise
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


FEATURE_ROWS_PATH    = Path("data/feature_store/feature_rows.parquet")
SYNTHETIC_PRICES_PATH = Path("data/feature_store/synthetic_prices.parquet")
MODEL_PATH           = Path("models/saved/run_predictor.pkl")

# Train ends Nov 30 2025; only backtest on val+test data (Dec 2025 onward)
# to avoid in-sample results on the RunPredictorStrategy
BACKTEST_START = "2025-12-01"


@dataclass
class SweepResult:
    name: str
    num_trades: int
    net_pnl: float
    win_rate: float
    avg_pnl_per_trade: float
    total_fees: float
    fee_drag_pct: float     # fees / gross_pnl (if positive)
    gross_pnl: float


def _fee_drag(result: SimulationResult) -> float:
    if result.gross_pnl <= 0:
        return float("inf")
    return result.total_fees / result.gross_pnl


def run_sweep() -> list[SweepResult]:
    print("Loading data...", flush=True)
    feature_rows    = pq.read_table(FEATURE_ROWS_PATH).to_pandas()
    synthetic_prices = pq.read_table(SYNTHETIC_PRICES_PATH).to_pandas()

    # Filter to out-of-sample period
    if "game_date" in feature_rows.columns:
        oos_games = feature_rows[feature_rows["game_date"] >= BACKTEST_START]["game_id"].unique().tolist()
    else:
        # Fall back: use all games (mix of in/out of sample)
        oos_games = None

    if oos_games is not None:
        print(f"Out-of-sample games: {len(oos_games)} (from {BACKTEST_START})", flush=True)
    else:
        print("No game_date column — using all games", flush=True)

    strategies: list[tuple[str, object]] = [
        # MeanReversion — fade runs after various thresholds
        ("MeanReversion(run≥6, sustainable_only=F)",  MeanReversionStrategy(run_threshold=6,  max_hold_possessions=8)),
        ("MeanReversion(run≥8, sustainable_only=F)",  MeanReversionStrategy(run_threshold=8,  max_hold_possessions=8)),
        ("MeanReversion(run≥10, sustainable_only=F)", MeanReversionStrategy(run_threshold=10, max_hold_possessions=8)),
        ("MeanReversion(run≥6, hold=12)",             MeanReversionStrategy(run_threshold=6,  max_hold_possessions=12)),
        ("MeanReversion(run≥8, hold=12)",             MeanReversionStrategy(run_threshold=8,  max_hold_possessions=12)),

        # LineupEdge — enter on lineup delta mismatches
        ("LineupEdge(delta≥3)",   LineupEdgeStrategy(delta_threshold=3.0, max_hold_possessions=8)),
        ("LineupEdge(delta≥5)",   LineupEdgeStrategy(delta_threshold=5.0, max_hold_possessions=8)),
        ("LineupEdge(delta≥7)",   LineupEdgeStrategy(delta_threshold=7.0, max_hold_possessions=8)),
        ("LineupEdge(delta≥5,h12)", LineupEdgeStrategy(delta_threshold=5.0, max_hold_possessions=12)),
        ("LineupEdge(delta≥7,h12)", LineupEdgeStrategy(delta_threshold=7.0, max_hold_possessions=12)),

        # Momentum — ride sustainable runs
        ("Momentum(run≥6)",   MomentumStrategy(run_threshold=6,  max_hold_possessions=8)),
        ("Momentum(run≥8)",   MomentumStrategy(run_threshold=8,  max_hold_possessions=8)),
        ("Momentum(run≥6,h12)", MomentumStrategy(run_threshold=6, max_hold_possessions=12)),

        # RunPredictor — model-based
        ("RunPredictor(thresh=0.12)", RunPredictorStrategy(entry_prob_threshold=0.12)),
        ("RunPredictor(thresh=0.15)", RunPredictorStrategy(entry_prob_threshold=0.15)),
        ("RunPredictor(thresh=0.18)", RunPredictorStrategy(entry_prob_threshold=0.18)),
    ]

    # Skip RunPredictor variants — per-possession XGBoost inference is too slow for sweep
    strategies = [(n, s) for n, s in strategies if "RunPredictor" not in n]

    if not MODEL_PATH.exists():
        print(f"WARNING: {MODEL_PATH} not found — skipping RunPredictorStrategy variants", flush=True)

    results: list[SweepResult] = []

    for i, (name, strategy) in enumerate(strategies):
        print(f"[{i+1}/{len(strategies)}] {name}...", flush=True)
        try:
            sim = BacktestSimulator(strategy, feature_rows, synthetic_prices)
            result = sim.run(game_ids=oos_games)

            if result.num_trades == 0:
                results.append(SweepResult(
                    name=name, num_trades=0, net_pnl=0, win_rate=0,
                    avg_pnl_per_trade=0, total_fees=0, fee_drag_pct=float("inf"), gross_pnl=0,
                ))
                continue

            results.append(SweepResult(
                name=name,
                num_trades=result.num_trades,
                net_pnl=result.net_pnl,
                win_rate=result.win_rate,
                avg_pnl_per_trade=result.net_pnl / result.num_trades,
                total_fees=result.total_fees,
                fee_drag_pct=_fee_drag(result),
                gross_pnl=result.gross_pnl,
            ))
        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)

    return results


def print_summary(results: list[SweepResult]) -> None:
    # Sort by net PnL descending
    ranked = sorted(results, key=lambda r: r.net_pnl, reverse=True)

    print("\n" + "="*80)
    print("STRATEGY SWEEP RESULTS  (out-of-sample, sorted by net PnL)")
    print("="*80)
    print(f"{'Strategy':<45} {'Trades':>6} {'Net PnL':>9} {'Win%':>6} {'$/trade':>8} {'Fee drag':>9}")
    print("-"*80)
    for r in ranked:
        fd = f"{r.fee_drag_pct:.0%}" if r.fee_drag_pct != float("inf") else "  N/A"
        print(
            f"{r.name:<45} {r.num_trades:>6} "
            f"${r.net_pnl:>8.2f} {r.win_rate:>5.1%} "
            f"${r.avg_pnl_per_trade:>7.4f} {fd:>9}"
        )
    print("="*80)

    # Top 3 with commentary
    profitable = [r for r in ranked if r.net_pnl > 0 and r.num_trades >= 20]
    print(f"\nProfitable strategies (≥20 trades): {len(profitable)}")
    for r in profitable[:3]:
        print(f"  ★ {r.name}")
        print(f"    {r.num_trades} trades | net ${r.net_pnl:.2f} | "
              f"win {r.win_rate:.1%} | ${r.avg_pnl_per_trade:.4f}/trade | "
              f"fee drag {r.fee_drag_pct:.0%}")

    if not profitable:
        print("  None profitable — all strategies losing money out-of-sample.")
        best = ranked[0]
        print(f"  Least bad: {best.name}  (net ${best.net_pnl:.2f}, {best.num_trades} trades)")


if __name__ == "__main__":
    results = run_sweep()
    print_summary(results)
