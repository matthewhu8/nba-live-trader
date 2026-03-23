"""
BacktestEvaluator — computes and reports full strategy performance metrics.

Produces:
  - Overall metrics (PnL, Sharpe, win rate, drawdown, fee drag)
  - Context breakdowns (by quarter, score_diff bucket, lineup delta, run length,
    shot sustainability)
  - JSON report written to backtesting/results/

Usage:
    python backtesting/evaluator.py
    # → runs MeanReversionStrategy and LineupEdgeStrategy, prints + saves report
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.simulator import BacktestSimulator, SimulationResult, TradeRecord, run_backtest

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("backtesting/results")


@dataclass
class ContextBucket:
    label: str
    num_trades: int
    win_rate: float
    net_pnl: float
    avg_hold: float        # in possessions


@dataclass
class DKValidationSummary:
    trades_with_dk_data: int
    total_trades: int
    avg_dk_move_favorable_cents: float   # avg DK move in signal direction (cents)
    avg_synthetic_pnl: float
    avg_dk_net_pnl: float
    signal_quality_pct: float            # % of trades where DK moved in our direction
    synthetic_vs_dk_correlation: float   # how good is our synthetic proxy?


@dataclass
class StrategyReport:
    strategy_name: str
    timestamp: str
    num_trades: int
    gross_pnl: float
    net_pnl: float
    total_fees: float
    fee_drag_pct: float      # fees / gross_pnl × 100
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    avg_hold_possessions: float
    by_quarter: list[ContextBucket]
    by_score_diff: list[ContextBucket]
    by_lineup_delta: list[ContextBucket]
    by_run_length: list[ContextBucket]
    by_sustainability: list[ContextBucket]
    by_exit_reason: list[ContextBucket]
    dk_validation: DKValidationSummary | None = None


def _sharpe(net_pnls: list[float], annualize: bool = True) -> float:
    if len(net_pnls) < 2:
        return 0.0
    arr = np.array(net_pnls)
    mean, std = arr.mean(), arr.std()
    if std == 0:
        return 0.0
    ratio = mean / std
    # Approximate annualization: ~80 NBA games/season × ~50 trades/game
    if annualize:
        ratio *= np.sqrt(4000)
    return float(ratio)


def _max_drawdown(net_pnls: list[float]) -> float:
    if not net_pnls:
        return 0.0
    cumulative = np.cumsum(net_pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    return float(drawdowns.max())


def _bucket_stats(
    trades: list[TradeRecord],
    label: str,
    filter_fn,
) -> ContextBucket:
    subset = [t for t in trades if filter_fn(t)]
    if not subset:
        return ContextBucket(label=label, num_trades=0, win_rate=0.0, net_pnl=0.0, avg_hold=0.0)

    wins = sum(1 for t in subset if t.net_pnl > 0)
    net_pnl = sum(t.net_pnl for t in subset)
    avg_hold = np.mean([t.exit_possession - t.entry_possession for t in subset])

    return ContextBucket(
        label=label,
        num_trades=len(subset),
        win_rate=wins / len(subset),
        net_pnl=net_pnl,
        avg_hold=float(avg_hold),
    )


def _compute_dk_validation(trades: list[TradeRecord]) -> DKValidationSummary | None:
    """
    Compute DK line validation metrics for trades that have DK data attached.

    Returns None if no trades have DK data (e.g. using old strategies without model).
    """
    dk_trades = [t for t in trades if t.dk_wp_entry is not None and t.dk_wp_exit is not None]
    if not dk_trades:
        return None

    # DK move in signal direction: positive = DK moved our way
    favorable_moves: list[float] = []
    for t in dk_trades:
        move_cents = (t.dk_wp_exit - t.dk_wp_entry) * 100.0
        if t.direction == "NO":
            move_cents = -move_cents
        favorable_moves.append(move_cents)

    avg_favorable = float(np.mean(favorable_moves))
    signal_quality = float(np.mean([m > 0 for m in favorable_moves]))

    synthetic_pnls = [t.net_pnl for t in dk_trades]
    dk_pnls = [t.dk_net_pnl for t in dk_trades if t.dk_net_pnl is not None]

    avg_synthetic = float(np.mean(synthetic_pnls)) if synthetic_pnls else 0.0
    avg_dk = float(np.mean(dk_pnls)) if dk_pnls else 0.0

    # Correlation between synthetic and DK PnL (how good is our proxy?)
    correlation = 0.0
    if len(synthetic_pnls) >= 2 and len(dk_pnls) == len(synthetic_pnls):
        corr_matrix = np.corrcoef(synthetic_pnls, dk_pnls)
        correlation = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

    return DKValidationSummary(
        trades_with_dk_data=len(dk_trades),
        total_trades=len(trades),
        avg_dk_move_favorable_cents=avg_favorable,
        avg_synthetic_pnl=avg_synthetic,
        avg_dk_net_pnl=avg_dk,
        signal_quality_pct=signal_quality,
        synthetic_vs_dk_correlation=correlation,
    )


def evaluate(result: SimulationResult, strategy_name: str) -> StrategyReport:
    trades = result.trades

    if not trades:
        logger.warning("No trades to evaluate for %s", strategy_name)

    net_pnls = [t.net_pnl for t in trades]
    hold_times = [t.exit_possession - t.entry_possession for t in trades]
    fee_drag = (result.total_fees / result.gross_pnl * 100) if result.gross_pnl != 0 else 0.0

    # --- Context breakdowns ---

    by_quarter = [
        _bucket_stats(trades, f"Q{q}", lambda t, q=q: t.period == q)
        for q in [1, 2, 3, 4]
    ]

    score_diff_buckets = [
        ("±0-5",   lambda t: abs(t.score_diff_at_entry) <= 5),
        ("±6-12",  lambda t: 6 <= abs(t.score_diff_at_entry) <= 12),
        ("±13-20", lambda t: 13 <= abs(t.score_diff_at_entry) <= 20),
        ("blowout",lambda t: abs(t.score_diff_at_entry) > 20),
    ]
    by_score_diff = [_bucket_stats(trades, label, fn) for label, fn in score_diff_buckets]

    lineup_buckets = [
        ("delta_0-3", lambda t: abs(t.lineup_delta_at_entry) < 3),
        ("delta_3-7", lambda t: 3 <= abs(t.lineup_delta_at_entry) < 7),
        ("delta_7+",  lambda t: abs(t.lineup_delta_at_entry) >= 7),
    ]
    by_lineup_delta = [_bucket_stats(trades, label, fn) for label, fn in lineup_buckets]

    run_buckets = [
        ("run_1-3", lambda t: 1 <= t.run_points_at_entry <= 3),
        ("run_4-6", lambda t: 4 <= t.run_points_at_entry <= 6),
        ("run_7+",  lambda t: t.run_points_at_entry >= 7),
    ]
    by_run_length = [_bucket_stats(trades, label, fn) for label, fn in run_buckets]

    by_sustainability = [
        _bucket_stats(trades, "sustainable",   lambda t: t.shot_sustainable_at_entry),
        _bucket_stats(trades, "unsustainable", lambda t: not t.shot_sustainable_at_entry),
    ]

    exit_reasons = sorted({t.exit_reason for t in trades})
    by_exit_reason = [
        _bucket_stats(trades, reason, lambda t, r=reason: t.exit_reason == r)
        for reason in exit_reasons
    ]

    dk_validation = _compute_dk_validation(trades)

    return StrategyReport(
        strategy_name=strategy_name,
        timestamp=datetime.now().isoformat(),
        num_trades=result.num_trades,
        gross_pnl=result.gross_pnl,
        net_pnl=result.net_pnl,
        total_fees=result.total_fees,
        fee_drag_pct=fee_drag,
        win_rate=result.win_rate,
        sharpe_ratio=_sharpe(net_pnls),
        max_drawdown=_max_drawdown(net_pnls),
        avg_hold_possessions=float(np.mean(hold_times)) if hold_times else 0.0,
        by_quarter=by_quarter,
        by_score_diff=by_score_diff,
        by_lineup_delta=by_lineup_delta,
        by_run_length=by_run_length,
        by_sustainability=by_sustainability,
        by_exit_reason=by_exit_reason,
        dk_validation=dk_validation,
    )


def print_report(report: StrategyReport) -> None:
    print(f"\n{'='*60}")
    print(f"  {report.strategy_name}")
    print(f"  {report.timestamp}")
    print(f"{'='*60}")
    print(f"  Trades:         {report.num_trades}")
    print(f"  Gross PnL:      ${report.gross_pnl:.2f}")
    print(f"  Fees:           ${report.total_fees:.2f}")
    print(f"  Net PnL:        ${report.net_pnl:.2f}")
    print(f"  Fee drag:       {report.fee_drag_pct:.1f}%")
    print(f"  Win rate:       {report.win_rate:.1%}")
    print(f"  Sharpe ratio:   {report.sharpe_ratio:.2f}")
    print(f"  Max drawdown:   ${report.max_drawdown:.2f}")
    print(f"  Avg hold:       {report.avg_hold_possessions:.1f} possessions")

    def print_buckets(title: str, buckets: list[ContextBucket]) -> None:
        print(f"\n  {title}:")
        for b in buckets:
            if b.num_trades == 0:
                continue
            print(f"    {b.label:<15} n={b.num_trades:>4}  win={b.win_rate:.1%}  pnl=${b.net_pnl:>7.2f}  hold={b.avg_hold:.1f}p")

    print_buckets("By quarter",       report.by_quarter)
    print_buckets("By score diff",    report.by_score_diff)
    print_buckets("By lineup delta",  report.by_lineup_delta)
    print_buckets("By run length",    report.by_run_length)
    print_buckets("By sustainability",report.by_sustainability)
    print_buckets("By exit reason",   report.by_exit_reason)

    if report.dk_validation is not None:
        dk = report.dk_validation
        print(f"\n  {'='*50}")
        print(f"  DK Line Validation")
        print(f"  {'='*50}")
        print(f"  Trades with DK data:         {dk.trades_with_dk_data} / {dk.total_trades}")
        print(f"  Avg DK move in our favor:    {dk.avg_dk_move_favorable_cents:+.2f}¢")
        print(f"  Signal quality:              {dk.signal_quality_pct:.1%}  (DK moved our direction)")
        print(f"  Avg synthetic PnL:           ${dk.avg_synthetic_pnl:.4f}")
        print(f"  Avg DK-actual PnL:           ${dk.avg_dk_net_pnl:.4f}  ← what we'd actually make")
        print(f"  Synthetic vs DK correlation: {dk.synthetic_vs_dk_correlation:.3f}")
    print()


def save_report(report: StrategyReport) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{report.strategy_name}_{ts}.json"

    def _serialize(obj: Any) -> Any:
        if isinstance(obj, ContextBucket):
            return asdict(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    data: dict[str, Any] = {
        "strategy_name": report.strategy_name,
        "timestamp": report.timestamp,
        "overall": {
            "num_trades": report.num_trades,
            "gross_pnl": report.gross_pnl,
            "net_pnl": report.net_pnl,
            "total_fees": report.total_fees,
            "fee_drag_pct": report.fee_drag_pct,
            "win_rate": report.win_rate,
            "sharpe_ratio": report.sharpe_ratio,
            "max_drawdown": report.max_drawdown,
            "avg_hold_possessions": report.avg_hold_possessions,
        },
        "by_quarter":       [asdict(b) for b in report.by_quarter],
        "by_score_diff":    [asdict(b) for b in report.by_score_diff],
        "by_lineup_delta":  [asdict(b) for b in report.by_lineup_delta],
        "by_run_length":    [asdict(b) for b in report.by_run_length],
        "by_sustainability":[asdict(b) for b in report.by_sustainability],
        "by_exit_reason":   [asdict(b) for b in report.by_exit_reason],
    }
    if report.dk_validation is not None:
        data["dk_validation"] = asdict(report.dk_validation)

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Report saved to %s", out_path)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from strategies.mean_reversion import MeanReversionStrategy
    from strategies.lineup_edge import LineupEdgeStrategy
    from strategies.momentum import MomentumStrategy
    from strategies.run_predictor_strategy import RunPredictorStrategy

    strategies = [
        MeanReversionStrategy(),
        LineupEdgeStrategy(),
        MomentumStrategy(),
        RunPredictorStrategy(),
    ]

    reports = []
    for strategy in strategies:
        logger.info("Running %s ...", strategy.name)
        result = run_backtest(strategy)
        report = evaluate(result, strategy.name)
        print_report(report)
        save_report(report)
        reports.append(report)

    # Cross-strategy summary
    print("\n" + "=" * 70)
    print("CROSS-STRATEGY SUMMARY")
    print("=" * 70)
    print(f"  {'Strategy':<32} {'Trades':>7} {'Net PnL':>9} {'Win%':>7} {'Sharpe':>7}")
    print(f"  {'-'*32} {'-'*7} {'-'*9} {'-'*7} {'-'*7}")
    for r in reports:
        print(
            f"  {r.strategy_name:<32} {r.num_trades:>7} "
            f"${r.net_pnl:>8.2f} {r.win_rate*100:>6.1f}% {r.sharpe_ratio:>7.2f}"
        )

    # DK signal quality comparison (if available)
    dk_rows = [(r.strategy_name, r.dk_validation) for r in reports if r.dk_validation is not None]
    if dk_rows:
        print(f"\n  {'Strategy':<32} {'DK sig%':>8} {'Avg DK move':>12} {'Synthetic corr':>15}")
        print(f"  {'-'*32} {'-'*8} {'-'*12} {'-'*15}")
        for name, dk in dk_rows:
            print(
                f"  {name:<32} {dk.signal_quality*100:>7.1f}% "
                f"{dk.avg_dk_move_cents:>11.2f}¢ {dk.synthetic_dk_correlation:>14.3f}"
            )
