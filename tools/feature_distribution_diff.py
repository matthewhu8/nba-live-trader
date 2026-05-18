"""
Compare the feature distribution of a live run against the model's validation
distribution.

What this answers:
  "Tonight the model behaved unexpectedly — was its INPUT distribution
   meaningfully different from what it saw during validation, or are we
   debugging a different problem (model regression, gate config, market
   regime in the price feed)?"

How it works:
  1. Build a reference: aggregate the 10 z-scored decision-relevant features
     from `features.possession_flat` over Matt's Head B validation window
     (Apr 7 – Apr 12, 2026, regular season only). Cache the per-feature
     mean/std to disk so we only query MotherDuck once.
  2. Read a live run's inference.jsonl, collect the same features from each
     possession record's `features_zscored` block (raw value, not z-score —
     so we compare apples to apples against the reference's raw values).
  3. Print a per-feature side-by-side table with drift annotation.

Usage:
  tools/feature_distribution_diff.py <run_id>                  # use cached reference
  tools/feature_distribution_diff.py <run_id> --rebuild-ref    # re-query reference
  tools/feature_distribution_diff.py --rebuild-ref-only        # just rebuild cache

The reference cache lives at tools/.reference_distribution.json. Safe to delete.

Drift interpretation:
  abs(drift_sigma) < 0.5    no concern
  0.5 ≤ drift_sigma < 1.0   notable, log it
  1.0 ≤ drift_sigma < 2.0   meaningful — investigate
  drift_sigma ≥ 2.0         alarming — likely regime shift or live bug
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

# These match _ZSCORE_FEATURES in live-trader/inference/main.py exactly.
ZSCORE_FEATURES = [
    "score_diff",
    "lineup_net_rating_delta",
    "current_run_length",
    "current_run_points",
    "home_points_last_5_poss",
    "away_points_last_5_poss",
    "pace_last_10_possessions",
    "home_xPPP_last_5",
    "away_xPPP_last_5",
    "garbage_time_risk",
]

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
REFERENCE_CACHE = PROJECT_ROOT / "tools" / ".reference_distribution.json"

# Matt's Head B validation window from CLAUDE.md: Mar 23 – Apr 6 2026 train /
# Apr 7+ 2026 val. We use Apr 7 – Apr 12 (~6 days, regular-season only — game
# IDs '0022%') as the closest analog to "what the model was validated on".
REF_DATE_START = "2026-04-07"
REF_DATE_END   = "2026-04-13"  # exclusive in the query


# ── reference build ──────────────────────────────────────────────────────────

def build_reference() -> dict:
    """Query possession_flat for the val window, compute per-feature stats."""
    import duckdb

    token = os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        sys.exit(
            "ERROR: MOTHERDUCK_TOKEN not set. Source .env first:\n"
            "  set -a; source .env; set +a"
        )

    conn = duckdb.connect(f"md:kalshi_trading?motherduck_token={token}", read_only=True)

    cols_sql = ", ".join(ZSCORE_FEATURES)
    query = f"""
        SELECT {cols_sql}
        FROM features.possession_flat
        WHERE game_id LIKE '0022%'
          AND wall_clock_ts >= TIMESTAMP '{REF_DATE_START}'
          AND wall_clock_ts <  TIMESTAMP '{REF_DATE_END}'
    """
    print(f"Querying val window {REF_DATE_START} → {REF_DATE_END} (regular season)…")
    df = conn.execute(query).fetchdf()
    conn.close()

    print(f"  {len(df):,} possessions across the window")
    if len(df) == 0:
        sys.exit("ERROR: no rows returned. Check the date range and game-id prefix.")

    stats: dict[str, dict] = {}
    for col in ZSCORE_FEATURES:
        vals = df[col].dropna().astype(float).tolist()
        if not vals:
            print(f"  WARN: {col} all NaN — skipping")
            continue
        stats[col] = {
            "n":    len(vals),
            "mean": statistics.fmean(vals),
            "std":  statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "p10":  _percentile(vals, 10),
            "p50":  _percentile(vals, 50),
            "p90":  _percentile(vals, 90),
        }

    out = {
        "meta": {
            "date_start": REF_DATE_START,
            "date_end":   REF_DATE_END,
            "rows":       len(df),
            "season_type": "regular",
        },
        "features": stats,
    }
    REFERENCE_CACHE.write_text(json.dumps(out, indent=2))
    print(f"Wrote reference to {REFERENCE_CACHE.relative_to(PROJECT_ROOT)}")
    return out


def _percentile(vals: list[float], p: float) -> float:
    """Linear-interpolation percentile. Vals need not be sorted."""
    s = sorted(vals)
    if not s:
        return 0.0
    k = (len(s) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


# ── live run aggregation ─────────────────────────────────────────────────────

def aggregate_run(run_id: str) -> dict:
    """Parse inference.jsonl for the run, compute distribution stats from the
    raw value in each features_zscored entry."""
    path = _locate_run(run_id)
    print(f"Reading {path.relative_to(PROJECT_ROOT)}…")

    collected: dict[str, list[float]] = {f: [] for f in ZSCORE_FEATURES}
    n_poss = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("event") != "possession":
                continue
            n_poss += 1
            fz = d.get("features_zscored") or {}
            for name in ZSCORE_FEATURES:
                entry = fz.get(name)
                if entry is None:
                    continue
                # Each entry is {"value": v, "z": z}. We want the raw value
                # for cross-comparison with the reference's raw stats.
                v = entry.get("value")
                if v is None:
                    continue
                collected[name].append(float(v))

    print(f"  {n_poss:,} possession records read")

    stats: dict[str, dict] = {}
    for name in ZSCORE_FEATURES:
        vals = collected[name]
        if not vals:
            continue
        stats[name] = {
            "n":    len(vals),
            "mean": statistics.fmean(vals),
            "std":  statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "p10":  _percentile(vals, 10),
            "p50":  _percentile(vals, 50),
            "p90":  _percentile(vals, 90),
        }
    return {"run_id": run_id, "rows": n_poss, "features": stats}


def _locate_run(run_id: str) -> Path:
    """Find the inference.jsonl for a run_id, searching the two common dirs."""
    candidates = [
        PROJECT_ROOT / "live-trader" / "go" / "logs" / "runs",
        PROJECT_ROOT / "logs" / "runs",
    ]
    for base in candidates:
        if not base.exists():
            continue
        for date_dir in base.iterdir():
            run_dir = date_dir / run_id
            if run_dir.exists():
                jsonl = run_dir / "inference.jsonl"
                if jsonl.exists():
                    return jsonl
    sys.exit(f"ERROR: no inference.jsonl found for run_id={run_id}")


# ── render diff ──────────────────────────────────────────────────────────────

def render_diff(reference: dict, run: dict) -> None:
    ref_features = reference["features"]
    run_features = run["features"]

    print()
    print(f"Reference: {reference['meta']['date_start']} → {reference['meta']['date_end']} "
          f"({reference['meta']['rows']:,} possessions, regular season)")
    print(f"Run:       {run['run_id']} ({run['rows']:,} possessions)")
    print()
    print(f"{'feature':<28} {'ref_mean':>10}  {'ref_std':>8}    {'live_mean':>10}  {'live_std':>8}    {'drift_σ':>9}  flag")
    print("─" * 100)

    for name in ZSCORE_FEATURES:
        ref = ref_features.get(name)
        run_f = run_features.get(name)
        if ref is None or run_f is None:
            print(f"{name:<28} {'(missing in one side — skipping)':>72}")
            continue

        drift = (run_f["mean"] - ref["mean"]) / ref["std"] if ref["std"] > 0 else 0.0
        flag = _drift_flag(abs(drift))

        print(
            f"{name:<28} "
            f"{ref['mean']:>10.3f}  {ref['std']:>8.3f}    "
            f"{run_f['mean']:>10.3f}  {run_f['std']:>8.3f}    "
            f"{drift:>+9.2f}  {flag}"
        )
    print()
    print("Drift interpretation:")
    print("  |drift| < 0.5   no concern")
    print("  0.5 ≤  < 1.0    notable")
    print("  1.0 ≤  < 2.0    investigate")
    print("  ≥ 2.0           likely regime shift / live bug")


def _drift_flag(abs_drift: float) -> str:
    if abs_drift < 0.5:
        return "ok"
    if abs_drift < 1.0:
        return "notable"
    if abs_drift < 2.0:
        return "INVESTIGATE"
    return "ALARMING"


# ── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_id", nargs="?", help="run_id to diff against the reference")
    p.add_argument("--rebuild-ref",      action="store_true", help="re-query MotherDuck for the reference distribution")
    p.add_argument("--rebuild-ref-only", action="store_true", help="rebuild the reference and exit (no run diff)")
    args = p.parse_args()

    if args.rebuild_ref_only:
        build_reference()
        return

    if not args.run_id:
        p.print_help()
        sys.exit(2)

    if args.rebuild_ref or not REFERENCE_CACHE.exists():
        reference = build_reference()
    else:
        reference = json.loads(REFERENCE_CACHE.read_text())
        meta = reference.get("meta", {})
        print(
            f"Using cached reference: {meta.get('date_start')} → "
            f"{meta.get('date_end')} ({meta.get('rows', 0):,} rows). "
            f"Pass --rebuild-ref to refresh."
        )

    run = aggregate_run(args.run_id)
    render_diff(reference, run)


if __name__ == "__main__":
    main()
