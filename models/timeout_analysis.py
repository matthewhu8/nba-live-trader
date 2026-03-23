"""
Timeout analysis: does calling a timeout stop an opponent's run?

Research question: After a team calls a timeout to stop a run of ≥6 points,
what happens to the score in the next 5 and 10 possessions?

Compares:
  - Post-timeout scoring margin (from targets in feature store)
  - vs. baseline: runs of same length with no timeout

Also breaks down by run_3pt_pct (composition of the run being stopped).

Output: console report + backtesting/results/timeout_analysis.json
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

FEATURE_ROWS_PATH = Path("data/feature_store/feature_rows.parquet")
TIMEOUTS_PATH     = Path("data/raw/timeout_events_202526.parquet")
RESULTS_DIR       = Path("backtesting/results")

# Minimum run length to analyze (points)
MIN_RUN_POINTS = 6


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FEATURE_ROWS_PATH.exists():
        raise FileNotFoundError(f"Feature store not found: {FEATURE_ROWS_PATH}")
    if not TIMEOUTS_PATH.exists():
        raise FileNotFoundError(
            f"Timeout events not found: {TIMEOUTS_PATH}\n"
            "Run: python -c \"from data.ingestion.nba_api_client import "
            "parse_timeouts_from_cache; parse_timeouts_from_cache()\""
        )

    features = pq.read_table(FEATURE_ROWS_PATH).to_pandas()
    timeouts = pq.read_table(TIMEOUTS_PATH).to_pandas()
    return features, timeouts


def _is_run_stopped_by_timeout(
    timeout_rows: pd.DataFrame,
    game_id: str,
    poss_period: int,
    poss_clock: float,
    run_team: str,
    home_team_tc: str,
) -> bool:
    """
    Was a timeout called by the non-run-team in the last 3 possessions before
    the current possession, while the run was active?
    """
    if timeout_rows.empty:
        return False

    run_tc = home_team_tc if run_team == "home" else None  # tricode of the team on the run

    # Look for a timeout within the recent window (simplified: same period, higher clock)
    recent = timeout_rows[
        (timeout_rows["period"] == poss_period) &
        (timeout_rows["game_clock_secs"] > poss_clock) &
        (timeout_rows["game_clock_secs"] < poss_clock + 90)  # within ~3 possessions
    ]
    if recent.empty:
        return False

    # Was the timeout called by the team NOT on the run?
    for _, to_row in recent.iterrows():
        tc = str(to_row["team_tricode"])
        if run_tc is None:
            return True  # can't determine run team, count it
        if tc != run_tc:
            return True  # timeout called by opponent of running team
    return False


def analyze_timeouts(features: pd.DataFrame, timeouts: pd.DataFrame) -> dict:
    """
    For each timeout-on-run event, record the post-timeout scoring margin.
    Compare to a control group (same run length, no timeout).
    """
    if "target_home_next_5_margin" not in features.columns:
        raise ValueError("Feature store missing target columns — rebuild first")
    if "current_run_3pt_pct" not in features.columns:
        raise ValueError("Feature store missing new features — rebuild first")

    games = features["game_id"].unique()
    timeout_game_ids = set(timeouts["game_id"].unique())

    # Per-game home team lookup from features (infer from score column patterns)
    # Use the games table if available, else infer from feature store
    games_path = Path("data/raw/games_202526.parquet")
    if games_path.exists():
        games_df  = pq.read_table(games_path).to_pandas()
        home_team_map = games_df.set_index("game_id")["home_team"].to_dict()
    else:
        home_team_map: dict[str, str] = {}

    timeout_events_list: list[dict] = []
    control_events_list: list[dict] = []

    for game_id in games:
        game_features = features[features["game_id"] == game_id].sort_values("possession_id")
        game_timeouts = timeouts[timeouts["game_id"] == game_id] if game_id in timeout_game_ids else pd.DataFrame()
        home_tc = home_team_map.get(game_id, "")

        for _, row in game_features.iterrows():
            run_points = int(row.get("current_run_points", 0))
            run_team   = str(row.get("current_run_team", ""))

            if run_points < MIN_RUN_POINTS or not run_team:
                continue

            # Skip blowout / garbage time
            if bool(row.get("is_blowout", False)) or bool(row.get("is_garbage_time", False)):
                continue

            poss_period = int(row.get("period", 0))
            poss_clock  = float(row.get("game_clock_secs", 0))
            run_3pt_pct = float(row.get("current_run_3pt_pct", 0.0))

            # Check if a timeout was called by the non-run team recently
            called_timeout = _is_run_stopped_by_timeout(
                game_timeouts, game_id, poss_period, poss_clock, run_team, home_tc
            )

            event = {
                "game_id":               game_id,
                "possession_id":         int(row.get("possession_id", 0)),
                "run_team":              run_team,
                "run_points":            run_points,
                "run_length":            int(row.get("current_run_length", 0)),
                "run_3pt_pct":           run_3pt_pct,
                "score_diff":            int(row.get("score_diff", 0)),
                "period":                poss_period,
                "next_5_margin":         int(row.get("target_home_next_5_margin", 0))
                                         * (1 if run_team == "home" else -1),
                "next_10_margin":        int(row.get("target_home_next_10_margin", 0))
                                         * (1 if run_team == "home" else -1),
                "meaningful_run_5":      bool(row.get("target_meaningful_run_5", False)),
            }

            if called_timeout:
                timeout_events_list.append(event)
            else:
                control_events_list.append(event)

    return {
        "timeout_events": timeout_events_list,
        "control_events": control_events_list,
    }


def _stats(events: list[dict], key: str) -> dict:
    vals = [e[key] for e in events]
    if not vals:
        return {"n": 0, "mean": 0.0, "std": 0.0, "pct_negative": 0.0}
    arr = np.array(vals, dtype=float)
    return {
        "n":            len(arr),
        "mean":         float(np.mean(arr)),
        "std":          float(np.std(arr)),
        "pct_negative": float(np.mean(arr < 0)),  # run continued after timeout
    }


def print_report(data: dict) -> None:
    to_evts  = data["timeout_events"]
    ctrl_evts = data["control_events"]

    print("\n" + "="*65)
    print("TIMEOUT ANALYSIS — Does a timeout stop an opponent's run?")
    print("="*65)
    print(f"\nTimeout-on-run events: {len(to_evts)}")
    print(f"Control events (no timeout): {len(ctrl_evts)}")

    for label, events in [("TIMEOUT", to_evts), ("CONTROL (no timeout)", ctrl_evts)]:
        if not events:
            continue
        s5  = _stats(events, "next_5_margin")
        s10 = _stats(events, "next_10_margin")
        run_5 = np.mean([e["meaningful_run_5"] for e in events])
        print(f"\n{label} (n={len(events)}):")
        print(f"  Next 5 poss scoring margin (run team): mean={s5['mean']:+.2f}  "
              f"std={s5['std']:.2f}  run_continued={s5['pct_negative']:.1%}")
        print(f"  Next 10 poss scoring margin:           mean={s10['mean']:+.2f}  "
              f"std={s10['std']:.2f}")
        print(f"  Meaningful run continued in 5 poss:    {run_5:.1%}")

    # Breakdown by run composition
    if to_evts:
        print("\n--- Timeout events by run 3pt% ---")
        for low, high, label in [(0.0, 0.33, "≤33% 3pt (paint-dominant)"),
                                  (0.33, 0.67, "33-67% 3pt (mixed)"),
                                  (0.67, 1.01, "≥67% 3pt (3pt-dominant)")]:
            subset = [e for e in to_evts if low <= e["run_3pt_pct"] < high]
            if not subset:
                continue
            s = _stats(subset, "next_5_margin")
            print(f"  {label}: n={s['n']}  next_5_mean={s['mean']:+.2f}  "
                  f"run_continued={s['pct_negative']:.1%}")

    # Breakdown by quarter
    print("\n--- Timeout events by quarter ---")
    for q in [1, 2, 3, 4]:
        subset = [e for e in to_evts if e["period"] == q]
        ctrl_q = [e for e in ctrl_evts if e["period"] == q]
        if not subset:
            continue
        s_to   = _stats(subset, "next_5_margin")
        s_ctrl = _stats(ctrl_q, "next_5_margin")
        print(f"  Q{q}: timeout n={s_to['n']}  next_5={s_to['mean']:+.2f}  |  "
              f"control n={s_ctrl['n']}  next_5={s_ctrl['mean']:+.2f}")

    print("\n" + "="*65)
    print("STRATEGY IMPLICATION:")
    if to_evts and ctrl_evts:
        to_mean   = np.mean([e["next_5_margin"] for e in to_evts])
        ctrl_mean = np.mean([e["next_5_margin"] for e in ctrl_evts])
        diff = to_mean - ctrl_mean
        print(f"  Timeout reduces run team's next-5 margin by {abs(diff):.2f} pts "
              f"vs no timeout (lower = run more likely stopped)")
        if diff < -1.0:
            print("  → timeout_on_opponent_run should be a STRONG EXIT signal")
        elif diff < 0:
            print("  → timeout_on_opponent_run is a modest exit signal")
        else:
            print("  → No clear effect — timeouts may not reliably stop runs in this dataset")
    print("="*65 + "\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    try:
        features, timeouts = load_data()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("Analyzing %d timeout events across %d unique games...",
                len(timeouts), timeouts["game_id"].nunique())

    data = analyze_timeouts(features, timeouts)
    print_report(data)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "timeout_analysis.json"
    with open(out_path, "w") as f:
        # Truncate events list to summary stats for the JSON
        summary = {
            "n_timeout_events":        len(data["timeout_events"]),
            "n_control_events":        len(data["control_events"]),
            "timeout_next5_mean":      float(np.mean([e["next_5_margin"] for e in data["timeout_events"]])) if data["timeout_events"] else 0.0,
            "control_next5_mean":      float(np.mean([e["next_5_margin"] for e in data["control_events"]])) if data["control_events"] else 0.0,
            "timeout_run_continued_5": float(np.mean([e["next_5_margin"] < 0 for e in data["timeout_events"]])) if data["timeout_events"] else 0.0,
            "control_run_continued_5": float(np.mean([e["next_5_margin"] < 0 for e in data["control_events"]])) if data["control_events"] else 0.0,
        }
        json.dump(summary, f, indent=2)
    logger.info("Summary written → %s", out_path)


if __name__ == "__main__":
    main()
