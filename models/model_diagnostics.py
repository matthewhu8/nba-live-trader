"""
Model diagnostics for the run predictor.

Answers the key question before building more on top:
  - Where does the signal actually live?
  - Is the model well-calibrated?
  - At what threshold does precision > 2× base rate?
  - Which contexts have useful signal vs. noise?

Reads: models/saved/run_predictor.pkl, data/feature_store/feature_rows.parquet
       data/feature_store/synthetic_prices.parquet (optional, for DK context)
Writes: backtesting/results/diagnostics_{timestamp}.json + printed report
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/saved/run_predictor.pkl")
FEATURE_STORE = Path("data/feature_store/feature_rows.parquet")
RESULTS_DIR = Path("backtesting/results")

# Must match run_predictor.py
VAL_START = "2025-12-01"
VAL_END = "2026-01-12"
TEST_START = "2026-01-13"  # DO NOT TOUCH

# Exact columns the model was trained on — must match run_predictor.py preprocessing
FEATURE_COLS = [
    "lineup_net_rating_delta",
    "home_lineup_net_rating",
    "away_lineup_net_rating",
    "home_lineup_sample_size",
    "away_lineup_sample_size",
    "home_lineup_just_changed",
    "away_lineup_just_changed",
    "home_points_last_5_poss",
    "away_points_last_5_poss",
    "home_points_last_10_poss",
    "away_points_last_10_poss",
    "current_run_team_encoded",  # encoded from current_run_team string
    "current_run_length",
    "current_run_points",
    "home_scoring_sustainable",
    "away_scoring_sustainable",
    "home_key_foul_count",
    "away_key_foul_count",
    "score_diff",
    "period",
    "minutes_into_game",
    "pace_last_10_possessions",
    "pace_season_baseline",
    "home_back_to_back",
    "away_back_to_back",
    "shot_distance",
    "shot_value",
]
TARGET_COL = "target_meaningful_run_5"
BASE_RATE = 0.0760  # from training data


def _encode_run_team(series: pd.Series) -> pd.Series:
    """Encode current_run_team string → int, matching run_predictor.py encoding."""
    mapping = {"home": 1, "away": -1, None: 0, float("nan"): 0}
    return series.map(lambda x: mapping.get(x, 0)).astype(float)


def _load_val_set() -> pd.DataFrame:
    """Load validation set with game dates, excluding test set."""
    df = pd.read_parquet(FEATURE_STORE)

    # game_id encodes date as YYYYMMDD prefix (e.g., 0022501001)
    # Use the same date logic as run_predictor.py: game_id prefix comparison
    # game_id format from nba_api: '002YYMMDD...' — but our data uses a numeric ID
    # Safer: join on games table which has the date column
    games_path = Path("data/raw/games_202526.parquet")
    if games_path.exists():
        games = pd.read_parquet(games_path)[["game_id", "game_date"]]
        games["game_date"] = pd.to_datetime(games["game_date"])
        df = df.merge(games, on="game_id", how="left")
    else:
        # Fallback: derive from game_id (YYYYMMDD is digits 3-10 in nba_api game IDs)
        def _date_from_game_id(gid: str) -> pd.Timestamp:
            s = str(gid)
            if len(s) >= 10:
                try:
                    return pd.Timestamp(s[3:11])
                except Exception:
                    pass
            return pd.NaT

        df["game_date"] = df["game_id"].apply(_date_from_game_id)

    val_mask = (df["game_date"] >= VAL_START) & (df["game_date"] <= VAL_END)
    val = df[val_mask].copy()
    logger.info("Validation set: %d rows from %d games", len(val), val["game_id"].nunique())
    return val


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same preprocessing as run_predictor.py before prediction."""
    df = df.copy()
    df["current_run_team_encoded"] = _encode_run_team(df["current_run_team"])
    return df


def _predict(model, df: pd.DataFrame) -> np.ndarray:
    df = _prepare_features(df)
    X = df[FEATURE_COLS].fillna(0).astype(float)
    return model.predict_proba(X)[:, 1]


def precision_recall_summary(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Precision-recall at key thresholds + AUCPR."""
    aucpr = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # Build threshold table
    rows = []
    for thresh in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        mask = y_prob >= thresh
        n = mask.sum()
        if n == 0:
            rows.append({"threshold": thresh, "n_signals": 0, "precision": None, "recall": None, "lift": None})
            continue
        prec = y_true[mask].mean()
        rec = y_true[mask].sum() / y_true.sum()
        rows.append({
            "threshold": thresh,
            "n_signals": int(n),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "lift": round(float(prec / baseline), 2),
        })

    return {
        "aucpr": round(float(aucpr), 4),
        "baseline_aucpr": round(float(baseline), 4),
        "threshold_table": rows,
    }


def calibration_summary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Calibration: does predicted prob match actual frequency?"""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    rows = [
        {"predicted_prob": round(float(pp), 4), "actual_freq": round(float(pt), 4),
         "ratio": round(float(pt / pp) if pp > 0 else 0, 2)}
        for pp, pt in zip(prob_pred, prob_true)
    ]
    # Mean absolute calibration error
    mace = float(np.mean(np.abs(prob_true - prob_pred)))
    return {"mace": round(mace, 4), "bins": rows}


def context_breakdown(val: pd.DataFrame, y_prob: np.ndarray, threshold: float = 0.15) -> dict:
    """Signal quality (precision) and count by context bucket.

    For each context, compute:
      - n_signals: how many signals fired above threshold
      - precision: how often the target was actually True
      - lift: precision / base_rate
    """
    val = val.copy()
    val["_prob"] = y_prob
    val["_signal"] = y_prob >= threshold
    val["_target"] = val[TARGET_COL].astype(bool)

    def bucket_stats(subset: pd.DataFrame) -> dict:
        signals = subset[subset["_signal"]]
        n = len(signals)
        if n == 0:
            return {"n_signals": 0, "precision": None, "lift": None}
        prec = float(signals["_target"].mean())
        return {
            "n_signals": n,
            "precision": round(prec, 4),
            "lift": round(prec / BASE_RATE, 2),
        }

    results: dict = {}

    # By quarter (period)
    results["by_quarter"] = {}
    for q in sorted(val["period"].dropna().unique()):
        results["by_quarter"][f"Q{int(q)}"] = bucket_stats(val[val["period"] == q])

    # By score_diff bucket
    def score_bucket(sd: float) -> str:
        sd = abs(sd)
        if sd <= 5:
            return "close (±5)"
        if sd <= 12:
            return "medium (6-12)"
        if sd <= 20:
            return "large (13-20)"
        return "blowout (>20)"

    val["_score_bucket"] = val["score_diff"].apply(score_bucket)
    results["by_score_diff"] = {b: bucket_stats(val[val["_score_bucket"] == b])
                                 for b in ["close (±5)", "medium (6-12)", "large (13-20)", "blowout (>20)"]}

    # By run length at entry
    def run_bucket(rl: float) -> str:
        if rl <= 3:
            return "short (1-3)"
        if rl <= 6:
            return "medium (4-6)"
        return "long (7+)"

    val["_run_bucket"] = val["current_run_length"].fillna(0).apply(run_bucket)
    results["by_run_length"] = {b: bucket_stats(val[val["_run_bucket"] == b])
                                 for b in ["short (1-3)", "medium (4-6)", "long (7+"]}

    # By shot sustainability
    results["by_sustainability"] = {
        "home_sustainable": bucket_stats(val[val["home_scoring_sustainable"] == True]),
        "home_unsustainable": bucket_stats(val[val["home_scoring_sustainable"] == False]),
    }

    # By lineup_net_rating_delta magnitude
    def delta_bucket(d: float) -> str:
        ad = abs(d)
        if ad <= 3:
            return "small (0-3)"
        if ad <= 7:
            return "medium (3-7)"
        return "large (7+)"

    val["_delta_bucket"] = val["lineup_net_rating_delta"].apply(delta_bucket)
    results["by_lineup_delta"] = {b: bucket_stats(val[val["_delta_bucket"] == b])
                                   for b in ["small (0-3)", "medium (3-7)", "large (7+"]}

    # Close games only (<= 5pt), by quarter — most actionable context
    close = val[val["score_diff"].abs() <= 5]
    results["close_games_by_quarter"] = {}
    for q in sorted(close["period"].dropna().unique()):
        results["close_games_by_quarter"][f"Q{int(q)}"] = bucket_stats(close[close["period"] == q])

    return results


def print_report(pr: dict, cal: dict, ctx: dict) -> None:
    print("\n" + "=" * 60)
    print("RUN PREDICTOR DIAGNOSTICS — VALIDATION SET")
    print("=" * 60)

    print(f"\nAUCPR: {pr['aucpr']}  (baseline: {pr['baseline_aucpr']})")
    print(f"Lift over baseline: {round(pr['aucpr'] / pr['baseline_aucpr'], 2)}×\n")

    print("Threshold sweep (looking for lift > 2.0×):")
    header = f"  {'thresh':>8} {'signals':>8} {'precision':>10} {'recall':>8} {'lift':>6}"
    print(header)
    for row in pr["threshold_table"]:
        if row["precision"] is None:
            print(f"  {row['threshold']:>8.2f} {'0':>8} {'—':>10} {'—':>8} {'—':>6}")
        else:
            flag = " ← 2×+" if (row["lift"] or 0) >= 2.0 else ""
            print(f"  {row['threshold']:>8.2f} {row['n_signals']:>8} "
                  f"{row['precision']:>10.4f} {row['recall']:>8.4f} {row['lift']:>6.2f}×{flag}")

    print(f"\nCalibration (MACE: {cal['mace']:.4f} — lower is better):")
    print(f"  {'predicted':>10} {'actual':>10} {'ratio':>8}")
    for b in cal["bins"]:
        print(f"  {b['predicted_prob']:>10.4f} {b['actual_freq']:>10.4f} {b['ratio']:>8.2f}×")

    print("\nContext breakdown (threshold=0.15, lift = precision / 7.6% base rate):")
    for section, buckets in ctx.items():
        print(f"\n  {section}:")
        for label, stats in buckets.items():
            if stats["n_signals"] == 0:
                print(f"    {label:<30} no signals")
            else:
                flag = " ★" if (stats["lift"] or 0) >= 2.0 else ""
                print(f"    {label:<30} n={stats['n_signals']:>4}  "
                      f"prec={stats['precision']:.4f}  lift={stats['lift']:.2f}×{flag}")

    print("\n" + "=" * 60)


def main() -> None:
    if not MODEL_PATH.exists():
        logger.error("Model not found at %s — run models/run_predictor.py first", MODEL_PATH)
        return

    logger.info("Loading model from %s", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    logger.info("Loading validation set from %s", FEATURE_STORE)
    val = _load_val_set()

    if len(val) == 0:
        logger.error("Validation set is empty — check date range and game_date column")
        return

    y_true = val[TARGET_COL].fillna(0).astype(int).values
    y_prob = _predict(model, val)

    logger.info("Running diagnostics on %d val rows (base rate %.4f)", len(val), y_true.mean())

    pr = precision_recall_summary(y_true, y_prob)
    cal = calibration_summary(y_true, y_prob)
    ctx = context_breakdown(val, y_prob, threshold=0.15)

    print_report(pr, cal, ctx)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"diagnostics_{ts}.json"
    with open(out, "w") as f:
        json.dump({"precision_recall": pr, "calibration": cal, "context_breakdown": ctx}, f, indent=2)
    logger.info("Saved diagnostics to %s", out)


if __name__ == "__main__":
    main()
