"""
XGBoost run predictor.

Trains a binary classifier to predict target_meaningful_run_5:
whether the home team will have a meaningful scoring run (6+ points)
in the next 5 possessions.

For away-run prediction, features are symmetrically swapped before inference
(same model, flipped perspective).

Train / val / test split — time-based, never random:
  Train:  Oct 21 – Nov 30, 2025  (~250 games)
  Val:    Dec 1,  2025 – Jan 12, 2026  (~200 games)
  Test:   Jan 13 – Mar 5, 2026   (~340 games) — touch once at final eval

Usage:
    python models/run_predictor.py
"""

import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

FEATURE_STORE_PATH = Path("data/feature_store/feature_rows.parquet")
GAMES_PATH = Path("data/raw/games_202526.parquet")
MODEL_OUTPUT_PATH = Path("models/saved/run_predictor.pkl")

TRAIN_END = pd.Timestamp("2025-11-30")
VAL_END   = pd.Timestamp("2026-01-12")
# test: everything after VAL_END

# Backward-looking feature columns only. Targets and identity columns excluded.
FEATURE_COLS: list[str] = [
    # Lineup
    "lineup_net_rating_delta",
    "home_lineup_net_rating",
    "away_lineup_net_rating",
    "home_lineup_sample_size",
    "away_lineup_sample_size",
    "home_lineup_just_changed",
    "away_lineup_just_changed",
    # Momentum
    "home_points_last_5_poss",
    "away_points_last_5_poss",
    "home_points_last_10_poss",
    "away_points_last_10_poss",
    "current_run_team_encoded",   # derived: 1=home, -1=away, 0=none
    "current_run_length",
    "current_run_points",
    "home_scoring_sustainable",
    "away_scoring_sustainable",
    # Shot composition of run
    "current_run_3pt_count",
    "current_run_3pt_pct",
    "current_run_paint_pct",
    # xPPP shot quality
    "home_xPPP_last_5",
    "away_xPPP_last_5",
    "home_actual_vs_expected_PPP",
    "away_actual_vs_expected_PPP",
    "home_shot_quality_trend",
    "away_shot_quality_trend",
    # Foul state
    "home_key_foul_count",
    "away_key_foul_count",
    "home_in_bonus",
    "away_in_bonus",
    "home_fouls_until_bonus",
    "away_fouls_until_bonus",
    # Score × time
    "score_diff",
    "period",
    "minutes_into_game",
    "trailing_team_urgency",
    "q4_close_game",
    "garbage_time_risk",
    # Pace
    "pace_last_10_possessions",
    "pace_season_baseline",
    # Fatigue
    "home_back_to_back",
    "away_back_to_back",
    # Shot info
    "shot_distance",
    "shot_value",
    # Timeout signals
    "possessions_since_last_timeout",
    "home_called_timeout_in_last_3_poss",
    "away_called_timeout_in_last_3_poss",
    "timeout_on_opponent_run",
    "home_full_timeouts_remaining",
    "away_full_timeouts_remaining",
    # Player APM
    "home_best_player_apm",
    "away_best_player_apm",
    "home_worst_player_apm",
    "away_worst_player_apm",
    "home_apm_spread",
    "away_apm_spread",
    "home_lineup_apm_sum",
    "away_lineup_apm_sum",
    "home_off_court_best_apm",
    "away_off_court_best_apm",
    "apm_delta",
]

TARGET_COL = "target_meaningful_run_5"


def _encode_run_team(series: pd.Series) -> pd.Series:
    """Map current_run_team → numeric: 'home'=1, 'away'=-1, None/other=0."""
    return series.map({"home": 1, "away": -1}).fillna(0).astype(float)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns and return a clean numeric feature matrix."""
    out = df.copy()
    out["current_run_team_encoded"] = _encode_run_team(out["current_run_team"])

    # Booleans → int for XGBoost
    bool_cols = [
        "home_lineup_just_changed", "away_lineup_just_changed",
        "home_scoring_sustainable", "away_scoring_sustainable",
        "home_back_to_back", "away_back_to_back",
        "home_in_bonus", "away_in_bonus",
        "q4_close_game",
        "home_called_timeout_in_last_3_poss", "away_called_timeout_in_last_3_poss",
        "timeout_on_opponent_run",
    ]
    for col in bool_cols:
        if col in out.columns:
            out[col] = out[col].astype(int)

    return out[FEATURE_COLS].fillna(0)


def build_away_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Flip home/away perspective for predicting away-team runs.

    Returns a new DataFrame with home ↔ away swapped so the same
    model can predict away scoring runs.
    """
    away = X.copy()

    swap_pairs = [
        ("home_lineup_net_rating",    "away_lineup_net_rating"),
        ("home_lineup_sample_size",   "away_lineup_sample_size"),
        ("home_lineup_just_changed",  "away_lineup_just_changed"),
        ("home_points_last_5_poss",   "away_points_last_5_poss"),
        ("home_points_last_10_poss",  "away_points_last_10_poss"),
        ("home_scoring_sustainable",  "away_scoring_sustainable"),
        ("home_key_foul_count",       "away_key_foul_count"),
    ]
    for home_col, away_col in swap_pairs:
        if home_col in away.columns and away_col in away.columns:
            away[home_col], away[away_col] = X[away_col].copy(), X[home_col].copy()

    # Flip signed fields
    away["lineup_net_rating_delta"]   = -X["lineup_net_rating_delta"]
    away["current_run_team_encoded"]  = -X["current_run_team_encoded"]
    away["score_diff"]                = -X["score_diff"]

    return away


@dataclass
class TrainResult:
    model: XGBClassifier
    val_aucpr: float
    feature_importances: dict[str, float]
    n_train: int
    n_val: int


def train(feature_store_path: Path = FEATURE_STORE_PATH) -> TrainResult:
    logger.info("Loading feature store from %s", feature_store_path)
    fr = pq.read_table(feature_store_path).to_pandas()
    games = pq.read_table(GAMES_PATH).to_pandas()

    # Join game_date for time-based split
    games["game_date"] = pd.to_datetime(games["game_date"])
    fr = fr.merge(games[["game_id", "game_date"]], on="game_id", how="left")

    missing_date = fr["game_date"].isna().sum()
    if missing_date > 0:
        logger.warning("%d rows have no game_date — dropping", missing_date)
        fr = fr.dropna(subset=["game_date"])

    # Drop blowout / garbage time from training (model is off during these)
    fr = fr[~fr["is_blowout"] & ~fr["is_garbage_time"]].copy()
    logger.info("After blowout/GT filter: %d rows", len(fr))

    X_all = prepare_features(fr)
    y_all = fr[TARGET_COL].astype(int)
    dates = fr["game_date"]

    train_mask = dates <= TRAIN_END
    val_mask   = (dates > TRAIN_END) & (dates <= VAL_END)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val,   y_val   = X_all[val_mask],   y_all[val_mask]

    logger.info(
        "Split: train=%d (%d positive, %.1f%%), val=%d (%d positive, %.1f%%)",
        len(X_train), y_train.sum(), y_train.mean() * 100,
        len(X_val),   y_val.sum(),   y_val.mean() * 100,
    )

    # Positive class weight: ratio of negatives to positives
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    logger.info("scale_pos_weight = %.1f", scale_pos_weight)

    # Train without class weighting so output probabilities are calibrated to the
    # true base rate (~7.6%). Imbalance is handled via AUCPR metric + early stopping.
    # A predicted probability of 0.15 is then ~2× the base rate — a meaningful signal.
    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    val_aucpr = _compute_aucpr(y_val.values, val_proba)
    baseline_aucpr = y_val.mean()  # random classifier AUCPR ≈ positive rate

    logger.info("Val AUCPR: %.4f  (baseline random: %.4f)", val_aucpr, baseline_aucpr)

    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    top10 = sorted(importances.items(), key=lambda x: -x[1])[:10]
    logger.info("Top 10 features: %s", top10)

    return TrainResult(
        model=model,
        val_aucpr=val_aucpr,
        feature_importances=importances,
        n_train=len(X_train),
        n_val=len(X_val),
    )


def _compute_aucpr(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Area under precision-recall curve via trapezoidal integration."""
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, y_proba))


def save_model(model: XGBClassifier, path: Path = MODEL_OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", path)


def load_model(path: Path = MODEL_OUTPUT_PATH) -> XGBClassifier:
    with open(path, "rb") as f:
        return pickle.load(f)


class RunPredictor:
    """
    Thin wrapper around the trained XGBClassifier.

    Provides predict_home_run() and predict_away_run() so strategies
    don't need to know about feature flipping.
    """

    def __init__(self, model: XGBClassifier) -> None:
        self.model = model

    @classmethod
    def from_file(cls, path: Path = MODEL_OUTPUT_PATH) -> "RunPredictor":
        return cls(load_model(path))

    def predict_home_run(self, feature_row_dict: dict) -> float:
        """Return probability that home team has a meaningful run in next 5 possessions."""
        X = self._dict_to_frame(feature_row_dict)
        return float(self.model.predict_proba(X)[0, 1])

    def predict_away_run(self, feature_row_dict: dict) -> float:
        """Return probability that away team has a meaningful run in next 5 possessions."""
        X = self._dict_to_frame(feature_row_dict)
        X_flipped = build_away_features(X)
        return float(self.model.predict_proba(X_flipped)[0, 1])

    def _dict_to_frame(self, feature_row_dict: dict) -> pd.DataFrame:
        """Convert a FeatureRow-like dict into a model-ready DataFrame row."""
        row = {col: feature_row_dict.get(col, 0) for col in FEATURE_COLS}

        # current_run_team might still be a string here
        if "current_run_team_encoded" not in row or row["current_run_team_encoded"] == 0:
            raw = feature_row_dict.get("current_run_team")
            row["current_run_team_encoded"] = {"home": 1, "away": -1}.get(raw, 0)

        return pd.DataFrame([row], columns=FEATURE_COLS).fillna(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = train()
    save_model(result.model)

    print(f"\n=== Run Predictor Training Complete ===")
    print(f"  Train rows:   {result.n_train:,}")
    print(f"  Val rows:     {result.n_val:,}")
    print(f"  Val AUCPR:    {result.val_aucpr:.4f}")
    print(f"  Model saved:  {MODEL_OUTPUT_PATH}")
