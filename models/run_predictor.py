"""
XGBoost run predictor.

Trains a binary classifier to predict target_meaningful_run_5_scoring:
whether the home team will have a meaningful scoring run (6+ net points)
in the next 5 scoring possessions.

Data source: features.possession_flat in local DuckDB (kalshi_trading.duckdb).
Pull latest with:
    python -m data.ingestion.duckdb_loader --pull-possession-flat

Train / val / test split — time-based, never random:
  Train:  Oct 21, 2025 – Jan 31, 2026
  Val:    Feb 1  – Mar 5, 2026
  Test:   Mar 6, 2026 – present  ← touch once at final eval only

Usage:
    python models/run_predictor.py
"""

import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.features.targets import add_targets

logger = logging.getLogger(__name__)

DB_PATH         = Path("kalshi_trading.duckdb")
MODEL_OUTPUT_PATH = Path("models/saved/run_predictor.pkl")

TRAIN_END = pd.Timestamp("2026-01-31")
VAL_END   = pd.Timestamp("2026-03-05")
# Test: everything after VAL_END — do not touch during development

FEATURE_COLS: list[str] = [
    # Score × time
    "score_diff",
    "period",
    "minutes_into_game",            # derived: (period-1)*12 + elapsed_in_period
    "trailing_team_urgency",
    "comeback_probability_proxy",
    "q4_close_game",
    "garbage_time_risk",
    # Shot info
    "shot_value",
    "shot_distance",
    # Momentum
    "home_points_last_5_poss",
    "away_points_last_5_poss",
    "home_points_last_10_poss",
    "away_points_last_10_poss",
    "current_run_team_encoded",     # derived: 1=home, -1=away, 0=none
    "current_run_length",
    "current_run_points",
    "current_run_3pt_pct",
    "current_run_paint_pct",
    # Pace
    "pace_last_10_possessions",
    "pace_season_baseline",
    # Shot quality
    "home_scoring_sustainable",
    "away_scoring_sustainable",
    "home_xPPP_last_5",
    "away_xPPP_last_5",
    "home_actual_vs_expected_PPP",
    "away_actual_vs_expected_PPP",
    "home_shot_quality_trend",
    "away_shot_quality_trend",
    # Foul state
    "home_team_fouls_q",            # team fouls this quarter (>= 5 = in bonus)
    "away_team_fouls_q",
    "home_cum_fouls",               # cumulative team fouls this game
    "away_cum_fouls",
    "home_in_bonus",                # derived: home_team_fouls_q >= 5
    "away_in_bonus",
    "home_fouls_until_bonus",       # derived: max(0, 5 - home_team_fouls_q)
    "away_fouls_until_bonus",
    "home_star_in_foul_trouble",
    "away_star_in_foul_trouble",
    "home_star_on_court",
    "away_star_on_court",
    # Event context at this possession
    "was_foul",
    "was_sub",
    "had_shooting_foul",
    "had_personal_foul",
    "home_sub_count",
    "away_sub_count",
    # Timeout signals
    "possessions_since_last_timeout",
    "home_called_timeout_in_last_3_poss",
    "away_called_timeout_in_last_3_poss",
    "home_full_timeouts_remaining",  # derived: max(0, 4 - home_timeouts_used)
    "away_full_timeouts_remaining",
    # Lineup signal
    "home_lineup_net_rating",
    "away_lineup_net_rating",
    "lineup_net_rating_delta",        # THE core signal — matchup quality delta
    "home_lineup_sample_size",        # possessions together (confidence weight)
    "away_lineup_sample_size",
    "home_lineup_just_changed",       # substitution just occurred this possession
    "away_lineup_just_changed",
]

TARGET_COL = "target_meaningful_run_5_scoring"


def _load_data() -> pd.DataFrame:
    """
    Load all possessions from features.possession_flat joined with game_date.

    All possession types are included — scoring, fouls, timeouts, subs — so the
    model can predict at any game state, not just after a made shot.
    Results are sorted by game_id, event_id to guarantee chronological order.
    """
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute("""
        SELECT pf.*, dg.game_date
        FROM features.possession_flat pf
        JOIN main.dim_games dg ON pf.game_id = dg.game_id
        ORDER BY pf.game_id, pf.event_id
    """).df()
    conn.close()
    logger.info("Loaded %d possessions from possession_flat", len(df))
    return df


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute feature columns derived from possession_flat fields."""
    out = df.copy()

    # Minutes elapsed since tip-off (game_clock_secs counts DOWN from 720 per period)
    elapsed_in_period = (720 - out["game_clock_secs"].clip(0, 720)) / 60
    out["minutes_into_game"] = (out["period"] - 1) * 12 + elapsed_in_period

    # Bonus state: 5+ team fouls in this quarter
    out["home_in_bonus"] = (out["home_team_fouls_q"] >= 5).astype(int)
    out["away_in_bonus"] = (out["away_team_fouls_q"] >= 5).astype(int)
    out["home_fouls_until_bonus"] = (5 - out["home_team_fouls_q"]).clip(lower=0)
    out["away_fouls_until_bonus"] = (5 - out["away_team_fouls_q"]).clip(lower=0)

    # Remaining timeouts (each team starts with 4 full timeouts)
    out["home_full_timeouts_remaining"] = (4 - out["home_timeouts_used"]).clip(lower=0)
    out["away_full_timeouts_remaining"] = (4 - out["away_timeouts_used"]).clip(lower=0)

    # Run team: string → numeric
    out["current_run_team_encoded"] = (
        out["current_run_team"].map({"home": 1, "away": -1}).fillna(0)
    )

    # Star foul trouble: possession_flat stores trouble_star_tier (0–3); derive bool
    out["home_star_in_foul_trouble"] = (out["home_trouble_star_tier"] > 0).astype(float)
    out["away_star_in_foul_trouble"] = (out["away_trouble_star_tier"] > 0).astype(float)

    return out


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add forward-looking target variables, computed per game in chronological order."""
    parts = []
    for _, game_df in df.groupby("game_id", sort=False):
        parts.append(add_targets(game_df.reset_index(drop=True)))
    return pd.concat(parts, ignore_index=True)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean numeric feature matrix aligned to FEATURE_COLS."""
    out = df.copy()

    bool_cols = [
        "home_scoring_sustainable", "away_scoring_sustainable",
        "q4_close_game",
        "home_in_bonus", "away_in_bonus",
        "home_star_in_foul_trouble", "away_star_in_foul_trouble",
        "home_star_on_court", "away_star_on_court",
        "was_foul", "was_sub",
        "had_shooting_foul", "had_personal_foul",
        "home_called_timeout_in_last_3_poss", "away_called_timeout_in_last_3_poss",
        "home_lineup_just_changed", "away_lineup_just_changed",
    ]
    for col in bool_cols:
        if col in out.columns:
            out[col] = out[col].astype(float)

    # Only keep FEATURE_COLS that exist — allows graceful degradation
    available = [c for c in FEATURE_COLS if c in out.columns]
    missing = [c for c in FEATURE_COLS if c not in out.columns]
    if missing:
        logger.warning("Features missing from data (will be 0): %s", missing)

    result = out[available].copy()
    for col in missing:
        result[col] = 0.0

    return result[FEATURE_COLS].fillna(0)


def build_away_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Flip home/away perspective for predicting away-team runs.

    Returns a new DataFrame with home ↔ away swapped so the same
    model can predict away scoring runs without retraining.
    """
    away = X.copy()

    swap_pairs = [
        ("home_points_last_5_poss",           "away_points_last_5_poss"),
        ("home_points_last_10_poss",          "away_points_last_10_poss"),
        ("home_scoring_sustainable",          "away_scoring_sustainable"),
        ("home_team_fouls_q",                 "away_team_fouls_q"),
        ("home_cum_fouls",                    "away_cum_fouls"),
        ("home_in_bonus",                     "away_in_bonus"),
        ("home_fouls_until_bonus",            "away_fouls_until_bonus"),
        ("home_star_in_foul_trouble",         "away_star_in_foul_trouble"),
        ("home_star_on_court",                "away_star_on_court"),
        ("home_sub_count",                    "away_sub_count"),
        ("home_called_timeout_in_last_3_poss","away_called_timeout_in_last_3_poss"),
        ("home_full_timeouts_remaining",      "away_full_timeouts_remaining"),
        ("home_xPPP_last_5",                  "away_xPPP_last_5"),
        ("home_actual_vs_expected_PPP",       "away_actual_vs_expected_PPP"),
        ("home_shot_quality_trend",           "away_shot_quality_trend"),
        ("home_lineup_net_rating",            "away_lineup_net_rating"),
        ("home_lineup_sample_size",           "away_lineup_sample_size"),
        ("home_lineup_just_changed",          "away_lineup_just_changed"),
    ]
    for home_col, away_col in swap_pairs:
        if home_col in away.columns and away_col in away.columns:
            away[home_col], away[away_col] = X[away_col].copy(), X[home_col].copy()

    away["current_run_team_encoded"] = -X["current_run_team_encoded"]
    away["score_diff"]               = -X["score_diff"]
    away["lineup_net_rating_delta"]  = -X["lineup_net_rating_delta"]

    return away


@dataclass
class TrainResult:
    model: XGBClassifier
    val_aucpr: float
    feature_importances: dict[str, float]
    n_train: int
    n_val: int


def train() -> TrainResult:
    df = _load_data()
    df = _add_derived_features(df)
    df = _add_targets(df)

    df["game_date"] = pd.to_datetime(df["game_date"])

    missing_date = df["game_date"].isna().sum()
    if missing_date > 0:
        logger.warning("%d rows have no game_date — dropping", missing_date)
        df = df.dropna(subset=["game_date"])

    # Drop blowout / garbage time — model is off during these
    df = df[~df["is_blowout"] & ~df["is_garbage_time"]].copy()
    logger.info("After blowout/garbage-time filter: %d rows", len(df))

    X_all = prepare_features(df)
    y_all = df[TARGET_COL].astype(int)
    dates = df["game_date"]

    train_mask = dates <= TRAIN_END
    val_mask   = (dates > TRAIN_END) & (dates <= VAL_END)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val,   y_val   = X_all[val_mask],   y_all[val_mask]

    logger.info(
        "Split: train=%d (%d positive, %.1f%%), val=%d (%d positive, %.1f%%)",
        len(X_train), y_train.sum(), y_train.mean() * 100,
        len(X_val),   y_val.sum(),   y_val.mean() * 100,
    )

    # Train without class weighting so output probabilities stay calibrated to
    # the true base rate (~7-8%). A threshold of 0.15 then means ~2× base rate.
    # Imbalance is handled via AUCPR metric + early stopping.
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

    val_proba  = model.predict_proba(X_val)[:, 1]
    val_aucpr  = _compute_aucpr(y_val.values, val_proba)
    baseline   = float(y_val.mean())

    logger.info("Val AUCPR: %.4f  (baseline random: %.4f)", val_aucpr, baseline)

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
        row = {col: feature_row_dict.get(col, 0) for col in FEATURE_COLS}

        # current_run_team might still be a string at inference time
        if not row.get("current_run_team_encoded"):
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
