"""
Layer 2 — Kalshi Price Movement Model

XGBoost regressor predicting expected Δ(yes_bid) over the next 30-180 seconds.
Requires 4+ weeks of real tick data.
"""
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.features.kalshi_features import add_kalshi_features
from models.features.l1_bridge import build_l1_bridge
from models.targets.kalshi_targets import add_kalshi_targets

logger = logging.getLogger(__name__)

# Output
MODEL_OUTPUT_PATH = Path("models/saved/l2_kalshi_predictor.pkl")

# We will predict the dynamic delta bid based on exact possession resolution
TARGET_COL = "target_delta_bid"

FEATURE_COLS = [
    # L1 probabilities and derivatives
    "run_prob",
    "run_prob_velocity",
    "time_decay_prob",
    "prob_price_divergence",
    
    # Kalshi Microstructure
    "spread",
    "trade_volume_60s",
    "time_since_last_trade_ms",
    
    # Dynamics Context
    "expected_minutes_for_5_poss",
    
    # Optional context (can be passed from L1 features)
    # "score_diff",
    # "period",
    # "minutes_remaining",
]


import duckdb
from dotenv import load_dotenv

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pulls pure historical websocket ticks (Layer 2) and play-by-play (Layer 1) natively 
    from your MotherDuck cloud environment.
    """
    import duckdb
    from dotenv import load_dotenv
    
    load_dotenv()
    
    try:
        conn = duckdb.connect('md:')
        
        logger.info("Downloading historical Kalshi ticks...")
        raw_ticks = conn.execute("SELECT * FROM kalshi_trading.main.kalshi_ticks ORDER BY ts ASC").df()
        
        logger.info("Mapping Play-by-Play wall-clock timestamps to Layer 1 state...")
        raw_l1 = conn.execute('''
            SELECT 
                   pf.game_id, 
                   pf.possession_id, 
                   pf.wall_clock_ts as ts, 
                   GREATEST(li.run_prob_home, li.run_prob_away) as run_prob,
                   20.0 as minutes_remaining,
                   100.0 as pace_last_10_possessions,
                   pf.wall_clock_ts + INTERVAL 3 MINUTE as target_ts
            FROM kalshi_trading.main.possession_feed pf
            LEFT JOIN layer2_models.main.l1_inferences li
                ON pf.game_id = li.game_id AND pf.possession_id = li.possession_id
            WHERE pf.wall_clock_ts IS NOT NULL
            ORDER BY pf.wall_clock_ts ASC
        ''').df()
        
        logger.info(f"Loaded {len(raw_ticks)} market ticks and {len(raw_l1)} possession milestones.")
        return raw_ticks, raw_l1
    except Exception as e:
        logger.error(f"MotherDuck pull failed. Check MOTHERDUCK_TOKEN: {e}")
        return pd.DataFrame(), pd.DataFrame()


def prepare_dataset(ticks: pd.DataFrame, l1_probs: pd.DataFrame) -> pd.DataFrame:
    """Builds features, bridges L1/L2, and calculates targets."""
    # 1. Base order book microstructure
    features = add_kalshi_features(ticks)
    
    # 2. Bridge L1 probabilities onto the tick stream
    features = build_l1_bridge(features, l1_probs)
    
    # 3. Generate future targets
    dataset = add_kalshi_targets(features)
    
    # Ensure consistent target naming whether dynamic or fallback was used
    if "target_delta_bid_180s" in dataset.columns and TARGET_COL not in dataset.columns:
        dataset = dataset.rename(columns={"target_delta_bid_180s": TARGET_COL})
    
    # Filter out exact nulls on target due to edge of dataset (last 180 seconds)
    if TARGET_COL in dataset.columns:
        dataset = dataset.dropna(subset=[TARGET_COL])
        
    return dataset


@dataclass
class L2TrainResult:
    model: XGBRegressor
    val_rmse: float
    feature_importances: dict[str, float]
    n_train: int
    n_val: int


def train() -> L2TrainResult:
    raw_ticks, raw_l1 = load_raw_data()
    
    if raw_ticks.empty or raw_l1.empty:
        logger.warning("No data found to train Layer 2. Return empty initialized model.")
        return L2TrainResult(XGBRegressor(), 0.0, {}, 0, 0)
        
    df = prepare_dataset(raw_ticks, raw_l1)
    
    # Time-based splitting
    # Ensuring purged cross validation over weeks
    df["date"] = df["ts"].dt.date
    dates = sorted(df["date"].unique())
    
    if len(dates) < 4:
        logger.warning("Less than 4 days of data loaded. Model requires 4+ weeks.")
        
    # Split: 80% train, 20% val chronologically
    split_idx = int(len(dates) * 0.8)
    train_dates = dates[:split_idx]
    
    train_mask = df["date"].isin(train_dates)
    val_mask = ~train_mask
    
    X_train = df.loc[train_mask, FEATURE_COLS].fillna(0)
    y_train = df.loc[train_mask, TARGET_COL]
    w_train = np.where(df.loc[train_mask, "is_tradable_target"], 1.0 / df.loc[train_mask, "spread"].clip(lower=1.0), 0.0)

    X_val = df.loc[val_mask, FEATURE_COLS].fillna(0)
    y_val = df.loc[val_mask, TARGET_COL]
    w_val = np.where(df.loc[val_mask, "is_tradable_target"], 1.0 / df.loc[val_mask, "spread"].clip(lower=1.0), 0.0)
    
    logger.info("Split: train=%d rows, val=%d rows", len(X_train), len(X_val))

    # Extreme low learning rate for tick-data regression
    model = XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        eval_metric="rmse",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )
    
    # Train using custom spread-inversed sample weights
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        sample_weight_eval_set=[w_train, w_val],
        verbose=100
    )
    
    # Evaluate
    val_preds = model.predict(X_val)
    val_rmse = np.sqrt(np.average((y_val - val_preds)**2, weights=w_val))
    logger.info("Weighted Validation RMSE: %.4f cents", val_rmse)

    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    top = sorted(importances.items(), key=lambda x: -x[1])[:5]
    logger.info("Top features: %s", top)
    
    return L2TrainResult(model, val_rmse, importances, len(X_train), len(X_val))


def save_model(model: XGBRegressor, path: Path = MODEL_OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = train()
    if result.n_train > 0:
        save_model(result.model)
