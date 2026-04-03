import logging
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterSampler
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.l2_predictor import (
    load_raw_data, 
    prepare_dataset, 
    FEATURE_COLS, 
    TARGET_COL, 
    save_model
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_tuning(n_iter=10):
    logger.info("Initializing Layer 2 Hyperparameter Tuning...")
    
    # 1. Load the exact dataset logic natively from the predictor
    raw_ticks, raw_l1 = load_raw_data()
    if raw_ticks.empty or raw_l1.empty:
        logger.error("No data found! Tuning aborted.")
        return
        
    df = prepare_dataset(raw_ticks, raw_l1)
    df["date"] = df["ts"].dt.date
    dates = sorted(df["date"].unique())
    
    # Chronological Split (80% train, 20% validation)
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
    
    logger.info(f"Tuning Base Loaded! Train: {len(X_train)} rows | Val: {len(X_val)} rows.")
    
    # 2. Define the Tuning Parameter Grid
    param_grid = {
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'max_depth': [4, 5, 6, 7, 8],
        'n_estimators': [500, 800, 1000],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.5]
    }
    
    # 3. Create the random sampler sequence
    sampler = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))
    
    best_rmse = float('inf')
    best_model = None
    best_params = None
    
    logger.info(f"Iterating through {n_iter} theoretical architectures...")
    
    # 4. Sweep the architecture combinations natively against the inverse spread-weights
    for i, params in enumerate(sampler):
        logger.info(f"--- Iteration {i+1}/{n_iter} ---")
        
        model = XGBRegressor(
            **params,
            eval_metric="rmse",
            random_state=42,
            n_jobs=-1
        )
        
        # Fit with our exact financial business logic
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val],
            verbose=False
        )
        
        val_preds = model.predict(X_val)
        val_rmse = np.sqrt(np.average((y_val - val_preds)**2, weights=w_val))
        
        logger.info(f"Params: {params} => RMSE: {val_rmse:.4f}")
        
        if val_rmse < best_rmse:
            logger.info(f"🌟 NEW BEST RMSE FOUND: {val_rmse:.4f} (Previous: {best_rmse:.4f})")
            best_rmse = val_rmse
            best_model = model
            best_params = params
            
    # 5. Overwrite the final architecture into the original saved pipeline slot
    logger.info(f"Tuning complete. Best RMSE achieved: {best_rmse:.4f}")
    logger.info(f"Saving optimum mathematically computed parameters: {best_params}")
    
    save_model(best_model)
    logger.info("The production l2_kalshi_predictor.pkl Model is officially fully tuned!")

if __name__ == '__main__':
    # Start with 10 random iterations for safety/speed. 
    run_tuning(n_iter=10)
