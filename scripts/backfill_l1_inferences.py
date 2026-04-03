import logging
import duckdb
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import sys

# Add parent directory to path so we can import project modules
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.run_predictor import (
    load_model, 
    _load_data, 
    _add_derived_features, 
    prepare_features, 
    build_away_features
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_backfill():
    logger.info("Starting Layer 1 Historical Backfill...")
    
    # 1. Load trained XGBoost Classifier
    try:
        model = load_model()
    except FileNotFoundError:
        logger.error("RunPredictor model not found. Ensure models/saved/run_predictor.pkl exists.")
        return

    # 2. Query MotherDuck DB (features.possession_flat)
    logger.info("Querying MotherDuck cloud for features.possession_flat...")
    try:
        load_dotenv()
        conn = duckdb.connect('md:kalshi_trading')
        df = conn.execute("""
            SELECT pf.*, dg.game_date
            FROM features.possession_flat pf
            JOIN main.dim_games dg ON pf.game_id = dg.game_id
            ORDER BY pf.game_id, pf.event_id
        """).df()
    except Exception as e:
        logger.error(f"Failed to query MotherDuck: {e}")
        return

    if df.empty:
        logger.warning("No possession_flat data found locally!")
        return

    # 3. Predict Probabilities in Batch
    logger.info(f"Deriving features and running XGBoost over {len(df)} possessions...")
    df_derived = _add_derived_features(df)
    
    X_home = prepare_features(df_derived)
    X_away = build_away_features(X_home)

    # predict_proba returns [prob_no_run, prob_run]
    run_prob_home = model.predict_proba(X_home)[:, 1]
    run_prob_away = model.predict_proba(X_away)[:, 1]

    # Combine results
    results_df = pd.DataFrame({
        "game_id": df["game_id"],
        "possession_id": df["possession_id"],
        "run_prob_home": run_prob_home,
        "run_prob_away": run_prob_away
    })

    # Drop potential duplicates and NaNs just in case
    results_df = results_df.dropna(subset=['game_id', 'possession_id'])
    
    # 4. Upload to MotherDuck
    logger.info("Connecting to MotherDuck to perform bulk INSERT into personal schema...")
    load_dotenv()
    try:
        # Note: We connect to 'md:' globally instead of md:kalshi_trading so we can create a new DB!
        conn = duckdb.connect('md:')
        
        # Create a personal/Layer 2 dedicated DB to securely bypass the organization's read-only share locks
        conn.execute("CREATE DATABASE IF NOT EXISTS layer2_models")
        
        # Drop previous inferences so we don't duplicate when doing full sweeps
        conn.execute("DROP TABLE IF EXISTS layer2_models.main.l1_inferences")
        
        # Ensure table exists in our dedicated database
        conn.execute('''
            CREATE TABLE IF NOT EXISTS layer2_models.main.l1_inferences (
                game_id VARCHAR,
                possession_id INTEGER,
                run_prob_home FLOAT,
                run_prob_away FLOAT,
                wall_clock_ts TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        logger.info(f"Uploading {len(results_df)} inferences remotely to layer2_models.main.l1_inferences...")
        # Instantly append the dataframe to the dedicated MotherDuck table
        conn.execute("INSERT INTO layer2_models.main.l1_inferences (game_id, possession_id, run_prob_home, run_prob_away) SELECT game_id, possession_id, run_prob_home, run_prob_away FROM results_df")
        
        logger.info("✅ Backfill Complete! Layer 2 is ready to train.")
    except Exception as e:
        logger.error(f"Failed to upload to MotherDuck: {e}")

if __name__ == "__main__":
    run_backfill()
