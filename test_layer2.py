"""
Test script to verify Layer 2 features, target generation, and L1 bridging logic.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the project root to sys.path so we can import from models
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from models.features.kalshi_features import add_kalshi_features
from models.features.l1_bridge import build_l1_bridge
from models.targets.kalshi_targets import add_kalshi_targets

def generate_synthetic_data():
    # 1. Synthetic Kalshi Ticks
    # Ticks over 4 minutes
    base_time = pd.Timestamp("2026-03-30 20:00:00")
    
    ticks = pd.DataFrame({
        "ts": [
            base_time, 
            base_time + pd.Timedelta(seconds=30),
            base_time + pd.Timedelta(seconds=60),
            base_time + pd.Timedelta(seconds=180), # Expected target for first tick will look near this timestamp
            base_time + pd.Timedelta(seconds=200),
            base_time + pd.Timedelta(seconds=240),
            base_time + pd.Timedelta(seconds=300),
        ],
        "yes_bid":  [40.0, 42.0, 45.0, 50.0, 50.0, 48.0, 47.0],
        "yes_ask":  [43.0, 45.0, 47.0, 66.0, 53.0, 51.0, 49.0], # Notice spread spikes at 180s (16 cents)
        "volume":   [   0,  500,    0,    0,  300,    0,  100]
    })
    
    # 2. Synthetic Layer 1 Probabilities
    # Updates every minute roughly
    l1_probs = pd.DataFrame({
        "ts": [
            base_time - pd.Timedelta(seconds=10), # Start of game
            base_time + pd.Timedelta(seconds=45), # Game event
            base_time + pd.Timedelta(seconds=150),# Game event
        ],
        "run_prob": [0.35, 0.40, 0.55],
        "minutes_remaining": [20.0, 19.5, 18.0],
        "pace_last_10_possessions": [105.0, 95.0, 100.0],
        # The exact timestamp when that specific specific 5-possession run resolved!
        "target_ts": [
            base_time + pd.Timedelta(seconds=180),  # Fast resolution (190s elapsed)
            base_time + pd.Timedelta(seconds=240),  # Average resolution (195s elapsed)
            base_time + pd.Timedelta(seconds=300),  # Slow resolution (150s elapsed due to late game)
        ]
    })
    
    return ticks, l1_probs

def test_pipeline():
    ticks, l1_probs = generate_synthetic_data()
    
    print("=== SYNTHETIC MARKET TICKS ===")
    print(ticks[["ts", "yes_bid", "yes_ask", "volume"]])
    print("\\n=== SYNTHETIC LAYER 1 PROBS ===")
    print(l1_probs)
    
    # Phase 1: Market features
    f1 = add_kalshi_features(ticks)
    assert "spread" in f1.columns
    assert "trade_volume_60s" in f1.columns
    assert "time_since_last_trade_ms" in f1.columns
    
    # Phase 2: Bridge
    f2 = build_l1_bridge(f1, l1_probs)
    assert "run_prob" in f2.columns
    assert "run_prob_velocity" in f2.columns
    assert "time_decay_prob" in f2.columns
    
    # Phase 3: Targets
    # Now passing the merged dataframe which contains 'target_ts'
    f3 = add_kalshi_targets(f2)
    assert "target_delta_bid" in f3.columns
    assert "is_tradable_target" in f3.columns
    
    print("\\n=== RESULTING DATAFRAME (Selected Columns) ===")
    display_cols = [
        "ts", "yes_bid", "spread", "expected_minutes_for_5_poss", "target_ts", "target_delta_bid", "is_tradable_target"
    ]
    print(f3[display_cols].to_string(index=False))
    
    print("\\nLogic tests passed successfully!")

if __name__ == "__main__":
    test_pipeline()
