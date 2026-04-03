"""
Layer 1 Bridge.

Fuses the slow, possession-based Layer 1 probabilities with the
high-frequency market tick data. Utilizes merge_asof to ensure no
lookahead bias (forward-fills the most recent L1 prob onto the current L2 tick).
"""

import numpy as np
import pandas as pd


def imply_fair_price(run_prob: pd.Series) -> pd.Series:
    """
    Convert a 0-1 probability into a Kalshi implied price in cents (1-99).
    Assumes binary market payout of 100 cents.
    """
    return (run_prob * 100).clip(1.0, 99.0)


def build_l1_bridge(l2_ticks: pd.DataFrame, l1_probs: pd.DataFrame) -> pd.DataFrame:
    """
    Merge L1 probabilities with L2 ticks and compute cross-layer features.

    Inputs:
      - l2_ticks: DataFrame featuring at least 'ts' (datetime), 'bid' (cents)
      - l1_probs: DataFrame with 'ts' (datetime) and 'run_prob' (0.0 to 1.0),
                  and ideally 'minutes_remaining'.
    """
    if l2_ticks.empty or l1_probs.empty:
        return l2_ticks

    # Ensure temporal sorting
    ticks = l2_ticks.sort_values("ts").copy()
    probs = l1_probs.sort_values("ts").copy()
    
    # 1. Forward-fill the most recent Layer 1 probability onto the tick data
    # direction="backward" means: for each L2 tick timestamp,
    # find the latest L1 timestamp that is <= the L2 timestamp.
    merged = pd.merge_asof(
        ticks, 
        probs, 
        on="ts", 
        direction="backward",
        suffixes=("", "_l1")
    )
    
    # If there are ticks before the first L1 prediction, fill with base rate ~0.076
    if "run_prob" in merged.columns:
        merged["run_prob"] = merged["run_prob"].fillna(0.076)
        
        # 2. Probability Velocity (Change in prob over the last X seconds)
        # Assuming the index is temporal or we can use rolling windows
        temp = merged.set_index("ts")
        
        # Calculate velocity over last 15 seconds
        # velocity = current prob - probing from 15s ago
        prob_series = temp["run_prob"]
        # Use simple difference for simplicity (most recent vs 15s ago)
        # If there wasn't a tick 15 seconds ago exactly, rolling takes max within window minus min (or explicitly taking first)
        velocity_series = prob_series - prob_series.shift(1).bfill()
        merged["run_prob_velocity"] = velocity_series.values
        
        # 3. Market Mispricing (Prob-implied Price vs Actual Bid)
        if "yes_bid" in merged.columns:
            implied = imply_fair_price(merged["run_prob"])
            merged["prob_price_divergence"] = implied - merged["yes_bid"]
            
        # 4. Time-Decay Weighted Probability
        if "minutes_remaining" in merged.columns:
            k = 0.05 # Decay constant
            merged["time_decay_prob"] = merged["run_prob"] * np.exp(-k * merged["minutes_remaining"])
            
        # 5. Pace-Estimated Feature (Expected Run Duration)
        if "pace_last_10_possessions" in merged.columns:
            # Pace is possessions per 48 minutes per team (roughly 100 for a fast game, 90 for slow)
            # Duration for 5 possessions of target team = 5 / (pace / 48) = 240 / pace
            pace_clamped = merged["pace_last_10_possessions"].fillna(100.0).clip(lower=60.0, upper=140.0)
            merged["expected_minutes_for_5_poss"] = 240.0 / pace_clamped
            
    return merged
