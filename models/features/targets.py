"""
Forward-looking target variables.

TRAINING ONLY — never used as features in live system.
These columns must be stripped before any live inference.

All targets are labeled with a `target_` prefix and a `_DO_NOT_USE_AS_FEATURE`
comment to make misuse obvious during code review.
"""

import numpy as np
import pandas as pd


def add_targets(poss_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add forward-looking target columns to a single game's possession DataFrame.

    Each target is computed AFTER the current possession — uses future data.
    DO NOT USE THESE AS FEATURES. They exist only for training the run predictor.

    Targets added:
      target_home_next_5_margin    int  — home - away points in next 5 possessions
      target_home_next_10_margin   int  — same for next 10 possessions
      target_meaningful_run_5      bool — home outscores by 6+ in next 5 possessions
      target_meaningful_run_10     bool — home outscores by 8+ in next 10 possessions
    """
    df = poss_df.copy()
    n  = len(df)

    home_pts = np.where(df["team_scored"] == "home", df["points"].values, 0)
    away_pts = np.where(df["team_scored"] == "away", df["points"].values, 0)
    net_pts  = home_pts - away_pts  # positive = home team scored

    # Rolling sum over the NEXT N possessions (forward-looking)
    net_series = pd.Series(net_pts)

    def future_sum(series: pd.Series, window: int) -> pd.Series:
        """Sum of the next `window` rows (exclusive of current)."""
        # Reverse, rolling sum, reverse back, then shift to align
        return series[::-1].rolling(window, min_periods=1).sum()[::-1].shift(-(window)).fillna(0)

    df["target_home_next_5_margin"]  = future_sum(net_series, 5).astype(int).values
    df["target_home_next_10_margin"] = future_sum(net_series, 10).astype(int).values
    df["target_meaningful_run_5"]    = (df["target_home_next_5_margin"]  >= 6)
    df["target_meaningful_run_10"]   = (df["target_home_next_10_margin"] >= 8)

    return df
