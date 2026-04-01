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
      target_home_next_5_margin        int  — home - away points in next 5 possessions
      target_home_next_10_margin       int  — same for next 10 possessions
      target_meaningful_run_5_scoring  bool — home outscores by 6+ in next 5 SCORING possessions
      target_meaningful_run_10         bool — home outscores by 8+ in next 10 possessions
    """
    df = poss_df.copy()
    n  = len(df)

    home_pts = np.where(df["team_scored"] == "home", df["points"].values, 0)
    away_pts = np.where(df["team_scored"] == "away", df["points"].values, 0)
    net_pts  = home_pts - away_pts  # positive = home team scored

    # Compute targets over the next N SCORING possessions for every row.
    # Works correctly whether the input contains all possessions or scoring-only.
    scoring_mask    = df["points"].values > 0
    scoring_idx     = np.where(scoring_mask)[0]
    cum_scoring_net = np.concatenate([[0], np.cumsum(net_pts[scoring_idx])])

    # For each row i, find the first scoring possession strictly after it
    all_rows   = np.arange(n)
    next_start = np.searchsorted(scoring_idx, all_rows + 1)

    next_end_5  = np.minimum(next_start + 5,  len(scoring_idx))
    next_end_10 = np.minimum(next_start + 10, len(scoring_idx))

    df["target_home_next_5_margin"]       = (cum_scoring_net[next_end_5]  - cum_scoring_net[next_start]).astype(int)
    df["target_home_next_10_margin"]      = (cum_scoring_net[next_end_10] - cum_scoring_net[next_start]).astype(int)
    df["target_meaningful_run_5_scoring"] = (df["target_home_next_5_margin"] >= 6)
    df["target_meaningful_run_10"]        = (df["target_home_next_10_margin"] >= 8)

    return df
