"""
Target generator for Kalshi Layer 2 Models.

Computes forward-looking targets based on tick data.
Target variables should ONLY be used during training and MUST be excluded
during live inference.

Primary target: target_delta_bid_180s (change in the 'yes_bid' over 3 minutes).
Head C target: 10-horizon discrete survival (did run break by scoring possession k?).
"""

import pandas as pd


def add_kalshi_targets(df: pd.DataFrame, window_seconds: int = 180) -> pd.DataFrame:
    """
    Generate forward-looking target variables for Kalshi tick data.
    
    Inputs:
      - df: DataFrame featuring at least 'ts' (datetime) and 'yes_bid', 'yes_ask'.
      - window_seconds: The look-forward window for delta prediction.

    Outputs:
      - target_delta_bid: future_yes_bid - current_yes_bid.
      - is_tradable_target: Boolean mask. False if current spread is too wide.
    """
    if df.empty or "ts" not in df.columns or "yes_bid" not in df.columns:
        return df

    out = df.copy()
    out = out.sort_values("ts").reset_index(drop=True)
    
    # Needs to be proper datetime to use DateOffset
    if not pd.api.types.is_datetime64_any_dtype(out["ts"]):
        out["ts"] = pd.to_datetime(out["ts"])

    # Support dynamic possession-based target resolution if 'target_ts' provided
    if "target_ts" in out.columns:
        out["_target_ts"] = pd.to_datetime(out["target_ts"])
        target_col_name = "target_delta_bid"
    else:
        out["_target_ts"] = out["ts"] + pd.Timedelta(seconds=window_seconds)
        target_col_name = f"target_delta_bid_{window_seconds}s"

    # DataFrame representing just the future timestamps and the bid value to grab
    future_bids = out[["ts", "yes_bid"]].rename(columns={"ts": "_future_actual_ts", "yes_bid": "_future_yes_bid"})

    # Forward merge_asof: 
    # For every _target_ts, find the closest _future_actual_ts that is >= _target_ts.
    # We use direction='forward' because we want the state of the market AT or immediately AFTER the target resolution time.
    merged = pd.merge_asof(
        out,
        future_bids,
        left_on="_target_ts",
        right_on="_future_actual_ts",
        direction="forward"
    )

    # 1. Target column: diff between future bid and current bid
    merged[target_col_name] = merged["_future_yes_bid"] - merged["yes_bid"]
    
    # Drop temp columns
    if "target_ts" not in merged.columns:
        merged = merged.drop(columns=["_target_ts"], errors="ignore")
    merged = merged.drop(columns=["_future_actual_ts", "_future_yes_bid"], errors="ignore")

    # 2. Add tradability mask. 
    # If spread > 15 cents, market is illiquid (e.g., timeout). 
    # We assign zero weight to predicting this noise.
    max_tradable_spread = 15.0
    
    if "yes_ask" in merged.columns and "yes_bid" in merged.columns:
        spread = merged["yes_ask"] - merged["yes_bid"]
        merged["is_tradable_target"] = spread <= max_tradable_spread
    else:
        merged["is_tradable_target"] = True

    return merged


def add_hazard_targets(df: pd.DataFrame, n_horizons: int = 10) -> pd.DataFrame:
    """
    Generate discrete survival hazard targets (Target C) for each possession row.

    For each row, looks forward through the next n_horizons SCORING possessions
    and records whether the current run has broken at each horizon.

    A run "breaks" when current_run_team changes from the entry value to a different
    team or becomes neutral (None / NaN / "none").

    Columns added:
      haz_0 .. haz_{n-1}: 1.0 if run has broken by scoring possession k, else 0.0
                          Monotonically non-decreasing (once broken, stays broken).
                          Rows without an active run: all hazards = 1.0.

    Args:
        df: possession_flat rows for a single game, sorted chronologically.
            Must have: game_id, event_id, current_run_team.
            Optionally: team_scored (to identify scoring possessions).
    """
    if df.empty:
        return df

    out = df.copy().reset_index(drop=True)
    haz_cols = [f"haz_{k}" for k in range(n_horizons)]

    # Identify scoring possession indices (rows where a team scored)
    if "team_scored" in out.columns:
        scoring_mask = out["team_scored"].notna() & (out["team_scored"] != "")
    else:
        # Fallback: treat all possessions as scoring
        scoring_mask = pd.Series(True, index=out.index)

    scoring_indices = out.index[scoring_mask].tolist()

    hazards = pd.DataFrame(
        data=0.0,
        index=out.index,
        columns=haz_cols,
        dtype=float,
    )

    for idx in out.index:
        entry_run_team = out.at[idx, "current_run_team"]

        # No active run at entry → hazard = 1 at all horizons
        if pd.isna(entry_run_team) or entry_run_team in ("", "none", None):
            hazards.loc[idx, haz_cols] = 1.0
            continue

        # Find the next n_horizons scoring possessions after this row
        future_scoring = [i for i in scoring_indices if i > idx]

        broken = False
        for k in range(n_horizons):
            if broken:
                hazards.at[idx, f"haz_{k}"] = 1.0
                continue

            if k >= len(future_scoring):
                # Game ended before this horizon — treat as broken (no run to survive)
                broken = True
                hazards.at[idx, f"haz_{k}"] = 1.0
                continue

            future_idx = future_scoring[k]
            future_run_team = out.at[future_idx, "current_run_team"]

            if future_run_team != entry_run_team:
                broken = True
                hazards.at[idx, f"haz_{k}"] = 1.0
            else:
                hazards.at[idx, f"haz_{k}"] = 0.0

    return pd.concat([out, hazards], axis=1)
