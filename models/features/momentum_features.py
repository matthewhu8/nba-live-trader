"""
Momentum features: scoring runs, pace, shot quality.

All computed within a single game's sorted possession sequence.
No lookahead — every feature at position i uses only possessions 0..i-1.
"""

import numpy as np
import pandas as pd

# Expected points per possession by shot type (league average constants)
_XPPP: dict[str, float] = {
    "3pt":        1.05,   # 3 × ~35% 3P%
    "paint":      1.20,   # 2 × ~60% inside FG%
    "midrange":   0.80,   # 2 × ~40% mid-range FG%
    "free_throw": 0.75,   # approx per-foul FT value
}


def _shot_xppp(shot_val: int, shot_dist: float) -> float:
    """Expected PPP for a single scoring possession based on shot type."""
    if shot_val == 1:
        return _XPPP["free_throw"]
    elif shot_val == 3:
        return _XPPP["3pt"]
    elif shot_dist < 8:
        return _XPPP["paint"]
    else:
        return _XPPP["midrange"]


def add_momentum_features(poss_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling momentum features on a single game's possessions (already sorted
    by possession_id). Adds columns in place:

      home_points_last_5_poss      int   — home points in the 5 possessions before this one
      away_points_last_5_poss      int
      home_points_last_10_poss     int
      away_points_last_10_poss     int
      current_run_team             str   — "home" | "away" | "" (no active run)
      current_run_length           int   — consecutive scoring possessions by run team
      current_run_points           int   — points in the current run
      current_run_3pt_count        int   — 3-pointers made in current run
      current_run_3pt_pct          float — fraction of run points from 3s (volatile = revert)
      current_run_paint_pct        float — fraction of run points from paint (sustainable)
      pace_last_10_possessions     float — avg seconds per possession (last 10)
      pace_season_baseline         float — game's avg seconds per possession so far
      home_scoring_sustainable     bool  — True if home's recent scoring is high-quality
      away_scoring_sustainable     bool
      home_xPPP_last_5             float — expected PPP from home team's shot selection (last 5)
      away_xPPP_last_5             float
      home_actual_vs_expected_PPP  float — actual minus expected PPP (positive = overperforming)
      away_actual_vs_expected_PPP  float
      home_shot_quality_trend      float — sign of (xPPP_last_5 - xPPP_prev_5): improving = +1
      away_shot_quality_trend      float
    """
    df = poss_df.copy()
    n  = len(df)

    # Point arrays by team
    home_pts = np.where(df["team_scored"] == "home", df["points"].values, 0)
    away_pts = np.where(df["team_scored"] == "away", df["points"].values, 0)

    # Rolling sums of the PREVIOUS N possessions (exclusive of current row)
    # Use shift(1) + rolling(N) to exclude the current possession.
    home_series = pd.Series(home_pts)
    away_series = pd.Series(away_pts)

    df["home_points_last_5_poss"]  = home_series.shift(1).rolling(5,  min_periods=0).sum().fillna(0).astype(int).values
    df["away_points_last_5_poss"]  = away_series.shift(1).rolling(5,  min_periods=0).sum().fillna(0).astype(int).values
    df["home_points_last_10_poss"] = home_series.shift(1).rolling(10, min_periods=0).sum().fillna(0).astype(int).values
    df["away_points_last_10_poss"] = away_series.shift(1).rolling(10, min_periods=0).sum().fillna(0).astype(int).values

    # Current run: consecutive scoring by one team immediately before this possession
    # Process sequentially — this can't be vectorized cleanly
    run_team      = [""] * n
    run_length    = [0]  * n
    run_points    = [0]  * n
    run_3pt_count = [0]  * n
    run_3pt_pct   = [0.0] * n
    run_paint_pct = [0.0] * n

    cur_team      = ""
    cur_length    = 0
    cur_points    = 0
    cur_3pt_count = 0
    cur_paint_pts = 0   # points scored from paint shots (distance < 8ft) in current run

    teams     = df["team_scored"].values
    pts       = df["points"].values
    shot_val  = df["shot_value"].values
    shot_dist = df["shot_distance"].values

    for i in range(n):
        # Capture run STATE BEFORE this possession (backward-looking)
        run_team[i]      = cur_team
        run_length[i]    = cur_length
        run_points[i]    = cur_points
        run_3pt_count[i] = cur_3pt_count
        if cur_points > 0:
            run_3pt_pct[i]   = (3 * cur_3pt_count) / cur_points
            run_paint_pct[i] = cur_paint_pts / cur_points
        else:
            run_3pt_pct[i]   = 0.0
            run_paint_pct[i] = 0.0

        # Update run state with this possession.
        # A run is a scoring streak — it only extends or resets when points are scored.
        # Stops and turnovers (p == 0) don't change run state: the run continues
        # until the opponent scores.
        t   = teams[i]
        p   = int(pts[i])
        is3 = int(shot_val[i]) == 3
        is_paint = float(shot_dist[i]) < 8.0  # includes FTs (dist=0) and paint shots

        if p > 0:
            if t == cur_team:
                cur_length    += 1
                cur_points    += p
                if is3:
                    cur_3pt_count += 1
                if is_paint:
                    cur_paint_pts += p
            else:
                # Opponent scored — start a new run for them
                cur_team      = t
                cur_length    = 1
                cur_points    = p
                cur_3pt_count = 1 if is3 else 0
                cur_paint_pts = p if is_paint else 0

    df["current_run_team"]      = run_team
    df["current_run_length"]    = run_length
    df["current_run_points"]    = run_points
    df["current_run_3pt_count"] = run_3pt_count
    df["current_run_3pt_pct"]   = run_3pt_pct
    df["current_run_paint_pct"] = run_paint_pct

    # Pace: seconds per possession (game_clock_secs decreases as game progresses)
    # We estimate possession duration as the drop in game_clock between consecutive possessions
    # within the same quarter. Cross-quarter drops are excluded.
    clock = df["game_clock_secs"].values
    period = df["period"].values
    possession_durations = np.zeros(n, dtype=float)
    for i in range(1, n):
        if period[i] == period[i - 1]:
            gap = clock[i - 1] - clock[i]  # time elapsed between possessions
            if 0 < gap < 60:               # sanity bounds: ignore outliers
                possession_durations[i] = gap

    dur_series = pd.Series(possession_durations)
    # Use shift(1) rolling to exclude current possession
    valid_dur  = dur_series.where(dur_series > 0)
    df["pace_last_10_possessions"] = (
        valid_dur.shift(1).rolling(10, min_periods=1).mean().fillna(15.0).values
    )
    # Expanding WITHIN-GAME mean up to (but not including) this possession.
    #
    # Historically named `pace_season_baseline`, which is not what it is — there is
    # nothing seasonal about it. That name caused a real bug: the live path read the
    # name rather than the behavior and fed the pregame constant under it, so the
    # column was a different quantity in each path. The true pregame prior is
    # `expected_pace`; the two are combined by transforms.pace_ref().
    #
    # Both names are written while possession_flat still carries the old column.
    # Drop `pace_season_baseline` once the table has been rebuilt.
    pace_game_to_date = (
        valid_dur.shift(1).expanding(min_periods=1).mean().fillna(15.0).values
    )
    df["pace_game_to_date"]    = pace_game_to_date
    df["pace_season_baseline"] = pace_game_to_date

    # Shot quality / sustainability:
    # "sustainable" = scoring via 3-pointers or short shots (distance < 5 ft = paint)
    # "unsustainable" = contested mid-range (distance 5-22 ft, not 3PT)
    # Masks restrict to actual scoring possessions (points > 0) so that stops and
    # turnovers — which have shot_distance=0 and shot_value=0 — don't pollute the
    # shot quality rolling windows.
    shot_dist = df["shot_distance"].values
    shot_val  = df["shot_value"].values
    is_3pt    = shot_val == 3
    is_paint  = shot_dist < 5
    sustainable = is_3pt | is_paint  # True = high-quality shot

    sus_series = pd.Series(sustainable.astype(float))

    # Only count scoring possessions in quality windows
    home_scored_mask = ((df["team_scored"] == "home") & (df["points"] > 0)).values
    away_scored_mask = ((df["team_scored"] == "away") & (df["points"] > 0)).values

    home_sus = sus_series.where(home_scored_mask, other=np.nan)
    away_sus = sus_series.where(away_scored_mask, other=np.nan)

    # Rolling mean over last 5 scored possessions per team (shift to exclude current)
    home_sus_roll = home_sus.shift(1).rolling(5, min_periods=1).mean().fillna(0.5)
    away_sus_roll = away_sus.shift(1).rolling(5, min_periods=1).mean().fillna(0.5)

    df["home_scoring_sustainable"] = home_sus_roll.values >= 0.5
    df["away_scoring_sustainable"] = away_sus_roll.values >= 0.5

    # xPPP — expected points per possession based on shot type, vs actual.
    # High actual_vs_expected = team is overperforming their shot quality → expect reversion.
    xppp_vals = np.array([
        _shot_xppp(int(sv), float(sd))
        for sv, sd in zip(df["shot_value"].values, df["shot_distance"].values)
    ])
    xppp_series = pd.Series(xppp_vals)

    # Reuse scored masks (points > 0) so stops/turnovers don't pollute xPPP windows
    home_xppp = xppp_series.where(home_scored_mask, other=np.nan)
    away_xppp = xppp_series.where(away_scored_mask, other=np.nan)

    # Rolling mean over last 5 scoring possessions per team (shift to exclude current)
    home_xppp_last5 = home_xppp.shift(1).rolling(5, min_periods=1).mean().fillna(1.0)
    away_xppp_last5 = away_xppp.shift(1).rolling(5, min_periods=1).mean().fillna(1.0)

    # Actual points per scoring possession for the same window
    home_pts_f = pd.Series(home_pts.astype(float)).where(home_scored_mask, other=np.nan)
    away_pts_f = pd.Series(away_pts.astype(float)).where(away_scored_mask, other=np.nan)

    home_actual_last5 = home_pts_f.shift(1).rolling(5, min_periods=1).mean().fillna(1.0)
    away_actual_last5 = away_pts_f.shift(1).rolling(5, min_periods=1).mean().fillna(1.0)

    # xPPP from the window BEFORE the most-recent 5 (possessions 6–10 ago) for trend
    home_xppp_prev5 = home_xppp.shift(6).rolling(5, min_periods=1).mean().fillna(1.0)
    away_xppp_prev5 = away_xppp.shift(6).rolling(5, min_periods=1).mean().fillna(1.0)

    df["home_xPPP_last_5"]            = home_xppp_last5.values
    df["away_xPPP_last_5"]            = away_xppp_last5.values
    df["home_actual_vs_expected_PPP"]  = (home_actual_last5 - home_xppp_last5).values
    df["away_actual_vs_expected_PPP"]  = (away_actual_last5 - away_xppp_last5).values
    # RAW differences, not np.sign(). The signed form collapsed every magnitude onto
    # {-1, 0, +1}, so a shot-quality collapse and an imperceptible drift produced the
    # same input. The prev-5 windows are also emitted so consumers can rebuild the
    # trend without recomputing the rolling windows.
    df["home_xPPP_prev_5"]        = home_xppp_prev5.values
    df["away_xPPP_prev_5"]        = away_xppp_prev5.values
    df["home_shot_quality_trend"] = home_xppp_last5.values - home_xppp_prev5.values
    df["away_shot_quality_trend"] = away_xppp_last5.values - away_xppp_prev5.values

    return df
