"""
Forward-looking survival hazard targets for MMoE Head C.

Target variables are TRAINING ONLY — never used as features in live inference.
"""

import pandas as pd


def add_hazard_targets(df: pd.DataFrame, n_horizons: int = 10) -> pd.DataFrame:
    """
    Generate discrete survival hazard targets (Head C) for each possession row.

    For each row, looks forward through the next n_horizons SCORING possessions
    and records whether the current run has broken at each horizon.

    A run "breaks" when current_run_team changes to a different team or neutral.

    Columns added:
      haz_0 .. haz_{n-1}: 1.0 if run broken by scoring possession k, else 0.0
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

    if "team_scored" in out.columns:
        scoring_mask = out["team_scored"].notna() & (out["team_scored"] != "")
    else:
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

        if pd.isna(entry_run_team) or entry_run_team in ("", "none", None):
            hazards.loc[idx, haz_cols] = 1.0
            continue

        future_scoring = [i for i in scoring_indices if i > idx]

        broken = False
        for k in range(n_horizons):
            if broken:
                hazards.at[idx, f"haz_{k}"] = 1.0
                continue

            if k >= len(future_scoring):
                broken = True
                hazards.at[idx, f"haz_{k}"] = 1.0
                continue

            future_idx = future_scoring[k]
            if out.at[future_idx, "current_run_team"] != entry_run_team:
                broken = True
                hazards.at[idx, f"haz_{k}"] = 1.0
            else:
                hazards.at[idx, f"haz_{k}"] = 0.0

    return pd.concat([out, hazards], axis=1)
