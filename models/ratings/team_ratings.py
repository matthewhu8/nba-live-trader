"""
  Team-level EWMA offensive/defensive ratings.

  Called by Phase 3d of the nightly post-game pipeline to update
  features.team_ratings after each night's games.
  """

  from typing import Any

  import pandas as pd

  # Matches FIXED_ALPHA constant in post_game_pipeline.py
  _FIXED_ALPHA = 0.05  # floor — half-life ~20 games


  def compute_game_team_stats(possessions_df: pd.DataFrame) -> dict[str, dict[str, float]]:
      """
      Compute raw offensive and defensive ratings for both teams from one game's possessions.

      Offensive rating = points scored per 100 own possessions.
      Defensive rating = points conceded per 100 opponent possessions.

      Returns {"home": {"ortg": float, "drtg": float}, "away": {"ortg": float, "drtg": float}},
      or {} if the DataFrame is malformed or too sparse.
      """
      if possessions_df.empty or "possessing_team" not in possessions_df.columns:
          return {}

      home_poss_df = possessions_df[possessions_df["possessing_team"] == "home"]
      away_poss_df = possessions_df[possessions_df["possessing_team"] == "away"]

      home_poss = len(home_poss_df)
      away_poss = len(away_poss_df)
  
      if home_poss < 10 or away_poss < 10:
          return {}

      home_pts = float(home_poss_df["points"].sum())
      away_pts = float(away_poss_df["points"].sum())

      home_ortg = home_pts / home_poss * 100.0
      away_ortg = away_pts / away_poss * 100.0

      return {
          "home": {"ortg": home_ortg, "drtg": away_ortg},
          "away": {"ortg": away_ortg, "drtg": home_ortg},
      }


  def apply_team_ewma(
      state_dict: dict,
      ortg: float,
      drtg: float,
      league_avg: float,
  ) -> dict[str, Any]:
      """
      Apply one EWMA update step to a team's rating state.
  
      Uses adaptive alpha: max(_FIXED_ALPHA, 2/(n+2)).
      Early games trust new data fully; converges to ~0.05 after ~40 games.
      When state_dict is empty (first game for this team), initializes from league_avg.

      Args:
          state_dict: current state pulled from features.team_ratings
                      keys: games_played, ewma_off_rating, ewma_def_rating, last_5_net_ratings
          ortg:       this game's raw offensive rating (pts per 100 possessions)
          drtg:       this game's raw defensive rating (pts conceded per 100 possessions)
          league_avg: current league-wide average ortg (for warm-start on first game)

      Returns dict with all columns needed for the INSERT into features.team_ratings.
      """
      games_played  = int(state_dict.get("games_played", 0) or 0)
      prev_ewma_off = float(state_dict.get("ewma_off_rating") or league_avg)
      prev_ewma_def = float(state_dict.get("ewma_def_rating") or league_avg)
      last_5_str    = str(state_dict.get("last_5_net_ratings") or "")

      alpha    = max(_FIXED_ALPHA, 2.0 / (games_played + 2))
      ewma_off = alpha * ortg + (1.0 - alpha) * prev_ewma_off
      ewma_def = alpha * drtg + (1.0 - alpha) * prev_ewma_def

      raw_net = ortg - drtg
      last_5  = [float(x) for x in last_5_str.split(",") if x.strip()]
      last_5.append(raw_net)
      last_5  = last_5[-5:]

      return {
          "games_played":       games_played + 1,
          "ewma_off_rating":    ewma_off,
          "ewma_def_rating":    ewma_def,
          "ewma_net_rating":    ewma_off - ewma_def,
          "off_rating":         ortg,
          "def_rating":         drtg,
          "net_rating":         raw_net,
          "last_5_net_ratings": ",".join(f"{x:.2f}" for x in last_5),
      }

