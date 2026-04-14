import math
import numpy as np
import pandas as pd
from typing import Dict, Any

# Constants
TEAM_SHRINKAGE_K = 15   # Games before we trust the team's observed rating 50%
TEAM_EWMA_FLOOR  = 0.05 # Minimum alpha, handles infinite decay

def compute_game_team_stats(poss_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Given an individual game's possession records from possession_feed,
    computes the raw Offensive and Defensive Rating for both the home
    and away teams.
    
    Returns standard 100-possession ratings:
        {
            'home': {'ortg': ..., 'drtg': ...},
            'away': {'ortg': ..., 'drtg': ...}
        }
    """
    # Count possessions and sum points for home team
    home_poss = len(poss_df[poss_df['possessing_team'] == 'home'])
    home_pts = poss_df[poss_df['possessing_team'] == 'home']['points'].sum()
    
    # Count possessions and sum points for away team
    away_poss = len(poss_df[poss_df['possessing_team'] == 'away'])
    away_pts = poss_df[poss_df['possessing_team'] == 'away']['points'].sum()
    
    if home_poss == 0 or away_poss == 0:
        return {} # Edge case handling for malformed data
    
    home_ortg = (home_pts * 100.0) / home_poss
    home_drtg = (away_pts * 100.0) / away_poss # DRtg is what the opponent scored
    
    away_ortg = (away_pts * 100.0) / away_poss
    away_drtg = (home_pts * 100.0) / home_poss 
    
    return {
        'home': {'ortg': home_ortg, 'drtg': home_drtg},
        'away': {'ortg': away_ortg, 'drtg': away_drtg}
    }

def apply_team_ewma(
    prior_state: Dict[str, Any], 
    raw_ortg: float, 
    raw_drtg: float, 
    running_league_avg_ortg: float
) -> Dict[str, Any]:
    """
    Takes a team's prior EWMA rating state and tonight's raw observed ratings, 
    and applies the EWMA and Bayesian Shrinkage calculations.
    
    Args:
        prior_state: Dictionary containing ewma_off_rating, ewma_def_rating, games_played.
                     Pass an empty dict if the team has no prior games.
        raw_ortg: Tonight's raw observed offensive rating
        raw_drtg: Tonight's raw observed defensive rating
        running_league_avg_ortg: The dynamic league average ORtg to use for shrinkage
        
    Returns:
        Updated state dictionary ready to be saved to features.team_ratings
    """
    # Parse prior history
    games_played = prior_state.get('games_played', 0) + 1
    
    prev_ewma_ortg = prior_state.get('ewma_off_rating')
    prev_ewma_drtg = prior_state.get('ewma_def_rating')
    
    # Calculate adaptive Alpha
    alpha = max(TEAM_EWMA_FLOOR, 1.0 / games_played)
    
    # 1. Update EWMA
    if prev_ewma_ortg is None or prev_ewma_drtg is None:
        new_ewma_ortg = raw_ortg
        new_ewma_drtg = raw_drtg
    else:
        new_ewma_ortg = alpha * raw_ortg + (1 - alpha) * prev_ewma_ortg
        new_ewma_drtg = alpha * raw_drtg + (1 - alpha) * prev_ewma_drtg
        
    # 2. Apply Bayesian Shrinkage (W -> 1 as games increase)
    w = games_played / (float(games_played) + TEAM_SHRINKAGE_K)
    
    # Note: DRtg is shrunk to the exact same League Average ORtg.
    # An average team will score AND allow the league average.
    final_ortg = w * new_ewma_ortg + (1.0 - w) * running_league_avg_ortg
    final_drtg = w * new_ewma_drtg + (1.0 - w) * running_league_avg_ortg
    final_net  = final_ortg - final_drtg
    
    # Form Tracking (last 5 raw net ratings)
    raw_net = raw_ortg - raw_drtg
    prev_last_5_str = prior_state.get('last_5_net_ratings', "")
    if prev_last_5_str:
        last_5_list = [float(x) for x in prev_last_5_str.split(",")]
    else:
        last_5_list = []
    
    last_5_list.insert(0, raw_net)
    last_5_list = last_5_list[:5] # Keep only last 5
    new_last_5_str = ",".join([str(round(x, 2)) for x in last_5_list])
    
    return {
        'games_played': games_played,
        'ewma_off_rating': new_ewma_ortg,
        'ewma_def_rating': new_ewma_drtg,
        'ewma_net_rating': new_ewma_ortg - new_ewma_drtg,
        'off_rating': final_ortg,
        'def_rating': final_drtg,
        'net_rating': final_net,
        'last_5_net_ratings': new_last_5_str
    }
