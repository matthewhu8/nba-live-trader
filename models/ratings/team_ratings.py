import pandas as pd

def compute_game_team_stats(poss_df: pd.DataFrame) -> dict:
    """
    Computes single-game raw Offensive and Defensive ratings for both teams.
    """
    stats = {}
    for team in ['home', 'away']:
        team_poss = poss_df[poss_df['possessing_team'] == team]
        poss_count = len(team_poss)
        pts = team_poss['points'].sum() if 'points' in team_poss else 0
        ortg = (pts * 100.0 / poss_count) if poss_count > 0 else 0
        stats[team] = {'ortg': ortg}
        
    stats['home']['drtg'] = stats['away']['ortg']
    stats['away']['drtg'] = stats['home']['ortg']
    return stats

def apply_team_ewma(state_dict: dict, ortg: float, drtg: float, league_avg: float) -> dict:
    """
    Updates the Exponential Weighted Moving Average for a team's ratings,
    and applies Bayesian Shrinkage towards the league average for early season stability.
    """
    games_played = int(state_dict.get('games_played', 0) or 0)
    prev_ewma_ortg = state_dict.get('ewma_off_rating')
    prev_ewma_drtg = state_dict.get('ewma_def_rating')
    
    if prev_ewma_ortg is None or prev_ewma_ortg == 0:
        prev_ewma_ortg = league_avg
    if prev_ewma_drtg is None or prev_ewma_drtg == 0:
        prev_ewma_drtg = league_avg

    new_games_played = games_played + 1
    
    # α = max(0.05, (1 / games_played))
    # Ensures tonight's game gets at least 5% weight
    alpha = max(0.05, 1.0 / new_games_played)
    
    # UPDATE EWMA
    new_ewma_ortg = alpha * ortg + (1 - alpha) * float(prev_ewma_ortg)
    new_ewma_drtg = alpha * drtg + (1 - alpha) * float(prev_ewma_drtg)
    
    # GET W (Bayesian Shrinkage Weight)
    # K = 15 (number of games before we trust the team 50%)
    K = 15
    w = new_games_played / (new_games_played + K)
    
    # FINAL SHRINKAGE
    shrunk_ortg = w * new_ewma_ortg + (1 - w) * league_avg
    shrunk_drtg = w * new_ewma_drtg + (1 - w) * league_avg
    
    # Manage the sliding window of last 5 net ratings
    last_5_str = state_dict.get('last_5_net_ratings', "")
    if pd.isna(last_5_str): last_5_str = ""
    try:
        last_5 = [float(x) for x in str(last_5_str).split(',') if x.strip()]
    except Exception:
        last_5 = []
        
    net_rating = ortg - drtg
    last_5.append(net_rating)
    if len(last_5) > 5:
        last_5 = last_5[-5:]
        
    last_5_out = ",".join(str(round(x, 2)) for x in last_5)
    
    return {
        'games_played': new_games_played,
        'ewma_off_rating': new_ewma_ortg, 
        'ewma_def_rating': new_ewma_drtg,
        'ewma_net_rating': new_ewma_ortg - new_ewma_drtg, 
        'off_rating': shrunk_ortg, 
        'def_rating': shrunk_drtg,
        'net_rating': shrunk_ortg - shrunk_drtg,
        'last_5_net_ratings': last_5_out
    }
