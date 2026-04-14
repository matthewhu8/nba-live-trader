import logging
import duckdb
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)

def compute_pregame_features(
    conn: duckdb.DuckDBPyConnection,
    game_id: str,
    game_date: str,
    home_team: str,
    away_team: str
) -> Dict[str, Any]:
    """
    Computes all 10 X_pregame features for a single game.
    This runs at Phase 4 in the nightly pipeline, evaluating the matchups
    based strictly on point-in-time correct data (BEFORE this game).
    
    Returns exactly one dictionary representing the row to be inserted 
    into features.pregame.
    """
    features = {
        'game_id': game_id,
        'team_net_rating_delta': 0.0,
        'home_off_rating': 110.0,
        'away_off_rating': 110.0,
        'home_def_rating': 110.0,
        'away_def_rating': 110.0,
        'roster_rapm_gap': 0.0,
        'missing_rapm_impact': 0.0,
        'rest_advantage': 0.0,
        'expected_pace': 15.0,
        'form_delta': 0.0
    }
    
    # ---------------------------------------------------------
    # Features 1-5: Team Ratings (ASOF before this game)
    # ---------------------------------------------------------
    # In live pipeline, we just fetch the max as_of_game_id for the team.
    # We do a subquery against dim_games to ensure point-in-time correctness
    # for backfilling history.
    for team, side in [(home_team, 'home'), (away_team, 'away')]:
        row = conn.execute(f"""
            SELECT tr.off_rating, tr.def_rating, tr.net_rating
            FROM features.team_ratings tr
            JOIN dim_games g ON tr.as_of_game_id = g.game_id
            WHERE tr.team_tricode = '{team}' 
              AND g.game_date < '{game_date}'
            ORDER BY g.game_date DESC
            LIMIT 1
        """).fetchone()
        
        if row:
            features[f'{side}_off_rating'] = row[0]
            features[f'{side}_def_rating'] = row[1]
            
    # Computes pregame net rating delta (Home Net - Away Net)
    home_net = features['home_off_rating'] - features['home_def_rating']
    away_net = features['away_off_rating'] - features['away_def_rating']
    features['team_net_rating_delta'] = home_net - away_net

    # ---------------------------------------------------------
    # Feature 6: Roster RAPM Gap
    # ---------------------------------------------------------
    # Who is playing tonight? We get this from tonight's player_game_stats.
    # Note: RAPM used is already point-in-time (season_apm is backfilled natively)
    rapm_df = conn.execute(f"""
        SELECT team_tricode, AVG(season_apm) as avg_rapm
        FROM (
            SELECT DISTINCT player_id, team_tricode, season_apm 
            FROM player_game_stats 
            WHERE game_id='{game_id}' AND season_apm IS NOT NULL
        )
        GROUP BY team_tricode
    """).df()
    
    hr = rapm_df[rapm_df.team_tricode == 'home'].avg_rapm.iloc[0] if 'home' in rapm_df.team_tricode.values else 0.0
    ar = rapm_df[rapm_df.team_tricode == 'away'].avg_rapm.iloc[0] if 'away' in rapm_df.team_tricode.values else 0.0
    features['roster_rapm_gap'] = hr - ar

    # ---------------------------------------------------------
    # Feature 7: Missing RAPM Impact
    # ---------------------------------------------------------
    # Full strength roster = anyone who played for team in last 40 days
    full_roster = conn.execute(f"""
        SELECT DISTINCT pgs.player_id, 
               CASE WHEN pgs.team_tricode = 'home' THEN g.home_team ELSE g.away_team END as team
        FROM player_game_stats pgs
        JOIN dim_games g ON pgs.game_id = g.game_id
        WHERE g.game_date >= '{game_date}'::DATE - INTERVAL 40 DAY
          AND g.game_date < '{game_date}'
          AND g.home_final_score IS NOT NULL
          AND (g.home_team IN ('{home_team}', '{away_team}') OR g.away_team IN ('{home_team}', '{away_team}'))
    """).df()
    
    # Active tonight
    active_tonight = conn.execute(f"""
        SELECT DISTINCT player_id, 
               CASE WHEN team_tricode = 'home' THEN '{home_team}' ELSE '{away_team}' END as team
        FROM player_game_stats WHERE game_id='{game_id}'
    """).df()
    
    # Missing = Full - Active
    if not full_roster.empty and not active_tonight.empty:
        active_ids = set(active_tonight.player_id.tolist())
        missing_df = full_roster[~full_roster.player_id.isin(active_ids)]
        
        if not missing_df.empty:
            missing_ids_tuple = tuple(missing_df.player_id.tolist())
            if len(missing_ids_tuple) == 1:
                # Handle tuple formatting for single-element SQL IN clause
                missing_str = f"({missing_ids_tuple[0]})"
            else:
                missing_str = str(missing_ids_tuple)
                
            # Fetch Minutes Share & RAPM for missing players
            min_share = conn.execute(f"""
                WITH player_game_poss AS (
                    SELECT pgs.player_id, pgs.game_id, pgs.team_tricode, COUNT(*) as player_poss
                    FROM player_game_stats pgs
                    JOIN dim_games g ON pgs.game_id = g.game_id
                    WHERE pgs.player_id IN {missing_str} AND g.game_date < '{game_date}'
                    GROUP BY pgs.player_id, pgs.game_id, pgs.team_tricode
                ),
                team_game_poss AS (
                    SELECT pg.game_id, pg.team_tricode, SUM(pg.player_poss) as team_poss
                    FROM player_game_poss pg
                    GROUP BY pg.game_id, pg.team_tricode
                )
                SELECT pgp.player_id, 
                       AVG(pgp.player_poss * 1.0 / NULLIF(tgp.team_poss, 0)) as avg_min_share
                FROM player_game_poss pgp
                JOIN team_game_poss tgp ON pgp.game_id = tgp.game_id AND pgp.team_tricode = tgp.team_tricode
                GROUP BY pgp.player_id
            """).df()
            
            # Fetch most recent RAPM
            missing_rapms = conn.execute(f"""
                SELECT pr.player_id, pr.adjusted_plus_minus
                FROM features.player_ratings pr
                JOIN dim_games g ON pr.as_of_game_id = g.game_id
                WHERE pr.player_id IN {missing_str} AND g.game_date < '{game_date}'
                QUALIFY ROW_NUMBER() OVER(PARTITION BY pr.player_id ORDER BY g.game_date DESC) = 1
            """).df()
            
            missing_merged = min_share.merge(missing_rapms, on='player_id', how='inner')
            if not missing_merged.empty:
                # impact = sum of |RAPM x minutes_share|
                missing_merged['impact'] = (missing_merged['avg_min_share'] * missing_merged['adjusted_plus_minus']).abs()
                features['missing_rapm_impact'] = float(missing_merged['impact'].sum())

    # ---------------------------------------------------------
    # Feature 8: Rest Advantage
    # ---------------------------------------------------------
    rest_dict = {}
    for team in [home_team, away_team]:
        last_date = conn.execute(f"""
            SELECT MAX(game_date) FROM dim_games 
            WHERE (home_team='{team}' OR away_team='{team}') 
            AND game_date < '{game_date}' AND home_final_score IS NOT NULL
        """).fetchone()[0]
        
        if last_date:
            rest = (pd.Timestamp(game_date) - pd.Timestamp(last_date)).days - 1
            # Clip rest at 7 days so crossing season boundaries (216 days) doesn't blow out the neural network scaler
            rest_dict[team] = min(rest, 7)
        else:
            rest_dict[team] = 3 # Default to rested for season openers
            
    features['rest_advantage'] = rest_dict[home_team] - rest_dict[away_team]

    # ---------------------------------------------------------
    # Feature 9: Expected Pace & Feature 10: Form Delta
    # ---------------------------------------------------------
    pace_dict = {}
    form_dict = {}
    
    for team, side in [(home_team, 'home'), (away_team, 'away')]:
        # Pace + Form fetched from our features.team_ratings table
        row = conn.execute(f"""
            SELECT tr.avg_secs_per_poss, tr.last_5_net_ratings, tr.net_rating
            FROM features.team_ratings tr
            JOIN dim_games g ON tr.as_of_game_id = g.game_id
            WHERE tr.team_tricode = '{team}' 
              AND g.game_date < '{game_date}'
            ORDER BY g.game_date DESC
            LIMIT 1
        """).fetchone()
        
        if row:
            pace_dict[team] = row[0] if row[0] else 15.0
            
            last_5_str = row[1]
            season_net = row[2]
            
            if last_5_str and season_net is not None:
                last_5_list = [float(x) for x in last_5_str.split(',')]
                last_5_avg = sum(last_5_list) / len(last_5_list)
                form_dict[team] = last_5_avg - season_net
            else:
                form_dict[team] = 0.0
        else:
            pace_dict[team] = 15.0
            form_dict[team] = 0.0
            
    features['expected_pace'] = (pace_dict[home_team] + pace_dict[away_team]) / 2.0
    features['form_delta'] = form_dict[home_team] - form_dict[away_team]

    return features
