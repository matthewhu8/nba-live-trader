import logging
import duckdb
import pandas as pd
from models.features.pregame_features import compute_pregame_features
from models.ratings.team_ratings import compute_game_team_stats, apply_team_ewma
from data.ingestion.duckdb_loader import init_features_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Connecting to MotherDuck...")
    import os
    from dotenv import load_dotenv
    load_dotenv()
    token = os.environ.get("MOTHERDUCK_TOKEN")
    conn = duckdb.connect(f"md:kalshi_trading?motherduck_token={token}")
    
    logger.info("Initializing tables...")
    init_features_tables(conn)
    
    # Just to be safe, wipe them so we do a clean backfill
    conn.execute("DELETE FROM features.pregame")
    conn.execute("DELETE FROM features.team_ratings")
    
    logger.info("Fetching game dates chronologically...")
    dates_df = conn.execute("""
        SELECT DISTINCT game_date 
        FROM dim_games 
        WHERE season='2025-26' AND home_final_score IS NOT NULL 
        ORDER BY game_date ASC
    """).df()
    
    for date_val in dates_df['game_date']:
        date_str = str(date_val)[:10]
        
        # Get all games for this date
        day_games = conn.execute(f"""
            SELECT game_id, home_team, away_team 
            FROM dim_games 
            WHERE game_date = '{date_str}' AND home_final_score IS NOT NULL
        """).df()
        
        if day_games.empty:
            continue
            
        # ---------------------------------------------------------
        # PHASE 4: Compute pregame features BEFORE updating ratings
        # ---------------------------------------------------------
        for _, game in day_games.iterrows():
            features = compute_pregame_features(
                conn=conn, 
                game_id=game['game_id'], 
                game_date=date_str, 
                home_team=game['home_team'], 
                away_team=game['away_team']
            )
            df = pd.DataFrame([features])
            conn.execute("INSERT INTO features.pregame SELECT * FROM df")
            
        # ---------------------------------------------------------
        # PHASE 3d: Update team EWMA ratings WITH tonight's results
        # ---------------------------------------------------------
        # 1. Point-in-time league average ORtg (using all games up to and including today)
        league_avg_res = conn.execute(f"""
            SELECT AVG(pts * 100.0 / NULLIF(poss, 0)) FROM (
                SELECT SUM(pf.points) as pts, COUNT(*) as poss
                FROM possession_feed pf
                JOIN dim_games g ON pf.game_id = g.game_id
                WHERE pf.possessing_team IN ('home', 'away')
                  AND g.game_date <= '{date_str}'
                GROUP BY pf.game_id, pf.possessing_team
            )
        """).fetchone()
        league_avg_ortg = league_avg_res[0] if (league_avg_res and league_avg_res[0]) else 110.0
        
        # 2. Update ratings for each game played tonight
        for _, game in day_games.iterrows():
            gid = game['game_id']
            
            # Get possessing stats
            poss_df = conn.execute(f"SELECT possessing_team, points FROM possession_feed WHERE game_id='{gid}' AND possessing_team IN ('home', 'away')").df()
            if poss_df.empty: 
                continue
                
            raw_stats = compute_game_team_stats(poss_df)
            if not raw_stats: 
                continue
                
            as_of_game_id = gid
            
            for side, tricode in [('home', game['home_team']), ('away', game['away_team'])]:
                curr_state = conn.execute(f"""
                    SELECT games_played, ewma_off_rating, ewma_def_rating, last_5_net_ratings 
                    FROM features.team_ratings 
                    WHERE team_tricode='{tricode}' 
                    ORDER BY as_of_game_id DESC LIMIT 1
                """).df()
                state_dict = curr_state.iloc[0].to_dict() if not curr_state.empty else {}
                
                raw = raw_stats[side]
                updated = apply_team_ewma(state_dict, raw['ortg'], raw['drtg'], float(league_avg_ortg))
                
                # Pace
                pace_res = conn.execute(f"""
                    SELECT AVG(pf.pace_season_baseline) 
                    FROM possession_feed pf JOIN dim_games g ON pf.game_id = g.game_id 
                    WHERE (g.home_team='{tricode}' OR g.away_team='{tricode}') 
                      AND pf.pace_season_baseline > 0 AND g.game_date <= '{date_str}'
                """).fetchone()
                pace = pace_res[0] if (pace_res and pace_res[0]) else 15.0
                
                conn.execute(f"""
                    INSERT INTO features.team_ratings (
                        team_tricode, as_of_game_id, games_played, ewma_off_rating, ewma_def_rating, ewma_net_rating,
                        off_rating, def_rating, net_rating, avg_secs_per_poss, last_5_net_ratings
                    ) VALUES (
                        '{tricode}', '{as_of_game_id}', {updated['games_played']}, 
                        {updated['ewma_off_rating']}, {updated['ewma_def_rating']}, {updated['ewma_net_rating']},
                        {updated['off_rating']}, {updated['def_rating']}, {updated['net_rating']},
                        {pace}, '{updated['last_5_net_ratings']}'
                    )
                """)
                
    logger.info("Backfill complete!")
    
    # Simple sanity check
    pregame_count = conn.execute("SELECT COUNT(*) FROM features.pregame").fetchone()[0]
    rating_count = conn.execute("SELECT COUNT(*) FROM features.team_ratings").fetchone()[0]
    
    logger.info(f"Populated {pregame_count} games in features.pregame")
    logger.info(f"Populated {rating_count} team updates in features.team_ratings")
    
    conn.close()

if __name__ == "__main__":
    main()
