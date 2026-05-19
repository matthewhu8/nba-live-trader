import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "live-trader"))

from inference.pregame import load_pregame

# Set up logging to see the [PREGAME] output
logging.basicConfig(level=logging.INFO)

async def test_pregame():
    game_id = "0042500213"  # Knicks @ Sixers
    print(f"Testing pregame load for {game_id}...")
    
    data = await load_pregame(game_id)
    
    if data.get("has_pregame_data") == 1.0:
        print("SUCCESS: MotherDuck connected and returned pregame data!")
        print(f"Team Net Rating Delta: {data.get('team_net_rating_delta')}")
        print(f"Lineups loaded: {len(data.get('lineup_ratings', {}))}")
    else:
        print("WARNING: Connected to MotherDuck, but no data found for this specific game.")
        print("This is expected if the game hasn't been backfilled into dim_games yet.")

if __name__ == "__main__":
    # Ensure we are in the right directory to find .env
    asyncio.run(test_pregame())
