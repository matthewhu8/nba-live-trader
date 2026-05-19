import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "live-trader"))

from inference.pregame import load_pregame

async def test():
    logging.basicConfig(level=logging.INFO)
    game_id = "0042500202" # CLE @ DET
    print(f"--- Testing pregame load for game {game_id} ---")
    data = await load_pregame(game_id)
    print(f"Has data: {data.get('has_pregame_data')}")
    print(f"Lineup ratings count: {len(data.get('lineup_ratings', {}))}")
    print(f"Player APM count: {len(data.get('player_apm', {}))}")
    
    game_id_2 = "0042500222" # LAL @ OKC
    print(f"\n--- Testing pregame load for game {game_id_2} ---")
    data2 = await load_pregame(game_id_2)
    print(f"Has data: {data2.get('has_pregame_data')}")

if __name__ == "__main__":
    asyncio.run(test())
