import sys
import os
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "live-trader"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import asyncio
import logging
logging.basicConfig(level=logging.DEBUG)
from inference.pregame import load_pregame

async def go():
    for gid in ['0042500234', '0022501150']:
        print('=' * 50)
        print('Testing', gid)
        res = await load_pregame(gid, fallback_home_team_id=1610612750, fallback_away_team_id=1610612759)
        keys = ['team_net_rating_delta','home_off_rating','away_off_rating','home_def_rating','away_def_rating','expected_pace','form_delta','has_pregame_data','pace_baseline','home_b2b','away_b2b']
        for k in keys:
            print(f'  {k:30s} = {res.get(k)}')
        print('  lineup_ratings size:', len(res.get('lineup_ratings', {})))
        print('  player_apm size:', len(res.get('player_apm', {})))

asyncio.run(go())
