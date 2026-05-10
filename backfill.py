import sys
import subprocess
from datetime import date, timedelta

start_date = date(2026, 4, 15)
end_date = date(2026, 5, 5)

curr = start_date
while curr <= end_date:
    print(f"\n--- Backfilling {curr.isoformat()} ---")
    subprocess.run(["python", "-m", "data.ingestion.post_game_pipeline", curr.isoformat()])
    curr += timedelta(days=1)
