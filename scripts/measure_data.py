"""
Measure the actual size of every dataset the model trains on.

Documentation drifts. This script is the authority: run it and paste the output
into docs/DATA_INVENTORY.md rather than hand-editing counts.

    source venv/bin/activate && set -a && . ./.env && set +a
    python scripts/measure_data.py

Every query is a server-side aggregate - no table is pulled down - because the
MotherDuck plan has a daily compute limit that full scans burn quickly.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must match models/mmoe/dataset.py - imported rather than copied so a split
# change here can never silently disagree with what training actually does.
from models.mmoe.dataset import BBALL_TRAIN_END, BBALL_VAL_END, HEADB_SPLIT_DATE


def connect() -> duckdb.DuckDBPyConnection:
    load_dotenv()
    token = os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError(
            "MOTHERDUCK_TOKEN not set. Run: set -a && . ./.env && set +a"
        )
    return duckdb.connect(f"md:kalshi_trading?motherduck_token={token}", read_only=True)


def show(title: str, df: pd.DataFrame) -> None:
    print(f"\n### {title}")
    print(df.to_string(index=False))


def table_sizes(con: duckdb.DuckDBPyConnection) -> None:
    tables = [
        "features.possession_flat",
        "features.pregame",
        "features.lineup_ratings",
        "features.player_ratings",
        "features.team_ratings",
        "main.kalshi_ticks",
        "main.dim_games",
        "main.kalshi_settled",
        "main.kalshi_market_map",
    ]
    rows = [
        {"table": t, "rows": con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]}
        for t in tables
    ]
    show("Table sizes", pd.DataFrame(rows))


def possession_coverage(con: duckdb.DuckDBPyConnection) -> None:
    show("features.possession_flat", con.execute("""
        SELECT COUNT(*)                     AS n_rows,
               COUNT(DISTINCT game_id)      AS n_games,
               COUNT(wall_clock_ts)         AS n_rows_timestamped,
               COUNT(DISTINCT CASE WHEN wall_clock_ts IS NOT NULL THEN game_id END)
                                            AS n_games_timestamped,
               MIN(wall_clock_ts)           AS first_ts,
               MAX(wall_clock_ts)           AS last_ts
        FROM features.possession_flat
    """).fetchdf())

    show("possession_flat by season", con.execute("""
        SELECT CASE WHEN g.game_date < DATE '2025-08-01' THEN '2024-25' ELSE '2025-26' END AS season,
               COUNT(*)                AS n_rows,
               COUNT(DISTINCT p.game_id) AS n_games,
               MIN(g.game_date)        AS first_game,
               MAX(g.game_date)        AS last_game
        FROM features.possession_flat p
        JOIN main.dim_games g USING (game_id)
        GROUP BY 1 ORDER BY 1
    """).fetchdf())


def tick_coverage(con: duckdb.DuckDBPyConnection) -> None:
    show("main.kalshi_ticks", con.execute("""
        SELECT COUNT(*)                          AS n_ticks,
               COUNT(DISTINCT market_ticker)     AS n_tickers,
               COUNT(DISTINCT game_id)           AS n_games,
               SUM(CASE WHEN game_id IS NULL THEN 1 ELSE 0 END) AS n_ticks_unmapped,
               MIN(ts)                           AS first_ts,
               MAX(ts)                           AS last_ts
        FROM main.kalshi_ticks
    """).fetchdf())

    show("kalshi_ticks by month", con.execute("""
        SELECT date_trunc('month', ts) AS month,
               COUNT(*)                        AS n_ticks,
               COUNT(DISTINCT market_ticker)   AS n_tickers,
               COUNT(DISTINCT game_id)         AS n_games
        FROM main.kalshi_ticks GROUP BY 1 ORDER BY 1
    """).fetchdf())


def joint_coverage(con: duckdb.DuckDBPyConnection) -> None:
    """
    Upper bound on the joint set: games present in BOTH possession_flat (with a
    timestamp) and kalshi_ticks. The trainable count is lower - dataset.py drops
    overtime, blowouts, and possessions with no tick inside the staleness window.
    Run build_dataloaders() for the post-filter number.
    """
    show("Joint upper bound (games in both tables)", con.execute("""
        WITH pf AS (SELECT DISTINCT game_id FROM features.possession_flat
                    WHERE wall_clock_ts IS NOT NULL),
             kt AS (SELECT DISTINCT game_id FROM main.kalshi_ticks
                    WHERE game_id IS NOT NULL)
        SELECT (SELECT COUNT(*) FROM pf)                        AS games_timestamped,
               (SELECT COUNT(*) FROM kt)                        AS games_with_ticks,
               (SELECT COUNT(*) FROM pf JOIN kt USING (game_id)) AS games_both
    """).fetchdf())

    show("Possessions in tick games", con.execute("""
        SELECT COUNT(*) AS n_possessions, COUNT(DISTINCT game_id) AS n_games
        FROM features.possession_flat
        WHERE wall_clock_ts IS NOT NULL
          AND game_id IN (SELECT game_id FROM main.kalshi_ticks WHERE game_id IS NOT NULL)
    """).fetchdf())


def split_balance(con: duckdb.DuckDBPyConnection) -> None:
    """Head B's split is a fixed date; tick coverage keeps growing past it."""
    show(f"Head B split at {HEADB_SPLIT_DATE.date()} (games)", con.execute(f"""
        WITH kt AS (SELECT DISTINCT game_id FROM main.kalshi_ticks WHERE game_id IS NOT NULL)
        SELECT CASE WHEN g.game_date < DATE '{HEADB_SPLIT_DATE.date()}'
                    THEN 'train' ELSE 'val' END AS split,
               COUNT(*)         AS n_games,
               MIN(g.game_date) AS first_game,
               MAX(g.game_date) AS last_game
        FROM kt JOIN main.dim_games g USING (game_id)
        GROUP BY 1 ORDER BY 1
    """).fetchdf())

    show(f"Basketball split (train<={BBALL_TRAIN_END.date()}, val<={BBALL_VAL_END.date()})",
         con.execute(f"""
        SELECT CASE
                 WHEN g.game_date <= DATE '{BBALL_TRAIN_END.date()}' THEN 'train'
                 WHEN g.game_date <= DATE '{BBALL_VAL_END.date()}'   THEN 'val'
                 ELSE 'test (sacred)'
               END AS split,
               COUNT(*)                  AS n_rows,
               COUNT(DISTINCT p.game_id) AS n_games
        FROM features.possession_flat p
        JOIN main.dim_games g USING (game_id)
        GROUP BY 1 ORDER BY 1
    """).fetchdf())


def main() -> None:
    con = connect()
    try:
        print(f"# Measured data inventory - {date.today().isoformat()}")
        print("# Source: MotherDuck md:kalshi_trading")
        table_sizes(con)
        possession_coverage(con)
        tick_coverage(con)
        joint_coverage(con)
        split_balance(con)
        print(
            "\nNOTE: trainable joint rows are lower than the upper bound above. "
            "Run models.mmoe.dataset.build_dataloaders() and read its "
            "'Joint rows' / 'Train:' log lines for post-filter counts."
        )
    finally:
        con.close()


if __name__ == "__main__":
    main()
