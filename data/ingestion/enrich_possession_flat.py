"""
Enrich features.possession_flat with lineup net rating columns.

Adds 7 columns sourced from features.lineup_ratings:
    home_lineup_net_rating      float  COALESCE(match, 0.0)
    away_lineup_net_rating      float  COALESCE(match, 0.0)
    lineup_net_rating_delta     float  home - away
    home_lineup_sample_size     int    possessions_together (0 = pure predicted)
    away_lineup_sample_size     int
    home_lineup_just_changed    bool   LAG-based change detection
    away_lineup_just_changed    bool

Join key: pf.game_id = lr.as_of_game_id (backward-looking, no lookahead)

Expected behaviour by season:
    2024-25: all ratings → 0.0 (lineup_ratings only covers 2025-26)
    2025-26: ~87% non-zero (13% are new/rare lineups not yet seen → 0.0)

Usage:
    python -m data.ingestion.enrich_possession_flat
    python -m data.ingestion.enrich_possession_flat --dry-run   # validate only, no write
"""

import argparse
import logging
import sys
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

DB_PATH = Path("kalshi_trading.duckdb")


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

_ENRICH_SQL = """
CREATE OR REPLACE TABLE features.possession_flat AS
SELECT
    pf.*,
    COALESCE(lr_home.net_rating,          0.0) AS home_lineup_net_rating,
    COALESCE(lr_away.net_rating,          0.0) AS away_lineup_net_rating,
    COALESCE(lr_home.net_rating, 0.0)
      - COALESCE(lr_away.net_rating, 0.0)      AS lineup_net_rating_delta,
    COALESCE(lr_home.possessions_together,  0) AS home_lineup_sample_size,
    COALESCE(lr_away.possessions_together,  0) AS away_lineup_sample_size,
    COALESCE(
        LAG(pf.home_lineup_id) OVER (PARTITION BY pf.game_id ORDER BY pf.event_id)
            != pf.home_lineup_id,
        FALSE
    ) AS home_lineup_just_changed,
    COALESCE(
        LAG(pf.away_lineup_id) OVER (PARTITION BY pf.game_id ORDER BY pf.event_id)
            != pf.away_lineup_id,
        FALSE
    ) AS away_lineup_just_changed
FROM features.possession_flat pf
LEFT JOIN features.lineup_ratings lr_home
    ON pf.home_lineup_id = lr_home.lineup_id
   AND pf.game_id        = lr_home.as_of_game_id
LEFT JOIN features.lineup_ratings lr_away
    ON pf.away_lineup_id = lr_away.lineup_id
   AND pf.game_id        = lr_away.as_of_game_id
"""


def enrich(db_path: Path) -> None:
    """Run the enrichment SQL against local DuckDB. Overwrites possession_flat in place."""
    conn = duckdb.connect(str(db_path))

    before_cols = len(conn.execute("DESCRIBE features.possession_flat").fetchall())
    before_rows = conn.execute("SELECT COUNT(*) FROM features.possession_flat").fetchone()[0]

    logger.info(
        "Starting enrichment — possession_flat before: %d rows, %d cols",
        before_rows, before_cols,
    )

    conn.execute(_ENRICH_SQL)

    after_cols = len(conn.execute("DESCRIBE features.possession_flat").fetchall())
    after_rows = conn.execute("SELECT COUNT(*) FROM features.possession_flat").fetchone()[0]

    if after_rows != before_rows:
        conn.close()
        raise RuntimeError(
            f"Row count changed after enrichment: {before_rows} → {after_rows}. "
            "JOIN may have multiplied rows. Investigate before pushing to cloud."
        )

    expected_new_cols = 7
    if after_cols != before_cols + expected_new_cols:
        conn.close()
        raise RuntimeError(
            f"Column count unexpected: {before_cols} → {after_cols} "
            f"(expected +{expected_new_cols})"
        )

    conn.close()
    logger.info(
        "Enrichment complete — possession_flat after: %d rows, %d cols (+%d)",
        after_rows, after_cols, expected_new_cols,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(db_path: Path) -> bool:
    """
    Run comprehensive validation on the enriched possession_flat.
    Returns True if all checks pass, False otherwise.
    Logs warnings for soft failures and errors for hard failures.
    """
    conn = duckdb.connect(str(db_path), read_only=True)
    passed = True

    print("\n" + "=" * 65)
    print("VALIDATION — features.possession_flat lineup columns")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Column presence
    # ------------------------------------------------------------------
    print("\n[1] Column presence")
    expected_cols = [
        "home_lineup_net_rating",
        "away_lineup_net_rating",
        "lineup_net_rating_delta",
        "home_lineup_sample_size",
        "away_lineup_sample_size",
        "home_lineup_just_changed",
        "away_lineup_just_changed",
    ]
    actual_cols = {row[0] for row in conn.execute("DESCRIBE features.possession_flat").fetchall()}
    for col in expected_cols:
        present = col in actual_cols
        status = "OK" if present else "MISSING"
        print(f"    {status:8s}  {col}")
        if not present:
            passed = False

    # ------------------------------------------------------------------
    # 2. Null checks (must be 0% — COALESCE guarantees this)
    # ------------------------------------------------------------------
    print("\n[2] Null checks (must be 0 nulls)")
    total = conn.execute("SELECT COUNT(*) FROM features.possession_flat").fetchone()[0]
    null_checks = [
        "home_lineup_net_rating",
        "away_lineup_net_rating",
        "lineup_net_rating_delta",
        "home_lineup_sample_size",
        "away_lineup_sample_size",
        "home_lineup_just_changed",
        "away_lineup_just_changed",
    ]
    for col in null_checks:
        nulls = conn.execute(
            f"SELECT COUNT(*) FROM features.possession_flat WHERE {col} IS NULL"
        ).fetchone()[0]
        status = "OK" if nulls == 0 else "FAIL"
        print(f"    {status:8s}  {col}: {nulls:,} nulls / {total:,} rows")
        if nulls > 0:
            passed = False

    # ------------------------------------------------------------------
    # 3. Mathematical integrity: delta = home - away exactly
    # ------------------------------------------------------------------
    print("\n[3] Math integrity: lineup_net_rating_delta = home - away")
    delta_violations = conn.execute("""
        SELECT COUNT(*) FROM features.possession_flat
        WHERE ABS(lineup_net_rating_delta
                  - (home_lineup_net_rating - away_lineup_net_rating)) > 0.001
    """).fetchone()[0]
    status = "OK" if delta_violations == 0 else "FAIL"
    print(f"    {status:8s}  {delta_violations:,} rows violating delta = home - away")
    if delta_violations > 0:
        passed = False

    # ------------------------------------------------------------------
    # 4. Value distributions
    # ------------------------------------------------------------------
    print("\n[4] Value distributions")

    stats = conn.execute("""
        SELECT
            AVG(home_lineup_net_rating)    AS h_mean,
            STDDEV(home_lineup_net_rating) AS h_std,
            MIN(home_lineup_net_rating)    AS h_min,
            MAX(home_lineup_net_rating)    AS h_max,
            AVG(lineup_net_rating_delta)   AS d_mean,
            STDDEV(lineup_net_rating_delta)AS d_std,
            MIN(lineup_net_rating_delta)   AS d_min,
            MAX(lineup_net_rating_delta)   AS d_max
        FROM features.possession_flat
    """).fetchone()

    print(f"    home_lineup_net_rating   mean={stats[0]:.3f}  std={stats[1]:.3f}  "
          f"min={stats[2]:.2f}  max={stats[3]:.2f}")
    print(f"    lineup_net_rating_delta  mean={stats[4]:.3f}  std={stats[5]:.3f}  "
          f"min={stats[6]:.2f}  max={stats[7]:.2f}")

    # Soft warning: delta mean should be near 0.0 (symmetric)
    if abs(stats[4]) > 2.0:
        logger.warning("lineup_net_rating_delta mean=%.3f is far from 0 — investigate", stats[4])

    # Lineup just changed rate
    jc_stats = conn.execute("""
        SELECT
            AVG(home_lineup_just_changed::float) AS home_rate,
            AVG(away_lineup_just_changed::float) AS away_rate
        FROM features.possession_flat
    """).fetchone()
    print(f"    home_lineup_just_changed  rate={jc_stats[0]*100:.1f}%  "
          f"(expected ~8%, warn if <3% or >15%)")
    print(f"    away_lineup_just_changed  rate={jc_stats[1]*100:.1f}%")

    for rate, label in [(jc_stats[0], "home"), (jc_stats[1], "away")]:
        if rate < 0.03 or rate > 0.15:
            logger.warning("%s_lineup_just_changed rate=%.1f%% is outside expected 3-15%%", label, rate * 100)

    # Sample size distribution
    ss_stats = conn.execute("""
        SELECT
            AVG(home_lineup_sample_size)            AS mean,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY home_lineup_sample_size) AS median,
            MAX(home_lineup_sample_size)            AS max_val,
            SUM(CASE WHEN home_lineup_sample_size = 0 THEN 1 ELSE 0 END) * 100.0
              / COUNT(*)                            AS pct_zero
        FROM features.possession_flat
    """).fetchone()
    print(f"    home_lineup_sample_size  mean={ss_stats[0]:.1f}  median={ss_stats[1]:.0f}  "
          f"max={ss_stats[2]}  pct_zero={ss_stats[3]:.1f}%  (expected ~84.7%)")

    # ------------------------------------------------------------------
    # 5. Season breakdown: 2025-26 should have non-zero ratings
    # ------------------------------------------------------------------
    print("\n[5] Season breakdown")

    season_stats = conn.execute("""
        SELECT
            CASE WHEN game_id >= '0022500001' THEN '2025-26' ELSE '2024-25' END AS season,
            COUNT(*)                                                              AS rows,
            SUM(CASE WHEN home_lineup_net_rating != 0.0 THEN 1 ELSE 0 END)      AS nonzero_home,
            ROUND(
                SUM(CASE WHEN home_lineup_net_rating != 0.0 THEN 1 ELSE 0 END) * 100.0
                / COUNT(*), 1
            )                                                                    AS pct_nonzero
        FROM features.possession_flat
        GROUP BY 1
        ORDER BY 1
    """).fetchall()

    for row in season_stats:
        season, rows, nonzero, pct = row
        print(f"    {season}: {rows:>8,} rows  {nonzero:>7,} non-zero  ({pct:.1f}%)")
        if season == "2025-26" and pct < 70:
            logger.warning(
                "2025-26 non-zero rate=%.1f%% is below 70%% — join may be broken", pct
            )
            passed = False
        if season == "2024-25" and pct > 1:
            logger.warning(
                "2024-25 has %.1f%% non-zero ratings — expected 0%% (no 2024-25 lineup data)", pct
            )

    # ------------------------------------------------------------------
    # 6. Spot-check: 3 specific 2025-26 games — starting lineup ratings non-zero
    # ------------------------------------------------------------------
    print("\n[6] Spot-checks — starting lineup ratings (first possession per game)")

    spot_games = conn.execute("""
        SELECT DISTINCT game_id
        FROM features.possession_flat
        WHERE game_id >= '0022500001'
        ORDER BY game_id
        LIMIT 3
    """).fetchall()

    for (game_id,) in spot_games:
        first = conn.execute("""
            SELECT
                game_id, event_id,
                home_lineup_id, away_lineup_id,
                home_lineup_net_rating, away_lineup_net_rating,
                lineup_net_rating_delta,
                home_lineup_sample_size, away_lineup_sample_size
            FROM features.possession_flat
            WHERE game_id = ?
            ORDER BY event_id
            LIMIT 1
        """, [game_id]).fetchone()
        print(
            f"    game {game_id}  "
            f"home_nr={first[4]:.2f}  away_nr={first[5]:.2f}  "
            f"delta={first[6]:.2f}  "
            f"h_ss={first[7]}  a_ss={first[8]}"
        )

    # ------------------------------------------------------------------
    # 7. Verify first row of each game has lineup_just_changed = FALSE
    # ------------------------------------------------------------------
    print("\n[7] First-possession lineup_just_changed must always be FALSE")

    first_row_violations = conn.execute("""
        WITH first_poss AS (
            SELECT game_id, MIN(event_id) AS first_event
            FROM features.possession_flat
            GROUP BY game_id
        )
        SELECT COUNT(*)
        FROM features.possession_flat pf
        JOIN first_poss fp ON pf.game_id = fp.game_id AND pf.event_id = fp.first_event
        WHERE pf.home_lineup_just_changed = TRUE
           OR pf.away_lineup_just_changed = TRUE
    """).fetchone()[0]

    status = "OK" if first_row_violations == 0 else "FAIL"
    print(f"    {status:8s}  {first_row_violations:,} games with TRUE on first possession")
    if first_row_violations > 0:
        passed = False

    # ------------------------------------------------------------------
    # 8. Sample size sanity: rows with possessions_together > 0 should have non-zero net_rating
    # ------------------------------------------------------------------
    print("\n[8] Sample size vs net_rating consistency")

    inconsistent = conn.execute("""
        SELECT COUNT(*)
        FROM features.possession_flat
        WHERE home_lineup_sample_size > 0 AND home_lineup_net_rating = 0.0
    """).fetchone()[0]
    # Note: 107 such anomalies exist in lineup_ratings itself (pre-existing), so warn only
    print(f"    home rows with sample_size > 0 but net_rating = 0.0: {inconsistent:,}  "
          f"(up to ~107 expected from source anomalies)")

    conn.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    result = "ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED — do NOT push to cloud"
    print(f"RESULT: {result}")
    print("=" * 65 + "\n")

    return passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(DB_PATH), help="Path to local DuckDB file")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate only (no write). Assumes enrichment has already been run.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = _parse_args()
    db_path = Path(args.db)

    if not db_path.exists():
        logger.error("DuckDB not found at %s", db_path)
        sys.exit(1)

    if not args.dry_run:
        enrich(db_path)

    passed = validate(db_path)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
