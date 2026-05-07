"""
Spread market arbitrage analysis.

For each game, we have two markets per tick window:
  - TeamA at spread X (e.g., LAL1)
  - TeamB at spread Y (e.g., HOU2)

Key question: are these mutually exclusive? If LAL1 and HOU2 are both YES,
can they both pay out? No — if LAL wins by 1+, HOU cannot win by 2+.
So if YES_bid_LAL1 + YES_bid_HOU2 > 100, selling both YES captures riskless profit.

This script:
1. Maps out exact spread combinations per game
2. Computes the per-tick sum of the two YES bids
3. Measures how often sum > 100, by how much, and for how long
4. Applies fee-adjusted profitability filter
"""

import os
import duckdb
import pandas as pd
import numpy as np

MOTHERDUCK_TOKEN = os.environ.get("MOTHERDUCK_TOKEN", "")

MAKER_FEE_RATE = 0.0175  # 0.0175 × contracts × (price/100)
CONTRACTS = 100  # standard lot size for fee calc


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(f"md:kalshi_trading?motherduck_token={MOTHERDUCK_TOKEN}")
    return con


def explore_schema(con: duckdb.DuckDBPyConnection) -> None:
    print("=== Schema exploration ===")

    result = con.execute("""
        SELECT market_ticker, COUNT(*) as ticks
        FROM main.kalshi_ticks
        WHERE market_ticker LIKE 'KXNBASPREAD%'
        GROUP BY market_ticker
        ORDER BY ticks DESC
        LIMIT 20
    """).fetchdf()
    print("\nTop 20 markets by tick count:")
    print(result.to_string(index=False))

    result2 = con.execute("""
        SELECT
            REGEXP_EXTRACT(market_ticker, 'KXNBASPREAD-[0-9A-Z]+-([A-Z]+)([0-9]+)$', 1) AS team,
            REGEXP_EXTRACT(market_ticker, 'KXNBASPREAD-[0-9A-Z]+-([A-Z]+)([0-9]+)$', 2) AS spread,
            COUNT(DISTINCT market_ticker) AS market_count,
            COUNT(*) AS total_ticks
        FROM main.kalshi_ticks
        WHERE market_ticker LIKE 'KXNBASPREAD%'
        GROUP BY team, spread
        ORDER BY market_count DESC
    """).fetchdf()
    print("\nTeam + spread distribution:")
    print(result2.to_string(index=False))


def build_game_market_pairs(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    For each game_id, find the two markets and their spread values.
    Verify they are mutually exclusive.
    """
    print("\n=== Building game-market pairs ===")

    pairs = con.execute("""
        WITH market_meta AS (
            SELECT DISTINCT
                game_id,
                market_ticker,
                REGEXP_EXTRACT(market_ticker, '-([A-Z]+)([0-9]+)$', 1) AS team,
                CAST(REGEXP_EXTRACT(market_ticker, '-([A-Z]+)([0-9]+)$', 2) AS INTEGER) AS spread
            FROM main.kalshi_ticks
            WHERE market_ticker LIKE 'KXNBASPREAD%'
              AND game_id IS NOT NULL
        ),
        game_markets AS (
            SELECT
                game_id,
                COUNT(DISTINCT market_ticker) AS market_count,
                LIST(market_ticker ORDER BY market_ticker) AS tickers,
                LIST(team ORDER BY market_ticker)           AS teams,
                LIST(spread ORDER BY market_ticker)         AS spreads
            FROM market_meta
            GROUP BY game_id
        )
        SELECT *
        FROM game_markets
        WHERE market_count = 2
        ORDER BY game_id
    """).fetchdf()

    print(f"Games with exactly 2 markets: {len(pairs)}")

    # Inspect the spread combinations
    spread_combos = {}
    for _, row in pairs.iterrows():
        spreads = tuple(sorted(row["spreads"]))
        spread_combos[spreads] = spread_combos.get(spreads, 0) + 1

    print("\nSpread combination frequency (sorted spread pair → game count):")
    for combo, count in sorted(spread_combos.items()):
        print(f"  {combo}: {count} games")

    return pairs


def compute_tick_sums(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    For each game and each aligned timestamp window, compute sum of YES bids.
    We align on 1-second buckets.
    """
    print("\n=== Computing per-tick bid sums ===")

    df = con.execute("""
        WITH market_meta AS (
            SELECT
                game_id,
                market_ticker,
                ts,
                yes_bid,
                yes_ask,
                REGEXP_EXTRACT(market_ticker, '-([A-Z]+)([0-9]+)$', 2) AS spread_str
            FROM main.kalshi_ticks
            WHERE market_ticker LIKE 'KXNBASPREAD%'
              AND game_id IS NOT NULL
              AND yes_bid IS NOT NULL
              AND yes_bid > 0
        ),
        -- Rank markets within each game so we get market_1 and market_2
        market_ranked AS (
            SELECT
                game_id,
                market_ticker,
                ts,
                yes_bid,
                yes_ask,
                spread_str,
                ROW_NUMBER() OVER (
                    PARTITION BY game_id
                    ORDER BY market_ticker
                ) AS mkt_rank_static,
                -- per tick, rank within game+second bucket
                DATE_TRUNC('second', ts) AS ts_bucket
            FROM market_meta
        ),
        -- Pivot: for each game × second, get both markets' yes_bid
        pivoted AS (
            SELECT
                a.game_id,
                a.ts_bucket,
                a.market_ticker AS mkt1,
                b.market_ticker AS mkt2,
                a.yes_bid       AS bid1,
                b.yes_bid       AS bid2,
                a.yes_ask       AS ask1,
                b.yes_ask       AS ask2,
                a.bid1 + b.bid2 AS sum_yes_bids
            FROM market_ranked a
            JOIN market_ranked b
              ON  a.game_id    = b.game_id
              AND a.ts_bucket  = b.ts_bucket
              AND a.market_ticker < b.market_ticker  -- ensure a < b to avoid duplication
            -- Take latest tick per market per second bucket
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY a.game_id, a.ts_bucket, a.market_ticker, b.market_ticker
                ORDER BY a.ts DESC, b.ts DESC
            ) = 1
        )
        SELECT *
        FROM pivoted
        ORDER BY game_id, ts_bucket
    """).fetchdf()

    print(f"Total aligned tick-pairs: {len(df):,}")
    print(f"Games covered: {df['game_id'].nunique()}")
    return df


def compute_tick_sums_v2(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Alternative approach: take the last tick per market per second, then join.
    More explicit and avoids QUALIFY issues.
    """
    print("\n=== Computing per-tick bid sums (v2) ===")

    df = con.execute("""
        WITH base AS (
            SELECT
                game_id,
                market_ticker,
                DATE_TRUNC('second', ts) AS ts_sec,
                yes_bid,
                yes_ask
            FROM main.kalshi_ticks
            WHERE market_ticker LIKE 'KXNBASPREAD%'
              AND game_id IS NOT NULL
              AND yes_bid IS NOT NULL
              AND yes_bid > 0
        ),
        -- Keep last observation per market per second
        last_per_sec AS (
            SELECT
                game_id,
                market_ticker,
                ts_sec,
                yes_bid,
                yes_ask
            FROM base
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY game_id, market_ticker, ts_sec
                ORDER BY ts_sec DESC
            ) = 1
        ),
        -- Identify the two markets per game alphabetically
        mkt_ranked AS (
            SELECT
                game_id,
                market_ticker,
                ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY market_ticker) AS mkt_rank
            FROM (SELECT DISTINCT game_id, market_ticker FROM last_per_sec)
        ),
        mkt1 AS (SELECT game_id, market_ticker AS mkt1_ticker FROM mkt_ranked WHERE mkt_rank = 1),
        mkt2 AS (SELECT game_id, market_ticker AS mkt2_ticker FROM mkt_ranked WHERE mkt_rank = 2)
        SELECT
            a.game_id,
            a.ts_sec,
            m1.mkt1_ticker,
            m2.mkt2_ticker,
            a.yes_bid  AS bid1,
            a.yes_ask  AS ask1,
            b.yes_bid  AS bid2,
            b.yes_ask  AS ask2,
            a.yes_bid + b.yes_bid AS sum_yes_bids
        FROM last_per_sec a
        JOIN last_per_sec b ON a.game_id = b.game_id AND a.ts_sec = b.ts_sec
        JOIN mkt1 m1 ON a.game_id = m1.game_id AND a.market_ticker = m1.mkt1_ticker
        JOIN mkt2 m2 ON b.game_id = m2.game_id AND b.market_ticker = m2.mkt2_ticker
        ORDER BY a.game_id, a.ts_sec
    """).fetchdf()

    print(f"Total aligned second-pairs: {len(df):,}")
    print(f"Games covered: {df['game_id'].nunique()}")
    return df


def distribution_analysis(df: pd.DataFrame) -> None:
    print("\n=== Distribution of sum_yes_bids ===")

    s = df["sum_yes_bids"]
    print(f"Min:    {s.min()}")
    print(f"P1:     {s.quantile(0.01):.1f}")
    print(f"P5:     {s.quantile(0.05):.1f}")
    print(f"P25:    {s.quantile(0.25):.1f}")
    print(f"Median: {s.median():.1f}")
    print(f"Mean:   {s.mean():.2f}")
    print(f"P75:    {s.quantile(0.75):.1f}")
    print(f"P90:    {s.quantile(0.90):.1f}")
    print(f"P95:    {s.quantile(0.95):.1f}")
    print(f"P99:    {s.quantile(0.99):.1f}")
    print(f"Max:    {s.max()}")

    # Histogram buckets
    bins = [0, 80, 90, 95, 98, 100, 102, 105, 110, 120, 200]
    counts, edges = np.histogram(s, bins=bins)
    print("\nHistogram:")
    for i, cnt in enumerate(counts):
        pct = cnt / len(s) * 100
        print(f"  [{edges[i]:>3}–{edges[i+1]:>3}): {cnt:>8,}  ({pct:5.2f}%)")

    above_100 = (s > 100).sum()
    pct_above = above_100 / len(s) * 100
    print(f"\nRows with sum > 100: {above_100:,}  ({pct_above:.3f}%)")

    if above_100 > 0:
        excess = s[s > 100] - 100
        print(f"Excess above 100 (cents):")
        print(f"  Mean:   {excess.mean():.2f}¢")
        print(f"  Median: {excess.median():.2f}¢")
        print(f"  P90:    {excess.quantile(0.90):.2f}¢")
        print(f"  Max:    {excess.max():.2f}¢")


def fee_adjusted_analysis(df: pd.DataFrame) -> None:
    """
    Strategy: sell both YES at bid prices (as maker).
    Revenue = (bid1 + bid2) / 100 per contract pair.
    Fee on each leg = MAKER_FEE_RATE × contracts × (price/100).
    Net profit = revenue - fees - $1 (maximum possible payout, since mutually exclusive).

    Wait — need to think carefully:
    Selling YES at bid = you receive bid cents per contract.
    If neither pays out: you keep both premiums.
    If one pays out: you pay $1 on that leg, keep premium on other.
    Max payout obligation: $1 (only one can pay out).

    Net = (bid1 + bid2)/100 - 1.00 - fees_both_legs
    Fees = MAKER_FEE_RATE × contracts × (bid1/100 + bid2/100)
    """
    print("\n=== Fee-adjusted profitability analysis ===")
    print(f"Using: {CONTRACTS} contracts, maker fee rate {MAKER_FEE_RATE}")

    df = df.copy()

    # Revenue from selling both YES (in dollars, per CONTRACTS contracts)
    df["revenue"] = (df["bid1"] + df["bid2"]) / 100 * CONTRACTS

    # Fees on both sell legs
    df["fee_leg1"] = MAKER_FEE_RATE * CONTRACTS * (df["bid1"] / 100)
    df["fee_leg2"] = MAKER_FEE_RATE * CONTRACTS * (df["bid2"] / 100)
    df["total_fees"] = df["fee_leg1"] + df["fee_leg2"]

    # Max payout obligation = $1 per contract (one leg wins at most)
    df["max_payout"] = 1.0 * CONTRACTS

    # Net P&L
    df["net_pnl"] = df["revenue"] - df["max_payout"] - df["total_fees"]

    # Profit threshold in cents for sum_yes_bids
    # net_pnl > 0 iff (bid1+bid2)/100*C - C - fee_rate*C*(bid1+bid2)/100 > 0
    # iff (bid1+bid2)/100 * (1 - fee_rate) > 1
    # iff (bid1+bid2) > 100 / (1 - fee_rate) = 100/0.9825 ≈ 101.78
    breakeven = 100 / (1 - MAKER_FEE_RATE)
    print(f"\nBreak-even sum_yes_bids (fee-adjusted): {breakeven:.2f}¢")

    profitable = df[df["net_pnl"] > 0]
    print(f"Rows with net_pnl > 0: {len(profitable):,}  ({len(profitable)/len(df)*100:.3f}%)")

    if len(profitable) > 0:
        print(f"\nNet P&L stats (profitable rows, {CONTRACTS} contracts):")
        print(f"  Mean:   ${profitable['net_pnl'].mean():.2f}")
        print(f"  Median: ${profitable['net_pnl'].median():.2f}")
        print(f"  P90:    ${profitable['net_pnl'].quantile(0.90):.2f}")
        print(f"  Max:    ${profitable['net_pnl'].max():.2f}")

        # How many distinct games have any profitable window?
        print(f"\nGames with at least 1 profitable second: {profitable['game_id'].nunique()}")

        # Per-game profitable seconds
        game_stats = profitable.groupby("game_id").agg(
            profitable_seconds=("net_pnl", "count"),
            max_pnl=("net_pnl", "max"),
            total_pnl_if_all_entered=("net_pnl", "sum"),
        ).sort_values("profitable_seconds", ascending=False)
        print("\nTop games by profitable seconds:")
        print(game_stats.head(15).to_string())

    return df


def duration_analysis(df: pd.DataFrame) -> None:
    """
    How long do excess windows last? Are they fleeting (1-2s) or sustained?
    """
    print("\n=== Duration of sum > 100 windows ===")

    over = df[df["sum_yes_bids"] > 100].copy()
    if len(over) == 0:
        print("No windows with sum > 100.")
        return

    over = over.sort_values(["game_id", "ts_sec"])

    # Detect contiguous runs per game
    over["prev_ts"] = over.groupby("game_id")["ts_sec"].shift(1)
    over["gap_sec"] = (over["ts_sec"] - over["prev_ts"]).dt.total_seconds()
    over["new_run"] = (over["gap_sec"] > 2) | over["gap_sec"].isna()
    over["run_id"] = over.groupby("game_id")["new_run"].cumsum()

    run_lengths = over.groupby(["game_id", "run_id"]).agg(
        duration_sec=("ts_sec", lambda x: (x.max() - x.min()).total_seconds() + 1),
        max_sum=("sum_yes_bids", "max"),
        mean_sum=("sum_yes_bids", "mean"),
    )

    print(f"Total distinct excess runs: {len(run_lengths):,}")
    dl = run_lengths["duration_sec"]
    print(f"Duration stats (seconds):")
    print(f"  Min:    {dl.min():.0f}s")
    print(f"  Median: {dl.median():.0f}s")
    print(f"  Mean:   {dl.mean():.1f}s")
    print(f"  P75:    {dl.quantile(0.75):.0f}s")
    print(f"  P90:    {dl.quantile(0.90):.0f}s")
    print(f"  P99:    {dl.quantile(0.99):.0f}s")
    print(f"  Max:    {dl.max():.0f}s")

    # Distribution of run lengths
    bins = [0, 1, 2, 5, 10, 30, 60, 120, 300, 10000]
    labels = ["1s", "2s", "3-5s", "6-10s", "11-30s", "31-60s", "61-120s", "121-300s", ">300s"]
    dl_cut = pd.cut(dl, bins=bins, labels=labels)
    print("\nRun length histogram:")
    for lbl, cnt in dl_cut.value_counts().sort_index().items():
        pct = cnt / len(dl) * 100
        print(f"  {lbl:>10}: {cnt:>6,}  ({pct:5.1f}%)")


def ask_side_analysis(df: pd.DataFrame) -> None:
    """
    Selling YES at the bid means we hit the bid (maker order).
    But if we're posting as maker we need someone to hit our offer.
    In practice, to sell YES at market we're a taker on the YES ask side.

    Clarify the correct arbitrage:
    - We want to SELL YES on both markets
    - As maker: we post YES sell orders at ask prices, wait for fill
    - Revenue = ask1 + ask2 (if filled)
    - Breakeven still at ~101.78¢ with maker fee

    But there's also the question of whether we get filled at all.
    Let's check ask-side sums too.
    """
    print("\n=== Ask-side analysis (posting YES sell at ask) ===")

    df2 = df.copy()
    df2["sum_yes_asks"] = df2["ask1"] + df2["ask2"]

    s = df2["sum_yes_asks"].dropna()
    print(f"Sum of YES asks — Mean: {s.mean():.1f}¢, Median: {s.median():.1f}¢")
    above_breakeven = (s > 101.78).sum()
    print(f"Rows where ask-sum > 101.78¢ (fee breakeven): {above_breakeven:,}  ({above_breakeven/len(s)*100:.3f}%)")

    # Mid-price sum for cleaner signal
    df2["mid1"] = (df2["bid1"] + df2["ask1"]) / 2
    df2["mid2"] = (df2["bid2"] + df2["ask2"]) / 2
    df2["sum_mid"] = df2["mid1"] + df2["mid2"]
    sm = df2["sum_mid"].dropna()
    print(f"\nSum of mid prices — Mean: {sm.mean():.1f}¢, Median: {sm.median():.1f}¢")
    print(f"Max mid sum: {sm.max():.1f}¢")
    above_100_mid = (sm > 100).sum()
    print(f"Rows where mid-sum > 100¢: {above_100_mid:,}  ({above_100_mid/len(sm)*100:.3f}%)")


def spread_relationship_analysis(con: duckdb.DuckDBPyConnection) -> None:
    """
    When teams have DIFFERENT spreads (e.g., LAL1 vs HOU2),
    the markets are truly mutually exclusive.

    When teams have the SAME spread (e.g., LAL1 vs HOU1),
    we need to verify mutual exclusivity:
    P(LAL wins by 1+) + P(HOU wins by 1+) + P(tie) = 1
    So they ARE mutually exclusive.

    In all cases, sum of YES bids > 100 is an arbitrage.
    """
    print("\n=== Spread relationship: same vs different spreads ===")

    result = con.execute("""
        WITH mkt AS (
            SELECT DISTINCT
                game_id,
                market_ticker,
                CAST(REGEXP_EXTRACT(market_ticker, '-([A-Z]+)([0-9]+)$', 2) AS INTEGER) AS spread
            FROM main.kalshi_ticks
            WHERE market_ticker LIKE 'KXNBASPREAD%'
              AND game_id IS NOT NULL
        ),
        game_spreads AS (
            SELECT
                game_id,
                MIN(spread) AS spread_min,
                MAX(spread) AS spread_max,
                COUNT(DISTINCT spread) AS n_distinct_spreads
            FROM mkt
            GROUP BY game_id
            HAVING COUNT(DISTINCT market_ticker) = 2
        )
        SELECT
            CASE WHEN n_distinct_spreads = 1 THEN 'Same spread'
                 ELSE 'Different spreads (' || spread_min || ' vs ' || spread_max || ')'
            END AS spread_type,
            COUNT(*) AS game_count
        FROM game_spreads
        GROUP BY spread_type
        ORDER BY game_count DESC
    """).fetchdf()

    print(result.to_string(index=False))

    # For different spread games, what are the combos?
    result2 = con.execute("""
        WITH mkt AS (
            SELECT DISTINCT
                game_id,
                market_ticker,
                REGEXP_EXTRACT(market_ticker, '-([A-Z]+)([0-9]+)$', 1) AS team,
                CAST(REGEXP_EXTRACT(market_ticker, '-([A-Z]+)([0-9]+)$', 2) AS INTEGER) AS spread
            FROM main.kalshi_ticks
            WHERE market_ticker LIKE 'KXNBASPREAD%'
              AND game_id IS NOT NULL
        ),
        ranked AS (
            SELECT game_id, team, spread,
                   ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY market_ticker) AS rn
            FROM mkt
        ),
        pivoted AS (
            SELECT
                a.game_id,
                a.spread AS spread1,
                b.spread AS spread2
            FROM ranked a JOIN ranked b ON a.game_id = b.game_id
            WHERE a.rn = 1 AND b.rn = 2
        )
        SELECT
            LEAST(spread1, spread2) AS s1,
            GREATEST(spread1, spread2) AS s2,
            COUNT(*) AS games
        FROM pivoted
        GROUP BY s1, s2
        ORDER BY games DESC
    """).fetchdf()

    print("\nSpread pair combinations across all games:")
    print(result2.to_string(index=False))


def main() -> None:
    token = os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("MOTHERDUCK_TOKEN not set in environment")

    con = connect()

    # 1. Schema exploration
    explore_schema(con)

    # 2. Spread relationships
    spread_relationship_analysis(con)

    # 3. Game-market pairs
    pairs = build_game_market_pairs(con)

    # 4. Build tick-level aligned pairs
    df = compute_tick_sums_v2(con)

    if len(df) == 0:
        print("No aligned pairs found. Check data.")
        return

    # 5. Distribution
    distribution_analysis(df)

    # 6. Duration of excess windows
    duration_analysis(df)

    # 7. Fee-adjusted P&L
    df_pnl = fee_adjusted_analysis(df)

    # 8. Ask-side and mid analysis
    ask_side_analysis(df)

    con.close()
    print("\n=== Analysis complete ===")


if __name__ == "__main__":
    import sys
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1hdHRoZXdodThAZ21haWwuY29tIiwibWRSZWdpb24iOiJhd3MtdXMtZWFzdC0xIiwic2Vzc2lvbiI6Im1hdHRoZXdodTguZ21haWwuY29tIiwicGF0IjoiaTZTSC1SWkRIdURoZjZ3WGFpNnJlY0xjTlg2QVhqOGlGMndHZEFVbml1WSIsInVzZXJJZCI6IjkwMjM5MDkxLTBmZWItNDIxMy1hOTU0LTJjZTIyMmQ3NjMyYyIsImlzcyI6Im1kX3BhdCIsInJlYWRPbmx5IjpmYWxzZSwidG9rZW5UeXBlIjoicmVhZF93cml0ZSIsImlhdCI6MTc3NTQyNTQzMX0.A0JTHqJ0-JLn-IOLam23zP1LbiGUFKAvwRLLsn11af4"
    os.environ["MOTHERDUCK_TOKEN"] = token
    main()
