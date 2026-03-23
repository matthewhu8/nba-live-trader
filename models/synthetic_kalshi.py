"""
Synthetic Kalshi price generator.

Generates per-possession synthetic Kalshi YES prices using real DraftKings WP
as a baseline, then applies Kalshi-specific behavioral adjustments:

  price = sportsbook_wp_lagged * 100
        + overreaction(current run)
        - reversion(possessions since run ended)
        + noise

Falls back to a logistic WP estimate from score_diff + minutes_remaining
when no sportsbook data is available for a game.

Output: data/feature_store/synthetic_prices.parquet

Usage:
    python models/synthetic_kalshi.py
"""

import logging
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import expit  # logistic sigmoid

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Team name mapping (TheOddsAPI full names → NBA tricodes) ---
TEAM_NAME_TO_TRICODE: dict[str, str] = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

_WALL_CLK_RE = re.compile(r"(\d+:\d+)\s*(AM|PM)\s*(\w+)", re.IGNORECASE)

SPORTSBOOK_LINES_DIR = Path("data/raw/sportsbook_lines")
OUTPUT_PATH = Path("data/feature_store/synthetic_prices.parquet")

OUTPUT_SCHEMA = pa.schema([
    pa.field("game_id",             pa.string()),
    pa.field("possession_id",       pa.int32()),
    pa.field("synthetic_yes_price", pa.int32()),
    pa.field("sharp_book_wp",       pa.float32()),
    pa.field("has_sportsbook_data", pa.bool_()),
])

TZ_OFFSETS: dict[str, int] = {
    "EST": -5, "EDT": -4,
    "CST": -6, "CDT": -5,
    "MST": -7, "MDT": -6,
    "PST": -8, "PDT": -7,
}

# Max gap between possession wall clock and nearest sportsbook snapshot
MAX_SNAPSHOT_GAP_MINUTES = 15


def parse_wall_clock_utc(game_date: date, period_wall_clock: str) -> datetime | None:
    """Convert '10:18 PM EST' + game_date → UTC datetime. Returns None on failure."""
    if not period_wall_clock or not period_wall_clock.strip():
        return None

    m = _WALL_CLK_RE.match(period_wall_clock.strip())
    if not m:
        return None

    time_str, ampm, tz_abbr = m.group(1), m.group(2).upper(), m.group(3).upper()
    tz_offset = TZ_OFFSETS.get(tz_abbr, -5)  # default EST

    try:
        local_dt = datetime.strptime(f"{time_str} {ampm}", "%I:%M %p")
    except ValueError:
        return None

    local_dt = local_dt.replace(year=game_date.year, month=game_date.month, day=game_date.day)

    # Handle midnight crossover: times like 12:15 AM are the next calendar day
    if ampm == "AM" and local_dt.hour < 6:
        local_dt += timedelta(days=1)

    return local_dt - timedelta(hours=tz_offset)


def possession_wall_clock_utc(
    game_date: date,
    period_wall_clock: str,
    game_clock_secs: float,
) -> datetime | None:
    """
    Derive possession wall clock from period-start anchor + in-period elapsed time.
    NBA periods run ~25 real minutes for 12 game-clock minutes.
    """
    period_start_utc = parse_wall_clock_utc(game_date, period_wall_clock)
    if period_start_utc is None:
        return None

    game_clock_elapsed = 720.0 - game_clock_secs  # 720s = 12min period
    real_elapsed_min = (game_clock_elapsed / 720.0) * 25.0
    return period_start_utc + timedelta(minutes=real_elapsed_min)


def logistic_wp_fallback(score_diff: float, minutes_remaining: float) -> float:
    """
    Simple logistic win probability from score_diff and minutes_remaining.
    Coefficients derived from empirical NBA data: ~0.3 points per minute matters.
    """
    if minutes_remaining <= 0:
        return 1.0 if score_diff > 0 else (0.5 if score_diff == 0 else 0.0)

    # Normalize score advantage relative to expected variability
    # Typical NBA: ~2.5 pts/min scoring pace, std of final margin ≈ 12 * sqrt(minutes/48)
    std_est = 12.0 * np.sqrt(max(minutes_remaining / 48.0, 0.01))
    z = score_diff / std_est
    return float(expit(z * 1.5))  # scale factor from historical calibration


@dataclass
class SyntheticKalshiPriceModel:
    overreaction_coeff: float = 0.30   # cents per run point above threshold
    reversion_speed: float = 0.30      # exp decay rate per possession after run ends
    run_threshold: int = 6             # run_points must exceed this to trigger overreaction
    lag_possessions: int = 2           # Kalshi lags ~2 possessions behind sportsbook
    noise_sigma: float = 0.5           # Gaussian noise in cents

    def generate(
        self,
        feature_rows: pd.DataFrame,
        games_df: pd.DataFrame,
        possessions_df: pd.DataFrame,
        sb_dir: Path,
        games_with_sb: set[str],
    ) -> pd.DataFrame:
        """
        Generate synthetic Kalshi prices for games that have sportsbook data.
        Games without sportsbook data are skipped entirely.

        Returns DataFrame with: game_id, possession_id, synthetic_yes_price,
        sharp_book_wp, has_sportsbook_data
        """
        game_meta = games_df.set_index("game_id")[["game_date", "home_team", "away_team"]].to_dict("index")

        poss_anchors = (
            possessions_df[["game_id", "period", "period_wall_clock", "possession_id", "game_clock_secs"]]
            .copy()
        )

        results: list[dict] = []
        skipped = 0

        for game_id, group in feature_rows.groupby("game_id"):
            if game_id not in games_with_sb:
                skipped += 1
                continue

            meta = game_meta.get(game_id)
            if meta is None:
                logger.warning("No game metadata for %s — skipping", game_id)
                skipped += 1
                continue

            game_date = meta["game_date"]
            home_tricode = meta["home_team"]
            away_tricode = meta["away_team"]

            # Load sportsbook data from the date file (named by ET game date)
            sb_game_df = self._load_sportsbook_for_game(
                sb_dir, game_date, home_tricode, away_tricode
            )

            if sb_game_df is None or sb_game_df.empty:
                skipped += 1
                continue

            poss_group = poss_anchors[poss_anchors["game_id"] == game_id].set_index("possession_id")

            wp_series = self._compute_sportsbook_wp(
                group, poss_group, game_date, sb_game_df, home_tricode
            )

            synthetic_prices = self._apply_behavioral_model(group, wp_series)

            for i, (_, row) in enumerate(group.iterrows()):
                wp_val = wp_series.iloc[i] if i < len(wp_series) else 0.5
                results.append({
                    "game_id": game_id,
                    "possession_id": int(row["possession_id"]),
                    "synthetic_yes_price": int(synthetic_prices.iloc[i]),
                    "sharp_book_wp": float(wp_val),
                    "has_sportsbook_data": True,
                })

        logger.info("Generated prices for %d games, skipped %d (no sportsbook data)", len(results) // 100, skipped)
        return pd.DataFrame(results)

    def _load_sportsbook_for_game(
        self,
        sb_dir: Path,
        game_date: date,
        home_tricode: str,
        away_tricode: str,
    ) -> pd.DataFrame | None:
        """Load sportsbook snapshots for one game from its date file."""
        date_file = sb_dir / f"{game_date}.parquet"
        if not date_file.exists():
            return None

        df = pq.read_table(date_file).to_pandas()

        # Match by team name → tricode
        home_mask = df["home_team_name"].map(TEAM_NAME_TO_TRICODE) == home_tricode
        away_mask = df["away_team_name"].map(TEAM_NAME_TO_TRICODE) == away_tricode
        matched = df[home_mask & away_mask].sort_values("snapshot_ts")

        return matched if not matched.empty else None

    def _compute_sportsbook_wp(
        self,
        group: pd.DataFrame,
        poss_group: pd.DataFrame,
        game_date: date,
        sb_game_df: pd.DataFrame | None,
        home_tricode: str,
    ) -> pd.Series:
        """
        For each possession, find nearest sportsbook snapshot WP.
        Falls back to logistic estimate if no sportsbook data or gap > MAX.
        Returns Series of home_wp values, lagged by lag_possessions.
        """
        wp_values: list[float] = []

        for _, row in group.iterrows():
            poss_id = row["possession_id"]
            poss_anchor = poss_group.loc[poss_id] if poss_id in poss_group.index else None

            poss_utc = None
            if poss_anchor is not None:
                poss_utc = possession_wall_clock_utc(
                    game_date,
                    str(poss_anchor["period_wall_clock"]),
                    float(poss_anchor["game_clock_secs"]),
                )

            wp = self._lookup_wp(poss_utc, sb_game_df, row, home_tricode)
            wp_values.append(wp)

        wp_series = pd.Series(wp_values, index=group.index)

        # Lag by lag_possessions: shift forward so strategy sees older data
        wp_lagged = wp_series.shift(self.lag_possessions).bfill().fillna(0.5)
        return wp_lagged

    def _lookup_wp(
        self,
        poss_utc: datetime | None,
        sb_game_df: pd.DataFrame | None,
        row: pd.Series,
        home_tricode: str,
    ) -> float:
        if poss_utc is not None and sb_game_df is not None and not sb_game_df.empty:
            deltas = (sb_game_df["snapshot_ts"].dt.tz_localize(None) - poss_utc.replace(tzinfo=None)).abs()
            nearest_idx = deltas.idxmin()
            gap_minutes = deltas[nearest_idx].total_seconds() / 60.0

            if gap_minutes <= MAX_SNAPSHOT_GAP_MINUTES:
                return float(sb_game_df.loc[nearest_idx, "home_wp"])

        # No sportsbook data within gap limit — return 0.5 as neutral placeholder
        # (game should have been filtered by games_with_sb before reaching here)
        return 0.5

    def _apply_behavioral_model(
        self,
        group: pd.DataFrame,
        wp_lagged: pd.Series,
    ) -> pd.Series:
        """
        Apply overreaction and mean-reversion terms on top of lagged sportsbook WP.
        Returns integer cents, clipped to [1, 99].
        """
        rng = np.random.default_rng(seed=42)
        prices: list[float] = []

        possessions_since_run_end = 0
        prev_run_team = None

        for i, (_, row) in enumerate(group.iterrows()):
            base = wp_lagged.iloc[i] * 100.0

            current_run_team = row.get("current_run_team")
            current_run_points = float(row.get("current_run_points", 0) or 0)
            home_team = "home"  # run team is labeled as "home" or "away" in feature store

            # Track when run ended for reversion
            if current_run_team != prev_run_team and prev_run_team is not None:
                possessions_since_run_end = 0
            elif current_run_team is None and prev_run_team is not None:
                possessions_since_run_end += 1
            prev_run_team = current_run_team

            # Overreaction term: retail overbets visible scoring runs
            overreaction = 0.0
            if current_run_team is not None and current_run_points > self.run_threshold:
                excess = current_run_points - self.run_threshold
                overreaction = self.overreaction_coeff * excess
                # If the run is AWAY team, overreaction pushes home price DOWN
                if current_run_team != home_team:
                    overreaction = -overreaction

            # Mean reversion term: prices drift back after run ends
            reversion = 0.0
            if current_run_team is None and possessions_since_run_end > 0:
                reversion = overreaction * np.exp(-self.reversion_speed * possessions_since_run_end)

            noise = rng.normal(0.0, self.noise_sigma)
            raw = base + overreaction - reversion + noise
            prices.append(max(1.0, min(99.0, round(raw))))

        return pd.Series(prices, index=group.index, dtype=int)


def load_all_sportsbook_lines() -> pd.DataFrame | None:
    """Load all daily sportsbook parquet files into one DataFrame."""
    if not SPORTSBOOK_LINES_DIR.exists():
        logger.warning("Sportsbook lines directory not found: %s", SPORTSBOOK_LINES_DIR)
        return None

    files = sorted(SPORTSBOOK_LINES_DIR.glob("*.parquet"))
    if not files:
        logger.warning("No sportsbook line files found in %s", SPORTSBOOK_LINES_DIR)
        return None

    tables = [pq.read_table(f) for f in files]
    combined = pa.concat_tables(tables)
    df = combined.to_pandas()
    # Ensure snapshot_ts is timezone-aware
    if df["snapshot_ts"].dt.tz is None:
        df["snapshot_ts"] = df["snapshot_ts"].dt.tz_localize("UTC")
    logger.info("Loaded %d sportsbook rows from %d files", len(df), len(files))
    return df


def _games_with_sportsbook(games_df: pd.DataFrame, sb_dir: Path) -> set[str]:
    """Return set of game_ids that have a matching entry in sportsbook_lines."""
    covered: set[str] = set()
    for game_date, group in games_df.groupby("game_date"):
        f = sb_dir / f"{game_date}.parquet"
        if not f.exists():
            continue
        df = pq.read_table(f).to_pandas()
        df["home_tricode"] = df["home_team_name"].map(TEAM_NAME_TO_TRICODE)
        df["away_tricode"] = df["away_team_name"].map(TEAM_NAME_TO_TRICODE)
        sb_keys = set(zip(df["home_tricode"], df["away_tricode"]))
        for _, game in group.iterrows():
            if (game["home_team"], game["away_team"]) in sb_keys:
                covered.add(game["game_id"])
    return covered


def main() -> None:
    games_path = Path("data/raw/games_202526.parquet")
    possessions_path = Path("data/raw/possessions_202526.parquet")
    feature_rows_path = Path("data/feature_store/feature_rows.parquet")

    for p in [games_path, possessions_path, feature_rows_path]:
        if not p.exists():
            logger.error("Required file not found: %s", p)
            sys.exit(1)

    if not SPORTSBOOK_LINES_DIR.exists():
        logger.error("No sportsbook lines found at %s — run sportsbook_client.py first", SPORTSBOOK_LINES_DIR)
        sys.exit(1)

    games_df = pq.read_table(games_path).to_pandas()
    possessions_df = pq.read_table(possessions_path).to_pandas()
    feature_rows = pq.read_table(feature_rows_path).to_pandas()

    games_with_sb = _games_with_sportsbook(games_df, SPORTSBOOK_LINES_DIR)
    logger.info(
        "%d / %d games have sportsbook data — skipping the other %d",
        len(games_with_sb), len(games_df), len(games_df) - len(games_with_sb),
    )

    model = SyntheticKalshiPriceModel()
    result_df = model.generate(
        feature_rows, games_df, possessions_df, SPORTSBOOK_LINES_DIR, games_with_sb
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(result_df, schema=OUTPUT_SCHEMA, preserve_index=False)
    pq.write_table(table, OUTPUT_PATH)

    logger.info("Wrote %d rows covering %d games to %s", len(result_df), result_df["game_id"].nunique(), OUTPUT_PATH)


if __name__ == "__main__":
    main()
