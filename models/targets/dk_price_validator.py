"""
DKPriceValidator — maps possessions to DraftKings live line snapshots.

For each trade signal (game_id, possession_id, direction), looks up:
  - dk_wp_entry:  DK home_wp at the possession's wall-clock time
  - dk_wp_exit:   DK home_wp ~lookahead_minutes later

This gives us the honest upper bound on extractable edge: had Kalshi
tracked DK perfectly, what would this trade have returned?

Reuses parse_wall_clock_utc and possession_wall_clock_utc from
models/synthetic_kalshi.py rather than duplicating the time logic.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow.parquet as pq

from models.synthetic_kalshi import (
    TEAM_NAME_TO_TRICODE,
    possession_wall_clock_utc,
)

logger = logging.getLogger(__name__)

MAKER_FEE_RATE = 0.0175
MAX_SNAPSHOT_GAP_MINUTES = 15


@dataclass
class DKMoveResult:
    dk_wp_entry: float        # DK home_wp at possession time
    dk_wp_exit: float         # DK home_wp ~lookahead_minutes later
    dk_price_entry_cents: int  # round(dk_wp_entry * 100)
    dk_price_exit_cents: int   # round(dk_wp_exit * 100)
    dk_gross_pnl: float        # (exit - entry) × size / 100, signed for direction
    dk_net_pnl: float          # gross minus maker fees both sides
    gap_minutes_entry: float   # nearest snapshot gap (data quality flag)


class DKPriceValidator:
    """
    Preloads all sportsbook files and possession timestamps for fast lookup.

    Call once, then get_dk_move() per trade.
    """

    def __init__(
        self,
        sportsbook_dir: Path,
        possessions_df: pd.DataFrame,
        games_df: pd.DataFrame,
    ) -> None:
        self._sportsbook_dir = Path(sportsbook_dir)
        self._games_meta = self._build_games_meta(games_df)
        self._game_snapshots = self._load_all_snapshots(games_df)
        self._possession_utc = self._build_possession_utc(possessions_df, games_df)

        logger.info(
            "DKPriceValidator ready: %d games with snapshots, %d possession timestamps",
            len(self._game_snapshots),
            len(self._possession_utc),
        )

    def get_dk_move(
        self,
        game_id: str,
        possession_id: int,
        direction: Literal["YES", "NO"],
        lookahead_minutes: int = 10,
        size: int = 10,
    ) -> DKMoveResult | None:
        """
        Return DK line movement for a trade signal.

        Returns None if:
          - No sportsbook data for this game
          - No snapshot within MAX_SNAPSHOT_GAP_MINUTES of possession time
          - No exit snapshot within lookahead window
        """
        snapshots = self._game_snapshots.get(game_id)
        if snapshots is None or snapshots.empty:
            return None

        poss_utc = self._possession_utc.get((game_id, possession_id))
        if poss_utc is None:
            return None

        entry = self._nearest_snapshot(snapshots, poss_utc)
        if entry is None:
            return None

        wp_entry = float(entry["home_wp"])
        # snapshot_ts may be tz-aware; poss_utc is naive UTC
        entry_ts = entry["snapshot_ts"]
        if hasattr(entry_ts, "tzinfo") and entry_ts.tzinfo is not None:
            entry_ts = entry_ts.replace(tzinfo=None)
        gap_minutes = abs((entry_ts - poss_utc).total_seconds()) / 60.0

        if gap_minutes > MAX_SNAPSHOT_GAP_MINUTES:
            return None

        exit_target_utc = poss_utc + timedelta(minutes=lookahead_minutes)
        exit_entry = self._nearest_snapshot(snapshots, exit_target_utc)
        if exit_entry is None:
            return None

        wp_exit = float(exit_entry["home_wp"])
        entry_cents = round(wp_entry * 100)
        exit_cents = round(wp_exit * 100)

        # PnL from the home_wp perspective:
        # YES buyer profits when home_wp rises
        # NO buyer profits when home_wp falls
        if direction == "YES":
            gross_pnl = (wp_exit - wp_entry) * 100.0 * size / 100.0
        else:
            gross_pnl = (wp_entry - wp_exit) * 100.0 * size / 100.0

        entry_fee = MAKER_FEE_RATE * size * entry_cents / 100.0
        exit_fee = MAKER_FEE_RATE * size * exit_cents / 100.0
        net_pnl = gross_pnl - entry_fee - exit_fee

        return DKMoveResult(
            dk_wp_entry=wp_entry,
            dk_wp_exit=wp_exit,
            dk_price_entry_cents=entry_cents,
            dk_price_exit_cents=exit_cents,
            dk_gross_pnl=gross_pnl,
            dk_net_pnl=net_pnl,
            gap_minutes_entry=gap_minutes,
        )

    # --- Internal helpers ---

    def _build_games_meta(self, games_df: pd.DataFrame) -> dict[str, dict]:
        """game_id → {game_date, home_team, away_team}."""
        return games_df.set_index("game_id")[["game_date", "home_team", "away_team"]].to_dict("index")

    def _load_all_snapshots(self, games_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Preload sportsbook snapshots for every known game, keyed by game_id."""
        game_snapshots: dict[str, pd.DataFrame] = {}

        for game_id, meta in self._games_meta.items():
            game_date = meta["game_date"]
            home_tricode = meta["home_team"]
            away_tricode = meta["away_team"]

            date_file = self._sportsbook_dir / f"{game_date}.parquet"
            if not date_file.exists():
                continue

            try:
                df = pq.read_table(date_file).to_pandas()
            except Exception as exc:
                logger.warning("Failed to read %s: %s", date_file, exc)
                continue

            home_mask = df["home_team_name"].map(TEAM_NAME_TO_TRICODE) == home_tricode
            away_mask = df["away_team_name"].map(TEAM_NAME_TO_TRICODE) == away_tricode
            matched = df[home_mask & away_mask].sort_values("snapshot_ts").reset_index(drop=True)

            if not matched.empty:
                game_snapshots[game_id] = matched

        return game_snapshots

    def _build_possession_utc(
        self,
        possessions_df: pd.DataFrame,
        games_df: pd.DataFrame,
    ) -> dict[tuple[str, int], datetime]:
        """Precompute UTC timestamp for every (game_id, possession_id)."""
        game_dates: dict[str, str] = dict(zip(games_df["game_id"], games_df["game_date"]))
        possession_utc: dict[tuple[str, int], datetime] = {}

        for _, row in possessions_df.iterrows():
            game_id = str(row["game_id"])
            game_date_str = game_dates.get(game_id)
            if game_date_str is None:
                continue

            import datetime as dt_module
            game_date = dt_module.date.fromisoformat(str(game_date_str))

            utc_ts = possession_wall_clock_utc(
                game_date,
                str(row["period_wall_clock"]),
                float(row["game_clock_secs"]),
            )
            if utc_ts is not None:
                possession_utc[(game_id, int(row["possession_id"]))] = utc_ts

        return possession_utc

    def _nearest_snapshot(
        self,
        snapshots: pd.DataFrame,
        target_utc: datetime,
    ) -> dict | None:
        """Find the snapshot row closest in time to target_utc.

        snapshot_ts is tz-aware UTC; target_utc is naive UTC (from possession_wall_clock_utc).
        We strip tz from snapshots for comparison.
        """
        # Strip tz from snapshot timestamps so both sides are naive UTC
        ts_naive = snapshots["snapshot_ts"].dt.tz_localize(None)
        import pandas as pd
        target_ts = pd.Timestamp(target_utc)
        deltas = (ts_naive - target_ts).abs()
        idx = deltas.idxmin()
        return snapshots.loc[idx].to_dict()
