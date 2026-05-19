"""
Verify every import the post-game pipeline needs at runtime.

Covers both top-level imports (fail at module-load time) and the deferred
imports that fire inside each pipeline phase. Each import is its own test so
failures are isolated and the missing package is immediately obvious.

Run before every Fly.io deploy:
    ./venv/bin/python -m pytest tests/test_pipeline_imports.py -v
"""


def test_sklearn_ridge() -> None:
    from sklearn.linear_model import Ridge  # noqa: F401


def test_scipy_sparse() -> None:
    from scipy.sparse import lil_matrix  # noqa: F401


def test_numpy() -> None:
    import numpy  # noqa: F401


def test_pandas() -> None:
    import pandas  # noqa: F401


def test_duckdb() -> None:
    import duckdb  # noqa: F401


def test_pyarrow() -> None:
    import pyarrow  # noqa: F401


def test_pytz() -> None:
    import pytz  # noqa: F401


def test_aiohttp() -> None:
    import aiohttp  # noqa: F401


def test_dotenv() -> None:
    from dotenv import load_dotenv  # noqa: F401


def test_cryptography() -> None:
    from cryptography.hazmat.primitives import hashes  # noqa: F401


# --- pipeline module imports ---


def test_pipeline_top_level() -> None:
    """Triggers all top-level imports in post_game_pipeline.py."""
    from data.ingestion.post_game_pipeline import run_post_game_pipeline  # noqa: F401


def test_game_schedule() -> None:
    from data.ingestion.game_schedule import GameInfo, get_todays_games  # noqa: F401


def test_nba_api_client() -> None:
    from data.ingestion.nba_api_client import parse_game_to_events  # noqa: F401


def test_feature_builder() -> None:
    from models.features.builder import process_game  # noqa: F401


# --- deferred imports (fired at phase-execution time) ---


def test_nba_api_endpoint() -> None:
    """Phase 1: PBP fetch."""
    from nba_api.stats.endpoints import playbyplayv3  # noqa: F401


def test_team_ratings() -> None:
    """Phase 3d: team EWMA ratings."""
    from models.ratings.team_ratings import (  # noqa: F401
        apply_team_ewma,
        compute_game_team_stats,
    )


def test_pregame_features() -> None:
    """Phase 4: pregame feature computation."""
    from models.features.pregame_features import compute_pregame_features  # noqa: F401


def test_backfill_wall_clock() -> None:
    """Phase 2b: wall-clock timestamp backfill."""
    from data.ingestion.backfill_wall_clock_ts import backfill_game  # noqa: F401
