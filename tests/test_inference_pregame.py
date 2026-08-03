"""
Pregame loader smoke test.

This was previously a bare `async def test_pregame()` with no assertions and a
`__main__` block. pytest collected it by name but cannot run a coroutine without
an async plugin, so it failed on every run regardless of the code under test.

It is now a synchronous test that drives the coroutine itself, asserts the
contract the live path depends on, and skips cleanly when MotherDuck credentials
are absent rather than failing.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "live-trader"))

from inference.pregame import load_pregame

# Knicks @ Sixers
GAME_ID = "0042500213"

requires_motherduck = pytest.mark.skipif(
    not os.environ.get("MOTHERDUCK_TOKEN"),
    reason="MOTHERDUCK_TOKEN not set — pregame load needs the cloud warehouse",
)


@requires_motherduck
def test_pregame_returns_the_keys_the_live_path_reads():
    """
    load_pregame() degrades gracefully when a game is missing, so the assertion
    here is about shape, not content: main.py indexes these keys unconditionally
    at game start and a missing one is a startup crash.
    """
    data = asyncio.run(load_pregame(GAME_ID))

    for key in (
        "lineup_ratings",
        "lineup_sample_sizes",
        "player_apm",
        "star_players",
        "pace_baseline",
        "expected_pace",
        "has_pregame_data",
    ):
        assert key in data, f"load_pregame() must always return {key!r}"

    assert isinstance(data["lineup_ratings"], dict)
    assert isinstance(data["lineup_sample_sizes"], dict)
    assert float(data["pace_baseline"]) > 0.0


@requires_motherduck
def test_lineup_sample_sizes_align_with_ratings():
    """
    `lineup_confidence` reads sample sizes keyed by the same lineup ids as the
    ratings. Live previously hardcoded both sample sizes to 0.0, so this pairing
    was never checked.
    """
    data = asyncio.run(load_pregame(GAME_ID))
    ratings = data["lineup_ratings"]
    samples = data["lineup_sample_sizes"]

    if not ratings:
        pytest.skip("no lineup ratings for this game — nothing to align")

    assert set(samples) == set(ratings), "every rated lineup needs a sample size"
    assert all(v >= 0 for v in samples.values())
