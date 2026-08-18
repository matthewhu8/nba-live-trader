"""
Regression test for the sub_count reset behavior.

Background: prior to 2026-05-13, GameState.home_sub_count and away_sub_count
incremented monotonically over the whole game (reaching 30+ by Q4). The
training-data column with the same name is per-possession (typical value 0–3,
max ~5). Live values at z=+73σ pushed the model into an input region it had
never seen, collapsing Head A's gating network and pinning run_prob near 0.

The fix: GameState.advance() resets both counters to 0 after each possession's
features are extracted. This test asserts that invariant.

Run from project root:
    ./venv/bin/python live-trader/inference/test_sub_count_reset.py
"""
import sys
from pathlib import Path

# Make `inference` importable when running from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.game_state import GameState


HOME_ID = 1610612747  # any nonzero values
AWAY_ID = 1610612744


def _sub_event(team_id: int, player_id: int, sub_type: str) -> dict:
    return {
        "actionType": "substitution",
        "teamId": team_id,
        "personId": player_id,
        "subType": sub_type,
    }


class _FakePossession:
    """Minimal stand-in for PossessionRow; advance() reads only a few fields."""
    points = 0
    team_scored = ""
    shot_value = 0
    shot_distance = 0.0
    shot_area = ""
    is_paint = False
    xppp = 0.0
    period = 1
    game_clock_secs = 720.0


def test_sub_count_resets_on_advance() -> None:
    """The core invariant: advance() zeros both sub counters."""
    state = GameState(game_id="test", home_team_id=HOME_ID, away_team_id=AWAY_ID)

    state.update_from_event(_sub_event(HOME_ID, 1001, "out"))
    state.update_from_event(_sub_event(HOME_ID, 1002, "in"))
    state.update_from_event(_sub_event(AWAY_ID, 2001, "out"))

    assert state.home_sub_count == 2, f"expected 2 home subs pre-advance, got {state.home_sub_count}"
    assert state.away_sub_count == 1, f"expected 1 away sub pre-advance, got {state.away_sub_count}"

    state.advance(_FakePossession())

    assert state.home_sub_count == 0, f"home_sub_count not reset: {state.home_sub_count}"
    assert state.away_sub_count == 0, f"away_sub_count not reset: {state.away_sub_count}"


def test_sub_count_stays_bounded_over_simulated_game() -> None:
    """
    Simulate a Q4 worth of sub events spread across possessions.
    Pre-fix, sub_count would climb to ~50 by end of simulation.
    Post-fix, it must never exceed the number of subs in a single possession.
    """
    state = GameState(game_id="test", home_team_id=HOME_ID, away_team_id=AWAY_ID)

    max_observed_home = 0
    max_observed_away = 0

    # 60 possessions with a few subs sprinkled through each.
    sub_pattern = [0, 1, 0, 2, 0, 0, 3, 0, 1, 0]  # cycles through

    for poss_idx in range(60):
        n_subs = sub_pattern[poss_idx % len(sub_pattern)]
        for s in range(n_subs):
            team = HOME_ID if (poss_idx + s) % 2 == 0 else AWAY_ID
            state.update_from_event(_sub_event(team, 3000 + s, "out"))

        max_observed_home = max(max_observed_home, state.home_sub_count)
        max_observed_away = max(max_observed_away, state.away_sub_count)
        state.advance(_FakePossession())

    assert max_observed_home + max_observed_away > 0, "test produced no subs, broken setup"

    # The invariant: never exceed the training distribution's maximum.
    HARD_MAX = 5  # training data max observed was ~3; 5 leaves headroom
    assert max_observed_home <= HARD_MAX, (
        f"home_sub_count climbed to {max_observed_home} during simulated game, "
        f"reset is not firing. Training max is ~3."
    )
    assert max_observed_away <= HARD_MAX, (
        f"away_sub_count climbed to {max_observed_away} during simulated game"
    )

    # The last advance() must have cleared them too.
    assert state.home_sub_count == 0
    assert state.away_sub_count == 0


def test_sub_count_unchanged_for_non_sub_events() -> None:
    """Foul and timeout events should not affect sub counters."""
    state = GameState(game_id="test", home_team_id=HOME_ID, away_team_id=AWAY_ID)

    state.update_from_event({
        "actionType": "foul", "teamId": HOME_ID, "personId": 1001,
        "period": 1, "subType": "shooting",
    })
    state.update_from_event({
        "actionType": "timeout", "teamId": AWAY_ID, "period": 1,
        "clock": "PT05M00.00S",
    })

    assert state.home_sub_count == 0
    assert state.away_sub_count == 0


if __name__ == "__main__":
    tests = [
        test_sub_count_resets_on_advance,
        test_sub_count_stays_bounded_over_simulated_game,
        test_sub_count_unchanged_for_non_sub_events,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
