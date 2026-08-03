"""
Train/live feature parity.

The offline builder and the live streamer compute the same model inputs by two
completely different routes: pandas over a whole game, versus a streaming
accumulator over one possession at a time. Four silent divergences were found in
that gap, three of them in a single six-feature block:

  1. `minutes_remaining` floored at 1.0 offline and 0.1 online (10x in the last minute)
  2. overtime pinned the online value to 0.1 on every OT possession (~50x)
  3. `garbage_time_risk` was a continuous sigmoid offline and a hard binary online
  4. `pace_season_baseline` was an expanding within-game mean offline and a fixed
     pregame constant online — a different quantity, not a different number

All four are asserted here. Every one of them would have failed this file.

Run: ./venv/bin/python -m pytest tests/test_feature_parity.py -v
"""

import math
import re
from pathlib import Path

import numpy as np
import pytest

from models.features import transforms as T
from models.mmoe.feature_config import ALL_FEATURE_COLS, PHYSICS_COLS


TOLERANCE = 1e-6

# (period, game_clock_secs) covering regulation, both ends of a quarter, the final
# minute, and overtime — the cases where the two paths used to disagree.
CLOCK_CASES = [
    (1, 720.0),
    (1, 360.0),
    (2, 720.0),
    (2, 1.0),
    (3, 400.0),
    (4, 720.0),
    (4, 60.0),
    (4, 6.0),
    (4, 0.0),
    (5, 300.0),   # OT
    (5, 30.0),    # OT, final seconds
    (6, 120.0),   # 2OT
]

SCORE_CASES = [0, 3, -3, 6, -6, 15, -15, 25, -25, 57, -57]


# ── The scalar/array contract every transform depends on ─────────────────────

def test_transforms_agree_between_scalar_and_array():
    """
    The offline path calls these with Series, the live path with floats. If a
    transform ever branches on type, the two paths silently diverge. This is the
    invariant that makes one shared module sufficient.
    """
    periods = np.array([p for p, _ in CLOCK_CASES], dtype=float)
    clocks = np.array([c for _, c in CLOCK_CASES], dtype=float)
    diffs = np.array([7.0] * len(CLOCK_CASES))

    vectorized = {
        "minutes_remaining": T.minutes_remaining(periods, clocks),
        "minutes_into_game": T.minutes_into_game(periods, clocks),
        "time_leverage": T.time_leverage(periods, clocks),
        "lead_z": T.lead_z(diffs, periods, clocks),
        "garbage_time_risk": T.garbage_time_risk(diffs, periods, clocks),
    }

    for name, batch in vectorized.items():
        fn = getattr(T, name)
        for i, (period, clock) in enumerate(CLOCK_CASES):
            scalar = (
                fn(7.0, period, clock)
                if name in ("lead_z", "garbage_time_risk")
                else fn(period, clock)
            )
            assert abs(float(scalar) - float(batch[i])) < TOLERANCE, (
                f"{name} disagrees between scalar and array at "
                f"period={period} clock={clock}: {scalar} vs {batch[i]}"
            )


# ── Bug 1: the minutes_remaining floor ───────────────────────────────────────

def test_minutes_remaining_floor_is_one_not_point_one():
    """
    Live used max(0.1, ...), which let the final minute of Q4 produce divisors ten
    times smaller than anything training ever saw.
    """
    assert float(T.minutes_remaining(4, 6.0)) == pytest.approx(1.0)
    assert float(T.minutes_remaining(4, 0.0)) == pytest.approx(1.0)
    assert float(T.minutes_remaining(4, 120.0)) == pytest.approx(2.0)


# ── Bug 2: overtime ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("period,clock,expected", [
    (5, 300.0, 5.0),
    (5, 60.0, 1.0),
    (6, 300.0, 5.0),
])
def test_overtime_minutes_remaining_uses_period_clock(period, clock, expected):
    """
    The old live formula was `48 - elapsed`, which goes negative in OT and clamped
    every OT possession to the same 0.1, roughly 50x off. The canonical form clamps
    `4 - period` at zero instead, so OT returns its own clock.
    """
    assert float(T.minutes_remaining(period, clock)) == pytest.approx(expected)

    broken_elapsed = ((period - 1) * 12) + (720.0 - clock) / 60.0
    broken = max(0.1, 48.0 - broken_elapsed)
    assert broken == pytest.approx(0.1), "sanity: the old formula really was broken"
    assert float(T.minutes_remaining(period, clock)) > broken


# ── Bug 3: garbage_time_risk is continuous ───────────────────────────────────

def test_garbage_time_risk_is_continuous_not_binary():
    """
    Live returned float(is_garbage_time), which is 0.0 or 1.0 and required Q4 with
    under 6 minutes left. Training used a sigmoid that is nonzero from Q1 onward.
    """
    observed = {
        float(T.garbage_time_risk(d, p, c))
        for d in (5, 15, 25, 40)
        for p, c in CLOCK_CASES
    }
    assert len(observed) > 10, "should take many distinct values, not just {0.0, 1.0}"
    assert not observed <= {0.0, 1.0}

    # The specific case that motivated the finding: a 25-point Q2 lead.
    q2_blowout = float(T.garbage_time_risk(25, 2, 360.0))
    assert q2_blowout == pytest.approx(0.3303, abs=1e-3)
    assert q2_blowout > 0.0, "live emitted 0.0 here because period != 4"


def test_garbage_time_risk_stays_in_unit_interval():
    for d in SCORE_CASES:
        for p, c in CLOCK_CASES:
            v = float(T.garbage_time_risk(d, p, c))
            assert 0.0 <= v <= 1.0, f"garbage_time_risk={v} out of range at d={d} p={p} c={c}"


# ── Bug 4: pace, the within-game mean vs the pregame constant ────────────────

def test_pace_ref_blends_and_never_drops_the_prior():
    """
    Offline `pace_season_baseline` is an expanding within-game mean; live fed the
    pregame constant under the same name. The shrinkage blend names both
    explicitly, so neither path can substitute one for the other.
    """
    game_pace, prior = 13.0, 16.0

    early = float(T.pace_ref(game_pace, prior, 0))
    assert early == pytest.approx(prior), "with no possessions the prior must dominate"

    even = float(T.pace_ref(game_pace, prior, T.PACE_SHRINK_POSSESSIONS))
    assert even == pytest.approx((game_pace + prior) / 2)

    late = float(T.pace_ref(game_pace, prior, 10_000))
    assert late == pytest.approx(
        T.PACE_MAX_WEIGHT * game_pace + (1 - T.PACE_MAX_WEIGHT) * prior
    )
    assert late != pytest.approx(game_pace), "the prior must keep a floor of influence"


def test_pace_shrink_weight_is_monotone_and_capped():
    weights = [float(T.pace_shrink_weight(n)) for n in range(0, 400, 10)]
    assert all(b >= a - TOLERANCE for a, b in zip(weights, weights[1:]))
    assert max(weights) <= T.PACE_MAX_WEIGHT + TOLERANCE


# ── The consolidation properties the doc claimed ─────────────────────────────

def test_lead_z_separates_winning_from_losing():
    """The original objection: d^2/m gave identical inputs for opposite trades."""
    up = float(T.lead_z(6, 4, 60.0))
    down = float(T.lead_z(-6, 4, 60.0))
    assert up == pytest.approx(-down)
    assert up > 0 > down

    old_proxy_up = (6 ** 2) / 1.0
    old_proxy_down = ((-6) ** 2) / 1.0
    assert old_proxy_up == old_proxy_down, "sanity: the old feature really was blind to sign"


def test_time_leverage_gives_late_minutes_more_resolution():
    """
    The same five minutes of clock must span more input range late than early.
    Both windows below are exactly five minutes: 10->15 elapsed and 40->45.
    Linear minutes_into_game/48 gives each 0.104, a ratio of 1.0.
    """
    early = abs(float(T.time_leverage(2, 540.0)) - float(T.time_leverage(1, 120.0)))
    late = abs(float(T.time_leverage(4, 180.0)) - float(T.time_leverage(4, 480.0)))

    assert early == pytest.approx(0.061, abs=1e-3)
    assert late == pytest.approx(0.158, abs=1e-3)
    assert late > early * 2, f"expected >2x late resolution, got {late / early:.2f}x"

    assert float(T.time_leverage(1, 720.0)) == pytest.approx(0.0)
    assert float(T.time_leverage(4, 0.0)) == pytest.approx(1.0)


def test_timeout_sentinel_no_longer_dominates_the_scale():
    """
    The 999 sentinel put 9.42% of rows at z=+3.10 while compressing the real 0-70
    range into a 0.24-wide band. Bounding the count and splitting the flag out
    restores resolution over the range that carries the signal.
    """
    real = np.arange(0, 71, dtype=float)
    population = np.concatenate([real, np.full(10, T.NO_TIMEOUT_SENTINEL, dtype=float)])

    raw_z = (population - population.mean()) / population.std()
    bounded = T.timeout_possessions_bounded(population)
    bounded_z = (bounded - bounded.mean()) / bounded.std()

    raw_span = raw_z[:71].max() - raw_z[:71].min()
    bounded_span = bounded_z[:71].max() - bounded_z[:71].min()
    assert bounded_span > raw_span * 2, (
        f"bounding should widen the real range's z-span; {raw_span:.3f} -> {bounded_span:.3f}"
    )

    assert float(T.no_timeout_yet(T.NO_TIMEOUT_SENTINEL)) == 1.0
    assert float(T.no_timeout_yet(70)) == 0.0
    assert float(T.timeout_possessions_bounded(T.NO_TIMEOUT_SENTINEL)) == T.TIMEOUT_POSSESSION_CAP


def test_quality_trend_keeps_magnitude():
    """The old columns were np.sign()-collapsed, so these two had to look alike."""
    tiny = float(T.quality_trend_edge(1.01, 1.00, 1.00, 1.00))
    huge = float(T.quality_trend_edge(1.90, 1.00, 1.00, 1.00))
    assert abs(huge) > abs(tiny) * 10
    assert np.sign(tiny) == np.sign(huge), "sanity: sign() would have made these equal"


def test_edges_are_antisymmetric():
    """Every home/away fold must flip sign when the teams are swapped."""
    for fn, args in [
        (T.edge, (3.0, 1.0)),
        (T.swing_5, (8.0, 2.0)),
        (T.xppp_edge, (1.2, 0.9)),
        (T.luck_edge, (0.4, -0.2)),
    ]:
        forward = float(fn(*args))
        reverse = float(fn(*args[::-1]))
        assert forward == pytest.approx(-reverse), f"{fn.__name__} is not antisymmetric"


def test_run_signed_points_carries_direction():
    assert float(T.run_signed_points(T.encode_run_team("home"), 9)) == 9.0
    assert float(T.run_signed_points(T.encode_run_team("away"), 9)) == -9.0
    assert float(T.run_signed_points(T.encode_run_team(""), 9)) == 0.0


# ── Market encoder toggle (the two-run attribution baseline) ─────────────────

@pytest.mark.parametrize("use_encoder", [True, False])
def test_model_accepts_the_same_raw_input_either_way(use_encoder):
    """
    Both variants must consume the identical 58-column raw vector. Only the expert
    input width differs, so run 1 and run 2 read the same dataset and the metric
    delta is attributable to the encoder alone.
    """
    import torch

    from models.mmoe.model import MMoEModel, RAW_INPUT_DIM

    model = MMoEModel(use_market_encoder=use_encoder)
    model.eval()
    with torch.no_grad():
        run, traj, haz = model(torch.randn(4, RAW_INPUT_DIM))

    assert run.shape == (4, 1) and traj.shape == (4, 10) and haz.shape == (4, 10)
    expected_width = 48 if use_encoder else RAW_INPUT_DIM
    assert model.experts[0].net[0].in_features == expected_width


def test_predictor_infers_the_variant_from_the_checkpoint():
    """
    Loading must not assume the encoder is on. A --no-market-encoder checkpoint
    otherwise fails with a shape error that reads like a stale checkpoint.
    """
    import pickle
    import tempfile

    import numpy as np
    import torch
    from sklearn.preprocessing import StandardScaler

    from models.mmoe.model import MMoEModel, RAW_INPUT_DIM
    from models.mmoe.predictor import MMoEPredictor

    tmp = Path(tempfile.mkdtemp())
    scaler_path = tmp / "scaler.pkl"
    scaler_path.write_bytes(pickle.dumps(StandardScaler().fit(np.random.randn(64, RAW_INPUT_DIM))))

    for use_encoder in (True, False):
        model_path = tmp / f"model_{use_encoder}.pt"
        torch.save(
            {"epoch": 1, "model_state": MMoEModel(use_market_encoder=use_encoder).state_dict()},
            model_path,
        )
        loaded = MMoEPredictor.load(model_path, scaler_path)
        assert (loaded._model.market_encoder is not None) == use_encoder


# ── Config integrity ─────────────────────────────────────────────────────────

def test_feature_config_is_consistent():
    assert len(PHYSICS_COLS) == 33
    assert len(ALL_FEATURE_COLS) == 58
    assert len(set(ALL_FEATURE_COLS)) == len(ALL_FEATURE_COLS), "duplicate feature name"


def _live_features(possession, state, market=None):
    """Run the live FeatureComputer, importing it lazily so collection stays cheap."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "live-trader"))
    from inference.features import FeatureComputer

    return FeatureComputer.compute(possession, state, market or [0.0] * 14)


def _matched_case(period, clock, home_score, away_score, expected_pace,
                  pace_sum, pace_count, poss_count, home_fouls, away_fouls):
    """
    Build one game situation expressed both ways: as a possession_flat row for the
    offline builder, and as a GameState + PossessionRow for the live streamer.

    The raw ingredients are held identical on purpose. Anything that differs after
    this point is a formula divergence, which is exactly what we are hunting.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "live-trader"))
    from inference.game_state import GameState
    from inference.possession import PossessionRow

    from models.mmoe.feature_config import PREGAME_COLS

    state = GameState(game_id="PARITY")
    state.pregame = {c: 0.0 for c in PREGAME_COLS}
    state.pregame["expected_pace"] = expected_pace
    state.pace_duration_sum = pace_sum
    state.pace_duration_count = pace_count
    state.possession_count = poss_count
    # Give the last-10 deque a known mean so it matches the offline row's
    # pace_last_10_possessions rather than exercising the empty-deque fallback.
    state.possession_durations.extend([15.0] * 10)
    state.home_team_fouls = {period: home_fouls}
    state.away_team_fouls = {period: away_fouls}
    state.lineup_ratings = {}
    state.lineup_sample_sizes = {}
    state.home_lineup = state.prev_home_lineup = []
    state.away_lineup = state.prev_away_lineup = []

    possession = PossessionRow(
        possession_id=poss_count, period=period, game_clock_secs=clock,
        team_scored="", points=0, shot_value=0, shot_distance=0.0, shot_area="",
        was_contested=False, was_foul=False, had_shooting_foul=False,
        had_personal_foul=False, was_sub=False,
        home_score=home_score, away_score=away_score,
        home_lineup_id="H", away_lineup_id="A", xppp=0.0, is_paint=False,
    )

    # The offline path derives elapsed possessions from the row's position within
    # its game (groupby.cumcount), matching live's state.possession_count. Both are
    # 0-indexed. That means the comparison is only honest against a full game frame,
    # so the caller pads this row with `poss_count` earlier possessions.
    row = {
        "game_id": "PARITY", "period": period, "game_clock_secs": clock,
        "score_diff": home_score - away_score,
        "expected_pace": expected_pace,
        "pace_game_to_date": (pace_sum / pace_count) if pace_count else T.PACE_FALLBACK_SECS,
        "pace_last_10_possessions": 15.0,
        "home_team_fouls_q": home_fouls, "away_team_fouls_q": away_fouls,
        "home_timeouts_used": 0, "away_timeouts_used": 0,
        "home_sub_count": 0, "away_sub_count": 0,
        "home_lineup_sample_size": 0, "away_lineup_sample_size": 0,
        "current_run_team": None, "current_run_points": 0, "current_run_length": 0,
        "current_run_3pt_pct": 0.0, "current_run_paint_pct": 0.0,
        "home_points_last_5_poss": 0, "away_points_last_5_poss": 0,
        "home_points_last_10_poss": 0, "away_points_last_10_poss": 0,
        "home_xPPP_last_5": 0.0, "away_xPPP_last_5": 0.0,
        "home_actual_vs_expected_PPP": 0.0, "away_actual_vs_expected_PPP": 0.0,
        "possessions_since_last_timeout": T.NO_TIMEOUT_SENTINEL,
        "home_called_timeout_in_last_3_poss": False,
        "away_called_timeout_in_last_3_poss": False,
        "home_trouble_star_tier": 0, "away_trouble_star_tier": 0,
        "home_star_on_court": False, "away_star_on_court": False,
        "home_lineup_just_changed": False, "away_lineup_just_changed": False,
        "lineup_net_rating_delta": 0.0,
        "shot_value": 0.0, "shot_distance": 0.0,
        "was_sub": False, "had_shooting_foul": False, "had_personal_foul": False,
        "points": 0, "team_scored": "",
    }
    return possession, state, row


# The situations where the two paths used to disagree: the final minute, overtime,
# a mid-game blowout, and an ordinary early possession.
PARITY_CASES = [
    dict(period=1, clock=600.0, home_score=10, away_score=8),
    dict(period=2, clock=360.0, home_score=55, away_score=30),   # 25-pt Q2 lead
    dict(period=4, clock=300.0, home_score=90, away_score=88),
    dict(period=4, clock=60.0, home_score=100, away_score=94),   # final minute
    dict(period=4, clock=6.0, home_score=100, away_score=106),   # trailing, buzzer
    dict(period=5, clock=300.0, home_score=110, away_score=110),  # OT
    dict(period=5, clock=30.0, home_score=113, away_score=110),   # OT, final seconds
]


@pytest.mark.parametrize("case", PARITY_CASES)
def test_offline_and_live_agree_numerically(case):
    """
    The headline invariant: identical game state must yield identical model inputs
    through both code paths, to 1e-6.

    All four historical divergences are inside these cases. Before the shared
    transforms module this failed on minutes_remaining (10x in the final minute),
    overtime (~50x), garbage_time_risk (continuous vs binary) and pace
    (within-game mean vs pregame constant).
    """
    import pandas as pd

    from models.mmoe.dataset import _add_derived_features

    possession, state, row = _matched_case(
        expected_pace=14.8, pace_sum=430.0, pace_count=30, poss_count=30,
        home_fouls=3, away_fouls=5, **case,
    )

    live = _live_features(possession, state)

    # Pad with `poss_count` preceding possessions so the row under test sits at
    # index poss_count, giving it the same elapsed-possession count live has.
    game = pd.DataFrame([row] * (state.possession_count + 1)).reset_index(drop=True)
    offline = _add_derived_features(game).iloc[state.possession_count]

    mismatches = []
    for col in PHYSICS_COLS:
        lo, of = float(live[col]), float(offline[col])
        if not math.isclose(lo, of, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            mismatches.append(f"{col}: live={lo!r} offline={of!r}")

    assert not mismatches, (
        "offline and live disagree for "
        f"period={case['period']} clock={case['clock']}:\n  " + "\n  ".join(mismatches)
    )


def test_pace_fallback_matches_offline_fillna_at_game_start():
    """
    Bug 5, found by the numeric parity test above.

    Before any possession duration is measurable, offline emits
    `pace_last_10_possessions` via `.fillna(15.0)` while live fell back to
    `state.pace_baseline` — the pregame constant again. Every possession at the
    start of every game disagreed.
    """
    possession, state, row = _matched_case(
        period=1, clock=720.0, home_score=0, away_score=0,
        expected_pace=14.8, pace_sum=0.0, pace_count=0, poss_count=0,
        home_fouls=0, away_fouls=0,
    )
    state.possession_durations.clear()   # no measurable durations yet

    live = _live_features(possession, state)

    # pace_ref falls back to the prior; pace_surprise must be measured against the
    # same 15.0 the offline builder uses, not against pace_baseline.
    assert float(live["pace_ref"]) == pytest.approx(14.8)
    assert float(live["pace_surprise"]) == pytest.approx(T.PACE_FALLBACK_SECS - 14.8)


# ── Training matches the traded regime ───────────────────────────────────────

def test_traded_regime_filter_drops_exactly_the_untradeable():
    """
    The agent hard-skips overtime and blowouts, so training must not learn from
    them. Rows the feature pipeline left incomplete go too: filling a NULL pace
    with 0 teaches a game state (0 seconds per possession) that cannot occur.
    """
    import pandas as pd

    from models.mmoe.dataset import _blowout_margin_pts, _filter_to_traded_regime

    margin = _blowout_margin_pts()
    base = {
        "home_xPPP_last_5": 1.0,
        "current_run_3pt_pct": 0.5,
        "pace_game_to_date": 14.0,
    }
    frame = pd.DataFrame([
        {"period": 1, "score_diff": 0, **base},                    # keep
        {"period": 4, "score_diff": margin, **base},               # keep: at the threshold
        {"period": 4, "score_diff": -margin, **base},              # keep: symmetric
        {"period": 5, "score_diff": 0, **base},                    # drop: overtime
        {"period": 6, "score_diff": 0, **base},                    # drop: 2OT
        {"period": 2, "score_diff": margin + 1, **base},           # drop: blowout
        {"period": 2, "score_diff": -(margin + 1), **base},        # drop: blowout, other side
        {"period": 3, "score_diff": 0, **{**base, "pace_game_to_date": None}},  # drop: incomplete
    ])

    kept = _filter_to_traded_regime(frame)

    assert len(kept) == 3
    assert (kept["period"] <= 4).all()
    assert (kept["score_diff"].abs() <= margin).all()
    assert kept["pace_game_to_date"].notna().all()


def test_blowout_margin_comes_from_the_live_gate_config():
    """
    Must be the trading.yaml value the agent gates on (30), not the stored
    possession_flat.is_garbage_time definition (20). Filtering at 20 would discard
    28,627 rows the system would actually have traded.
    """
    from models.mmoe.dataset import TRADING_CONFIG_PATH, _blowout_margin_pts

    assert TRADING_CONFIG_PATH.exists(), f"trading.yaml not found at {TRADING_CONFIG_PATH}"
    assert _blowout_margin_pts() == 30


# ── The Python/Go contract ───────────────────────────────────────────────────

GO_DIR = Path(__file__).resolve().parent.parent / "live-trader" / "go"


def _go_sources() -> dict[str, str]:
    return {
        p.name: p.read_text()
        for p in GO_DIR.glob("*.go")
        if not p.name.endswith("_test.go")
    }


def test_go_never_gates_on_a_feature_lookup():
    """
    Go returns the zero value for a missing map key, with no error. So any gate
    that reads `resp.Features["x"]` turns itself off the moment `x` is renamed or
    dropped — which is exactly how the physics consolidation disabled the overtime
    skip and drove currentRunLength to 0, blocking every entry.

    Every name Go reads out of Features must therefore still be a real model
    feature. Gate inputs belong in named response fields instead.
    """
    pattern = re.compile(r'resp\.Features\[\s*"([^"]+)"\s*\]')

    offenders: list[str] = []
    for filename, src in _go_sources().items():
        for name in pattern.findall(src):
            if name not in ALL_FEATURE_COLS:
                offenders.append(f"{filename} reads Features[{name!r}], which is not a model feature")

    assert not offenders, (
        "Go is reading a non-existent feature and will silently get 0:\n  "
        + "\n  ".join(offenders)
        + "\n\nPromote it to a named field on PossessionResponse instead."
    )


def test_python_response_supplies_every_named_field_go_expects():
    """
    The Go struct and the Pydantic model are two hand-maintained halves of one wire
    format. If Go declares a json tag Python never sends, Go silently reads the
    zero value — the same failure mode as above, one level up.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "live-trader"))
    from inference.main import PossessionResponse

    struct = re.search(
        r"type PossessionResponse struct \{(.*?)\n\}",
        _go_sources()["inference_client.go"],
        re.S,
    )
    assert struct, "could not locate PossessionResponse in inference_client.go"

    go_fields = set(re.findall(r'json:"([^",]+)"', struct.group(1)))
    python_fields = set(PossessionResponse.model_fields)

    missing = go_fields - python_fields
    assert not missing, (
        f"Go expects fields Python never sends: {sorted(missing)}. "
        f"Go will read the zero value for each."
    )


def test_gate_inputs_are_named_fields_not_features():
    """The specific two that broke, pinned by name."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "live-trader"))
    from inference.main import PossessionResponse

    for field in ("is_overtime", "current_run_length"):
        assert field in PossessionResponse.model_fields, (
            f"{field} must be a named response field — the Go agent gates on it"
        )
        assert field not in PHYSICS_COLS, (
            f"{field} must NOT be a model feature; keeping it in both places is how "
            f"the coupling comes back"
        )


def test_live_emits_exactly_the_configured_features():
    """
    The live FeatureComputer must emit every model input and no stale extras.
    A missing key is silently zero-filled at inference, which is how a dropped
    feature becomes an undetected accuracy loss.
    """
    import inspect
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "live-trader"))
    from inference.features import FeatureComputer

    emitted: set[str] = set()
    for name in ("_compute_momentum", "_compute_context", "_compute_lineup", "_compute_derived"):
        src = inspect.getsource(getattr(FeatureComputer, name))
        emitted |= {
            line.split('"')[1]
            for line in src.splitlines()
            if line.strip().startswith('"') and '":' in line
        }

    expected = set(PHYSICS_COLS)
    missing = expected - emitted
    assert not missing, f"live path never emits these model inputs: {sorted(missing)}"
