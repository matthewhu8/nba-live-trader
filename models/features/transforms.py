"""
Shared feature transforms — the single source of truth for every derived quantity
that both the offline builder and the live streamer must produce identically.

Why this module exists
----------------------
Every model input is computed twice: offline in `models/features/*.py` and
`models/mmoe/dataset.py` (pandas, whole-game, vectorized), and online in
`live-trader/inference/features.py` (streaming, one possession at a time).
Four silent train/live divergences were traced to that duplication. Any formula
that lives in only one of those files is a future divergence.

Everything here is a pure function of numbers. Because the bodies are numpy
ufunc expressions, the same function serves both callers unchanged:
arrays/Series offline, plain floats online.

Rules for anything added here:
  - No pandas, no I/O, no state. Numbers in, numbers out.
  - Never branch on `isinstance`. Use numpy ops that work on scalars and arrays.
  - Every constant that both paths depend on is named here, not duplicated.
"""

from typing import Union

import numpy as np

# Accepts plain floats (live path) or numpy arrays / pandas Series (offline path).
# Return type always matches the input type because every body is a ufunc chain.
Numeric = Union[float, int, np.ndarray]


# ── Canonical constants ───────────────────────────────────────────────────────
# These were previously duplicated (and disagreed) across the offline and live
# paths. They now exist exactly once.

REGULATION_MINUTES: float = 48.0
PERIOD_MINUTES: float = 12.0
PERIOD_SECONDS: float = 720.0
REGULATION_PERIODS: int = 4

# Floor applied to minutes-remaining before it is used as a divisor. Training has
# always used 1.0; live used 0.1, which produced up to 10x divergence in the final
# minute. 1.0 is canonical.
MIN_MINUTES_REMAINING: float = 1.0

# Lower floor used only by `lead_z`, which divides by sqrt(m) rather than m and so
# stays well-conditioned closer to the buzzer. Keeps late-game resolution that a
# 1.0 floor would flatten.
LEAD_Z_MIN_MINUTES: float = 0.5

# Garbage-time sigmoid: centred at a 15-point margin, 5-point scale.
GARBAGE_MARGIN_CENTER: float = 15.0
GARBAGE_MARGIN_SCALE: float = 5.0

# Pace shrinkage: weight on the within-game mean reaches 0.5 at this many elapsed
# possessions, and plateaus at PACE_MAX_WEIGHT so the pregame prior always retains
# a floor of influence.
PACE_SHRINK_POSSESSIONS: float = 40.0
PACE_MAX_WEIGHT: float = 0.85
PACE_FALLBACK_SECS: float = 15.0

# possessions_since_last_timeout: 999 was used as a "no timeout yet" sentinel and
# destroyed the column under StandardScaler (9.4% of rows at z=+3.10, compressing
# the real 0-70 range into a 0.24-wide z band). The count is now capped and the
# sentinel is carried by a separate binary flag.
NO_TIMEOUT_SENTINEL: int = 999
TIMEOUT_POSSESSION_CAP: float = 30.0

FULL_TIMEOUTS_PER_GAME: int = 4
BONUS_FOUL_THRESHOLD: int = 5

# Expected points per possession by shot type (league averages). Previously these
# lived only in momentum_features.py; the live path had its own copy.
XPPP_FREE_THROW: float = 0.75
XPPP_3PT: float = 1.05
XPPP_PAINT: float = 1.20
XPPP_MIDRANGE: float = 0.80
PAINT_DISTANCE_FT: float = 8.0


def shot_xppp(shot_value: Numeric, shot_distance: Numeric) -> Numeric:
    """Expected points per possession implied by a single shot's type."""
    value = np.asarray(shot_value, dtype=float)
    distance = np.asarray(shot_distance, dtype=float)
    return np.where(
        value == 1.0,
        XPPP_FREE_THROW,
        np.where(
            value == 3.0,
            XPPP_3PT,
            np.where(distance < PAINT_DISTANCE_FT, XPPP_PAINT, XPPP_MIDRANGE),
        ),
    )


# ── Clock ─────────────────────────────────────────────────────────────────────

def minutes_into_game(period: Numeric, game_clock_secs: Numeric) -> Numeric:
    """Elapsed regulation-equivalent minutes since tip-off."""
    period_arr = np.asarray(period, dtype=float)
    clock_arr = np.minimum(np.asarray(game_clock_secs, dtype=float), PERIOD_SECONDS)
    return (period_arr - 1.0) * PERIOD_MINUTES + (PERIOD_SECONDS - clock_arr) / 60.0


def minutes_remaining_raw(period: Numeric, game_clock_secs: Numeric) -> Numeric:
    """
    Minutes left in regulation, unfloored.

    Overtime periods return only the current period's clock, which is correct:
    `4 - period` clamps at zero so OT never produces a negative remainder. The
    live path previously computed `48 - elapsed`, which goes negative in OT and
    pinned every OT possession to the same floor value.
    """
    period_arr = np.asarray(period, dtype=float)
    periods_left = np.maximum(REGULATION_PERIODS - period_arr, 0.0)
    return periods_left * PERIOD_MINUTES + np.asarray(game_clock_secs, dtype=float) / 60.0


def minutes_remaining(
    period: Numeric,
    game_clock_secs: Numeric,
    floor: float = MIN_MINUTES_REMAINING,
) -> Numeric:
    """Minutes left in regulation, floored so it is safe as a divisor."""
    return np.maximum(minutes_remaining_raw(period, game_clock_secs), floor)


def time_leverage(period: Numeric, game_clock_secs: Numeric) -> Numeric:
    """
    Nonlinear game clock: 0.0 at tip-off, 1.0 at the buzzer.

    Uses a sqrt scale so a minute late in the game occupies more of the input
    range than a minute early. Five minutes of clock spans 0.061 of the range at
    10-15 minutes elapsed but 0.158 at 40-45 minutes, a 2.6x resolution gain
    exactly where price sensitivity is highest.
    """
    raw = minutes_remaining_raw(period, game_clock_secs)
    ratio = np.clip(raw / REGULATION_MINUTES, 0.0, 1.0)
    return 1.0 - np.sqrt(ratio)


# ── Score leverage ────────────────────────────────────────────────────────────

def lead_z(score_diff: Numeric, period: Numeric, game_clock_secs: Numeric) -> Numeric:
    """
    Signed lead in units of remaining scoring randomness: `d / sqrt(m)`.

    If in-game scoring is approximately a random walk, the standard deviation of
    the remaining margin swing scales with sqrt(minutes remaining), so this is the
    natural "how safe is this lead" quantity and it is the standard win-probability
    construction.

    Replaces `trailing_team_urgency` (|d|/m) and `comeback_probability_proxy`
    (d^2/m), both of which discarded the sign: winning by 6 and losing by 6 with a
    minute left produced numerically identical inputs for opposite trades.
    """
    m = minutes_remaining(period, game_clock_secs, floor=LEAD_Z_MIN_MINUTES)
    return np.asarray(score_diff, dtype=float) / np.sqrt(m)


def garbage_time_risk(
    score_diff: Numeric, period: Numeric, game_clock_secs: Numeric
) -> Numeric:
    """
    Continuous 0-1 proximity to garbage time.

    Note the time factor uses the UNFLOORED minutes remaining, matching the
    original training definition. The live path previously emitted a hard binary
    requiring period == 4 and clock < 360, so a 25-point Q2 lead scored 0.33 in
    training and 0.00 live.
    """
    margin = np.abs(np.asarray(score_diff, dtype=float))
    sigmoid_in = (margin - GARBAGE_MARGIN_CENTER) / GARBAGE_MARGIN_SCALE
    sigmoid_out = 1.0 / (1.0 + np.exp(-np.clip(sigmoid_in, -20.0, 20.0)))
    raw_remaining = minutes_remaining_raw(period, game_clock_secs)
    time_factor = np.clip(1.0 - raw_remaining / REGULATION_MINUTES, 0.0, 1.0)
    return sigmoid_out * time_factor


# ── Scoring runs ──────────────────────────────────────────────────────────────

def run_signed_points(run_team_encoded: Numeric, run_points: Numeric) -> Numeric:
    """
    Run size with direction folded in: +N when home is on the run, -N when away.

    Previously the direction (`current_run_team_encoded`) and the magnitude
    (`current_run_points`) were separate columns, requiring the MLP to learn their
    product from the joint rows alone (15,383 in Head B's train split as of
    2026-08-11; see docs/DATA_INVENTORY.md).
    """
    return np.asarray(run_team_encoded, dtype=float) * np.asarray(run_points, dtype=float)


def run_efficiency(run_points: Numeric, run_length: Numeric) -> Numeric:
    """
    Points per possession within the current run.

    `run_length` and `run_points` correlate heavily (a k-possession run is roughly
    2.3k points), so feeding both raw is largely duplicated variance. The ratio is
    the orthogonal component: how hot the run is, independent of how long.
    """
    length = np.maximum(np.asarray(run_length, dtype=float), 1.0)
    return np.asarray(run_points, dtype=float) / length


def run_fragility(run_3pt_pct: Numeric, run_paint_pct: Numeric) -> Numeric:
    """
    Positive when the run is jumper-driven (likely to revert), negative when it is
    paint-driven (more repeatable).
    """
    return np.asarray(run_3pt_pct, dtype=float) - np.asarray(run_paint_pct, dtype=float)


# ── Recent scoring ────────────────────────────────────────────────────────────

def swing_5(home_points_last_5: Numeric, away_points_last_5: Numeric) -> Numeric:
    """Signed margin swing over the last 5 possessions."""
    return np.asarray(home_points_last_5, dtype=float) - np.asarray(away_points_last_5, dtype=float)


def swing_accel(
    home_points_last_5: Numeric,
    away_points_last_5: Numeric,
    home_points_last_10: Numeric,
    away_points_last_10: Numeric,
) -> Numeric:
    """
    Recent-half swing minus prior-half swing.

    The 5- and 10-possession windows are nested and correlate ~0.7, so feeding both
    levels is feeding the same trend twice. Expressing the wider window as an
    acceleration decorrelates them.
    """
    recent = swing_5(home_points_last_5, away_points_last_5)
    full = np.asarray(home_points_last_10, dtype=float) - np.asarray(away_points_last_10, dtype=float)
    prior = full - recent
    return recent - prior


# ── Shot quality ──────────────────────────────────────────────────────────────

def xppp_edge(home_xppp_last_5: Numeric, away_xppp_last_5: Numeric) -> Numeric:
    """Shot-selection quality differential, home minus away."""
    return np.asarray(home_xppp_last_5, dtype=float) - np.asarray(away_xppp_last_5, dtype=float)


def luck_edge(
    home_actual_vs_expected: Numeric, away_actual_vs_expected: Numeric
) -> Numeric:
    """
    Signed overperformance against shot quality, home minus away.

    The closest feature in the set to the core thesis: a team scoring above its
    shot quality is a mean-reversion candidate, and the market is watching the
    scoreboard rather than the shot chart.
    """
    return (
        np.asarray(home_actual_vs_expected, dtype=float)
        - np.asarray(away_actual_vs_expected, dtype=float)
    )


def quality_trend_edge(
    home_xppp_last_5: Numeric,
    home_xppp_prev_5: Numeric,
    away_xppp_last_5: Numeric,
    away_xppp_prev_5: Numeric,
) -> Numeric:
    """
    Differential change in shot quality, as a raw magnitude.

    Deliberately not `np.sign()`. The original columns collapsed to {-1, 0, +1},
    so a shot-quality collapse and an imperceptible drift produced identical
    inputs. The net can pick its own threshold; it cannot recover a magnitude that
    was already discarded.
    """
    home_trend = np.asarray(home_xppp_last_5, dtype=float) - np.asarray(home_xppp_prev_5, dtype=float)
    away_trend = np.asarray(away_xppp_last_5, dtype=float) - np.asarray(away_xppp_prev_5, dtype=float)
    return home_trend - away_trend


# ── Pace ──────────────────────────────────────────────────────────────────────

def pace_shrink_weight(elapsed_possessions: Numeric) -> Numeric:
    """
    Weight on the within-game pace mean, shrinking toward the pregame prior early
    and plateauing at PACE_MAX_WEIGHT so the prior never drops out entirely.
    """
    elapsed = np.maximum(np.asarray(elapsed_possessions, dtype=float), 0.0)
    weight = elapsed / (elapsed + PACE_SHRINK_POSSESSIONS)
    return np.minimum(weight, PACE_MAX_WEIGHT)


def pace_ref(
    pace_game_to_date: Numeric,
    expected_pace: Numeric,
    elapsed_possessions: Numeric,
) -> Numeric:
    """
    Reference pace: an empirical-Bayes blend of this game's realized pace and the
    pregame matchup prior.

    Early in the game the in-game sample is tiny and the prior dominates; late it
    plateaus at 0.85 so the prior keeps a floor of influence. `expected_pace` is the
    existing pregame feature (mean of the two teams' trailing avg_secs_per_poss),
    already computed as-of prior games only.
    """
    w = pace_shrink_weight(elapsed_possessions)
    return w * np.asarray(pace_game_to_date, dtype=float) + (1.0 - w) * np.asarray(
        expected_pace, dtype=float
    )


def pace_surprise(
    pace_last_10: Numeric,
    pace_game_to_date: Numeric,
    expected_pace: Numeric,
    elapsed_possessions: Numeric,
) -> Numeric:
    """How much faster or slower the last 10 possessions ran than the reference."""
    reference = pace_ref(pace_game_to_date, expected_pace, elapsed_possessions)
    return np.asarray(pace_last_10, dtype=float) - reference


# ── Timeouts ──────────────────────────────────────────────────────────────────

def timeout_possessions_bounded(possessions_since_last_timeout: Numeric) -> Numeric:
    """
    Possessions since the last timeout, capped so the sentinel cannot dominate the
    column's scale. Rows carrying the sentinel are clamped to the cap; the fact
    that no timeout has occurred is carried separately by `no_timeout_yet`.
    """
    raw = np.asarray(possessions_since_last_timeout, dtype=float)
    return np.minimum(raw, TIMEOUT_POSSESSION_CAP)


def no_timeout_yet(possessions_since_last_timeout: Numeric) -> Numeric:
    """1.0 when no timeout has been called yet this game, else 0.0."""
    raw = np.asarray(possessions_since_last_timeout, dtype=float)
    return (raw >= float(NO_TIMEOUT_SENTINEL)).astype(float)


# ── Home/away folds ───────────────────────────────────────────────────────────
# Each of these replaces a home column and an away column with one signed
# quantity. `lineup_net_rating_delta` already followed this pattern; these apply
# it consistently.

def edge(home_value: Numeric, away_value: Numeric) -> Numeric:
    """Generic signed home-minus-away fold."""
    return np.asarray(home_value, dtype=float) - np.asarray(away_value, dtype=float)


def lineup_confidence(
    home_sample_size: Numeric, away_sample_size: Numeric
) -> Numeric:
    """
    Log-scaled confidence in the lineup ratings, driven by the weaker of the two
    sample sizes. A rating pair is only as trustworthy as its thinner side.
    """
    weakest = np.minimum(
        np.asarray(home_sample_size, dtype=float),
        np.asarray(away_sample_size, dtype=float),
    )
    return np.log1p(np.maximum(weakest, 0.0))


def any_flag(home_flag: Numeric, away_flag: Numeric) -> Numeric:
    """1.0 when either side's flag is set."""
    home = np.asarray(home_flag, dtype=float)
    away = np.asarray(away_flag, dtype=float)
    return np.maximum(np.minimum(home, 1.0), np.minimum(away, 1.0))


# ── Bonus / foul state ────────────────────────────────────────────────────────

def fouls_until_bonus(team_fouls_this_quarter: Numeric) -> Numeric:
    """Fouls remaining before the opponent shoots bonus free throws."""
    fouls = np.asarray(team_fouls_this_quarter, dtype=float)
    return np.clip(BONUS_FOUL_THRESHOLD - fouls, 0.0, float(BONUS_FOUL_THRESHOLD))


def full_timeouts_remaining(timeouts_used: Numeric) -> Numeric:
    """Full timeouts a team has left, clamped to the legal range."""
    used = np.asarray(timeouts_used, dtype=float)
    return np.clip(FULL_TIMEOUTS_PER_GAME - used, 0.0, float(FULL_TIMEOUTS_PER_GAME))


def encode_run_team(run_team: str) -> float:
    """Map the run-team label to a sign. Scalar-only: this one takes a string."""
    return {"home": 1.0, "away": -1.0}.get(run_team, 0.0)
