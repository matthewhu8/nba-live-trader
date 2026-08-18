"""
FeatureComputer assembles the 58-feature vector for MMoE inference from a completed
PossessionRow, the current GameState and the Kalshi market snapshot. Keys align to
models/mmoe/feature_config.ALL_FEATURE_COLS.

    33 physics  computed here from the possession and GameState's rolling caches
    11 pregame  loaded once at game start into GameState.pregame
    14 market   computed by Go's ring buffer and passed in as kalshi_snapshot

Every feature is read from GameState before advance() runs, matching the shift(1)
pattern in training: possession N sees only state from before possession N.

This is the streaming reimplementation of the offline builders in
models/features/{momentum,context,lineup}_features.py.
"""

from typing import TYPE_CHECKING

from models.features import transforms as T
from models.mmoe.feature_config import ALL_FEATURE_COLS, MARKET_COLS

# Every formula here comes from models/features/transforms.py, which the offline
# builder also calls. Never reintroduce a local constant or formula: a live copy that
# drifts from the trained one is a silent train/serve divergence, not an error.

if TYPE_CHECKING:
    from inference.game_state import GameState
    from inference.possession import PossessionRow


class FeatureComputer:

    @staticmethod
    def compute(
        possession: "PossessionRow",
        state: "GameState",
        kalshi_snapshot: list[float],  # 14 floats in MARKET_COLS order
    ) -> dict[str, float]:
        """
        Assemble the 58-feature dict, whose keys must match ALL_FEATURE_COLS exactly.

        The assertion below earns its keep because MMoEPredictor.predict() defaults a
        missing key to 0.0: a zero-filled feature costs accuracy with no error raised.
        """
        features: dict[str, float] = {}

        features.update(FeatureComputer._compute_momentum(possession, state))
        features.update(FeatureComputer._compute_context(possession, state))
        features.update(FeatureComputer._compute_lineup(possession, state))
        features.update(FeatureComputer._compute_derived(possession, state))
        features.update(state.pregame)  # 11 pregame features (static)
        features.update(dict(zip(MARKET_COLS, kalshi_snapshot)))  # 14 market features

        assert set(features.keys()) >= set(ALL_FEATURE_COLS), (
            f"Missing features: {set(ALL_FEATURE_COLS) - set(features.keys())}"
        )

        return features

    # ── Physics feature groups ────────────────────────────────────────────────

    @staticmethod
    def _compute_momentum(possession: "PossessionRow", state: "GameState") -> dict[str, float]:
        """Streaming equivalent of momentum_features.add_momentum_features()."""
        home_last_5  = _sum_points(state.recent_possessions, "home", n=5)
        away_last_5  = _sum_points(state.recent_possessions, "away", n=5)
        home_last_10 = _sum_points(state.recent_possessions, "home", n=10)
        away_last_10 = _sum_points(state.recent_possessions, "away", n=10)

        # These percentages are fractions of run POINTS, not of shots, matching training.
        run_3pt_pct  = (3 * state.run_3pt_count / state.run_points) if state.run_points > 0 else 0.0
        run_paint_pct = (state.run_paint_pts / state.run_points) if state.run_points > 0 else 0.0

        # The empty-deque fallback must be PACE_FALLBACK_SECS to match the offline
        # fillna, not state.pace_baseline, which is the pregame constant.
        pace_last_10 = (
            sum(state.possession_durations) / len(state.possession_durations)
            if state.possession_durations else T.PACE_FALLBACK_SECS
        )

        # Shot quality over the last 5 scored possessions per team.
        home_xppp        = _mean_xppp(state.home_scored_poss)
        away_xppp        = _mean_xppp(state.away_scored_poss)
        home_actual_ppp  = _mean_actual_ppp(state.home_scored_poss)
        away_actual_ppp  = _mean_actual_ppp(state.away_scored_poss)

        # A raw xPPP difference rather than its sign, so the model can pick its own
        # threshold from the magnitude.
        home_traj = (home_xppp - state.home_prev_xppp) if state.home_prev_xppp > 0.0 else 0.0
        away_traj = (away_xppp - state.away_prev_xppp) if state.away_prev_xppp > 0.0 else 0.0

        # Expanding within-game mean, distinct from the pregame pace_baseline.
        pace_game_to_date = (
            state.pace_duration_sum / state.pace_duration_count
            if state.pace_duration_count > 0
            else T.PACE_FALLBACK_SECS
        )
        expected_pace = state.pregame.get("expected_pace", 0.0) or T.PACE_FALLBACK_SECS

        return {
            "shot_value":    float(possession.shot_value),
            "shot_distance": possession.shot_distance,
            # Runs
            "run_signed_points": float(
                T.run_signed_points(T.encode_run_team(state.run_team), state.run_points)
            ),
            "run_efficiency": float(T.run_efficiency(state.run_points, state.run_length)),
            "run_fragility":  float(T.run_fragility(run_3pt_pct, run_paint_pct)),
            # Recent scoring
            "swing_5": float(T.swing_5(home_last_5, away_last_5)),
            "swing_accel": float(
                T.swing_accel(home_last_5, away_last_5, home_last_10, away_last_10)
            ),
            # Pace
            "pace_ref": float(
                T.pace_ref(pace_game_to_date, expected_pace, state.possession_count)
            ),
            "pace_surprise": float(
                T.pace_surprise(
                    pace_last_10, pace_game_to_date, expected_pace, state.possession_count
                )
            ),
            # Shot quality
            "xppp_edge": float(T.xppp_edge(home_xppp, away_xppp)),
            "luck_edge": float(
                T.luck_edge(home_actual_ppp - home_xppp, away_actual_ppp - away_xppp)
            ),
            "quality_trend_edge": float(T.edge(home_traj, away_traj)),
        }

    @staticmethod
    def _compute_context(possession: "PossessionRow", state: "GameState") -> dict[str, float]:
        """Streaming equivalent of context_features.add_context_features()."""
        score_diff = possession.home_score - possession.away_score
        period     = possession.period
        clock_secs = possession.game_clock_secs

        home_team_fouls_q = state.home_team_fouls.get(period, 0)
        away_team_fouls_q = state.away_team_fouls.get(period, 0)

        home_trouble_tier = _star_foul_trouble_tier(state, "home", period)
        away_trouble_tier = _star_foul_trouble_tier(state, "away", period)

        poss_since_to = _possessions_since_last_timeout(state)
        home_to_last3, away_to_last3 = _timeout_in_last_3(state, possession.possession_id)

        return {
            # Score x time
            "score_diff":        float(score_diff),
            "lead_z":            float(T.lead_z(score_diff, period, clock_secs)),
            "time_leverage":     float(T.time_leverage(period, clock_secs)),
            "garbage_time_risk": float(T.garbage_time_risk(score_diff, period, clock_secs)),
            # Foul state
            "team_foul_edge":         float(T.edge(home_team_fouls_q, away_team_fouls_q)),
            "home_fouls_until_bonus": float(T.fouls_until_bonus(home_team_fouls_q)),
            "away_fouls_until_bonus": float(T.fouls_until_bonus(away_team_fouls_q)),
            "star_trouble_edge":  float(T.edge(home_trouble_tier > 0, away_trouble_tier > 0)),
            "star_on_court_edge": float(
                T.edge(_has_star(state, "home"), _has_star(state, "away"))
            ),
            # Event context
            "was_sub":           float(possession.was_sub),
            "had_shooting_foul": float(possession.had_shooting_foul),
            "had_personal_foul": float(possession.had_personal_foul),
            "sub_count_edge":    float(T.edge(state.home_sub_count, state.away_sub_count)),
            # Timeouts
            "poss_since_timeout":  float(T.timeout_possessions_bounded(poss_since_to)),
            "no_timeout_yet":      float(T.no_timeout_yet(poss_since_to)),
            "timeout_called_edge": float(T.edge(home_to_last3, away_to_last3)),
            "timeouts_remaining_edge": float(
                T.edge(
                    T.full_timeouts_remaining(state.home_timeouts_used),
                    T.full_timeouts_remaining(state.away_timeouts_used),
                )
            ),
        }

    @staticmethod
    def _compute_lineup(possession: "PossessionRow", state: "GameState") -> dict[str, float]:
        """Streaming equivalent of lineup_features.add_lineup_features()."""
        home_net = state.lineup_ratings.get(possession.home_lineup_id, 0.0)
        away_net = state.lineup_ratings.get(possession.away_lineup_id, 0.0)

        home_samples = state.lineup_sample_sizes.get(possession.home_lineup_id, 0.0)
        away_samples = state.lineup_sample_sizes.get(possession.away_lineup_id, 0.0)

        home_changed = state.home_lineup != state.prev_home_lineup
        away_changed = state.away_lineup != state.prev_away_lineup

        return {
            "lineup_net_rating_delta": float(T.edge(home_net, away_net)),
            "lineup_confidence":       float(T.lineup_confidence(home_samples, away_samples)),
            "lineup_changed_edge":     float(T.edge(home_changed, away_changed)),
            "lineup_changed_any":      float(T.any_flag(home_changed, away_changed)),
        }

    @staticmethod
    def _compute_derived(possession: "PossessionRow", state: "GameState") -> dict[str, float]:
        """Hook for features belonging to no other group. Currently empty."""
        return {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sum_points(recent: "deque", team: str, n: int) -> int:
    """Sum points scored by `team` in the last `n` possessions."""
    window = list(recent)[-n:]
    return sum(p.points for p in window if p.team_scored == team)


def _mean_xppp(scored_poss: "deque") -> float:
    if not scored_poss:
        return 0.0
    return sum(p.xppp for p in scored_poss) / len(scored_poss)


def _mean_actual_ppp(scored_poss: "deque") -> float:
    """Mean points per scoring possession over the last 5 scored possessions.

    The deque holds only scoring possessions, so it gives the same window as the
    masked rolling mean offline. Returns 0.0 rather than the offline fill of 1.0 at
    game start, because pairing 1.0 with an xppp of 0.0 emits a spurious edge.
    """
    if not scored_poss:
        return 0.0
    return sum(p.points for p in scored_poss) / len(scored_poss)


def _has_star(state: "GameState", team: str) -> bool:
    lineup = state.home_lineup if team == "home" else state.away_lineup
    return any(state.star_players.get(p, 0) >= 1 for p in lineup)


def _star_foul_trouble_tier(state: "GameState", team: str, period: int) -> int:
    lineup = state.home_lineup if team == "home" else state.away_lineup
    threshold = {1: 2, 2: 3, 3: 3, 4: 4}.get(period, 2)
    for player_id in lineup:
        tier = state.star_players.get(player_id, 0)
        if tier > 0 and state.player_fouls.get(player_id, 0) >= threshold:
            return tier
    return 0


def _possessions_since_last_timeout(state: "GameState") -> int:
    if not state.timeouts:
        return 999
    last_poss_id = state.timeouts[-1]["possession_id"]
    return state.possession_count - last_poss_id


def _timeout_in_last_3(state: "GameState", possession_id: int) -> tuple[bool, bool]:
    recent = state.timeouts[-10:]
    home_to = any(
        t["team"] == "home" and (possession_id - t["possession_id"]) <= 3
        for t in recent
    )
    away_to = any(
        t["team"] == "away" and (possession_id - t["possession_id"]) <= 3
        for t in recent
    )
    return home_to, away_to
