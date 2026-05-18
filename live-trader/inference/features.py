"""
FeatureComputer — assembles the full 83-feature X vector for MMoE inference.

This is where X vector computation happens in the live system.
Takes a completed PossessionRow + current GameState + Kalshi market snapshot
and returns a feature dict aligned to models/mmoe/feature_config.ALL_FEATURE_COLS.

Feature groups:
    58 physics  — computed here from PossessionRow + GameState rolling caches
    11 pregame  — loaded once at game start, stored in GameState.pregame (static)
    14 market   — pre-computed by Go's KalshiRingBuffer, passed in as kalshi_snapshot

Critical invariant: all features are extracted from GameState BEFORE advance() is called.
This matches the shift(1) pattern in training — possession N sees state before possession N.

Streaming reimplementation of:
    models/features/momentum_features.py  → _compute_momentum()
    models/features/context_features.py   → _compute_context()
    models/features/lineup_features.py    → _compute_lineup()
    models/mmoe/dataset.py _add_derived_features() → _compute_derived()
"""

from typing import TYPE_CHECKING

from models.mmoe.feature_config import ALL_FEATURE_COLS, MARKET_COLS

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
        Assemble the full 83-feature dict.
        Keys must exactly match models/mmoe/feature_config.ALL_FEATURE_COLS.
        Missing keys default to 0.0 in MMoEPredictor.predict().
        """
        features: dict[str, float] = {}

        features.update(FeatureComputer._compute_momentum(possession, state))
        features.update(FeatureComputer._compute_context(possession, state))
        features.update(FeatureComputer._compute_lineup(possession, state))
        features.update(FeatureComputer._compute_derived(possession, state))
        features.update(state.pregame)  # 11 pregame features (static)
        features.update(dict(zip(MARKET_COLS, kalshi_snapshot)))  # 14 market features

        # Validate alignment (only in debug — remove for prod)
        assert set(features.keys()) >= set(ALL_FEATURE_COLS), (
            f"Missing features: {set(ALL_FEATURE_COLS) - set(features.keys())}"
        )

        return features

    # ── Physics feature groups ────────────────────────────────────────────────

    @staticmethod
    def _compute_momentum(possession: "PossessionRow", state: "GameState") -> dict[str, float]:
        """
        Streaming equivalent of momentum_features.add_momentum_features().
        All values read from state BEFORE this possession is processed.
        """
        # Last 5 / last 10 points per team — read from recent_possessions deque
        home_last_5  = _sum_points(state.recent_possessions, "home", n=5)
        away_last_5  = _sum_points(state.recent_possessions, "away", n=5)
        home_last_10 = _sum_points(state.recent_possessions, "home", n=10)
        away_last_10 = _sum_points(state.recent_possessions, "away", n=10)

        # Run state — read before update (shift(1) invariant)
        run_3pt_pct  = (state.run_3pt_count / state.run_points) if state.run_points > 0 else 0.0
        run_paint_pct = (state.run_paint_pts / state.run_points) if state.run_points > 0 else 0.0

        # Pace — mean of possession_durations deque
        pace_last_10 = (
            sum(state.possession_durations) / len(state.possession_durations)
            if state.possession_durations else state.pace_baseline
        )

        # Shot quality — last 5 scored possessions per team
        home_xppp        = _mean_xppp(state.home_scored_poss)
        away_xppp        = _mean_xppp(state.away_scored_poss)
        home_actual_ppp  = _mean_actual_ppp(state.home_scored_poss)
        away_actual_ppp  = _mean_actual_ppp(state.away_scored_poss)
        home_sustain     = _is_sustainable(state.home_scored_poss)
        away_sustain     = _is_sustainable(state.away_scored_poss)

        # TODO: shot_quality_trend (needs prev-window xPPP — deque maxlen=5 doesn't keep it)

        return {
            "home_points_last_5_poss":   float(home_last_5),
            "away_points_last_5_poss":   float(away_last_5),
            "home_points_last_10_poss":  float(home_last_10),
            "away_points_last_10_poss":  float(away_last_10),
            "current_run_team_encoded":  _encode_run_team(state.run_team),
            "current_run_length":        float(state.run_length),
            "current_run_points":        float(state.run_points),
            "current_run_3pt_pct":       run_3pt_pct,
            "current_run_paint_pct":     run_paint_pct,
            "pace_last_10_possessions":  pace_last_10,
            "pace_season_baseline":      state.pace_baseline,
            "home_scoring_sustainable":  float(home_sustain),
            "away_scoring_sustainable":  float(away_sustain),
            "home_xPPP_last_5":          home_xppp,
            "away_xPPP_last_5":          away_xppp,
            "home_actual_vs_expected_PPP": home_actual_ppp - home_xppp,
            "away_actual_vs_expected_PPP": away_actual_ppp - away_xppp,
            "home_shot_quality_trend":   0.0,    # TODO
            "away_shot_quality_trend":   0.0,    # TODO
            "shot_value":                float(possession.shot_value),
            "shot_distance":             possession.shot_distance,
        }

    @staticmethod
    def _compute_context(possession: "PossessionRow", state: "GameState") -> dict[str, float]:
        """
        Streaming equivalent of context_features.add_context_features().
        Foul counts, timeout counts, score context — all from accumulators in GameState.
        """
        score_diff      = possession.home_score - possession.away_score
        period          = possession.period
        clock_secs      = possession.game_clock_secs
        minutes_elapsed = ((period - 1) * 12) + (720.0 - clock_secs) / 60.0
        minutes_remaining = max(0.1, 48.0 - minutes_elapsed)

        is_blowout      = abs(score_diff) > 20
        is_garbage_time = is_blowout and period == 4 and clock_secs < 360

        # Team fouls → bonus state
        home_team_fouls_q = state.home_team_fouls.get(period, 0)
        away_team_fouls_q = state.away_team_fouls.get(period, 0)
        home_in_bonus     = home_team_fouls_q >= 5
        away_in_bonus     = away_team_fouls_q >= 5

        # Star foul trouble: read from player_fouls + star_players
        home_trouble_tier = _star_foul_trouble_tier(state, "home", period)
        away_trouble_tier = _star_foul_trouble_tier(state, "away", period)

        # Timeout context
        poss_since_to = _possessions_since_last_timeout(state)
        home_to_last3, away_to_last3 = _timeout_in_last_3(state, possession.possession_id)

        # TODO: timeout_on_opponent_run

        return {
            "score_diff":                     float(score_diff),
            "period":                         float(period),
            "minutes_into_game":              minutes_elapsed,
            "trailing_team_urgency":          abs(score_diff) / minutes_remaining,
            "comeback_probability_proxy":     (score_diff ** 2) / minutes_remaining,
            "q4_close_game":                  float(period == 4 and abs(score_diff) <= 5),
            "garbage_time_risk":              float(is_garbage_time),
            "home_team_fouls_q":              float(home_team_fouls_q),
            "away_team_fouls_q":              float(away_team_fouls_q),
            "home_cum_fouls":                 float(sum(state.home_team_fouls.values())),
            "away_cum_fouls":                 float(sum(state.away_team_fouls.values())),
            "home_in_bonus":                  float(home_in_bonus),
            "away_in_bonus":                  float(away_in_bonus),
            "home_fouls_until_bonus":         float(max(0, 5 - home_team_fouls_q)),
            "away_fouls_until_bonus":         float(max(0, 5 - away_team_fouls_q)),
            "home_star_in_foul_trouble":      float(home_trouble_tier > 0),
            "away_star_in_foul_trouble":      float(away_trouble_tier > 0),
            "home_star_on_court":             float(_has_star(state, "home")),
            "away_star_on_court":             float(_has_star(state, "away")),
            "was_foul":                       float(possession.was_foul),
            "was_sub":                        float(possession.was_sub),
            "had_shooting_foul":              float(possession.had_shooting_foul),
            "had_personal_foul":              float(possession.had_personal_foul),
            "home_sub_count":                 float(state.home_sub_count),
            "away_sub_count":                 float(state.away_sub_count),
            "possessions_since_last_timeout": float(poss_since_to),
            "home_called_timeout_in_last_3_poss": float(home_to_last3),
            "away_called_timeout_in_last_3_poss": float(away_to_last3),
            "home_full_timeouts_remaining":   float(4 - state.home_timeouts_used),
            "away_full_timeouts_remaining":   float(4 - state.away_timeouts_used),
        }

    @staticmethod
    def _compute_lineup(possession: "PossessionRow", state: "GameState") -> dict[str, float]:
        """
        Streaming equivalent of lineup_features.add_lineup_features().
        All values from dict lookups into pre-loaded lineup_ratings and player_apm.
        """
        home_net = state.lineup_ratings.get(possession.home_lineup_id, 0.0)
        away_net = state.lineup_ratings.get(possession.away_lineup_id, 0.0)

        home_apms = [state.player_apm.get(p, 0.0) for p in state.home_lineup]
        away_apms = [state.player_apm.get(p, 0.0) for p in state.away_lineup]

        home_best  = max(home_apms) if home_apms else 0.0
        away_best  = max(away_apms) if away_apms else 0.0
        home_worst = min(home_apms) if home_apms else 0.0
        away_worst = min(away_apms) if away_apms else 0.0

        # TODO: home_lineup_sample_size from lineup_ratings table
        # TODO: off_court_best_apm (best player NOT currently on court)

        return {
            "home_lineup_net_rating":    home_net,
            "away_lineup_net_rating":    away_net,
            "lineup_net_rating_delta":   home_net - away_net,
            "home_lineup_sample_size":   0.0,   # TODO
            "away_lineup_sample_size":   0.0,   # TODO
            "home_lineup_just_changed":  float(state.home_lineup != state.prev_home_lineup),
            "away_lineup_just_changed":  float(state.away_lineup != state.prev_away_lineup),
        }

    @staticmethod
    def _compute_derived(possession: "PossessionRow", state: "GameState") -> dict[str, float]:
        """
        Minimal derived features from dataset.py _add_derived_features().
        These don't fit neatly into the other groups.
        """
        # back_to_back loaded once at game start
        return {
            "home_back_to_back": float(state.home_b2b),
            "away_back_to_back": float(state.away_b2b),
        }


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
    """Mean points-per-scoring-possession over the last 5 scored possessions.

    Matches offline momentum_features.add_momentum_features() — there, actual_vs_expected_PPP
    is `pd.Series(home_pts).where(home_scored_mask).shift(1).rolling(5).mean() - home_xppp_last5`
    where the masked + rolled value is the mean points-per-scoring-possession over the last 5
    home-scored possessions. The deque (maxlen=5, only scoring possessions ever appended) gives
    us the same window. Returns 0.0 to match the offline NaN→fillna(1.0) at game start, since
    pairing 1.0 here with xppp=0.0 would emit actual_vs_expected = 1.0 spuriously.
    """
    if not scored_poss:
        return 0.0
    return sum(p.points for p in scored_poss) / len(scored_poss)


def _is_sustainable(scored_poss: "deque") -> bool:
    """Sustainable if majority of recent scoring came from 3pt or paint."""
    if not scored_poss:
        return False
    paint_or_3 = sum(1 for p in scored_poss if p.was_paint or p.shot_value == 3)
    return paint_or_3 / len(scored_poss) > 0.5


def _encode_run_team(run_team: str) -> float:
    return {"home": 1.0, "away": -1.0, "": 0.0}.get(run_team, 0.0)


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
    recent = state.timeouts[-10:]   # only scan last 10 timeouts
    home_to = any(
        t["team"] == "home" and (possession_id - t["possession_id"]) <= 3
        for t in recent
    )
    away_to = any(
        t["team"] == "away" and (possession_id - t["possession_id"]) <= 3
        for t in recent
    )
    return home_to, away_to
