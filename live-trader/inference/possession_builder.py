"""
PossessionBuilder — raw NBA CDN event → PossessionRow

Converts a single NBAEvent dict (as received from Go's CDN poller) into a
typed PossessionRow, or returns None if the event does not complete a possession.

Possession boundary rules (must match nba_api_client.py exactly):
    A possession ends on:
      - Made field goal
      - Turnover
      - Defensive rebound after missed shot
      - Last free throw in a sequence (if missed → defensive rebound completes it)
    A possession does NOT end on:
      - Offensive rebound (same team retains possession)
      - Mid-possession foul (ball still in play)
      - Non-final free throws

The state machine must match nba_api_client.py's parsing exactly.
Validation: replay 3 historical games → diff against possession_flat in MotherDuck.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from inference.game_state import GameState


# xPPP lookup table by shot area (matches momentum_features._shot_xppp)
# Keys match NBA CDN `area` field values
_XPPP_BY_AREA: dict[str, float] = {
    "Restricted Area":              1.30,
    "In The Paint (Non-RA)":        0.82,
    "Mid-Range":                    0.78,
    "Left Corner 3":                1.07,
    "Right Corner 3":               1.07,
    "Above the Break 3":            1.05,
    "Backcourt":                    0.00,
}


@dataclass
class PossessionRow:
    """
    One completed possession — the unit of feature computation.
    Field names must match possession_flat schema from MotherDuck.
    """
    possession_id:    int
    period:           int
    game_clock_secs:  float   # seconds remaining in period at possession end
    team_scored:      str     # "home" | "away" | "none"
    points:           int     # 0, 1, 2, or 3
    shot_value:       int     # 2 or 3 (0 if no shot)
    shot_distance:    float   # feet (0 if no shot)
    shot_area:        str     # NBA CDN area string
    was_contested:    bool
    was_foul:         bool
    had_shooting_foul: bool
    had_personal_foul: bool
    was_sub:          bool    # substitution occurred during this possession
    home_score:       int
    away_score:       int
    home_lineup_id:   str     # MD5 hash of sorted player IDs
    away_lineup_id:   str
    xppp:             float   # expected PPP from shot selection
    is_paint:         bool    # shot from paint area


class PossessionBuilder:

    @staticmethod
    def parse(raw_event: dict, state: "GameState") -> Optional[PossessionRow]:
        """
        Attempt to build a PossessionRow from a raw NBA CDN event.
        Returns None if the event does not complete a possession.
        Side effects (fouls, subs, timeouts) are NOT handled here —
        those are handled by state.update_from_event() in main.py.
        """
        action_type = raw_event.get("actionType", "")

        if action_type in ("2pt", "3pt"):
            return PossessionBuilder._from_field_goal(raw_event, state)
        if action_type == "turnover":
            return PossessionBuilder._from_turnover(raw_event, state)
        if action_type == "rebound" and raw_event.get("subType") == "defensive":
            return PossessionBuilder._from_defensive_rebound(raw_event, state)
        # Free throws: only the final FT completes a possession (if missed → DR follows)
        # Offensive rebounds: return None (possession continues)
        return None

    @staticmethod
    def _from_field_goal(event: dict, state: "GameState") -> PossessionRow:
        # TODO: parse team, points (2 or 3), shot area, distance, contested flag
        # TODO: compute lineup IDs from state.home_lineup / state.away_lineup
        # TODO: compute xppp from _XPPP_BY_AREA
        raise NotImplementedError

    @staticmethod
    def _from_turnover(event: dict, state: "GameState") -> PossessionRow:
        # TODO: parse team (who turned it over), zero points
        raise NotImplementedError

    @staticmethod
    def _from_defensive_rebound(event: dict, state: "GameState") -> PossessionRow:
        # TODO: previous team's possession ended with missed shot — zero points
        raise NotImplementedError

    @staticmethod
    def _lineup_id(player_ids: list[int]) -> str:
        """MD5 hash of sorted player IDs — matches nba_api_client._lineup_id()."""
        import hashlib
        key = "_".join(str(p) for p in sorted(player_ids))
        return hashlib.md5(key.encode()).hexdigest()[:12]

    @staticmethod
    def _xppp(area: str, contested: bool) -> float:
        base = _XPPP_BY_AREA.get(area, 0.85)
        return base * (0.88 if contested else 1.0)
