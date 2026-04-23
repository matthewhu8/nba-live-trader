"""
PossessionBuilder — converts raw NBA CDN events into PossessionRow structs.

This is the streaming state machine equivalent of nba_api_client.parse_game_to_events().
It MUST produce the same possession boundaries as the training data pipeline.

Possession boundaries (matching nba_api_client.py exactly):
  Ends on:  made field goal, turnover, defensive rebound, last free throw in sequence
  Continues: offensive rebound, mid-possession foul, non-final free throws

CDN action types (different from nba_api):
  "2pt" / "3pt"      — field goal attempt (shotResult: "Made" | "Missed")
  "freethrow"        — free throw (shotResult: "Made" | "Missed")
  "rebound"          — rebound (subType: "offensive" | "defensive")
  "turnover"         — turnover
  "substitution"     — substitution (subType: "in" | "out")
  "foul"             — foul (subType: "personal", "shooting", "technical", etc.)
  "timeout"          — timeout
  "period"           — period start/end
  "jumpball"         — jump ball (determines first possession)
  "violation"        — goaltending or other violation

Validation: replay historical games through this builder and diff output vs
possession_flat in MotherDuck before going live.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from inference.game_state import GameState


# Expected PPP constants — must match momentum_features._XPPP exactly
_XPPP_FREE_THROW = 0.75
_XPPP_3PT        = 1.05
_XPPP_PAINT      = 1.20   # shot_distance < 8 ft
_XPPP_MIDRANGE   = 0.80


@dataclass
class PossessionRow:
    """
    One completed possession — the unit of feature computation.
    Field names match possession_flat schema from MotherDuck.
    """
    possession_id:     int
    period:            int
    game_clock_secs:   float
    team_scored:       str     # "home" | "away" | ""
    points:            int     # 0, 1, 2, or 3
    shot_value:        int     # 2 or 3 (0 if no shot / turnover / stop)
    shot_distance:     float
    shot_area:         str
    was_contested:     bool    # CDN doesn't provide this; always False
    was_foul:          bool    # a foul occurred during this possession
    had_shooting_foul: bool
    had_personal_foul: bool
    was_sub:           bool    # a substitution occurred during this possession
    home_score:        int
    away_score:        int
    home_lineup_id:    str
    away_lineup_id:    str
    xppp:              float   # expected PPP based on shot type
    is_paint:          bool    # shot_distance < 5 ft (sustainability threshold)


class PossessionBuilder:

    @staticmethod
    def parse(raw_event: dict, state: "GameState") -> Optional[PossessionRow]:
        """
        Process a single raw NBA CDN event.
        Mutates state's possession SM fields (possessing_team, missed_shot_team, ft_*).
        Returns a completed PossessionRow when a possession boundary is crossed,
        or None if the event does not end a possession.

        Feature-relevant state (run state, foul counts, scored buffers) is
        updated by state.advance() AFTER this returns a non-None row.
        """
        action_type = raw_event.get("actionType", "")

        if action_type == "period":
            return _handle_period(raw_event, state)
        if action_type == "jumpball":
            return _handle_jumpball(raw_event, state)
        if action_type in ("2pt", "3pt"):
            return _handle_field_goal(raw_event, state)
        if action_type == "freethrow":
            return _handle_free_throw(raw_event, state)
        if action_type == "rebound":
            return _handle_rebound(raw_event, state)
        if action_type == "turnover":
            return _handle_turnover(raw_event, state)
        if action_type == "violation":
            return _handle_violation(raw_event, state)

        # substitution, foul, timeout — handled by state.update_from_event()
        return None


# ── Event handlers ─────────────────────────────────────────────────────────────

def _handle_period(event: dict, state: "GameState") -> Optional[PossessionRow]:
    sub_type = (event.get("subType") or "").lower()
    desc     = (event.get("description") or "").lower()
    period   = event.get("period", state.current_period)
    clock    = _parse_clock(event.get("clock", ""))

    is_end   = "end" in sub_type or "end" in desc
    is_start = not is_end and ("start" in sub_type or "start" in desc)

    if is_end:
        # Flush any pending missed shot (the period ended with a miss + no rebound)
        if state.missed_shot_team:
            row = _build_row(state, clock, period, team_scored="", points=0,
                             shot_value=0, shot_distance=0.0, shot_area="")
            state.missed_shot_team = ""
            state.ft_in_seq        = False
            state._reset_poss_flags()
            return row
        state.ft_in_seq = False

    if is_start:
        state.missed_shot_team = ""
        state.ft_in_seq        = False

    return None


def _handle_jumpball(event: dict, state: "GameState") -> None:
    team_id = event.get("teamId", 0)
    winner  = state._team_side(team_id)
    if winner:
        state.possessing_team = winner
    return None


def _handle_field_goal(event: dict, state: "GameState") -> Optional[PossessionRow]:
    action_type = event.get("actionType", "")
    shot_result = event.get("shotResult", "")
    team_id     = event.get("teamId", 0)
    team_side   = state._team_side(team_id) or state.possessing_team
    shot_val    = 3 if action_type == "3pt" else 2
    shot_dist   = float(event.get("shotDistance") or 0.0)
    shot_area   = event.get("area") or ""
    clock       = _parse_clock(event.get("clock", ""))
    period      = event.get("period", state.current_period)

    _update_pending_score(event, state)

    if shot_result == "Made":
        state.ft_in_seq        = False
        state.missed_shot_team = ""
        row = _build_row(state, clock, period,
                         team_scored=team_side, points=shot_val,
                         shot_value=shot_val, shot_distance=shot_dist,
                         shot_area=shot_area)
        state.possessing_team = _flip(team_side)
        state._reset_poss_flags()
        return row
    else:
        # Missed — wait for rebound
        state.missed_shot_team = team_side
        state.possessing_team  = team_side
        return None


def _handle_free_throw(event: dict, state: "GameState") -> Optional[PossessionRow]:
    desc        = event.get("description") or ""
    shot_result = event.get("shotResult", "")
    team_id     = event.get("teamId", 0)
    team_side   = state._team_side(team_id) or state.possessing_team
    made        = shot_result == "Made"
    period      = event.get("period", state.current_period)
    clock       = _parse_clock(event.get("clock", ""))
    sub_type    = (event.get("subType") or "").lower()

    if "technical" in sub_type or "technical" in desc.lower():
        ft_n, ft_m = 1, 1
    else:
        ft_match = re.search(r"(\d+)\s+of\s+(\d+)", desc, re.IGNORECASE)
        if not ft_match:
            return None
        ft_n = int(ft_match.group(1))
        ft_m = int(ft_match.group(2))

    if ft_n == 1:
        state.ft_in_seq    = True
        state.ft_total     = ft_m
        state.ft_made      = 1 if made else 0
        state.ft_player_id = event.get("personId", 0)
        state.ft_clock_secs = clock
        state.ft_period    = period
        state.ft_team      = team_side
    else:
        state.ft_made += 1 if made else 0

    if ft_n < ft_m:
        return None

    # Last FT — possession ends
    state.ft_in_seq        = False
    state.missed_shot_team = ""
    _update_pending_score(event, state)

    scored_team = state.ft_team if state.ft_made > 0 else ""
    row = _build_row(state,
                     clock  = state.ft_clock_secs,
                     period = state.ft_period,
                     team_scored   = scored_team,
                     points        = state.ft_made,
                     shot_value    = 1,
                     shot_distance = 0.0,
                     shot_area     = "Free Throw")
    state.possessing_team = _flip(state.ft_team)
    state._reset_poss_flags()
    return row


def _handle_rebound(event: dict, state: "GameState") -> Optional[PossessionRow]:
    if not state.missed_shot_team:
        return None  # dead ball / end-of-period rebound

    sub_type  = (event.get("subType") or "").lower()
    team_id   = event.get("teamId", 0)
    reb_side  = state._team_side(team_id)
    clock     = _parse_clock(event.get("clock", ""))
    period    = event.get("period", state.current_period)

    # Determine offensive vs defensive.
    # CDN: subType = "offensive" or "defensive"
    # Fallback: compare reb_side with missed_shot_team
    if sub_type == "offensive":
        is_offensive = True
    elif sub_type == "defensive":
        is_offensive = False
    else:
        is_offensive = (reb_side == state.missed_shot_team)

    if is_offensive:
        state.missed_shot_team = ""
        return None

    # Defensive rebound — end the missed team's possession as a stop
    state.missed_shot_team = ""
    row = _build_row(state, clock, period,
                     team_scored="", points=0, shot_value=0,
                     shot_distance=0.0, shot_area="")
    state.possessing_team = reb_side or _flip(state.possessing_team)
    state._reset_poss_flags()
    return row


def _handle_turnover(event: dict, state: "GameState") -> Optional[PossessionRow]:
    team_id   = event.get("teamId", 0)
    team_side = state._team_side(team_id) or state.possessing_team
    clock     = _parse_clock(event.get("clock", ""))
    period    = event.get("period", state.current_period)

    state.missed_shot_team = ""
    state.ft_in_seq        = False
    row = _build_row(state, clock, period,
                     team_scored="", points=0, shot_value=0,
                     shot_distance=0.0, shot_area="")
    state.possessing_team = _flip(team_side)
    state._reset_poss_flags()
    return row


def _handle_violation(event: dict, state: "GameState") -> Optional[PossessionRow]:
    sub_type = (event.get("subType") or "").lower()
    clock    = _parse_clock(event.get("clock", ""))
    period   = event.get("period", state.current_period)
    _update_pending_score(event, state)

    if "goaltending" in sub_type:
        # Basket counts: the offensive team scores 2 points
        offense_side = state.missed_shot_team or state.possessing_team
        state.missed_shot_team = ""
        row = _build_row(state, clock, period,
                         team_scored=offense_side, points=2,
                         shot_value=2, shot_distance=0.0, shot_area="")
        state.possessing_team = _flip(offense_side)
        state._reset_poss_flags()
        return row

    # Kicked ball, lane violation, etc. → treat as turnover
    return _handle_turnover(event, state)


# ── Builder ────────────────────────────────────────────────────────────────────

def _build_row(
    state: "GameState",
    clock: float,
    period: int,
    team_scored: str,
    points: int,
    shot_value: int,
    shot_distance: float,
    shot_area: str,
) -> PossessionRow:
    home_lid = _lineup_id(state.home_lineup)
    away_lid = _lineup_id(state.away_lineup)
    xppp     = _shot_xppp(shot_value, shot_distance)
    is_paint = shot_distance < 5.0 and shot_value not in (1, 3)

    return PossessionRow(
        possession_id     = state.possession_count + 1,
        period            = period,
        game_clock_secs   = clock,
        team_scored       = team_scored,
        points            = points,
        shot_value        = shot_value,
        shot_distance     = shot_distance,
        shot_area         = shot_area,
        was_contested     = False,
        was_foul          = state.poss_had_foul,
        had_shooting_foul = state.poss_had_shooting_foul,
        had_personal_foul = state.poss_had_personal_foul,
        was_sub           = state.poss_had_sub,
        home_score        = state.pending_home_score,
        away_score        = state.pending_away_score,
        home_lineup_id    = home_lid,
        away_lineup_id    = away_lid,
        xppp              = xppp,
        is_paint          = is_paint,
    )


# ── Utilities ─────────────────────────────────────────────────────────────────

def _parse_clock(clock_str: str) -> float:
    m = re.match(r"PT(\d+)M([\d.]+)S", clock_str or "")
    if not m:
        return 0.0
    return int(m.group(1)) * 60 + float(m.group(2))


def _update_pending_score(event: dict, state: "GameState") -> None:
    """Update state's pending score from any event that carries scoreHome/scoreAway."""
    try:
        h = int(event.get("scoreHome") or -1)
        if h >= 0:
            state.pending_home_score = h
    except (ValueError, TypeError):
        pass
    try:
        a = int(event.get("scoreAway") or -1)
        if a >= 0:
            state.pending_away_score = a
    except (ValueError, TypeError):
        pass


def _flip(team_side: str) -> str:
    return "away" if team_side == "home" else "home"


def _lineup_id(player_ids: list[int]) -> str:
    """
    MD5 hash of sorted player IDs.
    Uses comma separator to match nba_api_client._lineup_id exactly.
    """
    key = ",".join(str(p) for p in sorted(player_ids))
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _shot_xppp(shot_val: int, shot_dist: float) -> float:
    """Expected PPP — matches momentum_features._shot_xppp exactly."""
    if shot_val == 1:
        return _XPPP_FREE_THROW
    if shot_val == 3:
        return _XPPP_3PT
    if shot_dist < 8.0:
        return _XPPP_PAINT
    return _XPPP_MIDRANGE
