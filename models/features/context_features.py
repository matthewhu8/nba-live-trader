"""
Game-context features: score state, foul state, fatigue, blowout flags,
bonus state, score×time interactions, and timeout signals.
"""

import numpy as np
import pandas as pd

from models.features.star_players import STAR_PLAYERS

# NBA teams start each game with 4 full timeouts (6 total including 2 short/20-sec)
_FULL_TIMEOUTS_PER_GAME = 4


def _player_foul_counts_at(
    foul_df: pd.DataFrame, period: int, clock: float
) -> dict[int, int]:
    """Count fouls committed by each player strictly before (period, clock)."""
    prior = foul_df[
        (foul_df["period"] < period) |
        ((foul_df["period"] == period) & (foul_df["game_clock_secs"] > clock))
    ]
    return prior.groupby("player_id").size().to_dict()


def _player_in_trouble(foul_count: int, period: int) -> bool:
    """Is this player's foul count problematic for the given period?"""
    if period <= 2 and foul_count >= 2:
        return True
    if period >= 3 and foul_count >= 4:
        return True
    return False


def _compute_b2b_flags(games: pd.DataFrame) -> dict[str, tuple[bool, bool]]:
    """
    Returns {game_id: (home_b2b, away_b2b)}.
    A team is on B2B if their previous game was the prior calendar day.
    """
    records = []
    for _, row in games.iterrows():
        records.append({"game_id": row["game_id"], "team": row["home_team"],
                        "date": pd.to_datetime(row["game_date"]), "slot": "home"})
        records.append({"game_id": row["game_id"], "team": row["away_team"],
                        "date": pd.to_datetime(row["game_date"]), "slot": "away"})

    tdf = pd.DataFrame(records).sort_values(["team", "date"])
    tdf["prev_date"] = tdf.groupby("team")["date"].shift(1)
    tdf["is_b2b"] = (tdf["date"] - tdf["prev_date"]).dt.days == 1

    result: dict[str, tuple[bool, bool]] = {}
    for game_id, grp in tdf.groupby("game_id"):
        home_b2b = grp.loc[grp["slot"] == "home", "is_b2b"].any()
        away_b2b = grp.loc[grp["slot"] == "away", "is_b2b"].any()
        result[str(game_id)] = (bool(home_b2b), bool(away_b2b))
    return result


def add_context_features(
    poss_df: pd.DataFrame,
    foul_events: pd.DataFrame,
    games: pd.DataFrame,
    game_id: str,
    timeout_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Adds game-context columns to a single game's possession DataFrame.

    Columns added:
      score_diff                        int   — home_score - away_score
      minutes_into_game                 float — elapsed real-game-time minutes
      time_remaining_period             float — game_clock_secs for this possession
      is_blowout                        bool  — |score_diff| > 20
      is_garbage_time                   bool  — is_blowout AND Q4 with < 6 min remaining
      home_key_foul_count               int   — cumulative home team fouls
      away_key_foul_count               int
      home_max_player_fouls             int   — max fouls by any single home player
      away_max_player_fouls             int
      home_player_in_trouble            bool
      away_player_in_trouble            bool
      home_trouble_player_id            int64
      away_trouble_player_id            int64
      home_star_in_foul_trouble         bool
      away_star_in_foul_trouble         bool
      home_back_to_back                 bool
      away_back_to_back                 bool
      home_in_bonus                     bool  — ≥5 team fouls this quarter
      away_in_bonus                     bool
      home_fouls_until_bonus            int   — fouls needed to reach bonus (0 if already in)
      away_fouls_until_bonus            int
      both_teams_in_bonus               bool
      trailing_team_urgency             float — |score_diff| / max(1, min_remaining)
      comeback_probability_proxy        float — score_diff² / max(1, min_remaining)
      q4_close_game                     bool  — Q4 AND |score_diff| ≤ 5
      garbage_time_risk                 float — 0–1 continuous probability approaching garbage
      possessions_since_last_timeout    int   — 999 if no timeout yet this game
      home_called_timeout_in_last_3_poss bool
      away_called_timeout_in_last_3_poss bool
      timeout_on_opponent_run           bool  — last timeout called by trailing team on a run
      home_full_timeouts_remaining      int
      away_full_timeouts_remaining      int
    """
    df = poss_df.copy()

    # Score differential
    df["score_diff"] = df["home_score"] - df["away_score"]

    # Minutes into game: each period is 12 minutes (720 seconds)
    # period 1: 0-12 min, period 2: 12-24 min, etc.
    period_elapsed = (df["period"] - 1) * 12.0
    clock_elapsed  = (720.0 - df["game_clock_secs"].clip(upper=720.0)) / 60.0
    df["minutes_into_game"]      = period_elapsed + clock_elapsed
    df["time_remaining_period"]  = df["game_clock_secs"]

    # Blowout / garbage time
    df["is_blowout"]      = df["score_diff"].abs() > 20
    df["is_garbage_time"] = df["is_blowout"] & (df["period"] >= 4) & (df["game_clock_secs"] < 360)

    # Foul counts: cumulative fouls up to (but not including) each possession
    game_fouls = foul_events[foul_events["game_id"] == game_id].copy()

    home_team = games.loc[games["game_id"] == game_id, "home_team"].values
    home_team = home_team[0] if len(home_team) else ""

    if not game_fouls.empty:
        home_fouls = game_fouls[game_fouls["team_tricode"] == home_team]
        away_fouls = game_fouls[game_fouls["team_tricode"] != home_team]

        poss_clocks = list(zip(df["period"].values, df["game_clock_secs"].values))

        def count_prior_fouls(foul_df: pd.DataFrame, poss_period: int, poss_clock: float) -> int:
            prior = foul_df[
                (foul_df["period"] < poss_period) |
                ((foul_df["period"] == poss_period) & (foul_df["game_clock_secs"] > poss_clock))
            ]
            return len(prior)

        home_foul_counts = [count_prior_fouls(home_fouls, p, c) for p, c in poss_clocks]
        away_foul_counts = [count_prior_fouls(away_fouls, p, c) for p, c in poss_clocks]

        # Per-player foul tracking
        home_max_fouls: list[int] = []
        away_max_fouls: list[int] = []
        home_in_trouble: list[bool] = []
        away_in_trouble: list[bool] = []
        home_trouble_pid: list[int] = []
        away_trouble_pid: list[int] = []
        home_star_trouble: list[bool] = []
        away_star_trouble: list[bool] = []

        for period, clock in poss_clocks:
            h_counts = _player_foul_counts_at(home_fouls, period, clock)
            a_counts = _player_foul_counts_at(away_fouls, period, clock)

            # Home
            h_max = max(h_counts.values(), default=0)
            h_trouble_pid = next(
                (pid for pid, cnt in sorted(h_counts.items(), key=lambda x: -x[1])
                 if _player_in_trouble(cnt, period)),
                -1,
            )
            home_max_fouls.append(h_max)
            home_in_trouble.append(h_trouble_pid != -1)
            home_trouble_pid.append(h_trouble_pid)
            home_star_trouble.append(h_trouble_pid != -1 and h_trouble_pid in STAR_PLAYERS)

            # Away
            a_max = max(a_counts.values(), default=0)
            a_trouble_pid = next(
                (pid for pid, cnt in sorted(a_counts.items(), key=lambda x: -x[1])
                 if _player_in_trouble(cnt, period)),
                -1,
            )
            away_max_fouls.append(a_max)
            away_in_trouble.append(a_trouble_pid != -1)
            away_trouble_pid.append(a_trouble_pid)
            away_star_trouble.append(a_trouble_pid != -1 and a_trouble_pid in STAR_PLAYERS)

    else:
        n = len(df)
        home_foul_counts = [0] * n
        away_foul_counts = [0] * n
        home_max_fouls = [0] * n
        away_max_fouls = [0] * n
        home_in_trouble = [False] * n
        away_in_trouble = [False] * n
        home_trouble_pid = [-1] * n
        away_trouble_pid = [-1] * n
        home_star_trouble = [False] * n
        away_star_trouble = [False] * n

    df["home_key_foul_count"]      = home_foul_counts
    df["away_key_foul_count"]      = away_foul_counts
    df["home_max_player_fouls"]    = home_max_fouls
    df["away_max_player_fouls"]    = away_max_fouls
    df["home_player_in_trouble"]   = home_in_trouble
    df["away_player_in_trouble"]   = away_in_trouble
    df["home_trouble_player_id"]   = home_trouble_pid
    df["away_trouble_player_id"]   = away_trouble_pid
    df["home_star_in_foul_trouble"] = home_star_trouble
    df["away_star_in_foul_trouble"] = away_star_trouble

    # Back-to-back flags — computed once from the full schedule and cached
    # to avoid O(n²) recomputation across all games.
    if not hasattr(add_context_features, "_b2b_cache"):
        add_context_features._b2b_cache: dict = {}  # type: ignore[attr-defined]
    cache_key = id(games)
    if cache_key not in add_context_features._b2b_cache:
        add_context_features._b2b_cache[cache_key] = _compute_b2b_flags(games)
    b2b_map = add_context_features._b2b_cache[cache_key]
    home_b2b, away_b2b = b2b_map.get(str(game_id), (False, False))
    df["home_back_to_back"] = home_b2b
    df["away_back_to_back"] = away_b2b

    # ── Bonus state ───────────────────────────────────────────────────────────
    # Team foul bonus: ≥5 team fouls in the current quarter → free throws on contact.
    # Resets to 0 at the start of each new quarter.
    if not game_fouls.empty:
        def count_quarter_fouls(foul_df: pd.DataFrame, poss_period: int, poss_clock: float) -> int:
            """Team fouls in the same quarter, strictly before (period, clock)."""
            prior = foul_df[
                (foul_df["period"] == poss_period) &
                (foul_df["game_clock_secs"] > poss_clock)
            ]
            return len(prior)

        home_q_fouls = [count_quarter_fouls(home_fouls, p, c) for p, c in poss_clocks]
        away_q_fouls = [count_quarter_fouls(away_fouls, p, c) for p, c in poss_clocks]
    else:
        n = len(df)
        home_q_fouls = [0] * n
        away_q_fouls = [0] * n

    home_in_bonus_arr = np.array(home_q_fouls) >= 5
    away_in_bonus_arr = np.array(away_q_fouls) >= 5

    df["home_in_bonus"]          = home_in_bonus_arr
    df["away_in_bonus"]          = away_in_bonus_arr
    df["home_fouls_until_bonus"] = np.clip(5 - np.array(home_q_fouls), 0, 4)
    df["away_fouls_until_bonus"] = np.clip(5 - np.array(away_q_fouls), 0, 4)
    df["both_teams_in_bonus"]    = home_in_bonus_arr & away_in_bonus_arr

    # ── Score × time interactions ────────────────────────────────────────────
    # A 5-point deficit in Q4 with 4 min left is urgent; same deficit in Q2 is not.
    minutes_remaining = (
        (4 - df["period"]).clip(lower=0) * 12.0 + df["game_clock_secs"] / 60.0
    )

    df["trailing_team_urgency"] = df["score_diff"].abs() / minutes_remaining.clip(lower=1.0)
    df["comeback_probability_proxy"] = (
        df["score_diff"].astype(float) ** 2 / minutes_remaining.clip(lower=1.0)
    )
    df["q4_close_game"] = (df["period"] == 4) & (df["score_diff"].abs() <= 5)

    # garbage_time_risk: continuous 0–1 proximity to garbage time
    # High score_diff AND late in the game → approaches 1.0
    sigmoid_in   = (df["score_diff"].abs().astype(float) - 15.0) / 5.0
    sigmoid_out  = 1.0 / (1.0 + np.exp(-sigmoid_in.clip(-20, 20)))
    time_factor  = (1.0 - (minutes_remaining / 48.0)).clip(lower=0.0, upper=1.0)
    df["garbage_time_risk"] = sigmoid_out * time_factor

    # ── Timeout features ─────────────────────────────────────────────────────
    game_timeouts = (
        timeout_events[timeout_events["game_id"] == game_id].copy()
        if timeout_events is not None and not timeout_events.empty
        else pd.DataFrame()
    )

    if not game_timeouts.empty:
        home_team = games.loc[games["game_id"] == game_id, "home_team"].values
        home_tc   = home_team[0] if len(home_team) else ""

        home_tos = game_timeouts[game_timeouts["team_tricode"] == home_tc]
        away_tos = game_timeouts[game_timeouts["team_tricode"] != home_tc]

        sorted_tos = sorted(
            zip(game_timeouts["period"].values, game_timeouts["game_clock_secs"].values),
            key=lambda x: (x[0], -x[1]),  # chronological: period asc, clock desc
        )
        sorted_home_tos = sorted(
            zip(home_tos["period"].values, home_tos["game_clock_secs"].values),
            key=lambda x: (x[0], -x[1]),
        )
        sorted_away_tos = sorted(
            zip(away_tos["period"].values, away_tos["game_clock_secs"].values),
            key=lambda x: (x[0], -x[1]),
        )

        def count_tos_before(sorted_list: list, poss_period: int, poss_clock: float) -> int:
            """Count timeouts strictly before (period, clock) in a sorted list."""
            count = 0
            for tp, tc in sorted_list:
                if (tp < poss_period) or (tp == poss_period and tc > poss_clock):
                    count += 1
                else:
                    break
            return count

        # possessions_since_last_timeout: sequential pass
        poss_since_to: list[int] = []
        to_ptr = 0
        poss_count = 999  # 999 = no timeout yet this game
        for period, clock in poss_clocks:
            while to_ptr < len(sorted_tos):
                tp, tc = sorted_tos[to_ptr]
                if (tp < period) or (tp == period and tc > clock):
                    poss_count = 0
                    to_ptr += 1
                else:
                    break
            poss_since_to.append(poss_count)
            if poss_count != 999:
                poss_count += 1

        # Called timeout in last 3 possessions: detect increase in timeout count
        home_to_before = pd.Series([
            count_tos_before(sorted_home_tos, p, c) for p, c in poss_clocks
        ])
        away_to_before = pd.Series([
            count_tos_before(sorted_away_tos, p, c) for p, c in poss_clocks
        ])
        df["home_called_timeout_in_last_3_poss"] = (
            (home_to_before - home_to_before.shift(3).fillna(0)) > 0
        )
        df["away_called_timeout_in_last_3_poss"] = (
            (away_to_before - away_to_before.shift(3).fillna(0)) > 0
        )

        # timeout_on_opponent_run: most recent timeout was called by trailing team
        # Use: last timeout's team_tricode matches the team NOT currently on a run
        # Approximate: look at the last timeout before each possession and check if
        # the calling team was trailing at that moment
        df["timeout_on_opponent_run"] = False  # default; set where applicable
        if "current_run_team" in df.columns and "current_run_points" in df.columns:
            last_to_team = game_timeouts.sort_values(
                ["period", "game_clock_secs"], ascending=[True, False]
            )
            for i, (period, clock) in enumerate(poss_clocks):
                run_points = int(df.iloc[i].get("current_run_points", 0))
                # Only flag as timeout-on-run when the run is substantial (≥6 points)
                # This avoids flagging every timeout as "on a run" since any consecutive
                # possession starts a run in our data model.
                if run_points < 6:
                    continue

                prior = last_to_team[
                    (last_to_team["period"] < period) |
                    ((last_to_team["period"] == period) & (last_to_team["game_clock_secs"] > clock))
                ]
                if prior.empty:
                    continue
                last_tc  = str(prior.iloc[-1]["team_tricode"])
                run_team = str(df.iloc[i].get("current_run_team", ""))
                # Timeout was called on opponent's run if calling team was NOT the run team
                if run_team and last_tc:
                    last_is_home = last_tc == home_tc
                    run_is_home  = run_team == "home"
                    df.iloc[i, df.columns.get_loc("timeout_on_opponent_run")] = (
                        last_is_home != run_is_home
                    )

        # Full timeouts remaining: started with 4, subtract used.
        # Clipped to [0, 4] since teams can call short timeouts that inflate the count.
        df["home_full_timeouts_remaining"] = (_FULL_TIMEOUTS_PER_GAME - home_to_before).clip(lower=0, upper=4)
        df["away_full_timeouts_remaining"] = (_FULL_TIMEOUTS_PER_GAME - away_to_before).clip(lower=0, upper=4)

    else:
        n = len(df)
        df["possessions_since_last_timeout"]     = 999
        df["home_called_timeout_in_last_3_poss"] = False
        df["away_called_timeout_in_last_3_poss"] = False
        df["timeout_on_opponent_run"]             = False
        df["home_full_timeouts_remaining"]        = _FULL_TIMEOUTS_PER_GAME
        df["away_full_timeouts_remaining"]        = _FULL_TIMEOUTS_PER_GAME

    df["possessions_since_last_timeout"] = poss_since_to if not game_timeouts.empty else [999] * len(df)

    return df
