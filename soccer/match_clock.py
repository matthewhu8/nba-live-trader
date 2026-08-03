"""
Estimate the match minute from the kickoff wall-clock time.

We have no live match-clock feed (and don't want one for this fun tool), so we
approximate: the in-game clock tracks real time within each half, freezes during
halftime, and resumes in the second half. This ignores stoppage time, so the real
break can land a couple of minutes later than the estimate — the strategy's
`entry_window_min` absorbs that slack.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class MatchClock:
    kickoff: datetime           # UTC
    first_half_len: float = 45.0
    halftime_len: float = 15.0

    def match_minute(self, now: datetime) -> float:
        elapsed = (now - self.kickoff).total_seconds() / 60.0
        if elapsed <= 0:
            return 0.0

        if elapsed <= self.first_half_len:
            return elapsed

        halftime_end = self.first_half_len + self.halftime_len
        if elapsed <= halftime_end:
            return self.first_half_len  # frozen through the interval

        return self.first_half_len + (elapsed - halftime_end)
