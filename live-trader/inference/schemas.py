"""
Pydantic schemas for the inference service HTTP API.
These define the contract between Go and Python.
"""

from pydantic import BaseModel


class GameStartRequest(BaseModel):
    market_ticker: str  # Kalshi ticker for this game (e.g. "KXNBA-25-BOS-MIA-SPREAD")


class PossessionRequest(BaseModel):
    raw_event:       dict               # NBA CDN action object as parsed by Go
    kalshi_snapshot: list[float]        # 14 floats in MARKET_COLS order from Go ring buffer
    wall_clock_ts:   str                # ISO timestamp


class PossessionResponse(BaseModel):
    action:        str          # "BUY_YES" | "BUY_NO" | "EXIT" | "WAIT"
    run_prob:      float
    trajectory:    list[float]  # 10 log-odds delta checkpoints
    hazard:        list[float]  # 10 survival hazard probabilities
    yes_bid:       int          # cents
    yes_ask:       int          # cents
    is_garbage_time: bool
    is_blowout:    bool
    features:      dict[str, float]  # full 83-feature dict (for Go's structured log)
    pipeline_ms:   int
