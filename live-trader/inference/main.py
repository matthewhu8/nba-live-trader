"""
Python Inference Service — FastAPI

This service owns all X vector computation and model inference.
Go sends raw NBA events + pre-computed Kalshi market features.
Python maintains per-game state, computes the 58 physics features,
assembles the full 83-feature vector, and runs the MMoE model.

Routes:
    POST /game/{game_id}/start       — initialize GameState, load pregame from MotherDuck
    POST /game/{game_id}/possession  — process event, compute X vector, run inference
    POST /game/{game_id}/end         — flush prediction cache, clean up state
    GET  /game/{game_id}/state       — current game state snapshot (dashboard/debug)
    GET  /health                     — service health + loaded model info
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from inference.game_state import GameState
from inference.feature_computer import FeatureComputer
from inference.possession_builder import PossessionBuilder
from inference.pregame_loader import load_pregame
from inference.schemas import PossessionRequest, PossessionResponse, GameStartRequest
from models.mmoe.predictor import MMoEPredictor

# ── Global state ─────────────────────────────────────────────────────────────

_predictor: MMoEPredictor | None = None
_games: dict[str, GameState] = {}           # game_id → GameState


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    _predictor = MMoEPredictor.load()
    yield
    _games.clear()


app = FastAPI(title="MMoE Inference Service", lifespan=lifespan)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/game/{game_id}/start")
async def start_game(game_id: str, req: GameStartRequest) -> dict[str, str]:
    """
    Initialize a new GameState for this game.
    Loads pregame features from MotherDuck and pre-loads lineup ratings.
    Called once by the Go coordinator at tip-off.
    """
    if game_id in _games:
        raise HTTPException(400, f"Game {game_id} already active")
    pregame = await load_pregame(game_id)
    _games[game_id] = GameState(game_id=game_id, pregame=pregame)
    return {"status": "started", "game_id": game_id}


@app.post("/game/{game_id}/possession")
async def process_possession(game_id: str, req: PossessionRequest) -> PossessionResponse:
    """
    Core endpoint. Called by Go on every possession event.

    Receives:
        raw_event       — NBA CDN action object (Go parsed JSON)
        kalshi_snapshot — 14 market features pre-computed by Go's KalshiRingBuffer
        wall_clock_ts   — timestamp for feed delay accounting

    Computes:
        1. PossessionBuilder parses raw_event → PossessionRow
        2. FeatureComputer extracts 58 physics features from GameState + PossessionRow
        3. Assembles full 83-feature vector (58 physics + 11 pregame + 14 market)
        4. MMoEPredictor runs inference → run_prob, trajectory[10], hazard[10]
        5. Updates GameState (run state, foul counts, buffers) AFTER feature extraction
        6. Appends to prediction_history cache

    Returns action recommendation + full MMoE output + assembled features (for Go logging).
    """
    state = _get_state(game_id)

    possession = PossessionBuilder.parse(req.raw_event, state)
    if possession is None:
        # Event did not complete a possession (e.g., mid-possession foul)
        # Update state side-effects (foul count, substitution) without inference
        state.update_from_event(req.raw_event)
        return PossessionResponse(action="WAIT", run_prob=0.0,
                                  trajectory=[0.0]*10, hazard=[1.0]*10,
                                  yes_bid=req.kalshi_snapshot[0],
                                  yes_ask=req.kalshi_snapshot[1],
                                  is_garbage_time=False, is_blowout=False,
                                  features={}, pipeline_ms=0)

    # X vector computation — features extracted BEFORE state update
    features = FeatureComputer.compute(possession, state, req.kalshi_snapshot)

    output = _predictor.predict(features)

    # State update AFTER feature extraction (preserves shift(1) invariant from training)
    state.advance(possession, req.raw_event)

    return PossessionResponse(
        action="WAIT",  # agent decision happens in Go
        run_prob=output.run_prob,
        trajectory=output.trajectory,
        hazard=output.hazard,
        yes_bid=int(features.get("yes_bid", 0)),
        yes_ask=int(features.get("yes_ask", 0)),
        is_garbage_time=bool(features.get("garbage_time_risk", 0) > 0.8),
        is_blowout=bool(abs(features.get("score_diff", 0)) > 20),
        features={k: float(v) for k, v in features.items()},
        pipeline_ms=0,
    )


@app.post("/game/{game_id}/end")
async def end_game(game_id: str) -> dict[str, Any]:
    """Clean up game state and return summary statistics."""
    state = _get_state(game_id)
    summary = {
        "game_id": game_id,
        "possessions_processed": len(state.recent_possessions),
        "predictions": len(state.prediction_history),
    }
    del _games[game_id]
    return summary


@app.get("/game/{game_id}/state")
async def get_state(game_id: str) -> dict[str, Any]:
    """Return current game state snapshot for dashboard/debugging."""
    state = _get_state(game_id)
    return {
        "game_id":          state.game_id,
        "run_team":         state.run_team,
        "run_length":       state.run_length,
        "home_lineup":      state.home_lineup,
        "away_lineup":      state.away_lineup,
        "last_prediction":  state.prediction_history[-1] if state.prediction_history else None,
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status":        "ok",
        "model_loaded":  _predictor is not None,
        "active_games":  list(_games.keys()),
    }


def _get_state(game_id: str) -> GameState:
    if game_id not in _games:
        raise HTTPException(404, f"Game {game_id} not started")
    return _games[game_id]
