"""
FastAPI inference service.

Takes raw NBA CDN events from the Go engine, runs them through the possession
parser, feature computer and MMoE model, and returns the results. The Go agent
makes the final buy and sell decisions.

Routes:
    POST /game/{game_id}/start       init GameState with pregame data
    POST /game/{game_id}/possession  event to inference to response
    POST /game/{game_id}/end         cleanup and log the game summary
    GET  /game/{game_id}/state       debug snapshot
    GET  /health                     liveness check
"""

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import jsonlog
from inference.features import FeatureComputer
from inference.game_state import GameState, PredictionRecord
from inference.possession import PossessionBuilder
from inference.pregame import load_pregame
from models.mmoe.predictor import MMoEPredictor, MODEL_PATH, SCALER_PATH, ROOT_DIR
from inference.dashboard import router as dashboard_router, broadcast_prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

_predictor: Optional[MMoEPredictor] = None
_games: dict[str, GameState] = {}

# Re-emitted into each run's inference.jsonl so every log file is self-contained.
_service_started_at: Optional[str] = None
_service_info_emitted_for_runs: set[str] = set()

# NBA regulation is 4 periods; anything beyond is overtime, which the Go agent
# skips and training excludes.
OVERTIME_FIRST_PERIOD = 5

# These features get a z-score attached to every possession record, so a reader can
# see at a glance whether a value is unusual against the training distribution. The
# means and standard deviations come from the training StandardScaler.
_ZSCORE_FEATURES: list[str] = [
    "score_diff",
    "lead_z",
    "time_leverage",
    "lineup_net_rating_delta",
    "run_signed_points",
    "swing_5",
    "pace_surprise",
    "xppp_edge",
    "luck_edge",
    "garbage_time_risk",
]

# {feature_name: (mean, std)}, populated when the predictor loads.
_zscore_stats: dict[str, tuple[float, float]] = {}

# Model paths come from trading.yaml via Go. Tracking what is currently loaded lets
# /start reload only on a real change, and the lock keeps concurrent game starts
# from loading the model twice.
_loaded_model_path: Optional[str] = None
_loaded_scaler_path: Optional[str] = None
_predictor_lock = asyncio.Lock()

# Gate thresholds forwarded by Go, so the dashboard mirrors the live agent rather
# than hardcoded literals.
_dashboard_gates: dict[str, float] = {
    "min_abs_traj":   0.08,
    "min_yes_bid":    30,
    "max_yes_bid":    70,
    "min_run_length": 0,
}


def _resolve_path(p: str) -> Path:
    """Resolve a config path: relative paths are anchored at the repo root."""
    path = Path(p)
    return path if path.is_absolute() else (ROOT_DIR / path)


def compute_gate_flags(
    score_diff: float,
    period: int,
    clock_secs: float,
    blowout_margin_pts: int,
    garbage_time_period: int,
    garbage_time_clock_secs: int,
) -> tuple[bool, bool]:
    """Return (is_garbage_time, is_blowout) from the config thresholds.

    Deliberately separate from the frozen `garbage_time_risk` model feature, so
    tuning the trade gate in trading.yaml never shifts a model input.
    """
    is_blowout = abs(score_diff) > blowout_margin_pts
    is_garbage_time = (
        is_blowout
        and period >= garbage_time_period
        and clock_secs < garbage_time_clock_secs
    )
    return is_garbage_time, is_blowout


def _requested_paths(model_path: Optional[str], scaler_path: Optional[str]) -> tuple[str, str]:
    """Resolve the model/scaler paths a /start asked for, falling back to defaults."""
    want_model = str(_resolve_path(model_path)) if model_path else str(MODEL_PATH)
    want_scaler = str(_resolve_path(scaler_path)) if scaler_path else str(SCALER_PATH)
    return want_model, want_scaler


def _needs_reload(model_path: Optional[str], scaler_path: Optional[str]) -> bool:
    """True if the requested paths differ from what's currently loaded."""
    want_model, want_scaler = _requested_paths(model_path, scaler_path)
    return want_model != _loaded_model_path or want_scaler != _loaded_scaler_path


def _build_zscore_stats(predictor: MMoEPredictor) -> dict[str, tuple[float, float]]:
    """Pull the tracked features out of the scaler's per-feature mean and std,
    skipping any that are missing or have a near-zero std."""
    stats: dict[str, tuple[float, float]] = {}
    all_stats = predictor.get_feature_stats()
    for name in _ZSCORE_FEATURES:
        if name not in all_stats:
            logging.warning("[MODEL] z-score feature %s not in scaler, skipped", name)
            continue
        mean, std = all_stats[name]
        if std < 1e-10:
            logging.warning("[MODEL] z-score feature %s has near-zero std, skipped", name)
            continue
        stats[name] = (mean, std)
    return stats


def _load_predictor(model_path: Optional[str] = None, scaler_path: Optional[str] = None) -> bool:
    """Load or reload the MMoE predictor and rebuild the z-score stats.

    Paths default to the packaged model. Returns False on a missing artifact; the
    service then keeps running without inference.
    """
    global _predictor, _zscore_stats, _loaded_model_path, _loaded_scaler_path
    mp = _resolve_path(model_path) if model_path else MODEL_PATH
    sp = _resolve_path(scaler_path) if scaler_path else SCALER_PATH
    try:
        predictor = MMoEPredictor.load(mp, sp)
    except FileNotFoundError as exc:
        logging.warning("[MODEL] artifact not found, running without inference: %s", exc)
        return False
    _predictor = predictor
    _zscore_stats = _build_zscore_stats(predictor)
    _loaded_model_path = str(mp)
    _loaded_scaler_path = str(sp)
    logging.info("[MODEL] loaded %s (z-score stats for %d features)", mp.name, len(_zscore_stats))
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service_started_at
    _service_started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    _load_predictor()  # Go can swap this later via /game/start
    yield
    jsonlog.shutdown()  # flush buffered JSONL writes


app = FastAPI(lifespan=lifespan)
app.include_router(dashboard_router)


# ── Pydantic models ────────────────────────────────────────────────────────────

class GameStartRequest(BaseModel):
    market_ticker: str
    home_team_id:  int
    away_team_id:  int
    # Without run_id and log_dir this run simply gets no JSONL logging.
    run_id:        Optional[str] = None
    log_dir:       Optional[str] = None
    # Forwarded from trading.yaml. Omitting these keeps the loaded model and the
    # default gates.
    model_path:    Optional[str] = None
    scaler_path:   Optional[str] = None
    min_abs_traj:   Optional[float] = None
    min_yes_bid:    Optional[int] = None
    max_yes_bid:    Optional[int] = None
    min_run_length: Optional[int] = None


class PossessionRequest(BaseModel):
    raw_event:       dict
    kalshi_snapshot: list[float]  # 14 floats in MARKET_COLS order
    wall_clock_ts:   str          # ISO-8601
    # Thresholds behind the is_garbage_time and is_blowout flags the agent skips on.
    # Distinct from the frozen `garbage_time_risk` model feature.
    blowout_margin_pts:      int = 30
    garbage_time_period:     int = 4
    garbage_time_clock_secs: int = 360


class PossessionResponse(BaseModel):
    action:          str          # always "WAIT"; the Go agent decides
    run_prob:        float
    trajectory:      list[float]  # 10 log-odds delta checkpoints from Head B
    hazard:          list[float]  # 10 survival hazard values from Head C
    yes_bid:         int          # from kalshi_snapshot[0] (cents)
    yes_ask:         int          # from kalshi_snapshot[1] (cents)
    is_garbage_time: bool
    is_blowout:      bool
    # Gate inputs are first-class fields rather than entries in `features`, so the
    # agent never depends on the model's feature set.
    is_overtime:        bool  = False
    current_run_length: float = 0.0
    # Model inputs. Go uses these for logging only.
    features:        dict[str, float]
    pipeline_ms:     int


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/game/{game_id}/start")
async def game_start(game_id: str, request: GameStartRequest):
    # Activate the run-scoped logger when Go supplied a run to log into.
    if request.run_id and request.log_dir:
        jsonlog.set_run(request.run_id, request.log_dir)
        _maybe_emit_service_info(request.run_id)

    # Reload only when the forwarded paths differ from what is loaded.
    if (request.model_path or request.scaler_path) and _needs_reload(request.model_path, request.scaler_path):
        async with _predictor_lock:
            # Re-check under the lock; another /start may have just loaded it.
            if _needs_reload(request.model_path, request.scaler_path):
                want_model, want_scaler = _requested_paths(request.model_path, request.scaler_path)
                logging.info("[MODEL] reload requested via /start: %s / %s", want_model, want_scaler)
                _load_predictor(request.model_path, request.scaler_path)

    if request.min_abs_traj is not None:
        _dashboard_gates["min_abs_traj"] = request.min_abs_traj
    if request.min_yes_bid is not None:
        _dashboard_gates["min_yes_bid"] = request.min_yes_bid
    if request.max_yes_bid is not None:
        _dashboard_gates["max_yes_bid"] = request.max_yes_bid
    if request.min_run_length is not None:
        _dashboard_gates["min_run_length"] = request.min_run_length

    pregame = await load_pregame(
        game_id,
        fallback_home_team_id=request.home_team_id,
        fallback_away_team_id=request.away_team_id,
    )

    state = GameState(
        game_id       = game_id,
        market_ticker = request.market_ticker,
        home_team_id  = pregame["home_team_id"],
        away_team_id  = pregame["away_team_id"],
    )

    # The 11 pregame floats, static for the whole game.
    pregame_float_keys = {
        "team_net_rating_delta", "home_off_rating", "away_off_rating",
        "home_def_rating", "away_def_rating", "roster_rapm_gap",
        "missing_rapm_impact", "rest_advantage", "expected_pace",
        "form_delta", "has_pregame_data",
    }
    state.pregame      = {k: v for k, v in pregame.items() if k in pregame_float_keys}
    state.lineup_ratings      = pregame["lineup_ratings"]
    state.lineup_sample_sizes = pregame.get("lineup_sample_sizes", {})
    state.player_apm     = pregame["player_apm"]
    state.star_players   = pregame["star_players"]
    state.home_b2b       = pregame["home_b2b"]
    state.away_b2b       = pregame["away_b2b"]
    state.pace_baseline  = pregame["pace_baseline"]

    _games[game_id] = state

    logging.info(
        "[PREGAME] game=%s market=%s home_id=%d away_id=%d b2b=(%s/%s) pace=%.1f",
        game_id,
        request.market_ticker,
        state.home_team_id,
        state.away_team_id,
        state.home_b2b,
        state.away_b2b,
        state.pace_baseline,
    )

    jl = jsonlog.get_logger()
    if jl is not None:
        jl.emit(
            "pregame_loaded",
            game_id,
            market_ticker = request.market_ticker,
            home_team_id  = state.home_team_id,
            away_team_id  = state.away_team_id,
            home_b2b      = state.home_b2b,
            away_b2b      = state.away_b2b,
            pace_baseline = state.pace_baseline,
            star_players  = state.star_players,
            pregame       = state.pregame,
        )

    return {"status": "ok", "game_id": game_id}


@app.post("/game/{game_id}/possession", response_model=PossessionResponse)
async def game_possession(game_id: str, request: PossessionRequest):
    state = _games.get(game_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not started")

    t0 = time.time()
    jl = jsonlog.get_logger()

    row = PossessionBuilder.parse(request.raw_event, state)

    # A None row means the event did not end a possession, e.g. an offensive
    # rebound or a non-final free throw. Recording those lets us diff the live
    # parser against the historical one for boundary divergences.
    if row is None:
        state.update_from_event(request.raw_event)
        pipeline_ms = int((time.time() - t0) * 1000)
        if jl is not None:
            jl.emit(
                "parser_skip",
                game_id,
                action_type   = request.raw_event.get("actionType", ""),
                sub_type      = request.raw_event.get("subType", ""),
                period        = request.raw_event.get("period", 0),
                clock         = request.raw_event.get("clock", ""),
                team_id       = request.raw_event.get("teamId", 0),
                pipeline_ms   = pipeline_ms,
            )
        return PossessionResponse(
            action          = "SKIP",
            run_prob        = 0.0,
            trajectory      = [0.0] * 10,
            hazard          = [0.0] * 10,
            yes_bid         = int(request.kalshi_snapshot[0]) if request.kalshi_snapshot else 0,
            yes_ask         = int(request.kalshi_snapshot[1]) if len(request.kalshi_snapshot) > 1 else 0,
            is_garbage_time = False,
            is_blowout      = False,
            features        = {},
            pipeline_ms     = pipeline_ms,
        )

    # Build features before advancing state, which preserves the shift(1) invariant.
    features = FeatureComputer.compute(row, state, request.kalshi_snapshot)

    # Capture the gate inputs in the same pre-advance window, so the gate sees the
    # state the model saw. They stay out of `features` on purpose: anything the agent
    # gates on must be immune to a change in the model's feature set.
    gate_is_overtime = row.period >= OVERTIME_FIRST_PERIOD
    gate_run_length  = float(state.run_length)

    if _predictor is not None:
        output = _predictor.predict(features)
    else:
        from models.mmoe.predictor import MMoEOutput
        output = MMoEOutput(run_prob=0.0, trajectory=[0.0] * 10, hazard=[0.0] * 10)

    state.advance(row)

    wall_clock = datetime.fromisoformat(request.wall_clock_ts)
    yes_bid = int(request.kalshi_snapshot[0]) if request.kalshi_snapshot else 0
    yes_ask = int(request.kalshi_snapshot[1]) if len(request.kalshi_snapshot) > 1 else 0

    state.prediction_history.append(PredictionRecord(
        possession_id = row.possession_id,
        wall_clock_ts = wall_clock,
        features      = features,
        run_prob      = output.run_prob,
        trajectory    = output.trajectory,
        hazard        = output.hazard,
        action        = "WAIT",
        yes_bid       = yes_bid,
        yes_ask       = yes_ask,
    ))

    is_garbage_time, is_blowout = compute_gate_flags(
        score_diff              = features.get("score_diff", 0.0),
        period                  = row.period,
        clock_secs              = row.game_clock_secs,
        blowout_margin_pts      = request.blowout_margin_pts,
        garbage_time_period     = request.garbage_time_period,
        garbage_time_clock_secs = request.garbage_time_clock_secs,
    )
    clock_str       = f"Q{row.period} {int(row.game_clock_secs // 60)}:{int(row.game_clock_secs % 60):02d}"
    traj_final      = output.trajectory[-1] if output.trajectory else 0.0
    pipeline_ms     = int((time.time() - t0) * 1000)

    logging.info(
        "[POSSESSION] game=%s poss_id=%d %s run_prob=%.2f traj_final=%+.3f pipeline=%dms",
        game_id,
        row.possession_id,
        clock_str,
        output.run_prob,
        traj_final,
        pipeline_ms,
    )

    response = PossessionResponse(
        action          = "WAIT",
        run_prob        = output.run_prob,
        trajectory      = output.trajectory,
        hazard          = output.hazard,
        yes_bid         = yes_bid,
        yes_ask         = yes_ask,
        is_garbage_time = is_garbage_time,
        is_blowout      = is_blowout,
        is_overtime        = gate_is_overtime,
        current_run_length = gate_run_length,
        features        = features,
        pipeline_ms     = pipeline_ms,
    )

    if jl is not None:
        # The model block records the gating weights (which experts each head
        # trusted) and the per-expert opinions (what each head would say on one
        # expert alone). Disagreement among experts marks a borderline call.
        model_block = {
            "gated": {
                "run_prob":   output.run_prob,
                "trajectory": output.trajectory,
                "hazard":     output.hazard,
            },
            "gating_weights":  output.gating_weights,
            "expert_opinions": {
                "run_prob":   output.expert_opinions_run,
                "trajectory": output.expert_opinions_traj,
                "hazard":     output.expert_opinions_haz,
            },
        }

        jl.emit(
            "possession",
            game_id,
            possession_id    = row.possession_id,
            period           = row.period,
            game_clock_secs  = row.game_clock_secs,
            team_scored      = row.team_scored,
            points           = row.points,
            shot_value       = row.shot_value,
            shot_distance    = row.shot_distance,
            home_score       = row.home_score,
            away_score       = row.away_score,
            yes_bid          = yes_bid,
            yes_ask          = yes_ask,
            has_market_data  = bool(features.get("has_market_data", 0.0)),
            traj_final       = traj_final,
            is_garbage_time  = is_garbage_time,
            is_blowout       = is_blowout,
            pipeline_ms      = pipeline_ms,
            model            = model_block,
            features         = features,
            features_zscored = _compute_zscores(features),
        )

    broadcast_prediction(game_id, {
        "possession_id": row.possession_id,
        "run_prob":      output.run_prob,
        "trajectory":    output.trajectory,
        "hazard":        output.hazard,
        "yes_bid":       yes_bid,
        "yes_ask":       yes_ask,
        "action":        "WAIT",
        "is_garbage_time": is_garbage_time,
        "is_blowout":    is_blowout,
        "pipeline_ms":   pipeline_ms,
        "clock_str":     clock_str,
        "period":        row.period,
        "features":      features,
        "market_ticker": state.market_ticker,
        "gates":         dict(_dashboard_gates),
    })

    return response


@app.post("/game/{game_id}/end")
async def game_end(game_id: str):
    state = _games.pop(game_id, None)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    logging.info(
        "[GAME END] game=%s total_possessions=%d total_predictions=%d",
        game_id,
        state.possession_count,
        len(state.prediction_history),
    )

    jl = jsonlog.get_logger()
    if jl is not None:
        jl.emit(
            "game_end",
            game_id,
            total_possessions  = state.possession_count,
            predictions_cached = len(state.prediction_history),
        )

    return {"status": "ok", "game_id": game_id, "total_possessions": state.possession_count}


@app.get("/game/{game_id}/state")
async def game_state(game_id: str):
    state = _games.get(game_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not started")

    return {
        "game_id":             state.game_id,
        "possession_count":    state.possession_count,
        "current_period":      state.current_period,
        "home_score":          state.pending_home_score,
        "away_score":          state.pending_away_score,
        "score_diff":          state.pending_home_score - state.pending_away_score,
        "run_team":            state.run_team,
        "run_length":          state.run_length,
        "run_points":          state.run_points,
        "home_lineup":         state.home_lineup,
        "away_lineup":         state.away_lineup,
        "home_team_fouls":     dict(state.home_team_fouls),
        "away_team_fouls":     dict(state.away_team_fouls),
        "home_timeouts_used":  state.home_timeouts_used,
        "away_timeouts_used":  state.away_timeouts_used,
        "home_b2b":            state.home_b2b,
        "away_b2b":            state.away_b2b,
        "pace_baseline":       state.pace_baseline,
        "predictions_cached":  len(state.prediction_history),
    }


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "games_active": len(_games),
        "model_loaded": _predictor is not None,
    }


# ── JSONL helpers ──────────────────────────────────────────────────────────────

def _compute_zscores(features: dict[str, float]) -> dict[str, dict[str, float]]:
    """
    Return {feature_name: {"value": v, "z": z}} for the tracked features. Each entry
    carries its own value so a log reader never has to cross-reference the raw
    feature block. Missing features default to 0.0, matching the predictor.
    """
    if not _zscore_stats:
        return {}
    out: dict[str, dict[str, float]] = {}
    for name, (mean, std) in _zscore_stats.items():
        v = float(features.get(name, 0.0))
        z = (v - mean) / std
        out[name] = {"value": v, "z": z}
    return out


def _maybe_emit_service_info(run_id: str) -> None:
    """Emit service_info once per run, so each run's inference.jsonl is self-describing."""
    if run_id in _service_info_emitted_for_runs:
        return
    _service_info_emitted_for_runs.add(run_id)

    jl = jsonlog.get_logger()
    if jl is None:
        return

    jl.emit(
        "service_info",
        None,
        service_started_at = _service_started_at,
        model_loaded       = _predictor is not None,
        model_path         = str(MODEL_PATH),
        scaler_path        = str(SCALER_PATH),
        python_version     = sys.version.split()[0],
    )
