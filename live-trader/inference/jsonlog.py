"""
The Python side of live-trader/go/jsonlog.go: one JSON object per line, written to
inference.jsonl in the run directory Go names on /game/start. A lock guards every
write so concurrent request handlers cannot interleave bytes inside a record.

Writes are best-effort. Errors warn at most once a minute and are then dropped, and
emit() on a closed logger is a no-op, so logging never breaks the service.

When a new run_id arrives the previous logger closes and a fresh file opens, which
lets one Python service serve several sequential Go runs.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ── JSONLogger ─────────────────────────────────────────────────────────────────

class JSONLogger:
    def __init__(self, run_id: str, log_dir: str) -> None:
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self._lock = threading.Lock()
        self._closed = False
        self._last_warn_ts = 0.0
        self._fp: Optional[Any] = None  # None if the file could not be opened

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._fp = open(self.log_dir / "inference.jsonl", "a", encoding="utf-8")
        except OSError as exc:
            logging.warning("[jsonlog] could not open inference.jsonl in %s: %s", log_dir, exc)
            self._closed = True

    def emit(self, event: str, game_id: Optional[str], **fields: Any) -> None:
        """Write one record. The envelope fields are added here; everything else comes
        from the caller's keyword args."""
        if self._closed or self._fp is None:
            return

        record: dict[str, Any] = dict(fields)
        # Envelope goes last so it wins any collision with a caller-supplied key.
        record["schema_version"] = 1
        record["ts"] = _utc_iso_ms()
        record["run_id"] = self.run_id
        record["event"] = event
        if game_id:
            record["game_id"] = game_id

        try:
            line = json.dumps(record, default=_json_default) + "\n"
        except (TypeError, ValueError) as exc:
            self._warn_rate_limited("marshal failed", exc)
            return

        with self._lock:
            if self._closed or self._fp is None:
                return
            try:
                self._fp.write(line)
                self._fp.flush()
            except OSError as exc:
                self._warn_rate_limited("write failed", exc)

    def close(self) -> None:
        """Idempotent. Safe to call from any thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._fp is not None:
                try:
                    self._fp.close()
                except OSError:
                    pass
                self._fp = None

    def _warn_rate_limited(self, msg: str, err: Exception) -> None:
        now = time.time()
        if now - self._last_warn_ts < 60.0:
            return
        self._last_warn_ts = now
        logging.warning("[jsonlog] %s: %s", msg, err)


# ── Current-run registry ───────────────────────────────────────────────────────
# The service handles one Go run at a time, so a single active logger suffices.

_lock = threading.Lock()
_current: Optional[JSONLogger] = None


def get_logger() -> Optional[JSONLogger]:
    """Return the current JSONLogger, or None if no run is active."""
    return _current


def set_run(run_id: str, log_dir: str) -> JSONLogger:
    """Activate a logger for this run, closing any logger from a different run.
    Idempotent for the same run_id."""
    global _current
    with _lock:
        if _current is not None and _current.run_id == run_id:
            return _current
        if _current is not None:
            _current.close()
        _current = JSONLogger(run_id, log_dir)
        return _current


def shutdown() -> None:
    """Close any active logger. Called from the FastAPI lifespan teardown."""
    global _current
    with _lock:
        if _current is not None:
            _current.close()
            _current = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _utc_iso_ms() -> str:
    """Match Go's millisecond ISO format, e.g. 2026-05-09T21:42:01.402Z."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_default(o: Any) -> Any:
    """Coerce the numpy and torch types that survive the model forward pass, so no
    call site has to convert before logging."""
    if hasattr(o, "tolist"):  # numpy arrays and torch tensors
        return o.tolist()
    if hasattr(o, "item"):  # numpy scalars
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
