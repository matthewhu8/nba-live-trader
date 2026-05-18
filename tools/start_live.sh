#!/usr/bin/env bash
# start_live.sh — bring up the live (paper) trading stack for a game night.
#
# What it does, in order:
#   1. Kills any existing uvicorn / go trader processes (safe — sends SIGTERM,
#      waits up to 5s, then SIGKILL).
#   2. Rebuilds the Go binary from source so the binary on disk matches HEAD.
#   3. Launches the Python inference service in the background, writing logs to
#      logs/inference/<timestamp>.log.
#   4. Polls /health until the model has loaded (15s timeout).
#   5. Launches the Go trader in the foreground.
#   6. On Ctrl+C, cleanly stops both processes.
#
# Usage:
#   tools/start_live.sh                                 # Coordinator mode (auto-detects tonight's games)
#   tools/start_live.sh --game 0042500223 --event KXNBASPREAD-26MAY09OKCLAL
#
# Any args you pass are forwarded to the Go binary.

set -euo pipefail

# ── resolve paths regardless of where the script is invoked from ───────────────
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )"
cd "${PROJECT_ROOT}"

INFERENCE_LOG_DIR="${PROJECT_ROOT}/logs/inference"
mkdir -p "${INFERENCE_LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
INFERENCE_LOG="${INFERENCE_LOG_DIR}/${TIMESTAMP}.log"

# ── pretty output helpers ──────────────────────────────────────────────────────
log()  { printf "\033[1;36m[start_live %s]\033[0m %s\n" "$(date +%H:%M:%S)" "$*"; }
warn() { printf "\033[1;33m[start_live %s WARN]\033[0m %s\n" "$(date +%H:%M:%S)" "$*"; }
err()  { printf "\033[1;31m[start_live %s ERR ]\033[0m %s\n" "$(date +%H:%M:%S)" "$*" >&2; }

# ── 1. kill any existing processes ─────────────────────────────────────────────
log "stopping any running inference / trader processes"
PIDS_UVI=$(pgrep -f "uvicorn inference.main:app"      || true)
PIDS_GO=$(pgrep -f "live-trader/go/go( |\$)|/go/go( |\$)" || true)

for PID in ${PIDS_UVI} ${PIDS_GO}; do
  if kill -0 "${PID}" 2>/dev/null; then
    log "  SIGTERM pid ${PID}"
    kill -TERM "${PID}" 2>/dev/null || true
  fi
done

# Wait up to 5s for graceful shutdown.
for _ in 1 2 3 4 5; do
  STILL=$(pgrep -f "uvicorn inference.main:app|live-trader/go/go( |\$)" || true)
  [[ -z "${STILL}" ]] && break
  sleep 1
done

# Anything still alive gets SIGKILL.
STILL=$(pgrep -f "uvicorn inference.main:app|live-trader/go/go( |\$)" || true)
if [[ -n "${STILL}" ]]; then
  warn "force-killing stragglers: ${STILL}"
  echo "${STILL}" | xargs -n1 kill -9 2>/dev/null || true
fi

# ── 2. rebuild Go binary ───────────────────────────────────────────────────────
log "rebuilding Go binary"
(
  cd "${PROJECT_ROOT}/live-trader/go"
  if ! go build .; then
    err "go build failed — aborting"
    exit 1
  fi
)

# ── 3. launch Python inference service ─────────────────────────────────────────
log "starting Python inference service → ${INFERENCE_LOG}"
(
  cd "${PROJECT_ROOT}"
  PYTHONPATH=.:live-trader \
  exec ./venv/bin/python -m uvicorn inference.main:app \
    --host 127.0.0.1 --port 8001 \
    >> "${INFERENCE_LOG}" 2>&1
) &
INFERENCE_PID=$!
log "  inference pid=${INFERENCE_PID}"

# ── 4. wait for /health to report model_loaded=true ────────────────────────────
log "waiting for inference service to be ready (max 15s)"
READY=false
for _ in $(seq 1 15); do
  if ! kill -0 "${INFERENCE_PID}" 2>/dev/null; then
    err "inference service died during startup — last log:"
    tail -30 "${INFERENCE_LOG}" >&2 || true
    exit 1
  fi
  RESP=$(curl -s --max-time 1 http://127.0.0.1:8001/health || true)
  if [[ "${RESP}" == *'"model_loaded":true'* ]]; then
    READY=true
    log "  ready: ${RESP}"
    break
  fi
  sleep 1
done

if [[ "${READY}" != "true" ]]; then
  err "inference service did not become ready within 15s"
  err "  last log:"
  tail -30 "${INFERENCE_LOG}" >&2 || true
  kill -TERM "${INFERENCE_PID}" 2>/dev/null || true
  exit 1
fi

# ── 5. launch Go trader (foreground) ───────────────────────────────────────────
# Clean shutdown handler: on script exit (Ctrl+C or natural), stop inference too.
cleanup() {
  log "cleaning up"
  if kill -0 "${INFERENCE_PID}" 2>/dev/null; then
    log "  stopping inference pid=${INFERENCE_PID}"
    kill -TERM "${INFERENCE_PID}" 2>/dev/null || true
    # Give it 3s, then SIGKILL.
    for _ in 1 2 3; do
      kill -0 "${INFERENCE_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -0 "${INFERENCE_PID}" 2>/dev/null && kill -9 "${INFERENCE_PID}" 2>/dev/null || true
  fi
  log "done. inference log: ${INFERENCE_LOG}"
}
trap cleanup EXIT INT TERM

log "starting Go trader (Ctrl+C to stop everything)"
log "  args: $*"
log "  run logs will appear in live-trader/go/logs/runs/$(date +%Y-%m-%d)/<run_id>/"

cd "${PROJECT_ROOT}/live-trader/go"
exec ./go "$@"
