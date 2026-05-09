"""
Kalshi live price recorder — WebSocket-based.

Records real-time bid/ask for every NBA game-winner market during live games
and bulk-INSERTs into the MotherDuck `kalshi_ticks` table.

Key design decisions:
  - Only records KXNBASPREAD markets with spread ≤ 3 (moneyline proxies).
  - Optionally filtered to a single game by team_codes (e.g. ("DAL", "BOS")).
    The daemon uses this to run one recorder per game concurrently.
  - Flushes to MotherDuck every 5 minutes OR every 2,000 ticks, whichever first.
  - Auto-exits when all subscribed markets settle AND MIN_RUNTIME_HOURS elapsed.

Normally launched by recorder_daemon.py (one instance per game).
Can still be run standalone for a full game-night sweep:
  python data/ingestion/kalshi_recorder.py
"""

import asyncio
import json
import logging
import os
import re
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import pyarrow as pa
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.ingestion.kalshi_historical_client import (
    KalshiAuth,
    kalshi_auth_from_env,
    kalshi_ssl_context,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL  = os.environ.get("KALSHI_REST_BASE_URL", "https://external-api.kalshi.com/trade-api/v2").rstrip("/")
WS_URL    = os.environ.get("KALSHI_WS_URL", "wss://external-api-ws.kalshi.com/trade-api/ws/v2")
SERIES    = "KXNBASPREAD"
LOCK_FILE = Path("logs/recorder/recorder.pid")

# Shared lock — only one DuckDB/MotherDuck connection opens at a time.
# Without this, N concurrent game recorders all flush simultaneously,
# each allocating ~700MB VM for a DuckDB connection → OOM.
_md_flush_lock: asyncio.Lock | None = None

def _get_flush_lock() -> asyncio.Lock:
    global _md_flush_lock
    if _md_flush_lock is None:
        _md_flush_lock = asyncio.Lock()
    return _md_flush_lock

# Only record the moneyline-proxy markets: spread ≤ 3.
# e.g., BOS1, BOS2, BOS3, DAL1, DAL2, DAL3 — the ones that behave like win/loss.
MAX_SPREAD = 3

FLUSH_INTERVAL_SECONDS    = 300   # 5 minutes — reduces MotherDuck connections
FLUSH_TICK_THRESHOLD      = 2000  # also flush if buffer exceeds this many ticks
RECONNECT_DELAY_SECONDS   = 5
MAX_MARKETS_PER_SUB       = 200
MIN_RUNTIME_HOURS         = 2.0   # don't check for settlement before this
MAX_RUNTIME_HOURS         = 7.0   # hard stop regardless (safety valve)

# A market is settled when yes_last reaches 1 (NO wins) or 99 (YES wins)
SETTLEMENT_THRESHOLD_LOW  = 2
SETTLEMENT_THRESHOLD_HIGH = 98

# Regex: KXNBASPREAD-26MAR06DALBOS-BOS1 → spread team=BOS, spread value=1
_TICKER_RE = re.compile(r"KXNBASPREAD-\w+-([A-Z]+)(\d+)$")

TICK_SCHEMA = pa.schema([
    pa.field("market_ticker", pa.string()),
    pa.field("ts",            pa.timestamp("us", tz="UTC")),
    pa.field("game_id",       pa.string()),
    pa.field("yes_bid",       pa.int32()),
    pa.field("yes_ask",       pa.int32()),
    pa.field("yes_last",      pa.int32()),
    pa.field("volume",        pa.int64()),
    pa.field("open_interest", pa.int64()),
])


# ---------------------------------------------------------------------------
# Lock file helpers
# ---------------------------------------------------------------------------

def _acquire_lock() -> bool:
    """
    Write a PID lockfile so only one recorder instance runs at a time.
    Returns True if this process should proceed, False if another is already running.
    """
    if LOCK_FILE.exists():
        try:
            existing_pid = int(LOCK_FILE.read_text().strip())
            os.kill(existing_pid, 0)  # raises OSError if process doesn't exist
            logger.warning(
                "Recorder already running as PID %d — exiting. "
                "Kill that process first if it is stuck: kill %d",
                existing_pid, existing_pid,
            )
            return False
        except (OSError, ValueError):
            logger.info("Stale lock file found (PID no longer running) — proceeding.")

    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))
    return True


def _release_lock() -> None:
    """Remove lockfile if it belongs to this process."""
    try:
        if LOCK_FILE.exists() and int(LOCK_FILE.read_text().strip()) == os.getpid():
            LOCK_FILE.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Market filter
# ---------------------------------------------------------------------------

def _is_moneyline_proxy(ticker: str) -> bool:
    """Return True if this ticker is a spread ≤ MAX_SPREAD market."""
    m = _TICKER_RE.search(ticker)
    if not m:
        return False
    spread_value = int(m.group(2))
    return spread_value <= MAX_SPREAD


# ---------------------------------------------------------------------------
# Tick buffer → parquet writer
# ---------------------------------------------------------------------------

class TickBuffer:
    """
    Accumulates tick rows in memory, bulk-INSERTs to MotherDuck on flush.
    Flushes every FLUSH_INTERVAL_SECONDS or when FLUSH_TICK_THRESHOLD rows
    accumulate — whichever comes first. One MotherDuck connection per flush
    batch to avoid holding long-lived connections across game sessions.
    """

    def __init__(self) -> None:
        self._rows: list[dict] = []

    @property
    def size(self) -> int:
        return len(self._rows)

    def add(self, ticker: str, tick: dict) -> None:
        self._rows.append(tick)

    async def flush(self) -> int:
        if not self._rows:
            return 0
        rows = self._rows
        self._rows = []
        # Serialize all concurrent game recorders — only one DuckDB connection at a time
        async with _get_flush_lock():
            return await asyncio.get_event_loop().run_in_executor(None, self._write, rows)

    def _write(self, rows: list[dict]) -> int:
        token = os.environ.get("MOTHERDUCK_TOKEN")
        if not token:
            raise RuntimeError("MOTHERDUCK_TOKEN not set — cannot flush ticks to MotherDuck")

        import duckdb

        batch = pa.table(
            {
                "market_ticker": pa.array([r["market_ticker"] for r in rows], pa.string()),
                "ts":            pa.array([r["ts"] for r in rows], pa.timestamp("us", tz="UTC")),
                "game_id":       pa.array([r["game_id"] for r in rows], pa.string()),
                "yes_bid":       pa.array([r["yes_bid"] for r in rows], pa.int32()),
                "yes_ask":       pa.array([r["yes_ask"] for r in rows], pa.int32()),
                "yes_last":      pa.array([r["yes_last"] for r in rows], pa.int32()),
                "volume":        pa.array([r["volume"] for r in rows], pa.int64()),
                "open_interest": pa.array([r["open_interest"] for r in rows], pa.int64()),
            },
            schema=TICK_SCHEMA,
        )

        conn = duckdb.connect(f"md:kalshi_trading?motherduck_token={token}")
        try:
            conn.execute("SET memory_limit='256MB'")
            conn.register("_tick_batch", batch)
            conn.execute("INSERT INTO kalshi_ticks BY NAME SELECT * FROM _tick_batch")
        finally:
            conn.close()

        return len(rows)


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class KalshiRecorder:

    def __init__(
        self,
        auth: KalshiAuth,
        team_codes: tuple[str, str] | None = None,
        game_id: str | None = None,
    ) -> None:
        self._auth       = auth
        self._buffer     = TickBuffer()
        self._running    = False
        self._start_time: float = 0.0
        self._game_id    = game_id or ""

        # When set, only record markets whose ticker contains both team codes.
        # e.g. ("DAL", "BOS") filters to KXNBASPREAD-26MAR06DALBOS-* markets only.
        self._team_codes = team_codes

        # Track last known yes_last per ticker to detect settlement
        self._last_price: dict[str, int] = {}
        self._subscribed_tickers: list[str] = []

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    async def _discover_moneyline_markets(
        self, session: aiohttp.ClientSession
    ) -> list[str]:
        """
        Return tickers for all open KXNBASPREAD markets with spread ≤ MAX_SPREAD.
        These are the moneyline-proxy markets: the contract pays if team wins
        by more than 1 (or 2, or 3) points — effectively a win/loss market.
        """
        all_tickers: list[str] = []
        cursor: str | None = None

        while True:
            params: dict[str, Any] = {
                "series_ticker": SERIES,
                "status":        "open",
                "limit":         200,
            }
            if cursor:
                params["cursor"] = cursor

            path = "/markets"
            headers = self._auth.headers("GET", path)

            async with session.get(f"{BASE_URL}{path}", headers=headers, params=params) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("Failed to list markets: %d %s", resp.status, body[:200])
                    break
                raw = await resp.json()

            batch  = raw.get("markets", [])
            cursor = raw.get("cursor")

            for m in batch:
                ticker = m.get("ticker", "")
                if not _is_moneyline_proxy(ticker):
                    continue
                if self._team_codes:
                    t1, t2 = self._team_codes
                    if t1 not in ticker or t2 not in ticker:
                        continue
                all_tickers.append(ticker)

            if not cursor or not batch:
                break

        logger.info(
            "Discovered %d moneyline-proxy markets (spread ≤ %d)",
            len(all_tickers), MAX_SPREAD,
        )
        return all_tickers

    # ------------------------------------------------------------------
    # Tick parsing
    # ------------------------------------------------------------------

    def _parse_tick(self, msg: dict) -> dict | None:
        msg_type = msg.get("type")
        inner    = msg.get("msg")

        if not isinstance(inner, dict) or msg_type != "ticker":
            return None

        ticker = inner.get("market_ticker")
        if not ticker:
            return None

        def to_cents(raw: str | float | None) -> int:
            if not raw:
                return 0
            return int(round(float(raw) * 100))

        def to_contracts(raw: str | float | None) -> int:
            if not raw:
                return 0
            return int(round(float(raw)))

        yes_bid  = to_cents(inner.get("yes_bid_dollars"))
        yes_ask  = to_cents(inner.get("yes_ask_dollars"))
        yes_last = to_cents(inner.get("price_dollars"))

        # Track last traded price for settlement detection — must happen before
        # the yes_bid filter so end-of-game settled prices are still registered.
        if yes_last > 0:
            self._last_price[ticker] = yes_last

        # Only record while market is actively quoted (game is live).
        # Pre-game: yes_bid == 0 (no quotes yet).
        # Post-game: market settles, bid collapses to 0.
        if yes_bid == 0:
            return None

        return {
            "market_ticker": ticker,
            "ts":            datetime.now(tz=timezone.utc),
            "game_id":       self._game_id,
            "yes_bid":       yes_bid,
            "yes_ask":       yes_ask,
            "yes_last":      yes_last,
            "volume":        to_contracts(inner.get("volume_fp")),
            "open_interest": to_contracts(inner.get("open_interest_fp")),
        }

    # ------------------------------------------------------------------
    # Settlement detection
    # ------------------------------------------------------------------

    def _all_games_settled(self) -> bool:
        """
        Returns True when every subscribed market has settled.
        A market is settled when its last traded price is ≤ 2 or ≥ 98.
        Only checked after MIN_RUNTIME_HOURS to avoid false positives
        from pre-game prices.
        """
        elapsed_hours = (time.monotonic() - self._start_time) / 3600.0
        if elapsed_hours < MIN_RUNTIME_HOURS:
            return False

        if not self._subscribed_tickers:
            return False

        # Need a last_price reading for every subscribed market
        for ticker in self._subscribed_tickers:
            price = self._last_price.get(ticker)
            if price is None:
                return False  # no trade yet — not settled
            if SETTLEMENT_THRESHOLD_LOW < price < SETTLEMENT_THRESHOLD_HIGH:
                return False  # still mid-game

        logger.info("All %d markets settled — games are over.", len(self._subscribed_tickers))
        return True

    # ------------------------------------------------------------------
    # WebSocket session
    # ------------------------------------------------------------------

    async def _subscribe(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        tickers: list[str],
        sub_id: int,
    ) -> None:
        for i in range(0, len(tickers), MAX_MARKETS_PER_SUB):
            batch = tickers[i: i + MAX_MARKETS_PER_SUB]
            await ws.send_json({
                "id":  sub_id + i,
                "cmd": "subscribe",
                "params": {
                    "channels":       ["ticker"],
                    "market_tickers": batch,
                },
            })

    async def _run_session(self, session: aiohttp.ClientSession) -> bool:
        """
        One WebSocket connection lifetime.
        Returns True if we should stop (all settled or max runtime),
        False if we should reconnect.
        """
        tickers = await self._discover_moneyline_markets(session)
        if not tickers:
            logger.info("No open moneyline markets found — waiting 60s")
            await asyncio.sleep(60)
            return False

        self._subscribed_tickers = tickers

        ws_path = "/trade-api/ws/v2"
        headers = self._auth.headers("GET", ws_path)

        async with session.ws_connect(WS_URL, headers=headers) as ws:
            await self._subscribe(ws, tickers, sub_id=1)
            logger.info("Subscribed to %d markets", len(tickers))

            flush_deadline     = time.monotonic() + FLUSH_INTERVAL_SECONDS
            ticks_this_session = 0

            async for raw_msg in ws:
                # Hard stop check
                elapsed_hours = (time.monotonic() - self._start_time) / 3600.0
                if elapsed_hours >= MAX_RUNTIME_HOURS:
                    logger.info("Max runtime (%.1fh) reached — stopping.", MAX_RUNTIME_HOURS)
                    await self._buffer.flush()
                    return True

                if raw_msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        msg = json.loads(raw_msg.data)
                    except json.JSONDecodeError:
                        continue

                    messages = msg if isinstance(msg, list) else [msg]
                    for m in messages:
                        if not isinstance(m, dict):
                            continue
                        tick = self._parse_tick(m)
                        if tick:
                            self._buffer.add(tick["market_ticker"], tick)
                            ticks_this_session += 1

                elif raw_msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    logger.warning("WebSocket closed/errored — will reconnect")
                    break

                # Flush on time interval OR tick count threshold
                should_flush = (
                    time.monotonic() >= flush_deadline
                    or self._buffer.size >= FLUSH_TICK_THRESHOLD
                )
                if should_flush:
                    flushed = await self._buffer.flush()
                    logger.info(
                        "Flushed %d rows | session ticks: %d | elapsed: %.1fh",
                        flushed, ticks_this_session,
                        (time.monotonic() - self._start_time) / 3600.0,
                    )
                    flush_deadline = time.monotonic() + FLUSH_INTERVAL_SECONDS

                    if self._all_games_settled():
                        return True

        flushed = await self._buffer.flush()
        logger.info("Session ended — flushed %d rows", flushed)
        return False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running    = True
        self._start_time = time.monotonic()

        consecutive_failures = 0

        connector = aiohttp.TCPConnector(ssl=kalshi_ssl_context())
        async with aiohttp.ClientSession(connector=connector) as session:
            while self._running:
                # Check max runtime at the top of every reconnect cycle.
                # This fires even when no WebSocket messages arrive (e.g. no active
                # markets, or a silent WebSocket that never sends ticks).
                elapsed_hours = (time.monotonic() - self._start_time) / 3600.0
                if elapsed_hours >= MAX_RUNTIME_HOURS:
                    logger.info("Max runtime (%.1fh) reached — stopping.", MAX_RUNTIME_HOURS)
                    await self._buffer.flush()
                    return

                remaining_secs = (MAX_RUNTIME_HOURS - elapsed_hours) * 3600.0

                try:
                    # asyncio.wait_for ensures _run_session is cancelled after
                    # remaining_secs even if no messages ever arrive (no-market nights,
                    # stale WebSocket). This is the primary fix for zombie accumulation.
                    done = await asyncio.wait_for(
                        self._run_session(session),
                        timeout=remaining_secs,
                    )
                    if done:
                        logger.info("Recorder finished cleanly.")
                        return
                    consecutive_failures = 0

                except asyncio.TimeoutError:
                    logger.info("Max runtime (%.1fh) reached — stopping.", MAX_RUNTIME_HOURS)
                    await self._buffer.flush()
                    return

                except asyncio.CancelledError:
                    logger.info("Cancelled — flushing and exiting")
                    await self._buffer.flush()
                    return

                except Exception as exc:
                    consecutive_failures += 1
                    delay = min(RECONNECT_DELAY_SECONDS * consecutive_failures, 60)
                    logger.error(
                        "Session error (%s) — reconnecting in %ds (failure #%d)",
                        exc, delay, consecutive_failures, exc_info=True,
                    )
                    await asyncio.sleep(delay)

    def stop(self) -> None:
        self._running = False
        # Flush is async — the run() loop will flush on its next CancelledError/exit
        logger.info("Recorder stopped")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    load_dotenv()

    if not _acquire_lock():
        return  # Another instance is already running

    try:
        auth = kalshi_auth_from_env()
    except RuntimeError:
        _release_lock()
        raise

    recorder = KalshiRecorder(auth)

    loop      = asyncio.get_running_loop()
    main_task = asyncio.current_task()

    def _handle_signal() -> None:
        logger.info("Signal received — flushing and shutting down")
        recorder.stop()
        if main_task and not main_task.done():
            main_task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    logger.info(
        "Starting Kalshi recorder | PID=%d | series=%s | max_spread=%d | max_runtime=%.1fh",
        os.getpid(), SERIES, MAX_SPREAD, MAX_RUNTIME_HOURS,
    )
    try:
        await recorder.run()
    except asyncio.CancelledError:
        pass  # already handled in run() / _handle_signal
    finally:
        _release_lock()


if __name__ == "__main__":
    asyncio.run(main())
