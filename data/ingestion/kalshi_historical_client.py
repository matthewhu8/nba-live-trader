"""
Kalshi settled market metadata backfill.

KXNBASPREAD launched October 21, 2025 (start of 2025-26 NBA season).
All settled markets post-date the historical API cutoff (March 2025),
so candlestick data is NOT available via the historical endpoint.

What IS available:
  - Settled market metadata: tickers, spread lines, open/close times,
    final resolution price, volume — via the live /events and /markets API.

This module downloads that metadata and writes it to:
  data/raw/kalshi_markets_settled.parquet

One row per market (each game generates ~11 spread-line markets).
Re-running is idempotent — re-fetches and overwrites the full file.

Once the historical API cutoff advances past October 2025 (likely 2026+),
the candlestick endpoint will become available for these markets.
The `fetch_candlesticks` method is retained for that future use.
"""

import asyncio
import base64
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

BASE_URL        = "https://api.elections.kalshi.com/trade-api/v2"
SETTLED_PATH    = Path("data/raw/kalshi_markets_settled.parquet")
CANDLESTICK_DIR = Path("data/raw/kalshi_candlesticks")

_EVENT_RE = re.compile(r"KXNBASPREAD-(\d{2})([A-Z]{3})(\d{2})([A-Z]{3})([A-Z]{3})")
_SPREAD_RE = re.compile(r"-([A-Z]{3})(\d+)$")

_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

SETTLED_SCHEMA = pa.schema([
    pa.field("event_ticker",    pa.string()),
    pa.field("market_ticker",   pa.string()),
    pa.field("game_date",       pa.date32()),
    pa.field("away_team",       pa.string()),
    pa.field("home_team",       pa.string()),
    pa.field("spread_team",     pa.string()),
    pa.field("spread_points",   pa.int32()),
    pa.field("open_time",       pa.timestamp("us", tz="UTC")),
    pa.field("close_time",      pa.timestamp("us", tz="UTC")),
    pa.field("result",          pa.string()),
    pa.field("last_price",      pa.int32()),
    pa.field("yes_bid",         pa.int32()),
    pa.field("yes_ask",         pa.int32()),
    pa.field("volume",          pa.int64()),
])

CANDLESTICK_SCHEMA = pa.schema([
    pa.field("market_ticker",   pa.string()),
    pa.field("game_id",         pa.string()),
    pa.field("period_end_ts",   pa.timestamp("us", tz="UTC")),
    pa.field("yes_bid_open",    pa.int32()),
    pa.field("yes_bid_high",    pa.int32()),
    pa.field("yes_bid_low",     pa.int32()),
    pa.field("yes_bid_close",   pa.int32()),
    pa.field("yes_ask_open",    pa.int32()),
    pa.field("yes_ask_high",    pa.int32()),
    pa.field("yes_ask_low",     pa.int32()),
    pa.field("yes_ask_close",   pa.int32()),
    pa.field("price_open",      pa.int32()),
    pa.field("price_high",      pa.int32()),
    pa.field("price_low",       pa.int32()),
    pa.field("price_close",     pa.int32()),
    pa.field("volume",          pa.int64()),
    pa.field("open_interest",   pa.int64()),
])


def iso_to_unix(s: str) -> int:
    s = s.rstrip("Z")
    if "." in s:
        s = s[: s.index(".")]
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())


def iso_to_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.rstrip("Z")
    if "." in s:
        s = s[: s.index(".")]
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def parse_event_ticker(event_ticker: str) -> dict:
    m = _EVENT_RE.search(event_ticker)
    if not m:
        return {}
    yy, mon, dd, away, home = m.groups()
    month = _MONTH_MAP.get(mon)
    if not month:
        return {}
    year = 2000 + int(yy)
    return {
        "game_date": datetime(year, month, int(dd), tzinfo=timezone.utc).date(),
        "away_team": away,
        "home_team": home,
    }


def parse_market_spread(market_ticker: str) -> tuple[str, int]:
    m = _SPREAD_RE.search(market_ticker)
    if not m:
        return ("", 0)
    return (m.group(1), int(m.group(2)))


class KalshiAuth:
    """Generates RSA-PSS signed headers for every Kalshi API request."""

    def __init__(
        self,
        key_id: str,
        private_key_path: str | None = None,
        private_key_pem: str | None = None,
    ) -> None:
        self._key_id = key_id
        if private_key_pem:
            pem_bytes = private_key_pem.encode()
        elif private_key_path:
            with open(private_key_path, "rb") as f:
                pem_bytes = f.read()
        else:
            raise ValueError("Either private_key_path or private_key_pem must be provided")
        self._private_key = serialization.load_pem_private_key(pem_bytes, password=None)

    def headers(self, method: str, path: str) -> dict[str, str]:
        ts_ms = str(int(time.time() * 1000))
        message = f"{ts_ms}{method.upper()}{path}".encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY":       self._key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        }


class KalshiHistoricalClient:

    def __init__(self, auth: KalshiAuth) -> None:
        self._auth                 = auth
        self._semaphore            = asyncio.Semaphore(4)
        self._last_request_time    = 0.0
        self._min_request_interval = 0.12

    async def _get(
        self,
        session: aiohttp.ClientSession,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict | None:
        url = f"{BASE_URL}{path}"
        async with self._semaphore:
            now = time.monotonic()
            gap = self._min_request_interval - (now - self._last_request_time)
            if gap > 0:
                await asyncio.sleep(gap)
            self._last_request_time = time.monotonic()

            headers = self._auth.headers("GET", path)
            try:
                async with session.get(url, headers=headers, params=params) as resp:
                    if resp.status == 429:
                        logger.warning("Rate limited on GET %s — backing off 5s", path)
                        await asyncio.sleep(5)
                        return None
                    if resp.status != 200:
                        body = await resp.text()
                        logger.debug("GET %s → %d: %s", path, resp.status, body[:200])
                        return None
                    return await resp.json()
            except aiohttp.ClientError as exc:
                logger.warning("Network error on GET %s: %s", path, exc)
                return None

    async def get_historical_cutoff(self, session: aiohttp.ClientSession) -> datetime:
        data = await self._get(session, "/historical/cutoff")
        if data is None:
            raise RuntimeError("Failed to fetch historical cutoff")
        ts_str = data["market_settled_ts"]
        return datetime.fromisoformat(ts_str.rstrip("Z")).replace(tzinfo=timezone.utc)

    async def list_settled_events(
        self,
        session: aiohttp.ClientSession,
        series_ticker: str = "KXNBASPREAD",
    ) -> list[dict]:
        events: list[dict] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {
                "series_ticker": series_ticker,
                "status":        "settled",
                "limit":         200,
            }
            if cursor:
                params["cursor"] = cursor
            data = await self._get(session, "/events", params=params)
            if data is None:
                logger.error("Failed to fetch events page — stopping pagination")
                break
            batch  = data.get("events", [])
            events.extend(batch)
            cursor = data.get("cursor")
            logger.debug("Fetched %d events (total: %d)", len(batch), len(events))
            if not cursor or not batch:
                break
        return events

    async def fetch_event_markets(
        self,
        session: aiohttp.ClientSession,
        event_ticker: str,
    ) -> list[dict]:
        data = await self._get(
            session,
            f"/events/{event_ticker}",
            params={"with_nested_markets": "true"},
        )
        if data is None:
            return []
        return data.get("event", {}).get("markets", [])

    async def fetch_candlesticks(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
    ) -> list[dict]:
        path   = f"/historical/markets/{ticker}/candlesticks"
        params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval}
        for attempt in range(2):
            data = await self._get(session, path, params=params)
            if data is not None:
                return data.get("candlesticks", [])
            if attempt == 0:
                await asyncio.sleep(2)
        return []

    def _market_to_row(self, market: dict, event_ticker: str) -> dict | None:
        ticker     = market.get("ticker", "")
        event_info = parse_event_ticker(event_ticker)
        spread_team, spread_pts = parse_market_spread(ticker)
        if not event_info or not spread_team:
            logger.debug("Could not parse ticker: %s", ticker)
            return None
        open_dt  = iso_to_dt(market.get("open_time"))
        close_dt = iso_to_dt(market.get("close_time"))
        return {
            "event_ticker":  event_ticker,
            "market_ticker": ticker,
            "game_date":     event_info["game_date"],
            "away_team":     event_info["away_team"],
            "home_team":     event_info["home_team"],
            "spread_team":   spread_team,
            "spread_points": spread_pts,
            "open_time":     open_dt,
            "close_time":    close_dt,
            "result":        market.get("result", ""),
            "last_price":    int(market.get("last_price", 0) or 0),
            "yes_bid":       int(market.get("yes_bid", 0) or 0),
            "yes_ask":       int(market.get("yes_ask", 0) or 0),
            "volume":        int(market.get("volume", 0) or 0),
        }

    async def _fetch_event_rows(
        self,
        session: aiohttp.ClientSession,
        event: dict,
        idx: int,
        total: int,
    ) -> list[dict]:
        event_ticker = event.get("event_ticker", "")
        markets = event.get("markets", [])
        if not markets:
            markets = await self.fetch_event_markets(session, event_ticker)
        rows = []
        for m in markets:
            row = self._market_to_row(m, event_ticker)
            if row:
                rows.append(row)
        logger.info("[%d/%d] %s — %d markets", idx, total, event_ticker, len(rows))
        return rows

    async def backfill_settled_metadata(self, series_ticker: str = "KXNBASPREAD") -> None:
        SETTLED_PATH.parent.mkdir(parents=True, exist_ok=True)
        async with aiohttp.ClientSession() as session:
            cutoff = await self.get_historical_cutoff(session)
            logger.info("Historical cutoff: %s", cutoff.isoformat())
            logger.info("Paginating settled %s events...", series_ticker)
            events = await self.list_settled_events(session, series_ticker)
            logger.info("Found %d settled events", len(events))
            total    = len(events)
            all_rows: list[dict] = []
            for i, ev in enumerate(events):
                rows = await self._fetch_event_rows(session, ev, i + 1, total)
                if not rows:
                    logger.debug("Retrying %s after 3s...", ev.get("event_ticker"))
                    await asyncio.sleep(3)
                    rows = await self._fetch_event_rows(session, ev, i + 1, total)
                all_rows.extend(rows)
        logger.info("Writing %d market rows to %s", len(all_rows), SETTLED_PATH)
        self._write_settled_parquet(all_rows)
        logger.info("Backfill complete — %d events, %d markets", total, len(all_rows))

    def _write_settled_parquet(self, rows: list[dict]) -> None:
        if not rows:
            logger.warning("No rows to write")
            return
        table = pa.table(
            {
                "event_ticker":  pa.array([r["event_ticker"]  for r in rows], pa.string()),
                "market_ticker": pa.array([r["market_ticker"] for r in rows], pa.string()),
                "game_date":     pa.array([r["game_date"]     for r in rows], pa.date32()),
                "away_team":     pa.array([r["away_team"]     for r in rows], pa.string()),
                "home_team":     pa.array([r["home_team"]     for r in rows], pa.string()),
                "spread_team":   pa.array([r["spread_team"]   for r in rows], pa.string()),
                "spread_points": pa.array([r["spread_points"] for r in rows], pa.int32()),
                "open_time":     pa.array([r["open_time"]     for r in rows], pa.timestamp("us", tz="UTC")),
                "close_time":    pa.array([r["close_time"]    for r in rows], pa.timestamp("us", tz="UTC")),
                "result":        pa.array([r["result"]        for r in rows], pa.string()),
                "last_price":    pa.array([r["last_price"]    for r in rows], pa.int32()),
                "yes_bid":       pa.array([r["yes_bid"]       for r in rows], pa.int32()),
                "yes_ask":       pa.array([r["yes_ask"]       for r in rows], pa.int32()),
                "volume":        pa.array([r["volume"]        for r in rows], pa.int64()),
            },
            schema=SETTLED_SCHEMA,
        )
        pq.write_table(table, SETTLED_PATH)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    load_dotenv()
    key_id   = os.environ.get("API_KEY_ID")
    pem_path = os.environ.get("PRIVATE_RSA_KEY")
    if not key_id or not pem_path:
        raise RuntimeError("API_KEY_ID and PRIVATE_RSA_KEY must be set in environment")
    auth   = KalshiAuth(key_id=key_id, private_key_path=pem_path)
    client = KalshiHistoricalClient(auth)
    await client.backfill_settled_metadata("KXNBASPREAD")


if __name__ == "__main__":
    asyncio.run(main())
