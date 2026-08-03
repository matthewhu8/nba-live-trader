"""
Thin synchronous Kalshi client for the soccer break-fade tool.

Read endpoints (events, markets, orderbook) work unauthenticated, so discovery
and live polling run without credentials. Order placement requires auth and is
only used in --live mode. Auth reuses the RSA-PSS signer from the historical
client so there is a single auth implementation in the repo.
"""

from __future__ import annotations

import logging
import math
import os
import time
import uuid
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get(
    "KALSHI_REST_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2"
).rstrip("/")


@dataclass
class TopOfBook:
    yes_bid: int   # best YES bid, cents
    yes_ask: int   # implied YES ask = 100 - best NO bid, cents

    @property
    def yes_mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2.0

    @property
    def spread(self) -> int:
        return self.yes_ask - self.yes_bid


def taker_fee_cents(price_cents: int, contracts: int) -> float:
    """Kalshi taker fee: ceil(0.07 * P * (1-P)) per contract, P in dollars."""
    p = price_cents / 100.0
    per_contract = math.ceil(0.07 * p * (1.0 - p) * 100.0) / 100.0
    return per_contract * contracts * 100.0  # return in cents


class KalshiClient:

    def __init__(self, auth=None, timeout: float = 10.0) -> None:
        self._auth = auth
        self._timeout = timeout
        self._session = requests.Session()

    def _headers(self, method: str, path: str) -> dict[str, str]:
        if self._auth is None:
            return {}
        return self._auth.headers(method, path)

    def _get(self, path: str, params: dict | None = None) -> dict | None:
        url = f"{BASE_URL}{path}"
        try:
            resp = self._session.get(
                url, headers=self._headers("GET", path), params=params, timeout=self._timeout
            )
        except requests.RequestException as exc:
            logger.warning("GET %s failed: %s", path, exc)
            return None
        if resp.status_code != 200:
            logger.debug("GET %s -> %d: %s", path, resp.status_code, resp.text[:200])
            return None
        return resp.json()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_open_events(self, series_ticker: str) -> list[dict]:
        events: list[dict] = []
        cursor: str | None = None
        while True:
            params = {"series_ticker": series_ticker, "status": "open", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            data = self._get("/events", params=params)
            if data is None:
                break
            batch = data.get("events", [])
            events.extend(batch)
            cursor = data.get("cursor")
            if not cursor or not batch:
                break
        return events

    def event_markets(self, event_ticker: str) -> list[dict]:
        data = self._get(f"/events/{event_ticker}", params={"with_nested_markets": "true"})
        if data is None:
            return []
        return data.get("event", {}).get("markets", [])

    def top_of_book(self, market_ticker: str) -> TopOfBook | None:
        data = self._get(f"/markets/{market_ticker}/orderbook")
        if data is None:
            return None
        book = data.get("orderbook", {})
        yes = book.get("yes") or []
        no = book.get("no") or []
        if not yes or not no:
            return None
        # Kalshi returns bid ladders ascending; best bid is the last (highest) price.
        best_yes_bid = int(yes[-1][0])
        best_no_bid = int(no[-1][0])
        return TopOfBook(yes_bid=best_yes_bid, yes_ask=100 - best_no_bid)

    # ------------------------------------------------------------------
    # Order placement (live mode only)
    # ------------------------------------------------------------------

    def place_market_order(self, ticker: str, side: str, contracts: int) -> dict | None:
        """
        Cross the spread to enter immediately at the break (a taker order).
        side is "yes" or "no". Requires auth. Returns the API response or None.
        """
        if self._auth is None:
            raise RuntimeError("Live order requires Kalshi auth — none configured")

        path = "/portfolio/orders"
        body = {
            "ticker": ticker,
            "action": "buy",
            "side": side.lower(),
            "count": contracts,
            "type": "market",
            "client_order_id": str(uuid.uuid4()),
        }
        headers = {**self._auth.headers("POST", path), "Content-Type": "application/json"}
        try:
            resp = self._session.post(
                f"{BASE_URL}{path}", headers=headers, json=body, timeout=self._timeout
            )
        except requests.RequestException as exc:
            logger.error("Order POST failed: %s", exc)
            return None
        if resp.status_code not in (200, 201):
            logger.error("Order rejected %d: %s", resp.status_code, resp.text[:300])
            return None
        return resp.json()
