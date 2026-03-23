"""
Probe: When did KXNBASPREAD launch?
- Pull settled markets back as far as the live API will paginate
- Find the oldest settled KXNBASPREAD market (series start date)
- Check if 2025-26 season events exist via the events endpoint
- Report how many games + markets are available
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from data.ingestion.kalshi_historical_client import KalshiAuth, BASE_URL

import aiohttp


async def get_raw(session, auth, path, params=None):
    resp = await session.get(
        f"{BASE_URL}{path}",
        headers=auth.headers("GET", path),
        params=params,
    )
    data = await resp.json()
    return resp.status, data


async def main() -> None:
    load_dotenv()
    auth = KalshiAuth(os.environ["API_KEY_ID"], os.environ["PRIVATE_RSA_KEY"])

    async with aiohttp.ClientSession() as session:

        # 1. Paginate ALL settled KXNBASPREAD events to find oldest
        print("=== Paginating ALL settled KXNBASPREAD events ===")
        cursor = None
        all_events = []
        page = 0
        while True:
            params = {"series_ticker": "KXNBASPREAD", "status": "settled", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            _, data = await get_raw(session, auth, "/events", params=params)
            batch = data.get("events", [])
            all_events.extend(batch)
            cursor = data.get("cursor")
            page += 1
            print(f"  Page {page}: {len(batch)} events (total so far: {len(all_events)})")
            if not cursor or not batch:
                break

        print(f"\nTotal settled KXNBASPREAD events: {len(all_events)}")
        if all_events:
            # Events come newest-first; last in list is oldest
            newest = all_events[0]
            oldest = all_events[-1]
            print(f"Newest: {newest.get('event_ticker')}  close={newest.get('close_date_iso') or newest.get('expected_expiration_time')}")
            print(f"Oldest: {oldest.get('event_ticker')}  close={oldest.get('close_date_iso') or oldest.get('expected_expiration_time')}")
            print("\nOldest 10 events:")
            for e in reversed(all_events[-10:]):
                print(f"  {e.get('event_ticker')}")

        # 2. Check how many markets per event on average
        if all_events:
            print(f"\n=== Markets per event (sample of 3) ===")
            for ev in all_events[:3]:
                et = ev.get("event_ticker")
                _, data = await get_raw(session, auth, "/events", params={
                    "series_ticker": "KXNBASPREAD",
                    "with_nested_markets": "true",
                    "limit": 1,
                })
                # We already fetched; just use the event directly
                _, nested = await get_raw(session, auth, f"/events/{et}",
                                          params={"with_nested_markets": "true"})
                mkts = nested.get("event", {}).get("markets", [])
                print(f"  {et}: {len(mkts)} markets")
                for m in mkts[:3]:
                    print(f"    {m['ticker']}  open={m['open_time']}  close={m['close_time']}")

        # 3. Try fetching candlesticks for the oldest event's first market
        if all_events:
            oldest_ticker = all_events[-1].get("event_ticker")
            print(f"\n=== Candlestick availability for oldest event: {oldest_ticker} ===")
            _, nested = await get_raw(session, auth, f"/events/{oldest_ticker}",
                                      params={"with_nested_markets": "true"})
            mkts = nested.get("event", {}).get("markets", [])
            if mkts:
                m0 = mkts[0]
                ticker = m0["ticker"]
                from data.ingestion.kalshi_historical_client import iso_to_unix
                start = iso_to_unix(m0["open_time"])
                end   = iso_to_unix(m0["close_time"])
                path  = f"/historical/markets/{ticker}/candlesticks"
                _, cdata = await get_raw(session, auth, path,
                                         params={"start_ts": start, "end_ts": end, "period_interval": 1})
                candles = cdata.get("candlesticks", [])
                print(f"  Market: {ticker}")
                print(f"  Candlestick status: {len(candles)} candles")
                if "error" in cdata:
                    print(f"  Error: {cdata['error']}")
                if candles:
                    print(f"  First candle: {json.dumps(candles[0], indent=4)}")

        # 4. Check if live settled markets have candlesticks
        print("\n=== Candlestick availability for a recently settled live market ===")
        _, mdata = await get_raw(session, auth, "/markets",
                                  params={"series_ticker": "KXNBASPREAD", "status": "settled", "limit": 1})
        recent = mdata.get("markets", [])
        if recent:
            m = recent[0]
            ticker = m["ticker"]
            from data.ingestion.kalshi_historical_client import iso_to_unix
            start = iso_to_unix(m["open_time"])
            end   = iso_to_unix(m["close_time"])
            _, cdata = await get_raw(session, auth, f"/historical/markets/{ticker}/candlesticks",
                                     params={"start_ts": start, "end_ts": end, "period_interval": 1})
            candles = cdata.get("candlesticks", [])
            print(f"  Market: {ticker}")
            print(f"  Status: {len(candles)} candles")
            if "error" in cdata:
                print(f"  Error: {cdata['error']}")


if __name__ == "__main__":
    asyncio.run(main())
