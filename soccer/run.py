"""
Live runner for the hydration-break fade strategy. Paper mode by default.

  # discover today's World Cup spread events and the near-50c market for each
  python -m soccer.run --discover

  # paper-trade one game (read-only orderbook polling, simulated fills)
  python -m soccer.run --event KXWCSPREAD-26JUN17ENGCRO --kickoff 2026-06-17T19:00:00Z

  # the same, actually sending taker orders (requires .env creds + explicit flag)
  python -m soccer.run --event ... --kickoff ... --live --contracts 10

Selecting the market: among the event's spread markets we pick the one whose YES
mid is closest to 50c — the "fifty-fifty" line — and trade only that one, whatever
its price. There is no band veto; the nearest-to-even line is always tradeable.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone

from soccer.espn_clock import EspnClock
from soccer.kalshi_client import KalshiClient, TopOfBook, taker_fee_cents
from soccer.match_clock import MatchClock
from soccer.strategy import BreakFadeConfig, BreakFadeStrategy, Decision, PriceSnapshot

logger = logging.getLogger("soccer.run")

SPREAD_SERIES = "KXWCSPREAD"
POLL_INTERVAL_SECONDS = 15.0
END_MATCH_MINUTE = 100.0  # stop after this estimated minute


def _parse_iso(s: str) -> datetime:
    s = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _teams_from_event(event_ticker: str) -> tuple[str, str]:
    """KXWCSPREAD-26JUN17ENGCRO -> ('ENG', 'CRO'). FIFA codes are 3 letters."""
    tail = event_ticker.split("-")[-1][-6:]
    return tail[:3], tail[3:]


def _load_auth_optional():
    """Return KalshiAuth if creds are present, else None (read-only mode)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from data.ingestion.kalshi_historical_client import kalshi_auth_from_env
        return kalshi_auth_from_env()
    except Exception as exc:  # noqa: BLE001 — creds optional for read-only use
        logger.info("No Kalshi auth (%s) — running read-only/paper.", exc)
        return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def pick_fifty_fifty_market(client: KalshiClient, event_ticker: str) -> tuple[str, TopOfBook] | None:
    """Return (ticker, book) for the spread market whose YES mid is nearest 50c."""
    markets = client.event_markets(event_ticker)
    best: tuple[str, TopOfBook] | None = None
    best_dist = 999.0
    for m in markets:
        ticker = m.get("ticker", "")
        book = client.top_of_book(ticker)
        if book is None:
            continue
        dist = abs(book.yes_mid - 50.0)
        if dist < best_dist:
            best_dist = dist
            best = (ticker, book)
    return best


def discover(client: KalshiClient) -> None:
    events = client.list_open_events(SPREAD_SERIES)
    if not events:
        print(f"No open {SPREAD_SERIES} events found.")
        return
    print(f"Open {SPREAD_SERIES} events: {len(events)}\n")
    for ev in events:
        et = ev.get("event_ticker", "")
        picked = pick_fifty_fifty_market(client, et)
        if picked is None:
            print(f"  {et:32s}  (no quotes yet)")
            continue
        ticker, book = picked
        print(f"  {et:32s}  -> {ticker:40s}  yes_mid={book.yes_mid:5.1f}c  spread={book.spread}c")


# ---------------------------------------------------------------------------
# Live loop
# ---------------------------------------------------------------------------

def _paper_entry_cost(side: str, book: TopOfBook) -> int:
    """Cents paid per contract crossing the spread to open `side`."""
    return book.yes_ask if side == "YES" else 100 - book.yes_bid


def _paper_exit_value(side: str, book: TopOfBook) -> int:
    """Cents received per contract crossing the spread to close `side`."""
    return book.yes_bid if side == "YES" else 100 - book.yes_ask


def run_live(
    client: KalshiClient,
    event_ticker: str,
    espn: EspnClock,
    clock: MatchClock | None,
    contracts: int,
    live: bool,
) -> None:
    picked = pick_fifty_fifty_market(client, event_ticker)
    if picked is None:
        print(f"Could not find a quoted spread market for {event_ticker}.")
        return
    ticker, book0 = picked

    strategy = BreakFadeStrategy(BreakFadeConfig())
    mode = "LIVE" if live else "PAPER"
    breaks = ", ".join(f"{b:.0f}'" for b in strategy.config.break_minutes)
    clock_src = "ESPN live clock" + (" + wall-clock fallback" if clock else "")
    print(
        f"[{mode}] {event_ticker}\n"
        f"  market : {ticker}  (yes_mid={book0.yes_mid:.1f}c at start)\n"
        f"  clock  : {clock_src}\n"
        f"  breaks : {breaks}  | contracts: {contracts}  | poll: {POLL_INTERVAL_SECONDS:.0f}s\n"
    )

    entry_cost: int | None = None

    while True:
        now = datetime.now(timezone.utc)

        minute = espn.live_minute(now)
        src = "espn"
        if espn.last_state == "post":
            print("ESPN reports the match has finished — stopping.")
            return
        if minute is None and clock is not None:
            minute = clock.match_minute(now)
            src = "est"
        if minute is None:
            state = espn.last_state or "not found"
            print(f"  waiting for live clock (ESPN: {state}) ...")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        if minute >= END_MATCH_MINUTE:
            print(f"Reached minute {minute:.0f} ({src}) — stopping.")
            return

        book = client.top_of_book(ticker)
        if book is None:
            print(f"  {minute:5.1f}'  no quotes")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        snap = PriceSnapshot(ts=now, match_minute=minute, yes_bid=book.yes_bid, yes_ask=book.yes_ask)
        decision = strategy.on_snapshot(snap)

        status = (
            f"  {minute:5.1f}'[{src}]  yes={book.yes_bid:2d}/{book.yes_ask:2d}c "
            f"mid={book.yes_mid:4.1f}  {decision.action}"
        )
        if decision.reason:
            status += f"  ({decision.reason})"
        print(status)

        if decision.action == "ENTER":
            entry_cost = _paper_entry_cost(decision.side, book)
            fee = taker_fee_cents(entry_cost, contracts)
            print(
                f"      >>> ENTER {decision.side} x{contracts} @ {entry_cost}c "
                f"(taker fee ~{fee/100:.2f}$)"
            )
            if live:
                resp = client.place_market_order(ticker, decision.side, contracts)
                print(f"      live order: {resp}")

        elif decision.action == "EXIT" and entry_cost is not None:
            exit_val = _paper_exit_value(decision.side, book)
            fees = taker_fee_cents(entry_cost, contracts) + taker_fee_cents(exit_val, contracts)
            gross = (exit_val - entry_cost) * contracts
            net = gross - fees
            print(
                f"      <<< EXIT {decision.side} @ {exit_val}c | "
                f"gross {gross/100:+.2f}$  net {net/100:+.2f}$ (after ~{fees/100:.2f}$ fees)"
            )
            entry_cost = None
            if live:
                opposite = "no" if decision.side == "YES" else "yes"
                resp = client.place_market_order(ticker, opposite, contracts)
                print(f"      live close: {resp}")

        time.sleep(POLL_INTERVAL_SECONDS)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="World Cup hydration-break fade trader")
    parser.add_argument("--discover", action="store_true", help="list open spread events + near-50c market")
    parser.add_argument("--event", help="KXWCSPREAD event ticker to trade")
    parser.add_argument("--kickoff", help="kickoff ISO-8601 for wall-clock fallback (e.g. 2026-06-17T19:00:00Z)")
    parser.add_argument("--league", default="fifa.world", help="ESPN soccer league slug")
    parser.add_argument("--contracts", type=int, default=10)
    parser.add_argument("--live", action="store_true", help="actually send orders (default: paper)")
    args = parser.parse_args()

    auth = _load_auth_optional()
    client = KalshiClient(auth=auth)

    if args.discover:
        discover(client)
        return

    if not args.event:
        parser.error("--event is required to trade (or use --discover)")

    if args.live and auth is None:
        parser.error("--live requires Kalshi credentials in .env")

    espn = EspnClock(team_codes=_teams_from_event(args.event), league=args.league)
    clock = MatchClock(kickoff=_parse_iso(args.kickoff)) if args.kickoff else None
    run_live(client, args.event, espn, clock, args.contracts, args.live)


if __name__ == "__main__":
    main()
