# FIFA / Soccer Momentum + Hydration-Break Trading on Kalshi — Feasibility Report

**Date:** 2026-06-17
**Context:** Exploring whether the NBA live-trading thesis (detect short-term momentum before the retail-driven market reprices) transfers to FIFA World Cup 2026 soccer matches on Kalshi, with a specific angle on momentum swings around mandatory hydration/cooling breaks.

---

## Verdict in one paragraph

The **execution layer is viable and ready** — Kalshi runs deep, liquid, live in-game soccer markets on the same REST + WebSocket API and fee structure we already use for NBA. The **two pillars of the strategy are both weak**, however: (1) there is **no cheap low-latency soccer data feed** — the affordable options (~15s polling) are no faster than, and probably slower than, Kalshi's own reprice, so the "we see it first" speed moat that powers the NBA system **does not exist here**; and (2) the **hydration-break momentum edge is statistically unproven** — the in-tournament evidence is ~50/50 and fully consistent with no effect. Recommendation: treat this as a **research/event-study project, not a build-and-deploy project.** Stand up a cheap recorder now (the World Cup is a non-repeatable data window), prove the lead/lag and the break effect on recorded data, and only then decide whether to build.

---

## 1. Execution layer — Kalshi soccer markets (STRONG / GREEN)

Live in-game soccer trading exists on Kalshi and is **more liquid than NBA** on marquee matches.

- **Live trading confirmed.** Match moneyline markets open weeks ahead and stay tradeable straight **through the match** — e.g. Sweden–Tunisia moneyline closed at `03:59Z` right after the match ended, not at kickoff. Markets show `status: active` with live orderbooks during play.
- **Liquidity.** England–Croatia moneyline live orderbook showed **$443k at top YES bid, $3.65M at top NO bid, ~1¢ inside spread**. World Cup winner market alone has cleared >$130M; tournament volume >$200M.
- **Series tickers** (parallel to `KXNBASPREAD`):
  - `KXWCGAME` — match moneyline, 3-way (TEAM / TEAM / TIE) ← **primary instrument**
  - `KXWCTOTAL` (O/U goals), `KXWCSPREAD` (handicap), `KXWCBTTS` (both teams score), `KXWCSCORE` (correct score), `KXWC1H` (first half), `KXWCFIRSTGOAL`, `KXWCGOAL`
  - Futures: `KXMENWORLDCUP` (winner), `KXWCGROUPWIN`
  - Event format: `KXWCGAME-26JUN17ENGCRO` → markets `-ENG`, `-CRO`, `-TIE`
- **API parity.** Same `api.elections.kalshi.com/trade-api/v2` REST (`/events`, `/markets`, `/orderbook`), same WebSocket channels (`orderbook_delta`, `ticker`, `trade`), same RSA-PSS auth, same integer-cents 1–99 / $1 settlement. **Our existing Go execution touchpoint and parser work essentially unchanged.**
- **Fees — one important nuance:** `KXWCGAME` (moneyline) and `KXMENWORLDCUP` use `quadratic_with_maker_fees`, so **makers ARE charged** (~1.75%) on our primary instrument — the 4× maker advantage we rely on in NBA is smaller here. The prop markets (`KXWCTOTAL/SPREAD/BTTS/SCORE`) are taker-fee-only (makers free).

**Caveats that matter for the thesis:**
- **No confirmed continuous "next goal" market** — the highest-frequency micro-momentum instrument may not exist; you'd be trading moneyline/total *drift*.
- **Soccer is low-event.** ~2.5 goals/match vs basketball's continuous scoring → far fewer discrete repricing triggers per match. A "scoring run" analog fires rarely.

---

## 2. Live momentum data feeds (WEAK / RED — this is the dealbreaker)

For an indie/retail developer there is **no true low-latency push feed**. This is the central risk.

| Tier | Providers | Latency | Access | Momentum data |
|---|---|---|---|---|
| Enterprise (push/WS) | Sportradar, Stats Perform/Opta, Genius, LSports | **sub-1s to ~11s** | Sales-gated, no public pricing | xG, possession value, full event taxonomy |
| **Indie (polling)** | **SportMonks**, API-Football | **~15s refresh, no WS** | **Self-serve** | SportMonks **Pressure Index** (composite momentum), dangerous attacks, xG |
| Free / scraped | SofaScore, FotMob, ESPN | 30–120s, fragile | ToS-gray, Cloudflare-blocked | SofaScore "Attack Momentum" graph, FotMob `momentum` |

- **Best official indie option: SportMonks + Pressure Index add-on** (~€78–98/mo, self-serve). Real-time composite momentum index (possession, dangerous attacks, corners, shots, tackles) — the closest analog to our basketball-run signal. But **15s polling, no push.**
- **Best free momentum signal: SofaScore `/api/v1/event/{id}/graph`** (literal Attack Momentum). Cloudflare-fragile, must throttle ≥30s, ToS-gray → not production-grade.
- **The latency math is brutal.** Kalshi reprices fast on live events; our affordable data is ~15s+ behind. In NBA we have a self-recorded tick moat and a predictive possession model; in soccer the affordable feed is likely **slower than the market we'd trade against.** Pure speed arbitrage is off the table — any edge must come from a *predictive* model on 15s event flow, not from seeing events first.

**Mandatory pre-build experiment:** measure the **lead/lag between SportMonks Pressure Index changes and Kalshi price moves** on recorded matches. If Kalshi moves first, there is no edge regardless of model quality.

---

## 3. Hydration / cooling breaks (MIXED — predictable timing, unproven edge)

The premise needs one correction and one reality check.

**Correction — the rule changed for 2026.** FIFA **scrapped the WBGT trigger.** Every one of the 104 matches now gets **two mandatory 3-minute breaks at fixed points (~22' and ~67')**, regardless of weather, roof, or temperature. This is *good* for us: **break occurrence and timing are known pre-match for every game** — no WBGT forecasting needed to know a break is coming. (WBGT only modulates *effect size*; irrelevant in roofed/AC venues — Dallas, Houston, Atlanta, SoFi, Vancouver.)

**Reality check — the momentum edge is statistically unproven.** The only evidence is in-tournament correlation:
- 1st half: 12 of 22 goals came *after* the first break. 2nd half: 12 of 24 *after* the second.
- **Those splits are ~50/50 — exactly what you'd expect from base rates if the break has no effect.** More goals fall in the later part of a half by construction.
- Coach quotes support the directional hypothesis ("momentum breaks… advantageous for the team losing momentum" — Emma Hayes), but **no published causal study exists**; Northeastern's group says it's still unstudied.

**Detection ≈ clock, not feed.** Feeds likely don't emit a discrete "cooling break" event. But because timing is mandatory/fixed, **you don't need detection — you have a schedule.** Confirm with a "no events for ~3 min around 22'/67'" stoppage proxy if desired.

**The only coherent tradeable form:** condition on **pre-break game state** (which side is under pressure / trailing) and test whether the trailing/pressured team's goal+xG rate in the 5–10 min post-break window beats baseline. Weight by WBGT only as an effect-size modifier. This must be validated on real data — be paranoid it's a base-rate illusion.

---

## Recommended next steps (cheap, reversible, data-first)

1. **Stand up a Kalshi soccer recorder now.** Clone the NBA recorder pattern to subscribe to `KXWCGAME` (+ `KXWCTOTAL`) orderbook/ticker WebSocket channels for live World Cup matches. The 2026 World Cup is a **non-repeatable data window** — every match not recorded is lost forever (same logic as our NBA tick recorder). This is the one time-critical action.
2. **In parallel, record a momentum feed.** Trial SportMonks (Pressure Index) for a few matches and/or scrape SofaScore Attack Momentum, timestamped, to align against the Kalshi ticks.
3. **Run two event studies on the recorded data before writing any strategy code:**
   - **Lead/lag:** does the Pressure Index move *before* Kalshi reprices, and by how many seconds? (Kills or confirms the whole idea.)
   - **Break effect:** conditioned on pre-break game state, is there an abnormal post-break goal/xG/price move? (Validates or debunks the hydration-break angle.)
4. **Only if both pass, scope a strategy.** Account for: maker fees on `KXWCGAME`, low event frequency, and the absence of a speed moat (model must be predictive, not reactive).

**Do not deploy capital on the hydration-break hypothesis as-is** — current evidence is consistent with no effect.

---

## Sources
- Kalshi live API queried directly (`/series/KXWCGAME`, `/events`, `/orderbook`); docs.kalshi.com (market data, websockets); kalshi.com/fee-schedule
- SportMonks (pricing, Pressure Index docs), api-sports.io, Sportradar latency indicator, Stats Perform/Opta feeds, LSports — feed latency/pricing
- SofaScore `/event/{id}/graph`, FotMob `matchDetails`, ESPN hidden API — unofficial momentum endpoints
- FIFA 2026 hydration-break rule: Kestrel Instruments, ESPN, NBC, Fortune; Northeastern (unstudied); IFAB footballrules.com
