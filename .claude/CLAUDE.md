# Kalshi Basketball Trading System

## End Goal
Make money live trading NBA games on Kalshi prediction markets. Everything in this project — data engineering, models, backtesting — exists to serve one outcome: a live system that places profitable maker orders during NBA games with a validated, understood edge.

This is a learning project. I'm new to algo trading but comfortable with Python.
- Prioritize readable, well-commented code — explain *why*, not just *what*
- When multiple approaches exist, explain tradeoffs before implementing
- Flag anything that could cause real financial loss

---

## Core Thesis

We predict short-term scoring runs — 3 to 5 minute windows — using lineup matchup data, entering Kalshi positions **before** the run shows up on the scoreboard and **before** the market reprices.

```
Detect run conditions early
  → Enter Kalshi limit order (maker, always)
    → Scoring run occurs → Kalshi price moves
      → Exit position → Capture the spread, minus fees
```

Retail bettors overreact to runs they just watched on TV. They underreact to lineup changes they didn't notice. We notice first. Kalshi's retail-dominated, emotionally-reactive market reprices slowly — that's our edge.

---

## Non-Negotiable Rules
- **Prefer maker orders** — limit orders by default (4x cheaper fees). Taker orders are allowed when the expected move justifies the higher taker fee — i.e. the edge still clears fees after accounting for the 0.07/contract taker cost. Maker remains the default; takers are the exception, only when it's clearly more profitable to cross the spread than to wait for a fill.
- **Fees in every signal calculation** — if edge doesn't clear fees, don't trade
- **Paper trade before real money** — weeks minimum, not days
- **Position limits before every order** — `live-trader/go/risk.go`, no exceptions
- **One execution touchpoint** — all Kalshi API calls go through `live-trader/go/` only
- **Kill switch must exist** before any live order is placed
- **Never hardcode credentials** — all keys via environment variables

## Credential Setup

Two env vars required for live/paper trading. Add to `.env` at project root (auto-loaded by Go trader at startup):

```
KALSHI_KEY_ID=<your-api-key-id>
KALSHI_PEM_PATH=/path/to/private_key.pem
```

Store the PEM file **outside the repo** — `~/.kalshi/private_key.pem` is the recommended location. Never place it inside the project directory. The `.env` file is gitignored.

Optional overrides (rarely needed):
```
KALSHI_REST_BASE_URL=https://trading-api.kalshi.com/trade-api/v2
KALSHI_WS_URL=wss://trading-api.kalshi.com/trade-api/ws/v2
```

Auth mechanism: RSA-PSS signed headers (`KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-TIMESTAMP`, `KALSHI-ACCESS-SIGNATURE`) generated in `live-trader/go/kalshi_auth.go`.

---

## Fee Structure
```
Maker fee:  ceil(0.0175 × contracts × P × (1-P))   (~$0.44 per 100 contracts at 50¢)
Taker fee:  ceil(0.07   × contracts × P × (1-P))   (~$1.75 per 100 contracts at 50¢)
        where P = price in dollars (50¢ → 0.50), rounded up to the cent
```
- The `P × (1-P)` term is required — an earlier version of this table omitted it and
  read `rate × contracts × price`, which overstates the fee ~4× at the 50¢ midpoint
  (the worked examples above have always assumed the correct formula). Canonical
  implementations: `backtesting/mmoe_backtest.py::_fee_one_leg`,
  `live-trader/inference/dashboard.py::kalshiMakerFee`.
- **Which leg pays what:** entry is always a resting limit (maker). Exits pay maker
  only on `take_profit` (PR #50 rests the TP limit); `stop_loss`, `momentum_flip`,
  `garbage_time` and `time_gate` all cross the book and pay taker.
- Taker fees are 4x higher — being a taker destroys edge on small price moves
- All prices are integers in cents (1–99). Never floats. Never decimals internally.
- A YES contract at 60¢ = market implies 60% win probability

---

## Current Status

- [x] Phase 1: Data foundation — recorder deployed on Fly.io, nba_api ingestion complete, possession_flat built (400K rows, 86 cols)
- [x] Phase 2: Feature store (58 features, lineup signals); player + lineup ratings (47M rows); nightly 3 AM pipeline on Fly.io
- [ ] Phase 2: Rotation tendency model — not built
- [x] Phase 3: MMoE backtest ⚠️ UNVERIFIED (exit-window lookahead — see warning above) — original best: `--use-traj-for-side --min-abs-traj 0.08 --min-run-length 2 --hold-seconds 240` → 207 trades, 47.8% win, +$37,409 (110 val games). **Superseded by Phase 1 sweep (see below).**
- [x] Phase 4: MMoE retrained (2026-05-15) — Head A AUCPR 0.1593, Head B dir acc 61.1% ⚠️ (Head B trained on 34% corrupt market rows), Head C Brier 0.0860; artifacts: `models/saved/mmoe_delay20.pt`
- [x] Phase 5: Go execution engine + Python inference service running end-to-end
- [x] Phase 5: Structured JSONL logging across Go and Python
- [x] Phase 5: Risk ledger fully enforced — `Check()`, `RecordFill()`, `RecordExit()` implemented; kill switch wired
- [x] Phase 6: Paper trading complete — multiple sessions through May 2026
- [x] Phase 6: Phase 1 aggregator sweep (2026-06-03) ⚠️ UNVERIFIED (lookahead) — winner `traj_aggregator: mean`, `min_abs_traj: 0.08`, `use_traj_for_side` → +$39.3K / 51.9% wr / 183 trades on 128 val games. Tool: `tools/sweep_traj_aggregator.py`
- [x] Phase 6: Entry threshold raised 0.08 → 0.12 (2026-06-06) ⚠️ tuned on biased backtests; re-derive — the 0.08–0.12 band had 33% win (below 55–56% break-even at TP=5/SL=3 with maker entry + maker TP + taker SL). At 0.12: ~60% fewer trades, ~60% win rate
- [x] Phase 6: Resting maker take-profits (PR #50) — on entry fill, an opposite-side `post_only=true` limit is placed at `entry + TP`. Captures ~4× maker discount on winners. Stops still cross the book (taker, bounded slippage by `exit_slippage_budget_cents`) — fixes 2026-05-28 post-mortem where 175 post-only stops were rejected as "post only cross"
- [x] Phase 6: Kelly-style sizing — `kelly_min_contracts: 60`, `kelly_slope: 100`, `kelly_anchor_traj: 0.12`. Floor 60C, ramps to 100C cap only at extreme conviction (`|traj| > 0.88`)
- [x] Phase 6: Market scanner hardened — won't swap markets while a position is open; stale-tick filter after swap; cancels routed to entry-market ticker (not new market)
- [x] Phase 6: Config-driven model paths — `inference.model_path` / `scaler_path` forwarded by Go on `/game/start`; Python only reloads if paths differ
- [x] Phase 6: Garbage-time GATE decoupled from MODEL feature — trade gate is config-tunable; `garbage_time_risk` model feature frozen at training values (30/4/360) to prevent train/serve skew
- [ ] **Phase 7: Live trading ← CURRENT FOCUS** — went live 2026-06-03 for NYK@SAS Game 1 (user-authorized). Daily kill: $40. Caps: $70 total / $70 per-game. Trailing-TP scaffold present but off (`trail_giveback_cents: 0`)
- [ ] Phase 2: Rotation tendency model — not built (deprioritized)

**Train/test split:** Jan 2026 cutoff for basketball; Head B: Mar 23–Apr 6 train / Apr 7–12 val. The 2024-25 season test set is sacred — never use during development.

---

## ⚠️ ALL BACKTEST P&L AND HEAD B ACCURACY ARE UNVERIFIED (as of 2026-08-03)

**Do not quote, build on, or tune against any P&L figure in this file until it is re-measured.** Every historical figure below was produced through the lookahead described in defect 1.

**THE CORRECTED BASELINE (2026-08-03, branch `fix/backtest-exit-window`):** on the 69-game
cache (2026-04-15 → 05-17, fully out-of-sample vs. both the Jan 2026 basketball cutoff and
Head B's Apr 7–12 val window), the Phase 6 config
(`--use-traj-for-side --min-abs-traj 0.08 --traj-aggregator mean --min-run-length 2 --hold-seconds 240`):

| | trades | win rate | net P&L |
|---|---|---|---|
| as previously reported | 81 | 51.9% | **+$16,100** (really +$161 — see defect 4) |
| corrected | 80 | 30.0% | **−$136.42** (gross **+$2.00**, fees **$138.42**) |

**The strategy loses money, and fees are the reason.** Gross edge over 80 trades is
**+$2.00**; fees are **$138.42** — a 69× drag. Net-of-fee win rate 30.0% against a
55–56% break-even; fees alone flip 6 gross-positive trades into losses. 70% of exits
are stop-outs or momentum flips.

This is not a tuning problem. A 5¢ take-profit cannot clear a $2.19 round trip on 100
contracts: you need ~55.7% accuracy and the model delivers ~37% on gross moves.
Either the edge per trade gets much larger (wider TP, better signal) or the fee per
trade gets much smaller (maker on both legs, i.e. never stopping out at market) —
tightening `min_abs_traj` cannot fix a 69× cost ratio.

**1. Exit-window lookahead — FIXED 2026-08-03** (`mmoe_backtest.py`, commit `92ebc6c`).
Entry price is read at `wall_clock_ts + feed_delay_s` (20s) but the exit search started at
`wall_clock_ts`, so a position could exit up to 20s **before it entered**, capturing
movement that had already happened. Measured on the 69-game event-trigger run: 25 of 64
trades held **< 20s** and produced **$13,500 of $14,000** at 84% win rate. On the mmoe run:
25 of 81 trades held < 20s and produced **+$17,300 at a perfect 100% win rate**, while the
56 legitimate trades **lost $1,200**.

The fix anchors the exit window, the possession window and the overlap guard on
`wct + feed_delay_s`, via a single `_tick_at_delay()` helper so the entry price and exit
anchor cannot drift apart again. A runtime invariant raises on any exit earlier than
`wct + feed_delay_s`; it is deliberately stated against `wct` rather than the anchor so
re-anchoring fails loudly instead of silently reinflating results. **Verified in both
directions** — it fires when the bug is reintroduced.

`backtesting/event_trigger_backtest.py` had the same pattern — **also fixed 2026-08-03**,
on `event_triggers` after merging. Corrected event-trigger result on the same 69-game cache:
**118 trades, 36.4% win rate, −$200.91**, versus its internal possession-close baseline of
208 trades / 33.2% / −$342.75. **Per trade the two are indistinguishable: −$1.70 vs −$1.65.**
Firing at subs/fouls/timeouts is not better; it just trades less. By trigger type:
`sub` −$29.13 at 44.1% wr (n=34), `timeout` −$20.68 at 35.3% (n=17), `foul` −$151.10 at
32.8% (n=67). The sub-vs-foul win-rate gap (44% vs 33%) is the only thing here worth a
second look, and n=34 cannot support a conclusion.

Why event triggers can't help much: mean earliness vs possession close is **2.4s**, against
a 20s feed delay. You cannot buy a timing edge measured in seconds when you are 20 seconds
behind the feed.

**Do not "filter" the bug out of old results.** Dropping sub-20s trades from a biased run
gave an estimate of −$2,600; actually re-running gave +$62. The corrected run re-simulates
those positions from the right anchor and the overlap guard admits a different trade set.
Only a real re-run counts.

**1b. Take-profits were over-credited — FIXED 2026-08-03.** TP exits were booked at the
price of the tick that breached the threshold rather than at the resting maker limit of
`entry + TP` (PR #50). 18 of 36 TP exits overshot the limit — median 6¢ past it, max 24¢
(one 44¢→20¢ fill booked as +$2,400) — crediting fills that were never available. Exit
price is now capped at the limit.

**2. Head B trained on partly-corrupt market features.** `wall_clock_ts` was one day late for 86 games (games tipping after 20:00 ET cross 00:00 UTC). `pd.merge_asof` never fails, so those possessions silently matched the last recorded tick — the settled price (1¢/99¢) — and were then quietly dropped by the 30–70¢ band. **34% of joint rows (16,955 of 49,576)** were affected. The data is now repaired (see below), but Head B was trained before the repair.

**Consequently suspect and needing re-derivation:**
- Phase 3 `+$37,409` and Phase 6 `+$39.3K / 51.9% / 183 trades`
- The aggregator sweep that selected `traj_aggregator: mean`
- **`min_abs_traj: 0.12`** in `live-trader/config/trading.yaml` — raised from 0.08 on win rates measured through the lookahead. This currently governs live orders. Note the 0.08 vs 0.12 comparison was between two buckets that were **both** inflated, so the threshold has no validated basis in either direction. Re-derive against the corrected backtest before the next live session.
- Head B's `Dir Acc 61.1%`

**3. Fees were zero — FIXED 2026-08-03.** `_compute_maker_fees()` returned `0.0`
unconditionally, which zeroed the taker cost on every stop-out, i.e. on every loser.
Replaced by `_compute_fees(entry, exit, contracts, exit_reason)`: maker entry always, maker
exit only on `take_profit`, taker otherwise. Worth $138 on the 80-trade corrected run —
more than twice the run's entire net P&L.

**4. P&L was reported in cents but labelled dollars — FIXED 2026-08-03.** Both backtests
computed `net = gross_cents * contracts - fees_dollars`, mixing units, then printed the
result with a `$`. A Kalshi contract settles at $1, so 100 contracts moving 5¢ is **$5.00,
not $500** — every historical figure in this file is **100× too large** (Phase 3's
`+$37,409` was $374; the sweep's `+$39.3K` was $393). Worse, because fees were computed in
real dollars and subtracted from a cents quantity, the fee looked **100× smaller than it
is** — which is exactly why the fee bug in defect 3 went unnoticed and why the corrected
baseline flips from +$62 to −$136. Fixed via `mmoe_backtest.pnl_dollars()`, now used
everywhere P&L is summed (both backtests plus `tools/sweep_dynamic_exit.py` and
`tools/sweep_traj_aggregator.py`).

Note the giveaway that resolved it: CLAUDE.md's own **55–56% break-even** for TP=5/SL=3 was
derived with *correct* units (win $5.00 − $0.88 = +$4.12; loss −$3.00 − $2.19 = −$5.19;
break-even = 5.19/9.31 = 55.7%). The doc and the backtest had disagreed by 100× all along.
The live Go trader is rigorously in cents (`max_total_exposure_cents: 7000 # $70`) and was
never affected.

**Known remaining bias (not fixed, smaller, and deliberate):** `simulate_exit` evaluates
momentum-flip and garbage-time exits at each possession's `wall_clock_ts`, with no feed
delay, so those exits fire ~20s earlier than a live trader could act. Fixing it means
changing `models/targets/exit_simulator.py`, which also generates Head B/C **training
labels** — so it would require a retrain and must not be done casually. 27.5% of corrected
exits are momentum flips, so this is material; scope it as its own task.

**⚠️ Separate finding, arguably bigger than any of the above: 18 of the 83 features are dead
in production.** To re-measure the event-trigger backtest, `models/mmoe/feature_config.py`
was restored to `main`'s 83-feature version on `event_triggers` — the branch's 65-feature
slim config cannot load `models/saved/mmoe_delay20.pt` (shape mismatch, 83 vs 65). But the
slimming commit (`cce25f8`, 2026-06-10) documents *why* those 18 were dropped, and the reason
is train/serve skew, not tidiness:

- **11 pregame features** — `has_pregame_data` was 0.0 across 1,605 live possessions
  (2026-05-25 → 06-09). The pregame join never populates in live.
- **5 lineup features** — `home/away_lineup_net_rating`, `lineup_net_rating_delta`,
  `home/away_lineup_sample_size` all 0.0 in live over the same window.
- **2 event flags** (`was_foul`, `was_sub`) — `train_std = 0` across 70k+ rows.

So the deployed 83-feature model is fed real pregame and lineup values in backtest and
**zeros in production**. Every backtest number, including the corrected ones above, is
optimistic relative to live for this additional reason. And the 5 dead lineup features are
the direct expression of the Core Thesis at the top of this file ("we predict runs using
lineup matchup data… they underreact to lineup changes they didn't notice") — the project's
central hypothesis is currently not wired up in production at all.

Resolving this means retraining a 65-feature model on the repaired timestamps, or fixing the
live lineup/pregame pipeline so the 83-feature model gets what it was trained on. Until then
the 83-feature config is a measurement convenience, not a decision.
⚠️ The supporting analysis (`scratch/feature_audit.csv`, `scratch/feature_audit_v2.py`) is
**untracked** — it exists only on the dev machine. Commit it before relying on it.

**Fixed and verified on 2026-08-03** (branch `fix/wall-clock-ts-date`, merged into `event_triggers`):
- 86 games' `wall_clock_ts` repaired in MotherDuck; all 2,127 timestamped games now at day-offset 0, no backwards periods, no implausible tip hours, row count preserved at 449,274. Backups in `data/backups/`.
- Midnight-crossover bug in `_parse_period_start_et` — lost a day when two consecutive periods started after midnight (late west-coast games). Was still live.
- Two guards added: `dataset.validate_possession_tick_overlap()` (first possession inside the tick window **and** ≥15% tick coverage) and `backfill_wall_clock_ts._validate_anchors()` (ET date matches `game_date`, plausible tip hour, monotonic periods, span < 6h).
- Row-fanout bug in the `possession_flat` CTAS rebuild — the join key repeats up to 9× per game; would have added 2,640 phantom rows.
- Backtest tick coverage went from 3% to 91% median; 69/69 games now pass.

---

## Directory Structure

```
kalshi-trader/
├── .env                              # credentials — never commit
│
├── data/
│   ├── ingestion/
│   │   ├── nba_api_client.py         # historical pull — play-by-play, lineups
│   │   ├── kalshi_recorder.py        # records live price ticks (deployed on Fly.io)
│   │   ├── post_game_pipeline.py     # nightly 3 AM pipeline (Fly.io)
│   │   └── duckdb_loader.py          # local DuckDB ↔ MotherDuck sync
│   ├── raw/                          # parquet snapshots (local mirror)
│   └── feature_store/
│       ├── player_ratings.parquet    # rolling point-in-time ratings
│       ├── lineup_ratings.parquet    # per lineup, per as_of_game
│       └── feature_rows.parquet      # THE feature store — one row per possession
│
├── models/
│   ├── features/
│   │   ├── builder.py                # orchestrates full feature row construction
│   │   ├── lineup_features.py
│   │   ├── momentum_features.py
│   │   ├── context_features.py
│   │   └── targets.py                # forward-looking targets (training only)
│   ├── ratings/
│   │   ├── player_rapm.py            # rolling regularized plus/minus
│   │   └── lineup_net_rating.py
│   ├── mmoe/
│   │   ├── model.py                  # MMoE architecture + predict_with_diagnostics
│   │   └── train.py
│   └── saved/
│       ├── mmoe_delay20.pt
│       └── mmoe_scaler_delay20.pkl
│
├── backtesting/
│   ├── simulator.py                  # core replay engine
│   ├── evaluator.py                  # metrics, context breakdown, attribution
│   └── mmoe_backtest.py              # validated backtest script
│
├── live-trader/
│   ├── go/                           # Go execution engine
│   │   ├── main.go
│   │   ├── coordinator.go
│   │   ├── game.go
│   │   ├── agent.go                  # entry/exit logic + gates
│   │   ├── risk.go                   # position limits (limit checks TODO)
│   │   ├── jsonlog.go                # structured JSONL logging
│   │   └── agent_test.go             # behavior tests for entry gates
│   ├── inference/                    # Python inference service (FastAPI)
│   │   ├── main.py
│   │   ├── features.py
│   │   ├── possession.py
│   │   ├── pregame.py
│   │   ├── jsonlog.py
│   │   └── dashboard.py
│   └── config/
│       └── trading.yaml              # live-tunable thresholds (no recompile needed)
│
├── pregame/
│   └── pregame_analyzer.py
│
├── risk/
│   └── position_limits.py
│
├── tools/
│   └── inspect_run.py                # post-game run analysis
│
└── logs/
    └── runs/{date}/{run_id}/
        ├── manifest.json
        ├── live-trader.jsonl
        ├── inference.jsonl
        └── stderr.log
```

---

## Data Sources

| Data | Source | Status |
|------|--------|--------|
| Historical play-by-play | `nba_api` | Complete |
| Historical lineup stats | `nba_api` | Complete |
| Live play-by-play | Kalshi WebSocket NBA feed | Live |
| Live Kalshi prices | Kalshi WebSocket | Recording (Fly.io) |
| Historical Kalshi prices | Self-recorded | Recording since Oct 2025 |

**The Kalshi recorder is time-critical.** Historical Kalshi tick data is not available from any vendor. Every game missed = lost training data permanently.

---

## Python Conventions
- Async (asyncio) for all ingestion, WebSocket, and live feed handling
- Type hints on all functions, always
- `logging` module everywhere — never `print()` for runtime output
- Explicit error handling — log what failed and why, never bare `except:`
- Prefer early returns over deeply nested conditionals
- All prices as integers in cents — never floats
- Parquet format for all stored data
- Never commit `.env`, `data/`, or `logs/` to git

---

## Common Commands

All commands run from the project root with `source venv/bin/activate` first.

### Data Pipeline

```bash
python -m data.ingestion.nba_api_client                        # fetch all missing games
python -m data.ingestion.nba_api_client --since 2026-03-13     # games on/after date
python -m data.ingestion.nba_api_client --game 0022501039      # single game

python -m data.ingestion.duckdb_loader                         # build/update local DuckDB
python -m data.ingestion.duckdb_loader --sync-motherduck       # push new rows to MotherDuck
python -m data.ingestion.duckdb_loader --pull-motherduck       # pull new rows from MotherDuck
python -m data.ingestion.duckdb_loader --pull-motherduck-full  # DESTRUCTIVE: replace local with remote
```

Both sync directions are additive — safe to run twice. Use `--pull-motherduck-full` only for fresh setup or corrupted local DB.

### Recorder

```bash
python data/ingestion/kalshi_recorder.py          # record live ticks on game days
python data/ingestion/game_schedule.py            # check today's schedule
python data/ingestion/game_schedule.py --date 2026-03-25
```

### Live Paper Trading (two-process setup)

```bash
# Terminal 1 — Python inference service (start first)
PYTHONPATH=.:live-trader ./venv/bin/python -m uvicorn inference.main:app \
  --host 127.0.0.1 --port 8001
# Wait for "Application startup complete."

# Terminal 2 — Go trader
cd live-trader/go
go build . && ./go                                    # auto-detects today's games
./go --game 0042500223 --event KXNBASPREAD-26MAY09OKCLAL  # single-game mode

# Dashboard: http://127.0.0.1:8001/dashboard
```

### Run Inspection

```bash
./venv/bin/python tools/inspect_run.py <run_id>

# Logs at: live-trader/go/logs/runs/{date}/{run_id}/
tail -f live-trader/go/logs/runs/{date}/{run_id}/live-trader.jsonl
```

---

## ML Model Stack

### MMoE (Multi-task Mixture-of-Experts) ← PRIMARY MODEL

83 input features (58 basketball + 10 pregame + 14 market + 1 flag), 3 expert networks (64-dim MLP), 3 gating networks, 3 task heads. ~37K params.

- **Head A — Run Classifier:** P(meaningful run in next 5 possessions) — AUCPR 0.1593 vs 0.0840 baseline (+90%). Entropy regularization (λ=0.015) fixed gating collapse.
- **Head B — Price Trajectory:** Δ(yes_bid) in log-odds over hold window — Dir Acc 61.1% ⚠️ UNVERIFIED (trained on ~42K joint rows with Kalshi ticks). w_b raised to 1.0.
- **Head C — Run Survival Hazard:** P(run still ongoing) at 10 time horizons — Brier 0.0860; used for dynamic exit timing.

**Trade only in 30–70¢ band.** Outside this range, market certainty is too high for basketball signal to move price.

**Head A as entry gate:** Tested at thr=0.10 and thr=0.15 — both reduced total P&L vs no gate. Head A's run_prob is NOT used as an entry gate. Head B trajectory is the entry signal; Head A is logged for diagnostics only.

**Retrain cadence:** rolling 60-game window every ~10 games. All three heads retrain together.

### Entry/Exit Agent
Currently a rule-based gate (see Live Entry Config). Full Thompson Sampling bandit planned once paper trading data accumulates.

---

## Live System Architecture

**Pre-game** (~30 min before tip): `pregame_analyzer.py` → `game_context.json` with lineup net ratings, player APM, coaching tendencies, watch flags. Nothing in `game_context.json` is recomputed mid-game.

**In-game loop:**
```
Kalshi WebSocket (NBA events, ~15-20s latency)
  → Go coordinator (event parsing, possession tracking)
  → Python inference service (feature assembly + MMoE inference)
    Head A: P(run) | Head B: Δ(yes_bid) | Head C: run survival
  → agent.go (entry gates → BUY YES / BUY NO / EXIT / WAIT)
  → risk.go (position limits — always checked)
  → Kalshi REST API (paper mode by default)
```

~15-25s from real-world event to order. Still faster than retail: we're predictive, they're reactive.

---

## Live Entry Config (aligned to backtest 2026-05-10)

| Gate | Status |
|------|--------|
| `\|traj_final\| ≥ 0.08` (Head B confidence) | required |
| `current_run_length ≥ 2` (no single-basket noise) | required |
| price band 30–70¢ | required |
| not garbage time / blowout | required |
| `run_prob ≥ 0.10` | **not used** — Head A gate reduces P&L at all tested thresholds |

Direction (BuyYes vs BuyNo) from `traj_final` sign. Thresholds in `live-trader/config/trading.yaml` — edit and restart Go, no recompile needed.

Backtest command (⚠️ results UNVERIFIED — exit-window lookahead): `python -m backtesting.mmoe_backtest --use-traj-for-side --min-abs-traj 0.08 --min-run-length 2 --hold-seconds 240 --threshold 0.0`
→ 207 trades, 47.8% win rate, +$37,409 net (110 val games)

---

## Nightly Post-Game Pipeline

Runs at 3 AM ET on Fly.io via `recorder_daemon.py`. All reads/writes go directly to MotherDuck — no local disk.

```
Phase 0  Upsert dim_games
Phase 1  Fetch + parse PBP via nba_api
Phase 2  Build possession_flat rows (ASOF ratings — point-in-time correct)
Phase 3  Update player_ratings + lineup_ratings
```

Idempotent — safe to re-run. Manual trigger:
```bash
python -m data.ingestion.post_game_pipeline 2026-03-31
```

**MotherDuck:** `kalshi_trading` database. Token in `.env` and Fly.io secrets as `MOTHERDUCK_TOKEN`.

**Gotchas:**
- Trust MotherDuck as source of truth; local DuckDB may be stale
- `--pull-motherduck` / `--pull-motherduck-full` flags broken locally (DuckDB 1.5.1 issue) — use `--pull-possession-flat` and `--pull-features-schema` instead

---

## Logging System

Structured JSONL across Go and Python, correlatable via shared `run_id`. Every Go `possession` event carries gate pass/fail and `first_blocking` gate name. Every Python `possession` event carries `model.gating_weights` (3×3), per-expert opinions for all heads, and `features_zscored` for 10 key features (`|z| ≥ 2` = unusual).

```bash
./venv/bin/python tools/inspect_run.py <run_id>   # full post-game summary
```

---

## Live Feature Risks

**Shot coordinate mismatch (low risk):** nba_api and the live feed use different coordinate systems. Fix before going live: one-time converter. Affects `shot_distance`, xPPP, shot sustainability features.

**Possession parser (HIGH RISK):** The live parser groups raw events into possessions. It must use the **exact same boundary definition as nba_api** — if it diverges, features are systematically wrong even though raw data is correct. Validate against a historical game possession-by-possession before going live.

---

## Key Reminders
- The end goal is live profitable trading. Every decision should serve that.
- Lookahead bias makes backtests look great and live trading look terrible — be paranoid about it
- When touching execution code: paper mode must be default, confirm before changing
- Only suggest a taker order when the edge still clears the higher taker fee — maker is the default, takers are the justified exception
- Blowout / garbage time = model off. Never trade garbage time.
- If a backtest result looks suspiciously good, assume lookahead bias first
- We are in Phase 6 (paper trading). Don't skip paper validation — minimum 4 weeks before real money.
