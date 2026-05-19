# Trader Development Log

---

## 2026-05-15 — Retrain v2: Entropy Fix + Scaler Fix + Risk Ledger

### What was wrong (pre-session)

**Head A gating collapse** — The gating network routed 99.4% of weight to one conservative expert. That expert learned to output near-zero `run_prob` always. The other two experts stopped receiving gradients. Result: `run_prob` was useless as an entry signal in production.

**Market scaler distortion** — StandardScaler was fit on all 400K+ rows, but 94% had market features zero-filled (no Kalshi tick data). This pulled `yes_bid`'s mean to ~3¢ and std to ~15¢. At inference (always `has_market_data=1`), a normal `yes_bid=50¢` produced a z-score of +3.9 — consistently out-of-distribution.

**Risk ledger not enforced** — `risk.go` had correct data structures but `Check()`, `RecordFill()`, `RecordExit()` were stubs. The game loop never called them. No position limits were actually in effect.

### Fixes applied

1. **Market scaler re-fit** (`models/mmoe/dataset.py`) — After global `StandardScaler.fit_transform()`, re-fit the 13 price/liquidity market features (indices 69–81) using only joint rows (`has_market_data==1`). `has_market_data` itself (index 82) is excluded intentionally — its z-score contrast (live≈+4, basketball-only≈-0.25) is a useful signal. Result: `yes_bid=50¢ → z≈0` at inference.

2. **Entropy regularization** (`models/mmoe/model.py`, `trainer.py`, `train_mmoe.py`) — Added `forward_with_gates()` to expose gate softmax weights during training. Added `_gate_entropy()` computing Shannon entropy H. Added `-λ×H` term to total loss (`lambda_entropy=0.015`). Forces diverse gate routing; prevents one expert from monopolizing.

3. **Head B weight increase** (`trainer.py`, `train_mmoe.py`) — `w_b: 0.5 → 1.0`. Head B (price trajectory) is the live entry signal and now gets equal gradient pressure with Head A.

4. **Risk ledger enforcement** (`live-trader/go/risk.go`, `game.go`) — `Check()` now enforces order size cap, per-game exposure, total exposure, and daily loss limit (trips kill switch). `RecordFill()` / `RecordExit()` update exposure and P&L. Game loop wired to call all three correctly.

### Retrain results

```
Retrain: python -m models.mmoe.train_mmoe --w-b 1.0 --lambda-entropy 0.015 --feed-delay 20 --max-epochs 200 --patience 15
Best epoch: 15 (early stopped at 30)
```

| Head | Metric | Prev model | New model |
|---|---|---|---|
| Head A | AUCPR | 0.1533 | **0.1593** |
| Head B | Dir Acc | 66.5% | 61.1% |
| Head C | Brier | 0.0939 | 0.0860 |

Head A AUCPR improved (+3.9%) and was healthy from epoch 1 — entropy fix confirmed working.

### Backtest results (val set, 110 tradeable games)

Config: `--use-traj-for-side --min-abs-traj 0.08 --min-run-length 2 --hold-seconds 240`

| Config | Trades | Win Rate | Net P&L | P&L/trade |
|---|---|---|---|---|
| No Head A gate (thr=0.0) | 207 | 47.8% | **+$37,409** | $181 |
| Head A gate thr=0.10 | 120 | 43.3% | +$17,771 | $148 |
| Head A gate thr=0.15 | 71 | 46.5% | +$9,865 | $139 |
| **Old model best** | 19 | 42.1% | +$2,463 | $130 |

### Decision: keep Head B only, no run_prob gate

Head A gate reduces P&L at every threshold — it filters profitable trades more than bad ones. Per-trade P&L drops monotonically as the threshold tightens. Head B's trajectory signal is a better filter for trade quality than Head A's run probability.

**Live config remains:** `|traj_final| ≥ 0.08`, `run_length ≥ 2`, price band 30–70¢, not garbage time/blowout.

### Model compatibility

New artifacts (`mmoe_delay20.pt`, `mmoe_scaler_delay20.pkl`) are drop-in replacements. Inference service picks them up on restart — no code changes needed.

---

## 2026-05-12 — Known Issues (pre-retrain)

### system issues/bugs (now fixed)

- ~~Need to work on the head A collapse: retrain with entropy regulation.~~ **FIXED 2026-05-15**
- ~~The StandardScaler is fitted on all 433K rows, but 94% of those rows have no Kalshi data. So for the 14 market features, when there's no data the values are all zero-filled.~~ **FIXED 2026-05-15**
- ~~Risk module kill switch — skeleton in `risk.go`, limit checks TODO~~ **FIXED 2026-05-15**

### next changes

1. How can we incorporate rotation tendencies (per-coach sub patterns — coaching philosophy, not team-based)
2. Improve features in the model
3. Configs aren't exactly aligned with backtests/training: possession hard cap, stop loss, take profit
4. Possession parser validation — must match nba_api boundary definition exactly before going live (HIGH RISK)
5. Shot coordinate mismatch — nba_api vs live feed use different coordinate systems, affects `shot_distance` / xPPP features
