# Feature Consolidation (2026-08-02)

Model input went **83 → 58 features** (physics 58 → 33; pregame 11 and market 14 unchanged).
Five train/live parity bugs fixed. Two live trading gates were broken by the change and restored.

Column names: `models/mmoe/feature_config.py`. Formulas: `models/features/transforms.py`.
`docs/FEATURES.md` describes the pre-consolidation 83-feature set and is retained for its
per-head importance analysis only.

---

## The rule that matters most

**Every derived feature has exactly one implementation, in `models/features/transforms.py`,
imported by both the offline builder and live inference.**

Features are computed twice by unrelated code: `models/mmoe/dataset.py` (pandas, whole game)
and `live-trader/inference/features.py` (streaming, one possession at a time). Five silent
divergences came from that duplication. Do not inline a formula in either path — add it to
`transforms.py` and call it. The functions are pure numpy ufunc chains, so the same function
takes Series offline and floats online.

`tests/test_feature_parity.py` asserts offline == live to 1e-6 across regulation, the final
minute, and overtime.

## What was wrong with the old feature set

1. **Direction and magnitude in separate columns** (5 places), forcing a 30K-param MLP to
   learn products from 15,383 Head B train rows (measured 2026-08-11, see
   `docs/DATA_INVENTORY.md`). `comeback_probability_proxy = score_diff² / m` made winning
   by 20 and losing by 20 numerically identical — opposite trades, same input.
2. **Seven columns were exact algebraic functions of others** — `lineup_net_rating_delta`'s
   own operands, `in_bonus == (fouls_until_bonus == 0)`, `was_foul` = OR of two present columns.
3. **Two columns destroyed by their own encoding.** `possessions_since_last_timeout` used
   `999` as a sentinel on 9.4% of rows; under StandardScaler that put the sentinel at z=+3.10
   and squeezed the real 0–70 range into a z-band **0.24 wide**. Splitting it into a capped
   count plus a `no_timeout_yet` flag widened that to **2.94** (measured, 400K rows).
   Separately, `np.sign()` on shot-quality trend discarded all magnitude.

Key replacements:

| New | Formula | Replaces |
|---|---|---|
| `lead_z` | `score_diff / √minutes_remaining` | `trailing_team_urgency`, `comeback_probability_proxy` |
| `time_leverage` | `1 − √(m/48)` | `minutes_into_game`, `q4_close_game` |
| `pace_ref` / `pace_surprise` | shrink in-game pace toward `expected_pace`, weight caps at 0.85 | `pace_season_baseline` used raw |
| `run_signed_points` | `sign(run_team) × run_points` | `current_run_team_encoded` + `current_run_points` |
| `*_edge` | `home − away` | 8 home/away level pairs |

## The five parity bugs

| # | Column | Offline | Live (was) |
|---|---|---|---|
| 1 | `minutes_remaining` floor | 1.0 | 0.1 — 10× off in the final minute |
| 2 | `minutes_remaining` in OT | actual OT clock | `48 − elapsed` goes negative, pinned 0.1 (~50×) |
| 3 | `garbage_time_risk` | continuous sigmoid | hard binary requiring Q4 |
| 4 | `pace_season_baseline` | expanding **within-game** mean | fixed **pregame** constant |
| 5 | `pace_last_10` empty fallback | `15.0` | `pace_baseline` |

**#4 is the cautionary one:** the column computes a within-game mean, not a season figure.
The live path implemented the *name*. A misleading name caused a real train/serve divergence.
It is now written as `pace_game_to_date`; `dataset.py` accepts either while `possession_flat`
still carries the legacy column. **#5 was found by the parity test**, after the other four.

## Gate inputs are not model features

`agent.go` read `resp.Features["period"]` and `resp.Features["current_run_length"]`. The
consolidation removed both. **Go returns 0 for a missing map key with no error**, so the
overtime gate became `0 >= 5` (never fires) and the run-length gate became `0 >= 2` (blocks
every entry). The system could not trade.

Both are now named fields on `PossessionResponse` alongside `is_garbage_time` / `is_blowout`,
captured pre-`advance()` so they match the state the model saw. **`Features` is logging-only.**
`test_go_never_gates_on_a_feature_lookup` parses the Go source and fails if any gate regresses
to a `Features[...]` lookup.

## Training restricted to the traded regime

`449,274 → 437,060 rows (97.3%)`: drops overtime (`period ≥ 5`), blowouts (`|score_diff| > 30`),
and rows with NULL core columns. The agent refuses to trade all of these.

The blowout threshold is read from `trading.yaml` at runtime. **Do not use the stored
`possession_flat.is_garbage_time` column** — it is built on a 20-pt margin while the live gate
uses 30, and filtering on it discards 28,627 rows the system would actually trade.

## Architecture

`nn.Linear(14, 4)` compresses the market block before the experts; physics stays
hand-engineered (strong domain priors there, none for LOB microstructure). 37K → 30K params.
`--no-market-encoder` trains the attribution baseline. `MMoEPredictor.load()` infers the
variant from the checkpoint. Pre-consolidation checkpoints are rejected with an explicit message.

## Retrain results (2026-08-02, 20 epochs each)

| | Run 1 (physics only) | Run 2 (+ encoder) | Prior |
|---|---|---|---|
| Head A AUCPR | **0.1675** | 0.1590 | 0.1593 |
| Head B dir acc | 0.613 | **0.625** | 0.611 |
| Head C Brier | **0.0848** | 0.0860 | 0.0860 |
| Val loss | **0.4185** | 0.4241 | — |

The encoder helped Head B (the head that depends on order-book data) and slightly hurt A and C.

**Val loss selects the wrong model.** It is a 3-head weighted sum, but entries are driven
almost entirely by Head B's trajectory sign (`agent.go`). Prefer Head B metrics when choosing
a checkpoint to trade.

## Open

- **Backtest not re-run.** The 207-trade / +$37,409 result is void — different features *and*
  a different training population.
- **Deltas may be seed noise.** One run each, no confidence intervals.
- **Both runs peaked at epoch 5 of 20** then early-stopped. Unhealthy curve, unrelated to this
  change, worth its own investigation.
- **`min_run_length_entry` / `min_abs_traj_entry` are stale** — tuned on old feature distributions.
- `docs/FEATURES.md` found that dropping all 11 pregame features *improved* val loss. Not acted on.
