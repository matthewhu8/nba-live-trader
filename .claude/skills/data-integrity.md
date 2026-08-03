# Data Integrity & Train/Serve Parity

Companion to `feature-engineering.md`. That file covers *how* a feature is computed (one
formula, two callers). This one covers whether the **values** actually arriving at the model
match what it trained on — a separate failure mode that `tests/test_feature_parity.py` does
not catch, because transform-level parity can hold while the inputs differ.

## Rule 0: use `scale_`, never `sqrt(var_)`

`models/mmoe/dataset.py:608-609` refits `scaler.mean_` and `scaler.scale_` on joint rows for
the 13 market features but never updates `var_`. `StandardScaler.transform` divides by
`scale_` (`predictor.py:136`), so `sqrt(var_)` is stale global variance:

| feature | `sqrt(var_)` | `scale_` | ratio |
|---|---|---|---|
| `yes_bid` | 14.49 | 35.25 | 2.4x |
| `time_since_last_trade_ms` | 32.51 | 138.70 | 4.3x |
| non-market features | — | — | 1.0x (identical) |

Any z-score computed the obvious way is 2.4–4.3x too large for market features. This has
already produced one wrong analysis (a reported 232 sigma that was really 8.6). Also note the
market block is **already conditional** on `has_market_data == 1` (`mean_[yes_bid]` = 53.74c,
not ~3c) — do not correct it for zero-fill a second time.

## The three environments disagree

Training, backtest and live each feed the model something different. Verified 2026-08-03
against 3,131 recorded live possessions and the 58-feature scaler.

| feature | train | backtest | live | live z | affects backtest? |
|---|---|---|---|---|---|
| `time_since_last_trade_ms` | 0.0 | 0.0 | 1164.8 | **+8.62** | no — live only |
| `trade_volume_60s` | 6.94e6 | 1.79e7 | 7.84e7 | +4.76 | yes, milder |
| `open_interest` | 141,647 | 330,871 | 1.32e6 | +3.76 | yes, milder |
| `has_market_data` | 0.055 | 1.0 | 1.0 | +4.09 | yes, by design |
| `home/away_sub_count` | 0.117 | **0.0** | 0.385 | +0.60 | three-way divergence |

`spread`, `bid_velocity_30s` and the velocity/divergence features are well aligned
(|z| <= 0.39 in both). The features dropped by the (invalid) June audit max out at 0.88 sigma
— see `model-provenance.md`.

## `time_since_last_trade_ms` is a dead constant in training

The tick store's `volume` column is **cumulative**, not per-tick. So `dataset.py:267-271`'s
`volume > 0` mask is always true, `last_trade_ts` is always the current row, and the feature
is identically 0.0 — measured in 770,054 of 770,054 local ticks. Live
(`live-trader/go/ring_buffer.go:93-100`) scans back for `Volume > 0` on the same cumulative
field, so it computes **feed staleness** (~643 ms median) instead. The
`// contracts traded this update` comment at `kalshi_feed.go:28` is wrong.

Compounding it: `ring_buffer.go:93` initialises to `float32(999999)`. Twenty of 3,131 live
rows carry that sentinel and contribute **86% of the raw live mean**. This is the exact
anti-pattern `feature-engineering.md` warns about ("no sentinels") and it is still live.

- Cheapest correct action: zero the feature in live to match training.
- Honest fix: difference `volume` per tick in both `dataset.py:268` and `ring_buffer.go:95`.
  Changes training inputs, so it requires a retrain.

## The local cache silently drops event context

`data/feature_store/possession_flat.parquet` has these all-NaN or 0.0 for **all 14,239 rows**:

```
sub_count  home_sub_count  away_sub_count  was_sub  was_foul
was_timeout  timeout_teams  foul_types  players_in_ids
```

And in `pregame.parquet`, `roster_rapm_gap`, `missing_rapm_impact` and `rest_advantage` are
literal 0.0 for all 68 games. The export appears to have dropped an event-parsing stage.

Consequences, in order of severity:

1. Any consolidated feature derived from these computes to a **constant without erroring** —
   `sub_count_edge`, `team_foul_edge`, `timeout_called_edge`, `poss_since_timeout`,
   `no_timeout_yet`, and `pace_ref` (needs `expected_pace`). Check before trusting them.
2. **It undercuts the event-trigger experiment**, which fires the model at substitutions and
   fouls while the features encoding those events read 0.0. The model cannot distinguish a
   sub-trigger row from any other. Re-export before concluding anything about event triggers.
3. The warehouse has these populated (`sub_count_edge` has train std 0.48), so the **cache is
   the broken artefact**, not the source.

## The cache's tick window sits outside the scaler's era

`feature_config.py:96` documents market coverage as Mar 23 – Apr 12 2026. The local
`kalshi_ticks.parquet` spans **Apr 17 – May 18** — entirely after the scaler's fit window and
entirely inside the Head B validation period. That is the mechanism behind the
`open_interest` / `trade_volume_60s` liquidity drift, and it means the backtest has never
evaluated the model on the era it was calibrated for.

## Pregame is not disabled — it silently fails

Nothing in the code disables pregame; it has been wired since `cdb2eea` (2026-04-23). The
zeros come from two independent failures:

- **Missing rows.** `features.pregame` is populated by a 3 AM ET prefill for *tomorrow's*
  games (`recorder_daemon.py:109-136`), so a same-day playoff game has no row at tip-off and
  `pregame.py:137-138` zero-fills. `has_pregame_data` has **never once been 1.0** in any
  recorded session.
- **Swallowed connection error.** The broad `except Exception` at the end of `load_pregame`
  downgrades a failed MotherDuck connect to a warning and continues with zeros. All four
  loaders are skipped together, which is why lineup ratings died at the same moment
  `MOTHERDUCK_TOKEN` disappeared from the run manifests (2026-05-25).

Two documented attempts to fix this — `bdf3a1c` (phase reorder) and `7b6b39f`
(`_run_pregame_prefill_sync`) — both failed. Neither the missing-row nor the swallowed-error
path is fixed as of 2026-08-03.

## Checklist before trusting any measurement

- [ ] Does the cache contain every raw input the config's derived features need, non-null?
- [ ] Does the tick window overlap the scaler's fit window?
- [ ] Was the training extract taken after the `wall_clock_ts` repair (2026-08-03)?
- [ ] Computed z-scores with `scale_`, not `sqrt(var_)`?
- [ ] Does `has_pregame_data` mean match between training and the evaluation set?
