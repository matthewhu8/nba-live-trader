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

Measured against the 58-feature config on this cache (2026-08-03), 7 of the 44 non-market
features are dead — 37 vary normally:

```
was_sub              all-NaN
had_shooting_foul    constant 0      had_personal_foul    constant 0
sub_count_edge       constant 0      roster_rapm_gap      constant 0
missing_rapm_impact  constant 0      rest_advantage       constant 0
```

Note `pace_ref`, `poss_since_timeout`, `no_timeout_yet`, `team_foul_edge` and
`timeout_called_edge` DO vary on this cache, so they are safe. The dead set is smaller than
feared — but a backtest on this cache is still feeding the model seven constants it saw
varying in training, so treat any P&L from it as provisional until the cache is re-exported.

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

## The exit-simulator feed delay — FIXED 2026-08-10 (Level 2)

`build_trajectory_targets` anchored every offset at each possession's `wall_clock_ts` with
**no feed delay**, while `yes_bid` on the same row came from the asof join at
`wall_clock_ts + feed_delay_seconds`. The labels were therefore anti-causal, not merely
early: TP and SL resolved against ticks that preceded the price the position entered at.

Now anchored at `wall_clock_ts + feed_delay_seconds`, which is a **required** argument with
no default — omitting it is the defect, so a new call site has to state it. The one name
feeds both `future_*` filters, the checkpoint grid and the time gate, so the hold is a true
`horizon_seconds` from entry (it was `wall_clock_ts + horizon`, i.e. 100s of real hold at a
20s delay). The take-profit clamp moved inside `simulate_exit` in the same change, so the
post-exit checkpoints freeze at the collectable price rather than the overshoot.

Verified in both directions: reverting the anchor fails 4 tests in
`test_backtest_invariants.py`, reverting the clamp fails 3. The backtest baseline is
unchanged at 181 trades / −$324.35, as expected — it clamped its own copy already.

**Head C was never affected.** Both this file and `CLAUDE.md` claimed the simulator built
"Head B and Head C" labels. Head C's hazards come from `kalshi_targets.add_hazard_targets`,
indexed over the next N *scoring possessions* rather than wall clock, and never touch the
exit simulator. Only `traj_0..9` (Head B) moved.

Retraining is unblocked. Read the comparability note in `model-provenance.md` first: Head B's
loss mask is a threshold on the labels themselves, so pre- and post-Level-2 `loss_b` figures
are not comparable in either direction.

### Possession-driven exits — FIXED 2026-08-10, on `fix/possession-event-delay`

**This section was here before Level 2, wrongly deleted by it, restored, and now resolved.**
Level 2 fixed the *entry* anchor, which governs which possessions the exit search can reach.
It did not delay the possession *events* themselves. That is now done, in both simulators at
once, keyed on a precomputed `_knowable_ts = wall_clock_ts + feed_delay_s`.

**It cost $22.57: −$324.35 → −$346.92** on the same config (181 trades both, 26.0% → 25.4%,
avg hold 70.7s → 76.3s, flip 59 → 53, stop 74 → 79). The early flips were acting as a lucky
exit; holding 20s longer changes the outcome of 6 of them (5 → `stop_loss`, 1 →
`take_profit`). This is the expected shape of an
honest correction — the old number was flattered by acting on information it did not have.

The filter also **widens**, which is the counter-intuitive half: a possession inside the delay
window has a raw wall clock *before* the anchor but becomes knowable *during* the hold, so
`wall_clock_ts > anchor` dropped events a live trader would have acted on. The correct filter
is on knowable time, which reduces algebraically to `wall_clock_ts > wct` — resembling the
reverted defect while being correct, because the tick filter stays at the anchor. Ticks are
market data observed live; possessions are game state on a delayed feed. Both call sites are
written in the unreduced form so that distinction survives review.

**Historical, as of 2026-08-09 — none of the code below exists any more.** Kept because
this section was once deleted while the bug was live. Do not act on it; verify against
the current source first. (Grepping this file and concluding the bug is live is exactly
how the deletion went wrong the first time.)


`simulate_exit` still selects `poss_so_far = window_possessions[wall_clock_ts <= tick["ts"]]`,
comparing raw possession wall clock against tick time, so a momentum flip or garbage-time
transition is acted on the moment its wall clock passes — not `feed_delay_s` later, when a live
trader would learn of it. Measured 2026-08-10: entry anchored at `T0+20`, run flips at
possession wall clock `T0+30`, first tick at `T0+35` → `momentum_flip` at `T0+35`, though the
flip is not knowable until `T0+50`. `dynamic_exit.simulate_exit_dynamic` has the same shape,
where it also gates re-inference (`window_poss.iloc[i]["wall_clock_ts"] <= tick["ts"]`).

It was material: `momentum_flip` was 32.6% of exits. Calling it a "modelling change" rather
than a bug (as an earlier revision of this file did) was too soft — the 20s is CDN polling
latency for game events, which is exactly how possessions and the score reach us, so the same
delay demonstrably applies. The entry signal already worked this way. Applying it to exit-side
possession events is the consistent completion of that, not a judgement call.

## Checklist before trusting any measurement

- [ ] Does the cache contain every raw input the config's derived features need, non-null?
- [ ] Does the tick window overlap the scaler's fit window?
- [ ] Was the training extract taken after the `wall_clock_ts` repair (2026-08-03)?
- [ ] Computed z-scores with `scale_`, not `sqrt(var_)`?
- [ ] Does `has_pregame_data` mean match between training and the evaluation set?
