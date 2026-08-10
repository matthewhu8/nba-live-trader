# Model Provenance — What Each Head Actually Learned From

Read this before concluding that a signal "doesn't work." Several heads were trained on far
less data than the documentation implies, so a null result may be a coverage problem rather
than a signal problem. All figures measured 2026-08-03 from the saved scalers, checkpoints
and the warehouse.

## Head B trained on ~19K rows, not 42K

`has_market_data` train mean x rows gives the joint (possession + tick) count directly:

| | rows fit | joint rows | joint % |
|---|---|---|---|
| 83-feature model | 350,449 | 19,811 | 5.7% |
| 58-feature model | 341,472 | **18,764** | 5.5% |

Three places in the docs disagree with this and with each other — `49,576`, `~42K`, `24K`.
The scaler is authoritative: **Head B's trading signal learned from under 19,000 possessions**,
before masking. `trainer.py` masks Head B further to `has_market_data * (|traj| > 0.02)`,
dropping ~42% more, so the effective sample is ~12K.

Also: if the total joint set is 49,576, then splitting at `HEADB_SPLIT_DATE` (2026-04-07)
gives **40% train / 60% val**, not the "~80/20" claimed at `dataset.py:50`. Reported Head B
metrics were measured on a val set *larger* than the train set.

## The Core Thesis has never been trainable

`features.pregame` and `features.lineup_ratings` stop being materialised after **2026-03-24**.
Kalshi tick recording starts **2026-03-23**. The overlap is two days.

Head B's training subset, measured:

| | Head B train | all train rows |
|---|---|---|
| `has_pregame_data = 1` | **12.3%** (2,519 rows, **12 games**) | 43.6% |
| `lineup_net_rating_delta != 0` | **31.9%** | ~51% |

**The marginal rate is misleading.** 43.6% looks healthy but is carried by basketball-only
rows the trading head never sees. The head that places trades saw pregame on 12 games.

From Mar 26 onward, `lineup_ratings` covers exactly **one game per date**. Every other game is
100% zero across all five lineup features — and unlike pregame, the lineup features have **no
presence flag**, and `0` is a legitimate value meaning "evenly matched." So ~68% of Head B's
rows carry an unflagged fake reading the model cannot learn to ignore, on
`lineup_net_rating_delta` — the column `models/features/builder.py:68` calls "THE core signal."

**This was not fixed by the 58-feature retrain.** `has_pregame_data` train mean is 0.4352
versus the old 0.4356, and the new `lineup_confidence` feature (`log1p(min(home_n, away_n))`)
has mean **0.1486**, implying real lineup sample sizes on only ~6% of rows. Keeping all 11
pregame features does not help while the upstream tables stop at 2026-03-24.

## The timestamp corruption hit validation, not training

86 games had `wall_clock_ts` one day late (repaired in MotherDuck 2026-08-03). Corruption
tracks late tip-offs, which cluster in the playoffs — i.e. after the split date:

| window | rows | corrupt | with pregame |
|---|---|---|---|
| Head B **train** (< Apr 7) | 20,458 | 1,869 (9.1%) | 2,519 (12.3%) |
| Head B **val** (>= Apr 7) | 27,080 | **14,899 (55.0%)** | 14,940 (55.2%) |

Of the 1,869 corrupt training rows, **zero** had pregame data. So the corruption did not
degrade what Head B learned. It degraded **early stopping, best-epoch selection, and every
reported Head B metric** — 55% of the val rows carried settled 1c/99c prices.

**Every checkpoint to date was trained pre-repair.** Fingerprint: the market scaler's
`yes_bid` std is 35.25 (new) and 35.92 (old), while repaired in-band ticks give 11.63. A std
of 35 requires heavy mass at both 1c and 99c — the settled-price signature. The new scaler is
statistically indistinguishable from the old one on every market feature.

## Checkpoint inventory (origin/main, 2026-08-03)

| file | epoch | val_loss | expert input dim | notes |
|---|---|---|---|---|
| `mmoe_delay20.pt` | 5 | 0.4241 | 48 | deployed; 58 feats + market encoder |
| `mmoe_delay20_nomktenc.pt` | 5 | 0.4185 | 58 | better val_loss, worse Head B |
| `mmoe_delay20_h360.pt` | 10 | 0.4306 | **83** | old architecture |
| `mmoe_delay20_h600.pt` | 3 | 0.4401 | **83** | old architecture |

Two cautions. The deployed model **stopped at epoch 5** on a validation set that was 55%
corrupt, so treat both the stopping point and the 0.4241 as unreliable rather than wrong. And
`h360`/`h600` are the **83-feature architecture** — they are not horizon variants of the
current model and cannot be compared to it directly.

Per `backtesting.md`, selection is on Head B (62.5% dir acc with the encoder vs 61.3%
without), not val loss. That is the right criterion; it just cannot be trusted until the val
set is clean.

## The June 2026 feature audit was invalid — do not act on it

`cce25f8` (dated **2026-08-03**, though its own docstring misdates it 2026-06-10) cut 83 -> 65
features citing `scratch/feature_audit.csv`. That artefact is unreliable:

- Its "training variance" came from raw warehouse tables **without** the trainer's
  `fillna(0)`, and never opened the scaler. A correct helper (`scratch/diff_zscore.py`) sat in
  the same folder, unused.
- `was_foul` / `was_sub` were dropped for `train_std = 0`. **False.** Those flags are stored
  sparsely (1 on the event, NULL otherwise) and the audit query appended
  `WHERE col IS NOT NULL`, deleting every negative case and leaving all ones. Real scaler stds
  are 0.3748 and 0.3267 — among the healthiest features in the set.
- **No feature in the 83 has scaler std < 1e-9.** Nothing was constant in training.
- Corrected, the 18 dropped features max out at **0.88 sigma** of live drift, while five
  *kept* market features exceed 3.7 sigma. The audit dropped the least-drifted features and
  retained the worst.
- Its live evidence ("1,605 possessions") is really 1,300 distinct possessions across **7
  games** with 19% duplicated rows, from a different host.

Restore `was_foul` / `was_sub` on sight. Do not use this artefact to justify a retrain.

## Statistical power — what a measurement can and cannot show

From the corrected backtest's per-trade variance (sd $3.96), at 80% power, two-sided 5%:

| true edge | trades needed | games needed |
|---|---|---|
| +$0.25/trade | ~1,970 | ~1,700 |
| +$0.50/trade | ~490 | ~425 |
| **+$1.00/trade** | **~123** | **~106** |
| +$2.00/trade | ~31 | ~27 |

The 71-game cache produced 80 trades across only 30 games. **A modest edge cannot be
demonstrated on it.** Size any validation set against this table first.

Relevant to future architecture work: an MMoE over 58 tabular features can survive ~12K
effective Head B rows. A recurrent or sequence model cannot — and with a 55%-corrupt val set
you could not detect the overfitting. Joint coverage, not model capacity, is the binding
constraint.

## Head B metrics are not comparable across Level 2 (2026-08-10)

Level 2 re-anchored the exit simulator at `wall_clock_ts + feed_delay`. Three things moved at
once, so **no pre-Level-2 Head B figure can be compared to a post-Level-2 one** — including
the 62.5% dir acc in `CLAUDE.md`:

1. **The targets changed.** The checkpoint grid shifted 20s later. `traj_0` was `wct+12s` —
   eight seconds *before* the position could exist — and is now a genuine 12s-ahead forecast.
2. **The row population changed too**, which is the part that gets missed.
   `trainer.py::_compute_loss` masks Head B to rows where `mean |traj| > 0.02` — a threshold on
   the very labels that moved. The anti-causal early TP/SL hits were exactly the
   high-magnitude rows that cleared it; many now exit flat and drop out. `loss_b` is an
   average of a different quantity over a different set of rows, with a different denominator.
3. **Selection is on total val loss** (`trainer.py:281-282`), so a change in `loss_b`'s scale
   re-weights the criterion and shifts which epoch gets kept — for reasons unrelated to any
   head improving. Head A and Head C are individually well-defined across the change; *which
   checkpoint you have* is not.

Direction is not predictable either: removing the leakage makes the task genuinely harder
(loss up) while shrinking target variance pushes MSE down (loss down). Do not read the sign.

### What to do instead

**Score the existing checkpoint on the new labels.** Level 2 touched only the label builder —
features, the scaler and the cache are untouched — so `models/saved/mmoe_delay20.pt` can be
evaluated against regenerated targets with no retrain, inference only. Old-model/new-truth vs
new-model/new-truth share targets, mask and rows, and the gap quantifies how much of Head B's
apparent skill *was* the leakage. That is the most informative number Level 2 can produce.

Three instrumentation fields on `EpochMetrics` support this, logged every epoch:

- `head_b_rows` / `head_b_rows_total` — mask retention. The hidden variable behind (2); a
  large drop is itself the finding about how much training signal was anti-causal.
- `rmse_b_zero` — RMSE of predicting 0.0 everywhere. `rmse_b` has no absolute meaning once the
  target variance moves; the logged `RMSE_B/zero-pred` ratio does. Above 1.0 means Head B is
  worse than predicting nothing.
- `rmse_b_per_checkpoint` — logged beside each saved checkpoint. `traj_0`/`traj_1` (12s/24s
  against a 20s delay) straddled the anchor and carried the most contamination; the aggregate
  averages away the effect under test.

And compare end-to-end in dollars against the backtest baseline, which did **not** move
(`backtesting.md`).
