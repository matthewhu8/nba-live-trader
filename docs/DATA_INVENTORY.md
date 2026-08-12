# Data Inventory (measured 2026-08-11)

Every number here was measured, not estimated.
Regenerate the warehouse section with `python scripts/measure_data.py`.
Regenerate the pipeline section by running `models.mmoe.dataset.build_dataloaders()` and reading its log lines.
Do not hand-edit counts in this file, and do not copy counts out of it into a docstring that will then rot.

This file supersedes the row counts previously carried in `dataset.py`, `model.py`, `feature_config.py`, `ARCHITECTURE_V2.md`, `FEATURES.md` and `FEATURE_CONSOLIDATION.md`.
Those said `~24K joint rows / 148 games` and `~390K basketball`.
The joint set is now **37,180 rows across 232 games**, 55% larger than the figure the architecture was sized against.

---

## 1. Warehouse (MotherDuck `md:kalshi_trading`)

| table | rows |
|---|---|
| `features.possession_flat` | 449,274 |
| `features.pregame` | 1,143 |
| `features.lineup_ratings` | 47,329,421 |
| `features.player_ratings` | 780,640 |
| `features.team_ratings` | 2,174 |
| `main.kalshi_ticks` | 2,345,274 |
| `main.dim_games` | 2,330 |
| `main.kalshi_settled` | 11,252 |
| `main.kalshi_market_map` | 11,288 |

### `features.possession_flat`

449,274 rows across 2,178 games, spanning 2024-10-22 to 2026-06-13.
409,872 rows across 2,127 games carry a non-null `wall_clock_ts`, which is the prerequisite for joining to market data.

| season | rows | games | window |
|---|---|---|---|
| 2024-25 | 180,062 | 875 | 2024-10-22 to 2025-02-28 |
| 2025-26 | 269,212 | 1,303 | 2025-10-21 to 2026-06-13 |

Every `game_id` in `possession_flat` resolves in `dim_games`.

### `main.kalshi_ticks`

2,345,274 ticks across 570 market tickers and 235 game IDs, spanning 2026-03-23 to 2026-06-13.
63,743 ticks (2.7%) carry no `game_id` and are unusable.

| month | ticks | tickers | games |
|---|---|---|---|
| 2026-03 | 434,146 | 126 | 59 |
| 2026-04 | 906,859 | 253 | 136 |
| 2026-05 | 795,143 | 171 | 40 |
| 2026-06 | 209,126 | 30 | 6 |

Tick recording now extends nearly two months past where the documentation assumed it stopped (2026-04-12).
The May and June games are postseason, which is a different market regime from the regular season.

### Joint upper bound

232 games appear in both `possession_flat` (timestamped) and `kalshi_ticks`, covering 47,441 possessions.
This is an upper bound; the trainable count after filtering is in section 2.

---

## 2. What the training pipeline actually produces

Measured by running `build_dataloaders()` with defaults: `feed_delay_seconds=20`, `horizon_seconds=120`, `tp=5`, `sl=3`.

```
Traded-regime filter   449,274 -> 437,060 rows (97.3% kept)
                       dropped 2,310 overtime, 9,904 blowout (|score_diff| > 30), 0 incomplete
Home-contract ticks    894,884 across 232 games (13,077 ticks unmatched to a game)
Tick join              8,301 of 45,481 candidate possessions dropped as stale (18.3%)

Joint rows             37,180 across 232 games
Basketball-only rows   391,579

Train                  338,091 rows  (322,708 basketball + 15,383 joint)
Val                     62,622 rows  ( 40,825 basketball +  21,797 joint)
```

Joint rows are **8.7%** of the 428,759 usable total, up from the 6% previously documented.

### Split boundaries

Basketball splits key on `game_date` across all of `possession_flat`:

| split | rows | games |
|---|---|---|
| train (<= 2026-01-31) | 330,638 | 1,599 |
| val (<= 2026-03-05) | 42,287 | 206 |
| test, sacred (> 2026-03-05) | 76,349 | 373 |

---

## 3. Defects this measurement exposed

### 3.1 The Head B split is inverted

`HEADB_SPLIT_DATE = 2026-04-07` was chosen to give roughly 80/20 when ticks covered 148 games ending 2026-04-12.
Ticks now run to 2026-06-13, so the same date gives:

| split | games | rows | share |
|---|---|---|---|
| train (< Apr 7) | 97 | 15,383 | 41.4% |
| val (>= Apr 7) | 137 | 21,797 | 58.6% |

Head B trains on the minority of its own data and validates on the majority.
Worse, the split is now also a regime boundary: every late-season and postseason game falls in validation, including all 46 games from May and June.
Any Head B metric measured under this split is reporting out-of-regime generalization on an oversized val set, which is not what the split was designed to measure.
Re-derive the date before the next Head B retrain.

### 3.2 The joint window sits entirely inside the basketball test period

The basketball test boundary is 2026-03-05.
Tick recording starts 2026-03-23.
So all 232 joint games fall inside what Heads A and C treat as the sacred test period.

There is no direct leak, because joint games (which have ticks) and basketball-only games (which do not) are disjoint sets.
But it does mean 232 games from the test window are being trained on by Head B.
Evaluating Heads A and C on the "sacred" test set is still clean; just do not describe the post-2026-03-05 period as globally untouched, because it is not.

### 3.3 Nine games are misaligned with their tick recording window

`validate_possession_tick_overlap` logs an ERROR for 9 of 232 games where the first possession falls outside the tick window, so the asof join returns settled end-of-game prices.
Worst offenders by tick coverage: `0022501047` (15.6%), `0022501041` (25.0%), `0022501052` (38.6%), `0022501070` (39.5%), `0022501038` (60.9%).
Training continues past this ERROR rather than failing.

### 3.4 The "std 35 = settled prices" fingerprint needs re-deriving

`skills/model-provenance.md` argues that a market-scaler `yes_bid` std near 35 is the signature of settled 1c/99c prices, citing 11.63 as the in-band reference.
Measured on the full current tick set, raw `yes_bid` has mean 47.4 and **std 28.0**, with only 2.4% of ticks at or below 2c and 1.1% at or above 98c.
The current refit gives std 31.1, which is close to the underlying distribution rather than far above it.
The 11.63 reference appears to have come from a much narrower window than the data now spans.
This does not clear the older checkpoints, but the fingerprint argument as written no longer holds and should be re-derived before it is used to judge a scaler again.

---

## 4. Sizing implication for the model

`MMoEModel` has 30,090 parameters: 22,656 in the three experts (75.3%), 6,933 in the heads (23.0%), 441 in the gates, 60 in the market encoder.

Heads A and C see all 338,091 training rows, so capacity is not the binding constraint for them.
Head B and the market encoder see only the 15,383 joint training rows, and `trainer.py` masks Head B further to rows where mean `|traj| > 0.02`.
Those rows come from 97 games, and possessions within a game are strongly autocorrelated, so the effective independent sample size for the trading head is far closer to 97 than to 15,383.

That is the real argument for questioning the expert width and the expert count, and it survives the corrected numbers.
It has not been tested: no width or expert-count ablation has been run.
