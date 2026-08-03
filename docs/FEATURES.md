# MMoE Feature Reference

> **SUPERSEDED for column names (2026-08-02).** The model now takes **58 features**
> (33 physics + 11 pregame + 14 market). Several columns named below no longer exist
> (`trailing_team_urgency`, `comeback_probability_proxy`, `minutes_into_game`,
> `current_run_team_encoded`, the home/away level pairs). See
> `docs/FEATURE_CONSOLIDATION.md` for what replaced them and why.
>
> **Still valid:** the per-head importance analysis and ablation results below — in particular
> that Head B depends almost entirely on the 14 market features, and that dropping all 11
> pregame features *improved* val loss (not acted on).

Reference for the 83 features fed to the MMoE model, with per-head importance and operational drop/keep recommendations from the 2026-06-08 feature analysis.

Authoritative source-of-truth for column names is `models/mmoe/feature_config.py`. This doc tells you which of those columns matter and why.

---

## TL;DR — operational recommendations

### Drop now (high confidence)

| Feature | Why |
|---|---|
| `rest_advantage` | Redundant with `team_net_rating_delta` + `form_delta`. L1 importance ≈ 0 across all heads + DCR. |
| `time_since_last_trade_ms` | Constant in val window (variance 0). Either broken or not varying in observed games. |
| `had_personal_foul` | Redundant with cumulative foul counts (`home_cum_fouls`, `away_cum_fouls`). |
| `home_lineup_just_changed` | Redundant with `lineup_net_rating_delta` — the *quality* of the change matters, not the fact of it. |
| `away_lineup_just_changed` | Same. |
| `d_spread` | Redundant with `spread` + `d_yes_bid`. |
| **All 11 pregame features** (except `has_pregame_data`) | L2 ablation: removing them **improved** total val loss (0.4137 vs 0.4220 baseline). They are constant within a game → wasted expert capacity. Keep `has_pregame_data` flag so the model knows pregame is absent for 2024-25 rows. |

### Keep — load-bearing for at least one head

| Group | Head it serves | Cost of removal (L2) |
|---|---|---|
| 14 MARKET features (`yes_bid`, `yes_ask`, `spread`, `d_yes_bid`, `bid_velocity_30s`, etc.) | **Head B (trajectory — live entry signal)** | DirAcc 0.600 → 0.552 (-7.9%); AUCPR -11.2%. Non-substitutable. |
| 11 LINEUP_SIGNAL features (`*_lineup_net_rating`, `*_star_on_court`, etc.) | Head A (run classifier) | AUCPR -9.2%. The "who's on the court" signal. |
| ~20 CONTEXT features (non-foul subset of context group) | Head C (hazard — exit timing) | Brier +22.6% when removed. Anchors `score_diff`, `period`, `minutes_into_game`, `trailing_team_urgency`, `garbage_time_risk`. |

### Watch out for these specifically

| Feature | Why it's special |
|---|---|
| `garbage_time_risk` | **Near-zero AUCPR importance but the highest decision-change-rate (DCR=0.115) of any feature** — 11.5% of trade decisions flip when this feature is permuted. Any future "auto-drop low-importance features" tooling would kill this and silently start entering blowout positions. Document in code that this stays. |
| `shot_value` | Dominates Head C Brier by ~10× (importance 0.037 vs next 0.004). The 0/2/3-point value of the last shot is the single strongest predictor of run survival. Unexpected but real. |
| `lineup_net_rating_delta` | 4× more important than either component (`home_lineup_net_rating`, `away_lineup_net_rating`) alone. The matchup *differential* carries the signal — directly validates the project thesis. |
| `team_net_rating_delta` | 4× more important in Q4 close games (importance 0.031) than mid-game (0.008). Pregame quality acts as a late-game tie-breaker only. |
| All 14 MARKET features collectively | Single point of failure for Head B. Monitor `has_market_data` freshness aggressively at inference time — stale market data degrades live entry to chance. |

### Candidates for slimming (medium confidence, needs more data)

- **MOMENTUM group (19 features)**: L2 ablation zeroed all 19 with no head-metric impact. The 4 subgroups (`points_last_5/10`, `xPPP`, `current_run_*`, `pace_*`) appear redundant with market+context. BUT keep `current_run_team_encoded` and `current_run_length` regardless — `agent.go` uses these directly for rule-based entry gating, not just for the model.
- **FOULS_ALONE subgroup (8 features)**: Negligible impact in isolation. Probably contributes only through interactions with the rest of CONTEXT.

---

## Per-head specialization (the big finding)

L2 ablation revealed strong per-head specialization that the shared-input MMoE architecture doesn't fully exploit:

| Head | Primary dependencies | Loss when zeroed (L2) |
|---|---|---|
| **A — Run classifier** | LINEUP + MARKET | Lineup ablation: -9.2% AUCPR. Market ablation: -11.2% AUCPR. |
| **B — Trajectory (entry signal)** | MARKET only | Market ablation: DirAcc 0.600 → 0.552 (-7.9%). All other groups left DirAcc flat or +. |
| **C — Hazard (exit timing)** | CONTEXT (non-foul subset) | Context ablation: Brier +22.6%. Anchored by score/time/period features. |

**Architectural implication**: Head B currently sees all 83 features but only needs 14. With only 24K joint training rows, the extra 69 features are likely overfitting noise. A model variant that restricts Head B's input to the 14 market features (or routes via per-head input projections) is plausibly a bigger win than feature-level pruning.

---

## Methodology

Two complementary analyses:

### Level 1 — Permutation importance

Shuffle each feature's column on the val set, measure metric drop per head. Done for each of 83 features singly *and* for 16 correlated feature groups (e.g., `market_orderbook` = bid/ask/spread/last/divergence shuffled together).

- Heads A + C eval: basketball val set (Feb 1 - Mar 5 2026, 68,759 rows).
- Head B eval: joint val set (Apr 7+ 2026, 314 rows — small).
- 10 permutation reps per feature, seeds 0-9, std reported.
- Per-head metrics: AUCPR (A), dir-acc & RMSE (B), Brier (C).
- Stratified by Q4-close and mid-game slices.
- **Decision-relevance score (DCR)**: % of decisions where the entry gate flips at threshold 0.12 OR the direction sign flips under permutation. The trading-relevant metric.

Caveat: trajectory targets weren't built in the L1 eval path, so per-feature Head B dir-acc importance is unreliable (most features show rank_b=1, indistinguishable). Group-level Head B importance via DCR is still trustworthy.

### Level 2 — Clean-retrain group ablation

For each of 6 feature groups (lineup_signal, momentum, context, pregame, market, fouls_alone) plus a baseline:
- Zero the group's columns in both train and val tensors (mean/scale also overwritten).
- Retrain MMoE from scratch with identical hyperparams (lr=1e-3, batch=512, max_epochs=200, patience=15, w_a=1.0, w_b=1.0, w_c=0.3, lambda_entropy=0.02, seed=42).
- Compare best-epoch val metrics to baseline.

Replication check (baseline retrain vs published `mmoe_delay20.pt`):
- AUCPR 0.1571 vs 0.1593 (1.4% off — within MPS non-determinism)
- DirAcc 0.600 vs 0.611 (1.8% off)
- Brier 0.0862 vs 0.0860 (0.3% off)
- ✓ Pipeline is sound; ablation deltas are interpretable.

### Level 3 — Top-K retraining (incomplete)

Was meant to find the smallest K (subset of L1-ranked features) where all heads stay within 5% of baseline. K=83 ran 9 epochs before session limit; K=20/30/40/50 never ran. Dataloaders are cached at `analysis/feature_importance/level3_topk/_cache/loaders.pkl` (177 MB) — resume via `analysis/feature_importance/level3_topk/run_topk.py --k {K}` for ~25 min total.

---

## Full L2 results table

| Run | Features zeroed | AUCPR_A | Δ % | DirAcc_B | Δ % | RMSE_B | Brier_C | Δ % | Total loss |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 0.1571 | — | 0.600 | — | 0.348 | 0.0862 | — | 0.4220 |
| ablate_lineup_signal | 11 | 0.1427 | **-9.2** | 0.624 | +4.0 | 0.417 | 0.0850 | -1.4 | 0.4205 |
| ablate_momentum | 19 | 0.1601 | +1.9 | 0.602 | +0.3 | 0.349 | 0.0870 | +0.9 | 0.4217 |
| ablate_context | 28 | 0.1561 | -0.6 | 0.616 | +2.8 | 0.240* | 0.1057 | **+22.6** | — |
| ablate_pregame | 11 | 0.1565 | -0.4 | 0.618 | +3.0 | 0.358 | 0.0845 | -2.0 | **0.4137** |
| ablate_market | 14 | 0.1395 | **-11.2** | **0.552** | **-7.9** | 0.354 | 0.0882 | +2.3 | — |
| ablate_fouls_alone | 8 | 0.1629 | +3.7 | 0.613 | +2.2 | 0.363 | 0.0857 | -0.6 | — |

*ablate_context RMSE drop is a regularization artifact (model collapses toward conservative near-zero trajectories), not actual gain — DirAcc didn't follow.

Pregame ablation total val loss (0.4137) is **lower** than baseline (0.4220) — this is the strongest "this group hurts us" signal in the entire study.

---

## L1 top-15 by merged ranking

From `analysis/feature_importance/level1_permutation/ranking_for_topk.csv`. Merged rank = min over (rank_a, rank_b, rank_c, rank_dcr), tie-broken by sum.

| Rank | Feature | Top-importance for |
|---|---|---|
| 1 | `trailing_team_urgency` | Head A |
| 2 | `score_diff` | Head A |
| 3 | `has_market_data` | Market gating |
| 4 | `current_run_team_encoded` | Head A (direction) |
| 5 | `yes_bid` | Head A + dominant for Head B |
| 6 | `yes_ask` | Same |
| 7 | `yes_last` | Same |
| 8 | `home_points_last_5_poss` | Momentum |
| 9 | `away_points_last_5_poss` | Momentum |
| 10 | `pace_last_10_possessions` | Pace |
| 11 | `minutes_into_game` | Time/regime |
| 12 | `away_points_last_10_poss` | Momentum |
| 13 | `home_full_timeouts_remaining` | Timeout state |
| 14 | `away_star_on_court` | Lineup |
| 15 | `home_star_on_court` | Lineup |

Full 83-feature ranking lives in `analysis/feature_importance/level1_permutation/ranking_for_topk.csv`.

---

## L1 group importance (AUCPR, full val set)

From `analysis/feature_importance/level1_permutation/ranking_groups.csv`. Higher = more important.

| Group | AUCPR importance | Brier importance | DCR |
|---|---|---|---|
| market_orderbook | **0.0313** | 0.0011 | 0.173 |
| lineup_apm | 0.0174 | 0.0001 | 0.065 |
| score_time | 0.0143 | 0.0016 | 0.265 |
| pregame_rating | 0.0111 | 0.0001 | 0.115 |
| stars | 0.0057 | 0.0001 | 0.064 |
| market_velocity | 0.0038 | 0.0000 | 0.052 |
| momentum_5_10 | 0.0030 | 0.0023 | 0.063 |
| pace | 0.0028 | 0.0002 | 0.051 |
| shot_quality | 0.0024 | 0.0006 | 0.056 |
| current_run | 0.0022 | 0.0018 | 0.064 |
| pregame_context | 0.0013 | 0.0002 | 0.072 |
| market_liquidity | 0.0008 | 0.0001 | 0.120 |
| shot_event | 0.0003 | 0.0389* | 0.036 |
| bonus_state | -0.0006 | 0.0001 | 0.059 |
| timeouts | 0.0000 | 0.0003 | 0.046 |
| lineup_change | 0.0000 | 0.0001 | 0.016 |

*`shot_event` Brier importance is driven almost entirely by `shot_value` — the 0/2/3-point value of the last shot. See "watch out" table above.

---

## Caveats

- **Head B dir-acc importance (L1) is unreliable** — trajectory targets weren't built in the L1 eval path, so per-feature Head B importance scores are essentially noise. Trust L2's Head B numbers instead.
- **Joint val set is small** (314 rows for Apr 7-12 window). Individual market-feature importance is noisy; group-level is trustworthy.
- **Baseline replication is within 1.4% of published**, but MPS non-determinism on Apple Silicon means deltas under 1% aren't meaningful. Treat differences below 2% as "no effect."
- **L3 plateau curve is missing** — we don't yet know the smallest K where all three heads stay within tolerance. Top-K recommendations above are derived from L1+L2 inference, not direct measurement.
- **agent.go uses `current_run_team_encoded` and `current_run_length` for rule-based entry gating** independent of the model. Keep them in the pipeline even if the model doesn't need them.

---

## Files

- Source: `models/mmoe/feature_config.py` (column-name truth)
- L1 artifacts: `analysis/feature_importance/level1_permutation/`
- L2 artifacts: `analysis/feature_importance/level2_ablation/`
- L3 artifacts (incomplete): `analysis/feature_importance/level3_topk/`
- Synthesis report: `analysis/feature_importance/synthesis/REPORT.md`
- Analysis date: 2026-06-08
