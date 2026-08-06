# Level 1 — Phase 0 Recon Report

**Date:** 2026-08-05
**Branch:** `fix/backtest-exit-window` @ `25587d9`
**Target:** `origin/main` @ `1f116b1`
**Status:** read-only. No files modified, no commits. Two remote refs created (Phase −1 backup only).

> **Headline:** the recon reproduced the documented `−$136.42` baseline *exactly* — but only after
> discovering the command written in `.claude/skills/backtesting.md` is not the command that
> produced it. With the correct config, branch HEAD measures **−$285.73 across 135 trades**, not
> −$136.42 across 80. The documented baseline went stale when the branch merged PR #51 and was
> never re-measured. Details in item 7.

---

## Phase −1 — completed

| check | expected | actual |
|---|---|---|
| `git rev-parse HEAD` | `25587d9` | `25587d940cf4d61a2fbd35802c5352b9baa4286a` ✓ |
| `git ls-remote origin fix/backtest-exit-window` | empty | **empty** ✓ (absent from all 21 remote heads) |
| `gh pr list --search "exit-window"` | nothing open | **could not run — `gh` is not installed** |

`gh` is absent from this machine (not in `/opt/homebrew/bin`, not on `PATH`). The open-PR check was
satisfied indirectly instead: a GitHub PR requires a live head ref, and no remote branch of that
name existed, so no open PR could exist. This is sound for the stop-trigger, but see
*spec omits #1* — `gh` is also required by Phase 6d.

Backup pushed, both refs at `25587d9`:

```
25587d94...  refs/heads/backup/exit-window-25587d9
25587d94...  refs/heads/fix/backtest-exit-window
```

The fix is off the laptop.

---

## Item 1 — `origin/main` HEAD

`1f116b1e4e52585ac7013473099d79de245f5011` — matches the spec exactly. No fifth PR landed.

---

## Item 2 — Is PR #52 in the local branch?

**Yes, by ancestry — the content check the spec prescribes was unnecessary.**

`9e799b3 Merge pull request #52` appears directly in `git log fix/backtest-exit-window`. #52 was a
*merge*, not a squash, so all five SHAs (`b37a4b9`, `9da6e31`, `0de4526`, `47156ae`, `18a5c29`) are
present on both sides.

The merge-base is `ef4c21f` (PR #51), which is **downstream of #52**. So #52 is shared history, not
something to reconcile. Divergence is small:

```
commits on branch not on main:  10  (9 real + 1 merge commit 5f8b723)
commits on main not on branch:   3  (63381ee, 8a40267, 1f116b1 — the #54 series)
```

---

## Item 3 — Documentation read

Read in full: `.claude/CLAUDE.md`, `.claude/skills/{backtesting,data-integrity,model-provenance,feature-engineering}.md`,
`backtesting/mmoe_backtest.py` (792 lines), `models/mmoe/dataset.py` (relevant regions),
`models/targets/exit_simulator.py` (entry region), `live-trader/config/trading.yaml`.

Contradictions and omissions are collected in the two dedicated sections below rather than repeated here.

One thing worth surfacing immediately, because it changes Phase 3a: **`data-integrity.md` already
documents the `exit_simulator.py` bug as deliberately unfixed** (lines 123–131, "Known remaining
bias — deliberately unfixed"). The repo and the spec agree on the Level 2 scope fence. Good.

---

## Item 4 — The four measurement defects

The spec says it "only names three with confidence (exit-window lookahead, missing `P×(1−P)` in the
fee formula, cents-vs-dollars)". **That list is wrong.** `.claude/skills/backtesting.md:31–36`
enumerates the actual four, and the missing `P×(1−P)` is not among them — that was a *documentation*
typo in `CLAUDE.md`, fixed in passing by commit `03940a9`, never a backtest defect.

| # | defect | fix location (branch @ `25587d9`) |
|---|---|---|
| 1 | **Exit-window lookahead** — entry priced at `wct + delay`, exit search started at `wct` | `mmoe_backtest.py:432` `_tick_at_delay(...)`; `:441` `simulate_exit(entry_wall_clock=entry_anchor_ts)`; guard `:464` |
| 2 | **TP over-crediting** — TP booked at the breaching tick, not the resting limit at `entry ± TP` | `mmoe_backtest.py:456–458` |
| 3 | **Fees hardcoded to `0.0`** — `_compute_maker_fees()` returned `0.0` unconditionally | `mmoe_backtest.py:297–323` (`_fee_one_leg` + `_compute_fees`) |
| 4 | **P&L in cents, labelled dollars** — 100× overstatement | `mmoe_backtest.py:279–289` (`pnl_dollars`) + `:650` |

All three line numbers the spec gives for defect 1 (~432 / ~441 / ~464) are **exact**. Line numbers
elsewhere in the spec are not — see below.

The fourth defect the spec asked me to identify is **defect 2, TP over-crediting**.

---

## Item 5 — What PR #54 actually changed

```
 .claude/CLAUDE.md                 | 15 +++--
 backtesting/mmoe_backtest.py      | 66 +++++++++++++-----
 live-trader/go/game.go            |  3 +-
 live-trader/go/orders.go          | 53 ++++++++++---
 live-trader/go/resting_tp_test.go | 24 ++++--
 live-trader/go/ring_buffer.go     | 35 ++++++--
 models/mmoe/dataset.py            | 62 ++++++++++++---
```

All three items the spec asked me to verify are **present on main**:

- ✓ `traded = out["volume"].diff()` → `.fillna(0.0).clip(lower=0.0)` before the rolling sum (`dataset.py`)
- ✓ `tolerance=pd.Timedelta(seconds=MARKET_STALENESS_TOLERANCE_SECONDS)` on `merge_asof`, with
  `MARKET_STALENESS_TOLERANCE_SECONDS = 30`
- ✓ bounded lookback in `_get_market_features_at_delay` using the same constant
- ✓ `ring_buffer.go` gained `tradedAt(i)` with the cross-reference comment to `dataset.py`

**But the spec badly understates what #54 did to the backtest.** Matt independently fixed
**two of the same four defects**, with a different implementation:

| defect | branch (`25587d9`) | main (`1f116b1`) |
|---|---|---|
| 3 — zeroed fees | `_fee_one_leg(rate, price, contracts)` + `_compute_fees(entry, exit, contracts, **exit_reason**)` | `kalshi_fee(contracts, price, rate)` + `_compute_maker_fees(entry, exit, contracts)` — **maker/maker only** |
| 4 — cents/dollars | `pnl_dollars()` helper | inline `gross * contracts / 100.0` |
| 1 — exit window | fixed | **not fixed** |
| 2 — TP over-credit | fixed | **not fixed** |

This is a genuine semantic reconciliation of two overlapping fixes, not a "keep mine / keep theirs"
on disjoint code. It is the main reason I recommend merge over rebase (item 6).

---

## Item 6 — Conflict preview

```
$ git merge-tree --write-tree --name-only fix/backtest-exit-window origin/main
.claude/CLAUDE.md
backtesting/mmoe_backtest.py
CONFLICT (content): Merge conflict in .claude/CLAUDE.md
CONFLICT (content): Merge conflict in backtesting/mmoe_backtest.py
```

Exactly the two predicted files. **No conflicts in `.claude/skills/*.md`** — the branch created
`data-integrity.md` and `model-provenance.md` new, and main never touched them. No Go conflicts.

The branch's full footprint vs merge-base confirms clean separation:

```
 .claude/CLAUDE.md                    |  33 +-
 .claude/skills/backtesting.md        |  82 ++-
 .claude/skills/data-integrity.md     | 139 +++
 .claude/skills/model-provenance.md   | 127 +++
 .gitignore                           |   4 +-
 backtesting/mmoe_backtest.py         | 239 ++++-
 backtesting/results/*.csv (4 files)  | 523 +++
 docs/STATE_2026-08-03.md             | 411 +++
 tools/sweep_dynamic_exit.py          |   7 +-
 tools/sweep_traj_aggregator.py       |   2 +-
```

The branch **does not touch `models/mmoe/dataset.py` or `models/targets/exit_simulator.py` at all.**
The scope fence is already respected by construction.

---

## Item 7 — Pre-rebase baseline  ⚠️ **STOP-TRIGGER FIRED, THEN DIAGNOSED**

### 7a. The documented command does not reproduce the documented number

Running the exact command block from `.claude/skills/backtesting.md:4–11`:

```bash
python -m backtesting.mmoe_backtest --use-traj-for-side --min-abs-traj 0.08 \
  --min-run-length 2 --hold-seconds 240 --threshold 0.0
```

| | trades | win rate | gross | fees | net |
|---|---|---|---|---|---|
| documented | 80 | 30.0% | +$2.00 | $138.42 | −$136.42 |
| **measured @ `25587d9`** | **466** | **31.1%** | **+$22.00** | **$803.40** | **−$781.40** |

Per the spec this is a hard stop. I diagnosed it rather than halting cold, because the spec says a
trade-count mismatch "needs to be diagnosed first."

### 7b. It is not the cache

The cache is byte-identical to the one the assessment describes: 71 games / 14,239 possession rows /
**770,054 ticks** (the exact figure the spec quotes), 69 tradeable, Apr 15 – May 17 2026. Cache
mtimes are `2026-08-03 15:00–15:01`, which **predates** the baseline commit `03940a9` at 16:01. Same
cache, same numbers available.

### 7c. It is not (only) the model swap

Hypothesis: the baseline was measured before PR #51 swapped the 83-feature config for the
58-feature one. Confirmed that the swap happened at `35de8ab` (17:11), which entered the branch via
merge `5f8b723` at **21:12**, hours after the baseline was recorded at 16:01. Verified feature counts
in a throwaway worktree:

```
85d8a66 (pre-#51):  ALL 83  = PHYSICS 58 + PREGAME 11 + MARKET 14
25587d9 (HEAD):     ALL 58  = PHYSICS 33 + PREGAME 11 + MARKET 14
```

Re-running the documented command at `85d8a66` gave **278 trades / −$482.75**. Closer, still not 80.

### 7d. Root cause: the documented command is the wrong command

The Aug-3 result CSVs were untracked by `613c090`, but are recoverable from `613c090^`. Three of
them hold exactly 80 rows. Reading the config back out of the saved columns:

| CSV | trades | win rate | net | fees | games | `traj_aggregator` | min `run_prob` |
|---|---|---|---|---|---|---|---|
| `20260803_155822` | 80 | 37.5% | +$61.58 | $138.42 | 30 | `mean` | 0.1508 |
| `20260803_162248` | 80 | **30.0%** | **−$136.42** | **$138.42** | 30 | `mean` | 0.1508 |
| `20260803_162417` | 80 | **30.0%** | **−$136.42** | **$138.42** | 30 | `mean` | 0.1508 |
| `20260803_170339` | 63 | 34.9% | −$82.53 | $103.53 | 32 | `mean` | 0.1506 (`--prod-features`) |

`traj_aggregator` is `mean`, not `final`. Minimum observed `run_prob` is `0.1508`, so the threshold
was `0.15` — the **default** — not `0.0`. (`155822` vs `162248` is the units fix landing: same 80
trades, same $138.42 fees, same +2.0¢ gross, `2.0*100−138.42 = +$61.58` before, `2.00−138.42 = −$136.42` after.)

The real command is:

```bash
python -m backtesting.mmoe_backtest --use-traj-for-side --min-abs-traj 0.08 \
  --min-run-length 2 --hold-seconds 240 --threshold 0.15 --traj-aggregator mean
```

The command printed in `backtesting.md` is the one that produced the **old 207-trade `+$37,409` run**,
left in place when the corrected table was written beneath it.

### 7e. Reproduced exactly

At `85d8a66` (83-feature) with the recovered config:

```
Trades:          80
Win rate:        30.0%
Total gross PnL: $+2.00
Total fees:      $138.42
Total net PnL:   $-136.42
```

**Every digit.** The measuring apparatus is trustworthy. The documentation was not.

### 7f. The true pre-rebase baseline

At branch HEAD `25587d9` (58-feature model), same recovered config:

| | trades | win rate | gross | fees | net |
|---|---|---|---|---|---|
| documented (83-feat, `85d8a66`) | 80 | 30.0% | +$2.00 | $138.42 | −$136.42 |
| **true pre-rebase (58-feat, `25587d9`)** | **135** | **25.2%** | **−$47.00** | **$238.73** | **−$285.73** |

**The `−$136.42` in `backtesting.md` and `CLAUDE.md` was already stale before Level 1 began.** The
branch absorbed a different model and feature set in its own merge commit and nobody re-ran the
backtest. This is a *fifth* reason the headline number was wrong, alongside the four defects — and
it is exactly the failure mode the spec warns about, one layer deeper than the spec looked.

Artifacts saved (in `$CLAUDE_JOB_DIR/tmp`, not the repo):
`baseline_pre_rebase_25587d9.txt`, `baseline_85d8a66_83feat.txt`, `repro_83feat_correctcfg.txt`,
`baseline_HEAD_correctcfg.txt`, `run_{154649,155822,162248,162417,170339}.csv`.

**Phase 5 must compare against −$285.73 / 135 trades, not −$136.42 / 80.**

---

## Item 8 — Working tree cleanliness

```
 M .DS_Store
?? data/backups/
?? docs/ASSESSMENT_2026-08-04.html
?? docs/ASSESSMENT_2026-08-04.pdf
?? scratch/
```

Matches the spec plus two files it did not predict (the assessment PDF and its HTML source). Nothing
else is dirty. `.DS_Store` is *already* in `.gitignore:26` — it shows as modified because it is
tracked, so Phase 0's "add `.DS_Store` to `.gitignore`" is a no-op; the correct action is
`git rm --cached .DS_Store`.

The verification worktree I created for item 7 has been removed; `git worktree list` shows only the
main checkout.

---

## Spec is wrong about

1. **The primary working directory.** The spec and the session config point at
   `~/Documents/nba_live_trader_1/nba-live-trader`, which contains nothing but an empty `.claude/`.
   The live repo is `~/projects/nba_live_trader_1/nba-live-trader` (moved off iCloud). All work below
   assumes the `projects` path.

2. **The four defects.** "Missing `P×(1−P)`" was never a backtest defect — it was a `CLAUDE.md` typo.
   The real fourth defect is **TP over-crediting**. This invalidates Phase 4c negative test #2 as
   written: dropping `P×(1−P)` tests a doc bug, not a code fix that this branch introduced. (It is
   still a worthwhile assertion, just not one of the four.)

3. **PR #52 is already in the branch** by ancestry. The whole content-check procedure in item 2 is
   unnecessary, and "even if the branch lacks #52's code" does not apply.

4. **PR #54 fixed two of the same four defects.** The spec frames the fee conflict as
   "yours charges taker, his doesn't." The deeper issue is that main *also* independently fixed
   cents-vs-dollars, so the conflict spans four regions of `mmoe_backtest.py`, not one.

5. **`_compute_maker_fees` is not the only function to delete.** Main also has `kalshi_fee`, which is
   functionally identical to the branch's `_fee_one_leg` modulo argument order. Deleting only
   `_compute_maker_fees` leaves `kalshi_fee` as dead code. The grep in Phase 2a would pass while
   leaving a duplicate fee implementation in the file — precisely the "one formula, two callers"
   violation `feature-engineering.md` exists to prevent.

6. **Phase 3a's location is wrong, and its stop-condition cannot fire.** `_join_pregame` /
   `_add_derived_features` are **not** in `_load_data()` — they are in `run_backtest()` at
   `mmoe_backtest.py:570–571`, already in the correct order, with a comment explaining why. The
   filter belongs at line 572, between `_add_derived_features` and `_select_home_best_contract`,
   matching `build_dataloaders` (`dataset.py:920–929`). Measured impact on the cache:

   ```
   Traded-regime filter: 14239 → 13737 rows (96.5% kept).
   Dropped 60 overtime (period>4), 442 blowout (|score_diff|>30), 0 incomplete (NULL pace)
   ```

   **3.5%, not "near-total"** — and zero NULL-pace rows, so the ordering hazard the spec warns
   about does not exist on this cache.

7. **Matt's "stale-timestamp warning" is not in the backtest.** It is in
   `models/mmoe/dataset.py::_join_ticks_to_possessions` (the `>20%` `logger.warning`). That is a
   different file from the one Phase 3b implies, though still outside the Level 2 region. See
   *spec omits #3* — I do not think it should simply be deleted.

8. **`min_abs_traj` is not the key name.** `live-trader/config/trading.yaml:45` reads
   `min_abs_traj_entry: 0.12`. A comment anchored on `min_abs_traj:` would attach to nothing.

9. **Test counts.** Six Python test files totalling **1,009** lines, not "7 files / ~1,280 lines".

10. **The Go test command is wrong.** The module is at `live-trader/go/`, not `live-trader/`.
    `cd live-trader && go test ./...` fails with "directory prefix . does not contain main module".

11. **`event_triggers` has no remote branch.** The spec's Phase 6e treats it as one of ~20 stale
    remotes; it is local-only. There are 21 remote heads, and pruning them is unaffected.

12. **Trade-count soft tripwire.** "Below ~40" was calibrated against 80. The real pre-rebase count
    is 135, so the equivalent floor is roughly 65–70.

---

## Spec omits

1. **`gh` is not installed.** Phase 6d (open PR, request review, merge) cannot execute. Options:
   `brew install gh` and authenticate; use the GitHub REST API with a token; or open the PR in a
   browser and have Wynn merge. This needs a decision before Phase 6.

2. **The fee formula is wrong in *both* implementations, and main's docs enshrine the bug.**
   The spec predicted a float-ceil trap at $0.20. It is real, and it breaks **three of five** taker
   rows, not one:

   | price | Kalshi published (taker) | both implementations | raw float before `ceil` |
   |---|---|---|---|
   | $0.10 | $0.63 | **$0.64** | `63.00000000000002` |
   | $0.20 | $1.12 | **$1.13** | `112.00000000000003` |
   | $0.50 | $1.75 | **$1.76** | `175.00000000000003` |
   | $0.85 | $0.90 | $0.90 ✓ | `89.25000000000001` |
   | $0.90 | $0.63 | $0.63 ✓ | `62.999999999999986` |

   Maker is wrong at $0.20 (`$0.29` vs `$0.28`). Consequences the spec did not anticipate:

   - Main's `.claude/CLAUDE.md` now documents **"Taker rate 0.07 → $1.76 per 100 @ 50¢"**. That is
     the buggy output written down as ground truth. It must not survive the merge.
   - The branch's own break-even arithmetic (55.7%, loss = `−$3.00 − $2.19` where
     `$2.19 = $0.44 + $1.75`) uses the *correct* $1.75 — so code and docs already disagree on the
     branch too.
   - Every fee total measured so far, including the $138.42 and the new $238.73, is inflated by up
     to a cent per leg. Fixing this **will move the Phase 5 baseline**, which is an argument for
     fixing it before the Phase 5 run rather than after.

3. **There is no Go fee implementation to check for parity.** Main's `CLAUDE.md` claims
   "Implemented once per language: `mmoe_backtest.py::kalshi_fee` and
   `live-trader/go/orders.go::kalshiFee`". **`kalshiFee` does not exist** — `orders.go` mentions the
   rates only in a header comment, and `calcNetPnL` explicitly "returns gross P&L with no fee
   deduction." Phase 4b's Go-parity check has no subject. The real finding is a false cross-reference
   in main's docs. The only other implementation is `kalshiMakerFee` (JavaScript, embedded in
   `live-trader/inference/dashboard.py:330`).

4. **Adding the traded-regime filter without removing the existing gate makes the backtest
   *stricter* than training, not aligned with it.** `mmoe_backtest.py:367` currently skips rows on
   the stored `is_blowout` / `is_garbage_time` columns. `feature-engineering.md:71–74` says
   explicitly:

   > Do **not** filter on the stored `is_garbage_time` column — it uses a 20-pt margin vs the gate's
   > 30 and discards 28,627 rows the system would really trade.

   So Phase 3a as written leaves both filters stacked: the 30-pt regime filter *and* the 20-pt column
   filter. To actually match training, line 367 should be removed at the same time. **This is a
   behavioural judgement call, so I am flagging rather than deciding** — but note the spec's stated
   rationale for 3a ("the backtest currently scores the model on all of them") is only true for
   overtime; blowouts are already excluded, more aggressively than training excludes them.

5. **PyYAML is not installed and is not in `requirements.txt`.** `_blowout_margin_pts()` therefore
   always fails its config read and falls back to the hardcoded 30:

   ```
   WARNING  Could not read agent.blowout_margin_pts from .../trading.yaml
            (No module named 'yaml') — falling back to 30.
            Training and the live gate may now disagree.
   ```

   `trading.yaml:77` happens to say `30`, so there is no behavioural difference **today** — but the
   filter silently ignores the config file it claims to read, and would diverge the moment anyone
   tunes that value. The warning fired on every run in this recon and presumably on every training
   run too.

6. **`pytest` is not installed** — not in the venv, not on `PATH`. Phase 4a and a definition-of-done
   checkbox cannot run until it is. `requirements.txt` has no test dependencies at all.

7. **Two Go tests already fail, for an environment reason.** 22 pass, 2 fail:
   `TestKalshiLiveAuth` and `TestKalshiLiveWebSocketAuth` both read
   `KALSHI_PEM_PATH=/Users/wynnsheridan/Documents/.../live-basketball.txt` from `.env` — the dead
   pre-move path. **The file does not exist at either location**, and `~/.kalshi/` does not exist
   either. These cannot pass on this machine without the key file, so "go test ./... pass" is
   unachievable as a gate. (Separately: `CLAUDE.md` requires the PEM to live *outside* the repo;
   this path points inside it.)

8. **16 backtest result CSVs are still tracked at branch HEAD.** `613c090` untracked 10 Aug-3 files
   and added `backtesting/results/*.csv` to `.gitignore`, but `.gitignore` does not untrack existing
   files — 11 from June and 4 from May/June remain, and 4 of them (523 lines) are *added* by this
   branch relative to the merge base. They will land in the PR unless removed. Given the commit
   message's own reasoning ("numbers that look authoritative and are not"), they should go.

9. **`_build_feature_dict`'s docstring is a fossil.** `mmoe_backtest.py:260–261` still says
   "Assemble 83-dim feature dict… Physics (58) + Pregame (11) + Market (14)". The code is correct
   (it iterates `PHYSICS_COLS + PREGAME_COLS`), but the docstring describes the pre-#51 world. It is
   incidental evidence for the item-7 diagnosis and should be corrected while the file is open.

10. **Rebase will drop the branch's merge commit.** `5f8b723` merged `ef4c21f` into the branch;
    since `ef4c21f` is already an ancestor of `origin/main`, `git rebase` will flatten it and replay
    9 commits. Expected, but the spec's post-rebase diff check should anticipate the commit count
    changing 10 → 9.

---

## Riskiest step, and how I would resequence

**Riskiest: Phase 2a, the `mmoe_backtest.py` conflict resolution.** Not because it is hard to make
*a* choice, but because both sides fixed defects 3 and 4 and the file has four independent regions in
tension (`kalshi_fee`/`_fee_one_leg`, `_compute_maker_fees`/`_compute_fees`, the inline `/100` vs
`pnl_dollars`, and the call site at `:472`). A resolution that keeps one of Matt's halves and one of
mine produces code that runs, passes tests, and silently double-converts or mis-rates a leg. The
`.claude/CLAUDE.md` conflict is second-riskiest for the same reason: main's version documents the
float-ceil bug as correct.

**I recommend `merge`, not `rebase`, and I would like this decided before Phase 1.** Reasons:

- `merge-tree` shows the *net* conflict is two files. A rebase replays 9 commits, several of which
  touch the same fee region in sequence (`92ebc6c` adds `_compute_fees`, `03940a9` amends it,
  `252e9ed` changes the call site), so the same semantic conflict must be resolved up to three times,
  each time against a partially-built version of my own change.
- The branch already contains a merge commit, so linear history is not preserved either way.
- One conflict resolution, in one reviewable commit, is much easier for Matt to review than nine
  replays — and reviewability matters here because the resolution is a real design decision about
  whose fee model wins.
- Phase −1 gave us an immutable `backup/exit-window-25587d9`, so the usual "rebase is safe because we
  can redo it" argument is satisfied either way.

**Resequencing I would propose:**

1. **Fix the fee float-ceil bug during Phase 2, not Phase 4.** As written, Phase 5 measures a
   baseline, then Phase 4b's assertions force a fee-code change that invalidates it. Fold the
   `Decimal`/rounding fix into the conflict resolution so the baseline is measured **once**, against
   fee code that already matches Kalshi's published table. This is the same argument the spec itself
   makes for bundling the traded-regime filter.
2. **Resolve item 4 above (the stacked garbage-time filter) before Phase 3a**, since it changes what
   the new baseline means.
3. **Install `pytest` and `pyyaml`, and add them to `requirements.txt`, before Phase 4.** Small, but
   Phase 4 is entirely blocked otherwise.

---

## What I believe will break

1. **Phase 4b will fail on first run** — on the $0.10, $0.20 and $0.50 taker rows and the $0.20 maker
   row, in both the branch's `_fee_one_leg` and main's `kalshi_fee`. This is the expected, correct
   outcome; the test is right and the code is wrong.
2. **The round-trip assertions will also fail** until that is fixed: the spec's loss case
   (`$0.44 + $1.75 = $2.19`) computes as `$0.44 + $1.76 = $2.20` today, so break-even is 55.72%, not
   55.7% — a difference too small to matter for strategy but enough to fail an exact assertion.
3. **Phase 5's number will not resemble −$136.42 and should not be expected to.** Starting point is
   −$285.73. The 30s tolerance and the regime filter will move it further, and the fee fix will move
   it *up* slightly (fees are currently overstated). I expect a final figure in the −$200 to −$300
   range on 110–135 trades, still significantly negative.
4. **Phase 4a cannot pass as specified** — 2 of 24 Go tests fail on a missing PEM that does not exist
   on this machine, and `pytest` is not installed.
5. **Phase 4c negative test #2 will not test what the spec thinks.** Removing `P×(1−P)` tests a
   documentation typo, not one of the four defects. The genuine fourth negative test is
   **TP over-crediting**: force `exit_price = sim.exit_price` on `take_profit` and assert that TP
   exits book above the resting limit.
6. **Phase 6b's premise is right but understated.** The #52 comment's "39 legitimate trades made $500
   at 38%" was filtered from a biased run, but it was also computed against the 83-feature model. Two
   independent reasons it is not comparable to the new number.
7. **Nothing will open a MotherDuck connection.** All four backtest runs logged
   "Loading from local parquet cache…" and completed in ~90s. Phase 4e is satisfiable as long as all
   three parquet files stay present.

---

## Questions requiring a decision before Phase 1

1. **Merge or rebase?** I recommend merge, for the reasons above.
2. **Fix the fee float-ceil bug inside Level 1?** I recommend yes, during Phase 2, so the baseline is
   measured once. It is arguably scope creep, but Phase 4b already mandates the fix — this is only a
   question of ordering.
3. **Remove the `is_blowout` / `is_garbage_time` skip at `mmoe_backtest.py:367` when adding the
   traded-regime filter?** Required for genuine train/backtest alignment; it is a behaviour change,
   so I will not do it unprompted.
4. **How should the PR be opened, given no `gh`?**
5. **Delete the 16 tracked result CSVs in this PR, or a separate one?**
6. **Matt's `>20%` stale-market warning: delete, or re-comment?** Its *attribution* to the
   un-backfilled `wall_clock_ts` bug is now false, but the guard itself is a generic
   "too many possessions lack fresh market data" check that can still fire for real reasons. I lean
   toward keeping the guard and rewriting the comment, rather than deleting as Phase 3b directs.
