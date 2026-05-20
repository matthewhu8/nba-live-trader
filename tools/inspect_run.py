#!/usr/bin/env python3
"""
inspect_run.py — print a human-readable summary of a single Run.

Usage:
    python tools/inspect_run.py <run_id_or_path>

Examples:
    python tools/inspect_run.py 20260509-214201-4d3091
    python tools/inspect_run.py live-trader/go/logs/runs/2026-05-09/20260509-214201-4d3091

Reads manifest.json + live-trader.jsonl + (optional) inference.jsonl from the
run dir and prints a structured summary covering: header, top-line stats,
per-game breakdown, gate blocker distribution, and anomaly counts.

This is just a friendlier face on the JSONL — every number printed here is
derivable from the raw files via jq. The script exists so you can answer
"how did this run go" with one command instead of three or four queries.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_BASES = [
    PROJECT_ROOT / "live-trader" / "go" / "logs" / "runs",
    PROJECT_ROOT / "logs" / "runs",  # fallback if logs/ live at project root
]


# ── Resolution ─────────────────────────────────────────────────────────────────

def resolve_run_dir(arg: str) -> Path:
    """Accept either a full path or a run_id; return the run directory."""
    p = Path(arg)
    if p.is_dir():
        return p
    # Treat as run_id — search known bases.
    for base in RUN_BASES:
        if not base.is_dir():
            continue
        for date_dir in base.iterdir():
            cand = date_dir / arg
            if cand.is_dir():
                return cand
    raise SystemExit(f"could not find run {arg!r} under {[str(b) for b in RUN_BASES]}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file. Skip malformed lines silently — don't fail a summary
    because one record is bad."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def fmt_money(v: float) -> str:
    sign = "+" if v >= 0 else "-"
    return f"{sign}${abs(v):.2f}"


def hr(char: str = "─", width: int = 70) -> str:
    return char * width


# ── Sections ───────────────────────────────────────────────────────────────────

def print_header(manifest: dict[str, Any]) -> None:
    print(hr("═"))
    print(f"  Run {manifest.get('run_id', '?')}")
    print(hr("═"))
    started   = manifest.get("started_at", "?")
    ended     = manifest.get("ended_at", "(still running)")
    paper     = manifest.get("paper_mode", "?")
    git_sha   = manifest.get("git_sha", "?")
    pid       = manifest.get("pid", "?")
    end_reason = manifest.get("end_reason", "—")
    print(f"  started    {started}")
    print(f"  ended      {ended}    ({end_reason})")
    print(f"  paper_mode {paper}    git_sha {git_sha[:12]}    pid {pid}")
    env_present = manifest.get("env_present") or {}
    if env_present:
        envs = ", ".join(k for k, v in env_present.items() if v) or "(none)"
        print(f"  env_set    {envs}")


def print_top_summary(manifest: dict[str, Any]) -> None:
    s = manifest.get("summary") or {}
    if not s:
        print("\n  (no summary in manifest — run may still be active)")
        return
    print()
    print(hr())
    print("  Top-line summary")
    print(hr())
    rows = [
        ("duration",         f"{s.get('duration_secs', 0)}s"),
        ("games run",        s.get("games_run", 0)),
        ("possessions",      s.get("total_possessions", 0)),
        ("backfill events",  s.get("backfill_seen", 0)),
        ("trades opened",    s.get("trades_opened", 0)),
        ("trades closed",    s.get("trades_closed", 0)),
        ("wins",             s.get("wins", 0)),
        ("net P&L",          fmt_money(s.get("net_pnl_dollars", 0))),
        ("errors",           s.get("errors", 0)),
        ("garbage events",   s.get("garbage_events", 0)),
        ("avg client ms",    f"{s.get('avg_client_ms', 0):.1f}"),
    ]
    for label, value in rows:
        print(f"  {label:<18} {value}")


def print_per_game(records: list[dict[str, Any]]) -> None:
    """Aggregate possession/entry/exit events by game_id."""
    games: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "possessions": 0,
        "backfill":    0,
        "entries":     0,
        "exits":       0,
        "wins":        0,
        "net_pnl":     0.0,
    })

    for r in records:
        gid = r.get("game_id")
        if not gid:
            continue
        ev = r.get("event")
        g = games[gid]
        if ev == "possession":
            if r.get("is_backfill"):
                g["backfill"] += 1
            else:
                g["possessions"] += 1
        elif ev == "entry":
            g["entries"] += 1
        elif ev == "exit":
            g["exits"] += 1
            pnl = float(r.get("net_pnl_dollars", 0))
            g["net_pnl"] += pnl
            if pnl > 0:
                g["wins"] += 1

    if not games:
        return
    print()
    print(hr())
    print("  Per-game breakdown")
    print(hr())
    print(f"  {'game_id':<14} {'poss':>5} {'bf':>4} {'entr':>5} {'exit':>5} {'wins':>5} {'net P&L':>10}")
    for gid, g in sorted(games.items()):
        print(
            f"  {gid:<14} {g['possessions']:>5} {g['backfill']:>4} "
            f"{g['entries']:>5} {g['exits']:>5} {g['wins']:>5} "
            f"{fmt_money(g['net_pnl']):>10}"
        )


def print_gate_blockers(records: list[dict[str, Any]]) -> None:
    """Count which gate blocked entry on each possession that ended in WAIT."""
    blockers: Counter = Counter()
    total = 0
    for r in records:
        if r.get("event") != "possession" or r.get("is_backfill"):
            continue
        total += 1
        gates = r.get("gates") or {}
        first = gates.get("first_blocking") or "(no block — action != WAIT)"
        blockers[first] += 1

    if total == 0:
        return
    print()
    print(hr())
    print(f"  Gate blocker breakdown  ({total} live possessions)")
    print(hr())
    for blocker, count in blockers.most_common():
        pct = 100.0 * count / total
        print(f"  {blocker:<35} {count:>5}   {pct:>5.1f}%")


def print_anomalies(records: list[dict[str, Any]],
                    inference_records: list[dict[str, Any]]) -> None:
    err_count   = sum(1 for r in records if r.get("event") == "error")
    parser_skip = sum(1 for r in inference_records if r.get("event") == "parser_skip")
    inf_errs    = sum(1 for r in inference_records if r.get("event") == "error")
    poss_count  = sum(1 for r in records if r.get("event") == "possession")
    inf_poss    = sum(1 for r in inference_records if r.get("event") == "possession")

    print()
    print(hr())
    print("  Anomalies / cross-stream sanity")
    print(hr())
    print(f"  Go-side errors                 {err_count}")
    print(f"  Inference-side errors          {inf_errs}")
    print(f"  Inference parser_skip events   {parser_skip}")
    print()
    # Go emits one `possession` per inbound NBA event; Python splits the same
    # stream into `possession` (events that closed a possession) + `parser_skip`
    # (mid-possession events like offensive rebounds, non-final FTs). The
    # correct cross-stream invariant is therefore Go.possession == Python.(
    # possession + parser_skip), not Go.possession == Python.possession.
    print(f"  Go possession events           {poss_count}")
    print(f"  Inference possession events    {inf_poss}  (+{parser_skip} parser_skip = {inf_poss + parser_skip} total)")
    if poss_count and (inf_poss or parser_skip):
        total_py = inf_poss + parser_skip
        if poss_count == total_py:
            print(f"  ✓ Go ↔ Python event counts match (possession + parser_skip)")
        else:
            print(f"  ⚠ MISMATCH: Go={poss_count}, Python possession+parser_skip={total_py} "
                  f"(diff = {poss_count - total_py})")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    run_dir = resolve_run_dir(sys.argv[1])
    print(f"  (loading from {run_dir})")
    print()

    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as e:
            print(f"  ⚠ manifest.json is malformed: {e}")

    go_records  = load_jsonl(run_dir / "live-trader.jsonl")
    inf_records = load_jsonl(run_dir / "inference.jsonl")

    print_header(manifest)
    print_top_summary(manifest)
    print_per_game(go_records)
    print_gate_blockers(go_records)
    print_anomalies(go_records, inf_records)

    print()
    print(hr("═"))


if __name__ == "__main__":
    main()
