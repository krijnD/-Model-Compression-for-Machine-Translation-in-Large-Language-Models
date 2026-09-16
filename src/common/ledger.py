#!/usr/bin/env python
"""Ledger helper: appends one JSON row per run to experiments/ledger.jsonl (atomic).

Schema (AGENTS.md §3): run_id, kind, method, variant, bits, group, languages,
benchmark, num_sentences, model_artifact, calibration_artifact, platform,
device, slurm_job_id, commit_sha, command, env_file, started_at, finished_at,
scores_path, efficiency_path, status.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

LEDGER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "experiments", "ledger.jsonl")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def commit_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", os.path.dirname(LEDGER), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return ""


def append(run_id: str, **fields) -> str:
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    row = {"run_id": run_id}
    row.update({k: v for k, v in fields.items() if v is not None})
    # guaranteed fields
    row.setdefault("commit_sha", commit_sha())
    row.setdefault("env_file", "")
    row.setdefault("started_at", "")
    row.setdefault("finished_at", "")
    row.setdefault("status", "planned")
    with open(LEDGER, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return row["run_id"]


def run_start(run_id: str, **fields) -> dict:
    """Append a started row; returns it for later run_finish."""
    fields.setdefault("started_at", utcnow())
    append(run_id, status="running", **fields)
    row = load(run_id)
    return row[-1]


def run_finish(run_id: str, status: str = "done",
               scores_path: str = "", efficiency_path: str = "",
               finished_at: str | None = None, **fields) -> dict:
    """Rewrite the last row for run_id with completion info (returns row)."""
    rows = load(run_id)
    if not rows:
        return {}
    row = rows[-1]
    row.update(fields)
    row["status"] = status
    row["finished_at"] = finished_at or utcnow()
    if scores_path:
        row["scores_path"] = scores_path
    if efficiency_path:
        row["efficiency_path"] = efficiency_path
    # rewrite whole file minus that row + appended
    all_rows = [r for r in read_all() if not (r.get("run_id") == run_id
                                              and r is not row)]
    all_rows.append(row)
    return row


def load(run_id: str) -> list[dict]:
    return [r for r in read_all() if r.get("run_id") == run_id]


def read_all() -> list[dict]:
    if not os.path.exists(LEDGER):
        return []
    out = []
    with open(LEDGER) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--field", action="append", default=[],
                    help="key=value to set (repeatable)")
    ap.add_argument("--status", default="planned")
    args = ap.parse_args()
    fields = {}
    for kv in args.field:
        k, _, v = kv.partition("=")
        fields[k] = v
    fields.setdefault("status", args.status)
    append(args.run_id, **fields)
    print(f"appended {args.run_id} -> {LEDGER}")


if __name__ == "__main__":
    main()