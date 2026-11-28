"""One command: PhysioNet 2019 originals -> converted JSON -> checks -> local DB"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import import_cli
from tools.physionet2019 import convert, validate
from tools.physionet2019.common import BACKEND_ROOT, DEFAULT_OUT, default_source_root


def _env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else fallback.resolve()


def _step(number: int, title: str) -> float:
    print(f"\n[{number}/4] {title}", flush=True)
    return time.monotonic()


def _done(started: float) -> str:
    return f"{time.monotonic() - started:.0f}s"


def _db_summary(database: Path) -> dict:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        one = lambda sql: connection.execute(sql).fetchone()[0]
        latest = (
            "SELECT p.result FROM predictions p JOIN "
            "(SELECT encounter_id, MAX(hour) h FROM predictions GROUP BY encounter_id) m "
            "ON p.encounter_id=m.encounter_id AND p.hour=m.h"
        )
        levels = {"high": 0, "low": 0, "unavailable": 0}
        for (result,) in connection.execute(latest):
            value = json.loads(result)
            level = (
                "unavailable"
                if value["risk_score"] is None
                else "high"
                if value["alert"]
                else "low"
            )
            levels[level] += 1
        return {
            "patients": one("SELECT COUNT(*) FROM encounters"),
            "observations": one("SELECT COUNT(*) FROM observations"),
            "alerts": one("SELECT COUNT(*) FROM alerts"),
            "latest_risk_level": levels,
            "replay_marked": one(
                "SELECT COUNT(*) FROM encounters WHERE "
                "json_extract(envelope,'$.provenance.time_basis')='research_replay'"
            ),
            "example_id": one("SELECT MIN(id) FROM encounters"),
        }
    finally:
        connection.close()


class _Progress:

    def __init__(self, handle, total: int, started: float):
        self.handle, self.total, self.started = handle, max(total, 1), started
        self.lines = 0
        self.step = max(self.total // 100, 1)

    def write(self, text: str) -> int:
        self.handle.write(text)
        for line in text.splitlines():
            # Count only per-file lines (success summaries and per-file errors).
            if '"encounter_id"' not in line and '"file_index"' not in line:
                continue
            self.lines += 1
            if self.lines % self.step == 0 and self.lines <= self.total:
                sys.__stdout__.write(
                    f"  ... {self.lines}/{self.total} files ({_done(self.started)})\n"
                )
                sys.__stdout__.flush()
        return len(text)

    def flush(self) -> None:
        self.handle.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--ai-root", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--limit", type=int, help="quick trial: first N records per set"
    )
    parser.add_argument(
        "--skip-convert", action="store_true", help="reuse existing JSON"
    )
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args(argv)

    source_root = (args.source_root or default_source_root()).resolve()
    ai_root = (
        args.ai_root.resolve()
        if args.ai_root
        else _env_path("SEWS_AI_REPO_ROOT", BACKEND_ROOT.parent / "AI")
    )
    database = (
        args.database.resolve()
        if args.database
        else _env_path("SEWS_DATABASE_PATH", BACKEND_ROOT / "data" / "sews_v2_16.sqlite")
    )
    out = args.out.resolve()
    reports = out.parent
    print("PhysioNet 2019 -> SEWS local DB")
    print(f"  source   {source_root}")
    print(f"  output   {out}")
    print(f"  AI root  {ai_root}")
    print(f"  database {database}")

    if not args.skip_convert:
        started = _step(1, "convert .psv -> import JSON")
        options = ["--source-root", str(source_root), "--out", str(out)]
        if args.limit:
            options += ["--limit", str(args.limit)]
        status = convert.main(options)
        report = json.loads((out / "_conversion_report.json").read_text("utf-8"))
        summary = report["summary"]
        print(
            f"  converted {summary['converted']}/{summary['records']}, "
            f"failed {summary['failed']} {summary['failure_codes'] or ''} ({_done(started)})"
        )
        if status != 0:
            print("stopped: conversion failed", file=sys.stderr)
            return 1
    else:
        print("\n[1/4] convert skipped (--skip-convert)")

    if not args.skip_validate:
        started = _step(2, "validate converted JSON against contract and originals")
        status = validate.main([str(out), "--source-root", str(source_root)])
        print(f"  ({_done(started)})")
        if status != 0:
            print("stopped: fix the FAIL lines above before importing", file=sys.stderr)
            return 1
    else:
        print("\n[2/4] validate skipped (--skip-validate)")

    started = _step(3, "import into the local DB (resumable; log in _import_log.jsonl)")
    import_report = reports / "_import_report.json"
    log = reports / "_import_log.jsonl"
    reports.mkdir(parents=True, exist_ok=True)
    import_report.unlink(missing_ok=True)  # never read a stale report
    total = len([p for p in out.glob("*.json") if not p.name.startswith("_")])
    with log.open("w", encoding="utf-8") as handle:
        progress = _Progress(handle, total, started)
        with redirect_stdout(progress), redirect_stderr(progress):
            import_cli.main(
                [
                    str(out),
                    "--database",
                    str(database),
                    "--ai-root",
                    str(ai_root),
                    "--report",
                    str(import_report),
                ]
            )
    if not import_report.exists():
        print(log.read_text(encoding="utf-8").strip(), file=sys.stderr)
        print(
            "stopped: import did not run. Check the AI model files under "
            f"{ai_root} (README 'Required files') or pass --ai-root.",
            file=sys.stderr,
        )
        return 1
    result = json.loads(import_report.read_text(encoding="utf-8"))
    s = result["summary"]
    print(
        f"  files {s['files_imported']}/{s['files_total']} imported, "
        f"{s['files_failed']} failed {s['failure_codes'] or ''}; "
        f"rows added {s['records_added']}, alerts added {s['alerts_added']} "
        f"({_done(started)})"
    )
    for item in [f for f in result["files"] if f["status"] == "failed"][:10]:
        print(f"  FAIL {Path(item['file']).name}: {item['code']} - {item['message']}")

    _step(4, "what the API will show")
    db = _db_summary(database)
    print(
        f"  /patients        total {db['patients']}  latest risk {db['latest_risk_level']}"
    )
    print(
        f"  /patients/{{id}}   e.g. /patients/{db['example_id']}  "
        f"(replay-time marked: {db['replay_marked']}/{db['patients']})"
    )
    print(f"  /alerts          total {db['alerts']}")
    print(f"  observations     {db['observations']} hourly rows")
    print("\nReports:")
    print(f"  {out / '_conversion_report.json'}")
    print(f"  {import_report}")
    print(f"  {log}")
    print(
        f'\nStart the API with the same DB:\n  export SEWS_DATABASE_PATH="{database}"'
    )
    return 0 if s["files_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
