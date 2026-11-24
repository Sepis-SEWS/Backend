from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from tools.physionet2019.common import (
    CONVERTER,
    DATASET_NAME,
    DATASET_URL,
    DATASET_VERSION,
    DEFAULT_OUT,
    DEFAULT_REPLAY_BASE,
    EXPECTED_COUNTS,
    SOURCE_SETS,
    ConversionError,
    assignment,
    build_import,
    check_import,
    default_source_root,
    dumps_import,
    find_sets,
    list_records,
    read_psv,
    write_text,
)


def convert_one(path: Path, source_set: str, out: Path, replay_base, now) -> dict:
    source = read_psv(path, source_set)
    data = build_import(source, replay_base)
    problems = check_import(data, now, source)
    if problems:
        raise ConversionError("validation_failed", "; ".join(problems[:5]))
    target = out / f"{data['encounter_id']}.json"
    write_text(target, dumps_import(data))
    if check_import(json.loads(target.read_text(encoding="utf-8")), now, source):
        target.unlink()
        raise ConversionError("round_trip_failed", "Written file failed re-validation")
    fio2 = [r["FiO2"] for r in source.rows if r["FiO2"] is not None]
    return {
        "sha256": source.sha256,
        "file": target.name,
        "rows": len(source.rows),
        "iculos_start": int(source.iculos[0]),
        "missing_cells": sum(v is None for row in source.rows for v in row.values()),
        # Reported only; source values are never altered.
        "warnings": ["fio2_outside_0_1"] if any(not 0 <= v <= 1 for v in fio2) else [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--replay-base",
        default=DEFAULT_REPLAY_BASE.isoformat(),
        help="ISO 8601 with timezone; keep the default so team DBs match",
    )
    parser.add_argument("--limit", type=int, help="smoke test: first N records per set")
    args = parser.parse_args(argv)

    source_root = args.source_root or default_source_root()
    replay_base = datetime.fromisoformat(args.replay_base)
    if replay_base.utcoffset() is None:
        parser.error("--replay-base must include a timezone")
    sets = find_sets(source_root)
    if set(sets) != set(SOURCE_SETS):
        print(
            f"training_setA/ and training_setB/ not found under {source_root} "
            "(use --source-root or set SEWS_AI_REPO_ROOT)",
            file=sys.stderr,
        )
        return 1
    now = datetime.now(UTC)
    listing = hashlib.sha256()
    counts, records, seen = {}, [], set()
    for source_set in SOURCE_SETS:
        files = list_records(sets[source_set])
        counts[source_set] = len(files)
        for path in files[: args.limit] if args.limit else files:
            entry = {"source_set": source_set, "record": path.stem}
            try:
                if path.stem in seen:
                    raise ConversionError(
                        "duplicate_record", "Record id appears in both sets"
                    )
                seen.add(path.stem)
                assignment(path.stem)  # validates the pNNNNNN id shape
                entry.update(
                    status="converted",
                    **convert_one(path, source_set, args.out, replay_base, now),
                )
            except ConversionError as error:
                entry.update(status="failed", code=error.code, message=error.message)
            except ValueError:
                entry.update(
                    status="failed",
                    code="bad_record_id",
                    message="File name is not a pNNNNNN record id",
                )
            listing.update(
                f"{source_set}/{path.name}:{entry.get('sha256', 'failed')}\n".encode()
            )
            records.append(entry)
            if len(records) % 2000 == 0:
                print(f"... {len(records)} records", file=sys.stderr)

    failed = [r for r in records if r["status"] == "failed"]
    for entry in failed:
        print(
            f"FAIL {entry['source_set']}/{entry['record']}: {entry['code']} - "
            f"{entry['message']}",
            file=sys.stderr,
        )
    full_run = not args.limit
    report = {
        "dataset": {
            "name": DATASET_NAME,
            "version": DATASET_VERSION,
            "url": DATASET_URL,
            "files_found": counts,
            "files_expected": EXPECTED_COUNTS,
            "complete_download": counts == EXPECTED_COUNTS,
            # Same value on every machine with the same source files (full runs only).
            "source_listing_sha256": listing.hexdigest() if full_run else None,
        },
        "converter": CONVERTER,
        "label_use": "SepsisLabel is never read into import files",
        "replay": {
            "base": replay_base.isoformat(),
            "time_basis": "research_replay",
            "note": "No calendar dates in the source. Stored time = base + ICULOS hours; "
            "these are research replay times, not original observation times.",
        },
        "generated_at": now.isoformat(),
        "summary": {
            "records": len(records),
            "converted": len(records) - len(failed),
            "failed": len(failed),
            "failure_codes": dict(sorted(Counter(r["code"] for r in failed).items())),
            "iculos_start_after_1": sum(r.get("iculos_start", 1) > 1 for r in records),
            "warnings": dict(
                Counter(w for r in records for w in r.get("warnings", []))
            ),
        },
        "records": records,
    }
    write_text(
        args.out / "_conversion_report.json",
        json.dumps(report, ensure_ascii=False, indent=1, allow_nan=False) + "\n",
    )
    if counts != EXPECTED_COUNTS:
        print(f"warning: found {counts}, expected {EXPECTED_COUNTS}", file=sys.stderr)
    print(
        json.dumps(
            {
                "conversion_summary": report["summary"],
                "source_listing_sha256": report["dataset"]["source_listing_sha256"],
            }
        )
    )
    return 0 if not failed and records else 1


if __name__ == "__main__":
    raise SystemExit(main())
