from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from tools.physionet2019.common import (
    DEFAULT_OUT,
    ConversionError,
    check_import,
    default_source_root,
    find_sets,
    read_psv,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="*", type=Path, default=[DEFAULT_OUT])
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--no-source", action="store_true")
    args = parser.parse_args(argv)
    source_root = None if args.no_source else args.source_root or default_source_root()
    files = []
    for path in args.paths:
        if path.is_dir():
            files.extend(
                sorted(p for p in path.glob("*.json") if not p.name.startswith("_"))
            )
        else:
            files.append(path)
    sets = find_sets(source_root) if source_root else {}
    if source_root is not None and len(sets) != 2:
        # One clear message instead of one FAIL line per file.
        print(
            f"training_setA/ and training_setB/ not found under {source_root} "
            "(fix --source-root, set SEWS_AI_REPO_ROOT, or use --no-source)"
        )
        return 2
    now = datetime.now(UTC)
    failed = 0
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            source = None
            if source_root is not None:
                provenance = data.get("provenance") or {}
                set_dir = sets.get(provenance.get("source_set", ""))
                if set_dir is None:
                    raise ConversionError("source_missing", "source set not found")
                source = read_psv(
                    set_dir / f"{provenance.get('source_record', '')}.psv",
                    provenance["source_set"],
                )
            problems = check_import(data, now, source)
        except ConversionError as error:
            problems = [error.message]
        except (OSError, ValueError, AttributeError, TypeError):
            problems = ["unreadable or not JSON"]
        if problems:
            failed += 1
            print(f"FAIL {path.name}: " + "; ".join(problems[:5]))
    print(f"valid {len(files) - failed}/{len(files)}")
    return 0 if files and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
