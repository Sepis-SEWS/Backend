import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from app.config import Settings
from app.errors import ServiceError
from app.schemas.import_file import ImportFile
from app.services.import_run import import_encounter
from app.services.inference import load_bundle
from app.store.sqlite import Store, canonical


def _error(code: str, message: str, index: int | None = None) -> None:
    body = {"error": {"code": code, "message": message}}
    if index is not None:
        body["file_index"] = index
    print(canonical(body), file=sys.stderr)


class _SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("Invalid command arguments")


def _expand(arguments: list[str]) -> list[Path]:
    """A directory expands to its *.json files; names starting with _ are reports."""
    files: list[Path] = []
    for argument in arguments:
        path = Path(argument)
        if path.is_dir():
            files.extend(
                sorted(
                    p
                    for p in path.glob("*.json")
                    if p.is_file() and not p.name.startswith("_")
                )
            )
        else:
            files.append(path)
    return files


def _details(error: ValidationError) -> list[dict]:
    # Location, type and message only: never the offending input values.
    return [
        {
            "loc": ".".join(str(part) for part in item["loc"]),
            "type": item["type"],
            "msg": item["msg"],
        }
        for item in error.errors(
            include_url=False, include_input=False, include_context=False
        )
    ][:20]


def _import_one(path: Path, store: Store, bundle) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeError:
        return {
            "status": "failed",
            "code": "invalid_encoding",
            "message": "File is not UTF-8 text",
        }
    except OSError:
        return {
            "status": "failed",
            "code": "file_unreadable",
            "message": "File could not be read",
        }
    try:
        payload = ImportFile.model_validate_json(text)
    except ValidationError as error:
        return {
            "status": "failed",
            "code": "invalid_import",
            "message": "File does not satisfy the import contract",
            "details": _details(error),
        }
    except ValueError:
        return {
            "status": "failed",
            "code": "invalid_import",
            "message": "File does not satisfy the import contract",
        }
    try:
        summary = import_encounter(payload, store, bundle)
    except ServiceError as error:
        # Only service-authored messages, never runtime exception details.
        return {"status": "failed", "code": error.code, "message": error.message}
    except (
        OSError,
        sqlite3.Error,
        RuntimeError,
        LookupError,
        TypeError,
        ArithmeticError,
        ValueError,
    ):
        return {
            "status": "failed",
            "code": "import_failed",
            "message": "File import failed",
        }
    return {"status": "imported", "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = _SafeParser(description=__doc__)
    parser.add_argument("files", nargs="+", metavar="FILE")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--ai-root", type=Path)
    parser.add_argument("--report", type=Path)
    try:
        args = parser.parse_args(argv)
    except ValueError:
        _error("invalid_arguments", "Command arguments are invalid")
        return 2
    try:
        settings = Settings.from_env()
        root = (
            args.ai_root.expanduser().resolve()
            if args.ai_root is not None
            else settings.ai_root
        )
        bundle = load_bundle(root)
        store = Store(
            args.database.expanduser()
            if args.database is not None
            else settings.database
        )
    except (OSError, ImportError, RuntimeError, ValueError, LookupError):
        _error("model_unavailable", "Import model is unavailable")
        return 1
    started_at = datetime.now(UTC).isoformat()
    files = _expand(args.files)
    results = []
    for index, path in enumerate(files, 1):
        outcome = _import_one(path, store, bundle)
        if outcome["status"] == "imported":
            print(json.dumps(outcome["summary"], ensure_ascii=False, allow_nan=False))
        else:
            _error(outcome["code"], outcome["message"], index)
        results.append({"file_index": index, "file": str(path), **outcome})
    imported = [r for r in results if r["status"] == "imported"]
    failed = [r for r in results if r["status"] == "failed"]
    summary = {
        "files_total": len(results),
        "files_imported": len(imported),
        "files_failed": len(failed),
        "records_added": sum(r["summary"]["records_added"] for r in imported),
        "alerts_added": sum(r["summary"]["alerts_added"] for r in imported),
        "failure_codes": dict(sorted(Counter(r["code"] for r in failed).items())),
    }
    print(canonical({"import_summary": summary}))
    if args.report is not None:
        report = {
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "database": str(store.path),
            "model_version": bundle.model_version,
            "threshold_version": bundle.threshold_version,
            "summary": summary,
            "files": results,
        }
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        except OSError:
            _error("report_unwritable", "Import report could not be written")
            return 1
    return 1 if failed or not results else 0


if __name__ == "__main__":
    raise SystemExit(main())
