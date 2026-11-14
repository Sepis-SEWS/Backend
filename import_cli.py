"""Import user-selected deidentified JSON files; never discover clinical data."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from pydantic import ValidationError

from app.config import Settings
from app.errors import ServiceError
from app.schemas.import_file import ImportFile
from app.services.import_run import import_encounter
from app.services.inference import load_bundle
from app.store.sqlite import Store, canonical


def _error(code: str, message: str) -> None:
    print(canonical({"error": {"code": code, "message": message}}), file=sys.stderr)


class _SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("Invalid command arguments")


def main(argv: list[str] | None = None) -> int:
    parser = _SafeParser(description=__doc__)
    parser.add_argument("files", nargs="+", metavar="FILE")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--ai-root", type=Path)
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
    for filename in args.files:
        try:
            payload = ImportFile.model_validate_json(
                Path(filename).read_text(encoding="utf-8")
            )
            summary = import_encounter(payload, store, bundle)
        except (ValidationError, UnicodeError, ValueError):
            _error("invalid_import", "File does not satisfy the import contract")
            return 1
        except ServiceError as error:
            # Only service-authored messages, never runtime exception details.
            _error(error.code, error.message)
            return 1
        except (
            OSError,
            sqlite3.Error,
            RuntimeError,
            LookupError,
            TypeError,
            ArithmeticError,
        ):
            _error("import_failed", "File import failed")
            return 1
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
