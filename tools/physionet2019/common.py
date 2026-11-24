from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from app.schemas.import_file import FEATURES, ImportFile

DATASET_NAME = "PhysioNet/CinC Challenge 2019 (Early Prediction of Sepsis)"
DATASET_VERSION = "1.0.0"
DATASET_URL = "https://physionet.org/content/challenge-2019/1.0.0/"
SOURCE_SETS = ("training_setA", "training_setB")
EXPECTED_COUNTS = {"training_setA": 20336, "training_setB": 20000}
CONVERTER = "physionet2019-psv-to-import-v2"
BACKEND_ROOT = Path(__file__).resolve().parents[2]


def default_source_root() -> Path:
    ai_root = os.environ.get("SEWS_AI_REPO_ROOT", "").strip()
    base = Path(ai_root).expanduser() if ai_root else BACKEND_ROOT.parent / "AI"
    return (base / "data" / "data-sepsis" / "training").resolve()


DEFAULT_OUT = BACKEND_ROOT / "data" / "import" / "physionet2019"
DEFAULT_REPLAY_BASE = datetime(2026, 1, 1, tzinfo=UTC)
ASSIGNEES = ("research-rn-1", "research-rn-2", "research-rn-3", "research-rn-4")
SOURCE_COLUMNS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess",
    "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
    "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
    "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
    "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets", "Age",
    "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS", "SepsisLabel",
]
LABEL = "SepsisLabel"
ROW_KEYS = frozenset(("ICULOS", *FEATURES))
assert LABEL not in ROW_KEYS


class ConversionError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class SourceRecord:
    record: str
    source_set: str
    sha256: str
    iculos: list[float]
    rows: list[dict[str, float | None]]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_sets(source_root: Path) -> dict[str, Path]:
    found = {}
    for source_set in SOURCE_SETS:
        for candidate in (
            source_root / source_set,
            source_root / "training" / source_set,
        ):
            if candidate.is_dir():
                found[source_set] = candidate
                break
    return found


def list_records(set_dir: Path) -> list[Path]:
    return sorted(p for p in set_dir.glob("p*.psv") if p.is_file())


def _number(text: str, column: str, line: int) -> float | None:
    if text == "NaN":
        return None
    try:
        value = float(text)
    except ValueError:
        raise ConversionError(
            "non_numeric_value", f"Non-numeric value in column {column}, line {line}"
        ) from None
    if not math.isfinite(value):
        raise ConversionError(
            "non_finite_value", f"Infinite value in column {column}, line {line}"
        )
    return value


def read_psv(path: Path, source_set: str) -> SourceRecord:
    try:
        data = path.read_bytes()
        lines = data.decode("utf-8").splitlines()
    except (OSError, UnicodeError):
        raise ConversionError(
            "source_unreadable", "Source file could not be read"
        ) from None
    if not lines or lines[0].split("|") != SOURCE_COLUMNS:
        raise ConversionError(
            "unexpected_header", "Header is not the 2019 41-column layout"
        )
    body = [line for line in lines[1:] if line.strip()]
    if not body:
        raise ConversionError("empty_record", "Record has no observation rows")
    position = {name: i for i, name in enumerate(SOURCE_COLUMNS)}
    iculos, rows = [], []
    for line_number, line in enumerate(body, 2):
        cells = line.split("|")
        if len(cells) != len(SOURCE_COLUMNS):
            raise ConversionError(
                "bad_row_width", f"Wrong column count on line {line_number}"
            )
        hour = _number(cells[position["ICULOS"]], "ICULOS", line_number)
        if hour is None or hour < 1 or hour != int(hour):
            raise ConversionError(
                "bad_iculos",
                f"ICULOS is not an integer hour >= 1 on line {line_number}",
            )
        iculos.append(hour)
        rows.append(
            {
                name: _number(cells[position[name]], name, line_number)
                for name in FEATURES
            }
        )
    for offset, hour in enumerate(iculos):
        if hour != iculos[0] + offset:
            raise ConversionError(
                "iculos_not_contiguous",
                f"ICULOS breaks the hourly grid at row {offset + 1}",
            )
    return SourceRecord(path.stem, source_set, sha256_bytes(data), iculos, rows)


def assignment(record: str) -> tuple[str, str, str]:
    number = int(record.lstrip("p"))
    return f"pn19-{record}", f"BED-{record}", ASSIGNEES[number % len(ASSIGNEES)]


def build_import(source: SourceRecord, replay_base: datetime) -> dict:
    encounter_id, bed, assignee = assignment(source.record)
    anchor = replay_base.astimezone(UTC).isoformat()
    return {
        "encounter_id": encounter_id,
        "patient_id": source.record,
        "bed": bed,
        "assignee": assignee,
        "icu_id": "ICU-A",
        "unit_profile": "physionet2019-source-v1",
        "admission_anchor": anchor,
        "provenance": {
            "source": "physionet2019",
            "dataset_version": DATASET_VERSION,
            "source_set": source.source_set,
            "source_record": source.record,
            "source_sha256": source.sha256,
            "converter": CONVERTER,
            "time_basis": "research_replay",
            "replay_base": anchor,
        },
        "observations": [
            {"ICULOS": hour, **values}
            for hour, values in zip(source.iculos, source.rows)
        ],
    }


def check_import(
    data: dict, now: datetime, source: SourceRecord | None = None
) -> list[str]:
    problems: list[str] = []
    try:
        payload = ImportFile.model_validate(data)
    except ValidationError as error:
        for item in error.errors(
            include_url=False, include_input=False, include_context=False
        ):
            location = ".".join(str(part) for part in item["loc"])
            problems.append(f"schema {location}: {item['msg']}")
        return problems
    for position, row in enumerate(data["observations"], 1):
        if LABEL in row:
            problems.append(
                f"row {position}: SepsisLabel must not be an inference input"
            )
        if set(row) != ROW_KEYS:
            problems.append(
                f"row {position}: keys differ from ICULOS + 16 model features"
            )
            break
    if payload.provenance is None or payload.provenance.time_basis != "research_replay":
        problems.append("provenance.time_basis must be research_replay")
    times = payload.times()
    if times[-1] > now:
        problems.append(
            "replay times extend into the future; choose an earlier replay base"
        )
    first = payload.observations[0].ICULOS
    if times[0] - payload.admission_anchor != timedelta(hours=first):
        problems.append("first row must be stored at admission_anchor + ICULOS hours")
    if source is not None:
        if payload.provenance and payload.provenance.source_sha256 != source.sha256:
            problems.append("provenance sha256 differs from the source file")
        if [row["ICULOS"] for row in data["observations"]] != source.iculos:
            problems.append("ICULOS differs from the source file")
        exported = [{k: row[k] for k in FEATURES} for row in data["observations"]]
        if exported != source.rows:
            problems.append("feature values differ from the source file")
    return problems


def dumps_import(data: dict) -> str:
    head = {k: v for k, v in data.items() if k != "observations"}
    text = json.dumps(head, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    rows = ",\n".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        for row in data["observations"]
    )
    return text[:-1] + ',"observations":[\n' + rows + "\n]}\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
