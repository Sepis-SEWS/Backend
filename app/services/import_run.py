"""Atomic, append-only replay of explicitly timestamped source observations."""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.errors import ServiceError
from app.schemas.import_file import ImportFile
from app.services.inference import ModelBundle
from app.store.sqlite import Store, canonical


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _hash(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _conflict() -> ServiceError:
    return ServiceError(
        409, "import_conflict", "Import conflicts with the stored encounter"
    )


def _existing(connection, encounter_id, envelope_hash, prefixes) -> int:
    encounter = connection.execute(
        "SELECT envelope_hash FROM encounters WHERE id=?", (encounter_id,)
    ).fetchone()
    if encounter is None:
        return 0
    if encounter["envelope_hash"] != envelope_hash:
        raise _conflict()
    rows = connection.execute(
        "SELECT hour,prefix_hash FROM observations WHERE encounter_id=? ORDER BY hour",
        (encounter_id,),
    ).fetchall()
    for index, row in enumerate(rows[: len(prefixes)]):
        if row["hour"] != index + 1 or row["prefix_hash"] != prefixes[index]:
            raise _conflict()
    return len(rows)


def _validate_result(result: dict, count: int) -> None:
    invalid = ServiceError(
        503, "invalid_model_response", "Model returned an invalid result"
    )
    if not isinstance(result, dict):
        raise invalid
    if (
        type(result.get("records_received")) is not int
        or result["records_received"] != count
        or type(result.get("minimum_records")) is not int
        or result["minimum_records"] != 24
        or type(result.get("score_threshold")) not in (int, float)
        or result["score_threshold"] != 0.2
        or "calibrated_probability" not in result
        or result["calibrated_probability"] is not None
        or type(result.get("alert")) is not bool
    ):
        raise invalid
    if (
        "calibration_available" in result
        and result["calibration_available"] is not False
    ):
        raise invalid
    score = result.get("risk_score")
    if count < 24:
        if (
            result.get("status") != "insufficient_history"
            or "risk_score" not in result
            or score is not None
            or result["alert"]
        ):
            raise invalid
    elif (
        result.get("status") != "scored"
        or type(score) not in (int, float)
        or not math.isfinite(score)
        or not 0 <= score <= 1
        or result["alert"] != (score >= 0.2)
    ):
        raise invalid
    try:
        canonical(result)
    except (ValueError, TypeError, OverflowError):
        raise invalid from None


def import_encounter(
    payload: ImportFile, store: Store, bundle: ModelBundle, now: datetime | None = None
) -> dict:
    """Score complete causal prefixes before taking the SQLite write lock."""
    times = payload.times()
    checked_at = now if now is not None else datetime.now(UTC)
    if checked_at.tzinfo is None or checked_at.utcoffset() is None:
        raise ServiceError(422, "invalid_time", "Import time must include a timezone")
    if any(when > checked_at for when in times):
        raise ServiceError(
            422, "future_observations", "Future observations are not allowed"
        )
    envelope = {
        "patient_id": payload.patient_id,
        "bed": payload.bed,
        "assignee": payload.assignee,
        "icu_id": payload.icu_id,
        "encounter_id": payload.encounter_id,
        "unit_profile": payload.unit_profile,
        "time_mode": "anchor" if payload.admission_anchor is not None else "rows",
        "anchor": _utc(times[0] - timedelta(hours=1)),
        "model_version": bundle.model_version,
        "threshold_version": bundle.threshold_version,
        "checkpoint_digest": bundle.checkpoint_digest,
        "policy_digest": bundle.policy_digest,
    }
    envelope_hash = _hash(envelope)
    raw = [row.values() for row in payload.observations]
    prefixes = []
    previous = envelope_hash
    for when, values in zip(times, raw):
        previous = _hash([previous, _utc(when), values])
        prefixes.append(previous)
    store.initialize()
    with store.connect() as connection:
        existing = _existing(connection, payload.encounter_id, envelope_hash, prefixes)
    results = {}
    for index in range(existing, len(raw)):
        try:
            # Give the runtime its own dictionaries; it must not mutate stored raw values.
            result = bundle.predict([dict(row) for row in raw[: index + 1]])
            _validate_result(result, index + 1)
            results[index + 1] = (canonical(result), datetime.now(UTC), dict(result))
        except (
            ServiceError,
            ArithmeticError,
            LookupError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            raise ServiceError(
                503, "inference_failed", "Import inference failed"
            ) from None
    added = alerts_added = 0
    with store.connect(write=True) as connection:
        existing = _existing(connection, payload.encounter_id, envelope_hash, prefixes)
        if existing < len(raw):
            connection.execute(
                "INSERT OR IGNORE INTO encounters(id,envelope,envelope_hash) VALUES(?,?,?)",
                (payload.encounter_id, canonical(envelope), envelope_hash),
            )
        imported_at = _utc(datetime.now(UTC))
        for index in range(existing, len(raw)):
            hour = index + 1
            result_json, computed_at, result = results[hour]
            connection.execute(
                "INSERT INTO observations(encounter_id,hour,observed_at,raw,prefix_hash) VALUES(?,?,?,?,?)",
                (
                    payload.encounter_id,
                    hour,
                    _utc(times[index]),
                    canonical(raw[index]),
                    prefixes[index],
                ),
            )
            connection.execute(
                "INSERT INTO predictions(id,encounter_id,hour,computed_at,imported_at,result,model_version,threshold_version) VALUES(?,?,?,?,?,?,?,?)",
                (
                    str(uuid4()),
                    payload.encounter_id,
                    hour,
                    _utc(computed_at),
                    imported_at,
                    result_json,
                    bundle.model_version,
                    bundle.threshold_version,
                ),
            )
            added += 1
            if result["alert"]:
                alert_id = str(uuid4())
                snapshot = {
                    "id": alert_id,
                    "patient_id": payload.encounter_id,
                    "bed": payload.bed,
                    "assignee": payload.assignee,
                    "risk_score": result["risk_score"],
                    "risk_level": "high",
                    "model_version": bundle.model_version,
                    "threshold_version": bundle.threshold_version,
                    "occurred_at": _utc(times[index]),
                    "status": "unacknowledged",
                    "acknowledged_by": None,
                    "acknowledged_at": None,
                    "repeat_count": 0,
                    "records": [],
                }
                connection.execute(
                    "INSERT INTO alerts(id,encounter_id,hour,snapshot) VALUES(?,?,?,?)",
                    (alert_id, payload.encounter_id, hour, canonical(snapshot)),
                )
                alerts_added += 1
    return {
        "encounter_id": payload.encounter_id,
        "records_received": len(raw),
        "records_added": added,
        "alerts_added": alerts_added,
    }
