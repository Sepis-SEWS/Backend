import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.errors import ServiceError
from app.schemas.dashboard import (
    SOFA,
    AlertEvent,
    CareRecord,
    Detail,
    Explanation,
    Listing,
    ManualAlertDraft,
    Observation,
    PatientDetail,
    PatientSummary,
    RecordDraft,
    Summary,
    TimePoint,
)
from app.schemas.import_file import FEATURES, UNITS
from app.services.mapping import prediction
from app.store.sqlite import Store, canonical


def utc_now() -> datetime:
    return datetime.now(UTC)


def not_found():
    raise ServiceError(404, "not_found", "Requested resource was not found")


def _record(row) -> CareRecord:
    return CareRecord(
        id=row["id"],
        alert_id=row["alert_id"],
        author=row["actor"],
        saved_at=row["saved_at"],
        **json.loads(row["body"]),
    )


def _alert(connection, row) -> AlertEvent:
    data = json.loads(row["snapshot"])
    data.update(
        status="acknowledged" if row["acknowledged_at"] else "unacknowledged",
        acknowledged_by=row["acknowledged_by"],
        acknowledged_at=row["acknowledged_at"],
        records=[
            _record(r)
            for r in connection.execute(
                "SELECT * FROM records WHERE alert_id=? ORDER BY saved_at,id",
                (row["id"],),
            )
        ],
    )
    return AlertEvent.model_validate(data)


def _patient(connection, encounter, now: datetime) -> PatientSummary:
    envelope = json.loads(encounter["envelope"])
    latest = connection.execute(
        "SELECT p.*, o.observed_at FROM predictions p JOIN observations o "
        "ON o.encounter_id=p.encounter_id AND o.hour=p.hour "
        "WHERE p.encounter_id=? ORDER BY p.hour DESC LIMIT 1",
        (encounter["id"],),
    ).fetchone()
    if latest is None:
        raise ServiceError(
            503, "store_inconsistent", "Stored encounter has no prediction"
        )
    observed_at = datetime.fromisoformat(latest["observed_at"])
    alerts = connection.execute(
        "SELECT id,acknowledged_at,json_extract(snapshot, '$.occurred_at') AS occurred_at "
        "FROM alerts WHERE encounter_id=?",
        (encounter["id"],),
    ).fetchall()
    latest_alert = max(
        alerts,
        key=lambda row: (datetime.fromisoformat(row["occurred_at"]), row["id"]),
        default=None,
    )
    replay = (envelope.get("provenance") or {}).get("time_basis") == "research_replay"
    delayed = replay or now - observed_at > timedelta(hours=2)
    return PatientSummary(
        id=encounter["id"],
        bed=envelope["bed"],
        assignee=envelope["assignee"],
        icu_id=envelope["icu_id"],
        last_observed_at=observed_at,
        prediction=prediction(latest, latest["observed_at"]),
        data_delayed=delayed,
        delay_reason="replay_historical" if delayed else None,
        latest_alert_id=latest_alert["id"] if latest_alert else None,
        unacknowledged_alert_count=sum(a["acknowledged_at"] is None for a in alerts),
    )


def patient_detail(store: Store, identifier: str) -> Detail:
    with store.connect() as connection:
        now = utc_now()
        encounter = connection.execute(
            "SELECT * FROM encounters WHERE id=?", (identifier,)
        ).fetchone()
        if encounter is None:
            not_found()
        patient = _patient(connection, encounter, now)
        rows = connection.execute(
            "SELECT o.*,p.id AS prediction_id,p.result FROM observations o JOIN predictions p "
            "ON p.encounter_id=o.encounter_id AND p.hour=o.hour "
            "WHERE o.encounter_id=? ORDER BY o.hour DESC LIMIT 24",
            (identifier,),
        ).fetchall()[::-1]
        last = json.loads(rows[-1]["raw"])
        observations = [
            Observation(
                feature_name=name,
                value=last[name],
                unit=UNITS[name],
                observed_at=rows[-1]["observed_at"] if last[name] is not None else None,
            )
            for name in FEATURES
        ]
        series = []
        for row in rows:
            raw, result = json.loads(row["raw"]), json.loads(row["result"])
            score = result["risk_score"]
            series.append(
                TimePoint(
                    observed_at=row["observed_at"],
                    predicted_at=row["observed_at"] if score is not None else None,
                    risk_score=score,
                    hr=raw["HR"],
                    map=raw["MAP"],
                    lactate=raw["Lactate"],
                )
            )
        detail = PatientDetail(
            **patient.model_dump(),
            observations=observations,
            series=series,
            explanation=Explanation(
                model_version=patient.prediction.model_version,
                prediction_id=patient.prediction.prediction_id,
            ),
            sofa=SOFA(),
        )
        return Detail(item=detail, as_of=now)


def _signature(kind: str, filters: dict) -> str:
    return hashlib.sha256(
        canonical({"kind": kind, "filters": filters}).encode()
    ).hexdigest()


def _page(
    items: list,
    key,
    cursor: str | None,
    signature: str,
    limit: int,
    key_size: int = 1,
    descending: bool = False,
):
    if cursor:
        try:
            if len(cursor) > 2048:
                raise ValueError
            value = json.loads(base64.b64decode(cursor, altchars=b"-_", validate=True))
            if set(value) != {"signature", "key"} or value["signature"] != signature:
                raise ValueError
            position = value["key"]
            if not isinstance(position, list) or not all(
                isinstance(v, str) for v in position
            ):
                raise ValueError
            if len(position) != key_size:
                raise ValueError
            items = [
                item
                for item in items
                if (key(item) < position if descending else key(item) > position)
            ]
        except (ValueError, TypeError, KeyError, AttributeError):
            raise ServiceError(
                422, "invalid_cursor", "Cursor is invalid for these filters"
            ) from None
    selected = items[:limit]
    next_cursor = None
    if len(items) > limit:
        next_cursor = base64.urlsafe_b64encode(
            canonical({"signature": signature, "key": key(selected[-1])}).encode()
        ).decode()
    return selected, next_cursor


def list_patients(
    store: Store, filters: dict, cursor: str | None, limit: int
) -> Listing[PatientSummary]:
    with store.connect() as connection:
        now = utc_now()
        patients = [
            _patient(connection, row, now)
            for row in connection.execute("SELECT * FROM encounters ORDER BY id")
        ]
        query = filters.get("query", "").casefold()
        patients = [
            p
            for p in patients
            if (not query or query in p.id.casefold() or query in p.bed.casefold())
            and (not filters.get("assignee") or p.assignee == filters["assignee"])
            and (
                not filters.get("risk_level")
                or p.prediction.risk_level == filters["risk_level"]
            )
            and (
                not filters.get("unacknowledged_only")
                or p.unacknowledged_alert_count > 0
            )
        ]
        unacked = []
        for patient in patients:
            unacked.extend(
                json.loads(r["snapshot"])
                for r in connection.execute(
                    "SELECT snapshot FROM alerts WHERE encounter_id=? AND acknowledged_at IS NULL",
                    (patient.id,),
                )
            )
        summary = Summary(
            scope="filtered" if any(filters.values()) else "global",
            total=len(patients),
            high=sum(p.prediction.risk_level == "high" for p in patients),
            warning=sum(p.prediction.risk_level == "warning" for p in patients),
            unavailable_count=sum(
                p.prediction.risk_level == "unavailable" for p in patients
            ),
            unacknowledged_alert_count=sum(
                p.unacknowledged_alert_count for p in patients
            ),
            unacknowledged_high_count=sum(
                alert["risk_level"] == "high" for alert in unacked
            ),
            oldest_unacknowledged_at=min(
                (datetime.fromisoformat(alert["occurred_at"]) for alert in unacked),
                default=None,
            ),
        )
        selected, next_cursor = _page(
            patients, lambda p: [p.id], cursor, _signature("patients", filters), limit
        )
        return Listing[PatientSummary](
            items=selected, as_of=now, next_cursor=next_cursor, summary=summary
        )


def list_alerts(
    store: Store, filters: dict, cursor: str | None, limit: int
) -> Listing[AlertEvent]:
    with store.connect() as connection:
        now = utc_now()
        alerts = [
            _alert(connection, row)
            for row in connection.execute("SELECT * FROM alerts")
        ]
        query = filters.get("query", "").casefold()
        alerts = [
            a
            for a in alerts
            if (
                not query
                or query in a.patient_id.casefold()
                or query in (a.bed or "").casefold()
            )
            and (not filters.get("assignee") or a.assignee == filters["assignee"])
            and (not filters.get("patient_id") or a.patient_id == filters["patient_id"])
            and (not filters.get("status") or a.status == filters["status"])
            and (
                not filters.get("from")
                or a.occurred_at >= datetime.fromisoformat(filters["from"])
            )
            and (
                not filters.get("to")
                or a.occurred_at < datetime.fromisoformat(filters["to"])
            )
        ]
        alerts.sort(key=lambda a: (a.occurred_at, a.id), reverse=True)

        def key(a):
            return [a.occurred_at.astimezone(UTC).isoformat(), a.id]

        unacked = [a for a in alerts if a.status == "unacknowledged"]
        summary = Summary(
            scope="filtered" if any(filters.values()) else "global",
            total=len(alerts),
            high=sum(a.risk_level == "high" for a in alerts),
            warning=sum(a.risk_level == "warning" for a in alerts),
            unacknowledged_alert_count=len(unacked),
            unacknowledged_high_count=sum(a.risk_level == "high" for a in unacked),
            oldest_unacknowledged_at=min(
                (a.occurred_at for a in unacked), default=None
            ),
            acknowledged_count=len(alerts) - len(unacked),
            recorded_count=sum(bool(a.records) for a in alerts),
        )
        selected, next_cursor = _page(
            alerts,
            key,
            cursor,
            _signature("alerts", filters),
            limit,
            key_size=2,
            descending=True,
        )
        return Listing[AlertEvent](
            items=selected, as_of=now, next_cursor=next_cursor, summary=summary
        )


def acknowledge(store: Store, identifier: str, actor: str) -> AlertEvent:
    with store.connect(write=True) as connection:
        row = connection.execute(
            "SELECT * FROM alerts WHERE id=?", (identifier,)
        ).fetchone()
        if row is None:
            not_found()
        if row["acknowledged_at"] is None:
            connection.execute(
                "UPDATE alerts SET acknowledged_by=?,acknowledged_at=? WHERE id=?",
                (actor, utc_now().isoformat(), identifier),
            )
        return _alert(
            connection,
            connection.execute(
                "SELECT * FROM alerts WHERE id=?", (identifier,)
            ).fetchone(),
        )


def _validate_key(key: str) -> None:
    if not key or not key.strip() or len(key) > 128:
        raise ServiceError(
            422,
            "invalid_idempotency_key",
            "A nonempty Idempotency-Key of at most 128 characters is required",
        )


def _insert_record(
    connection, alert_id: str, actor: str, key: str, body: str, when: str
) -> CareRecord:
    record_id = str(uuid4())
    connection.execute(
        "INSERT INTO records VALUES(?,?,?,?,?,?)",
        (record_id, alert_id, actor, key, body, when),
    )
    return _record(
        connection.execute("SELECT * FROM records WHERE id=?", (record_id,)).fetchone()
    )


def create_manual_alert(
    store: Store, patient_id: str, actor: str, key: str, draft: ManualAlertDraft
) -> AlertEvent:
    _validate_key(key)
    body_hash = hashlib.sha256(
        canonical(draft.model_dump()).encode("utf-8")
    ).hexdigest()
    with store.connect(write=True) as connection:
        encounter = connection.execute(
            "SELECT envelope FROM encounters WHERE id=?", (patient_id,)
        ).fetchone()
        if encounter is None:
            not_found()
        previous = connection.execute(
            "SELECT * FROM alerts WHERE encounter_id=? AND manual_actor=? AND manual_key=?",
            (patient_id, actor, key),
        ).fetchone()
        if previous is not None:
            if previous["manual_body_hash"] != body_hash:
                raise ServiceError(
                    409,
                    "idempotency_conflict",
                    "This key has already been used with a different body",
                )
            return _alert(connection, previous)

        envelope = json.loads(encounter["envelope"])
        alert_id = str(uuid4())
        when = utc_now().isoformat()
        snapshot = {
            "id": alert_id,
            "patient_id": patient_id,
            "bed": envelope["bed"],
            "assignee": envelope["assignee"],
            "source": "manual",
            "message": draft.message,
            "created_by": actor,
            "risk_score": None,
            "risk_level": None,
            "model_version": None,
            "threshold_version": None,
            "occurred_at": when,
        }
        connection.execute(
            "INSERT INTO alerts(id,encounter_id,snapshot,acknowledged_by,acknowledged_at,"
            "manual_actor,manual_key,manual_body_hash) VALUES(?,?,?,?,?,?,?,?)",
            (
                alert_id,
                patient_id,
                canonical(snapshot),
                actor if draft.record else None,
                when if draft.record else None,
                actor,
                key,
                body_hash,
            ),
        )
        if draft.record:
            _insert_record(
                connection,
                alert_id,
                actor,
                key,
                canonical(draft.record.model_dump()),
                when,
            )
        return _alert(
            connection,
            connection.execute(
                "SELECT * FROM alerts WHERE id=?", (alert_id,)
            ).fetchone(),
        )


def save_record(
    store: Store, identifier: str, actor: str, key: str, draft: RecordDraft
) -> CareRecord:
    _validate_key(key)
    body = canonical(draft.model_dump())
    with store.connect(write=True) as connection:
        alert = connection.execute(
            "SELECT * FROM alerts WHERE id=?", (identifier,)
        ).fetchone()
        if alert is None:
            not_found()
        if alert["acknowledged_at"] is None:
            raise ServiceError(
                409,
                "alert_not_acknowledged",
                "Acknowledge the alert before recording care",
            )
        row = connection.execute(
            "SELECT * FROM records WHERE actor=? AND alert_id=? AND key=?",
            (actor, identifier, key),
        ).fetchone()
        if row is not None:
            if row["body"] != body:
                raise ServiceError(
                    409,
                    "idempotency_conflict",
                    "This key has already been used with a different body",
                )
            return _record(row)
        return _insert_record(
            connection, identifier, actor, key, body, utc_now().isoformat()
        )
