"""Durable alert snapshots, acknowledgement, and idempotent care records."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import AwareDatetime

from app.errors import ServiceError
from app.routers.deps import authorize
from app.schemas.dashboard import AlertEvent, CareRecord, Listing, RecordDraft
from app.services import query as service

router = APIRouter(dependencies=[Depends(authorize)])


@router.get("/alerts", response_model=Listing[AlertEvent])
def alerts(
    request: Request,
    query: str = Query(default="", max_length=128),
    assignee: str = Query(default="", max_length=128),
    patient_id: str | None = None,
    status: Literal["unacknowledged", "acknowledged"] | None = None,
    from_time: Annotated[AwareDatetime | None, Query(alias="from")] = None,
    to_time: Annotated[AwareDatetime | None, Query(alias="to")] = None,
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=100, ge=1, le=100),
):
    if from_time and to_time and from_time >= to_time:
        raise ServiceError(422, "invalid_time_range", "from must precede to")
    return service.list_alerts(
        request.app.state.store,
        {
            "query": query,
            "assignee": assignee,
            "patient_id": patient_id,
            "status": status,
            "from": from_time.isoformat() if from_time else None,
            "to": to_time.isoformat() if to_time else None,
        },
        cursor,
        limit,
    )


@router.post("/alerts/{alert_id}/acknowledgement", response_model=AlertEvent)
def acknowledgement(alert_id: str, request: Request):
    return service.acknowledge(
        request.app.state.store, alert_id, request.app.state.settings.actor
    )


@router.post("/alerts/{alert_id}/records", response_model=CareRecord)
def record(
    alert_id: str,
    draft: RecordDraft,
    request: Request,
    key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=128)],
):
    return service.save_record(
        request.app.state.store, alert_id, request.app.state.settings.actor, key, draft
    )
