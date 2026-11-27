"""Stored encounter listing and raw observations, never request-time scoring."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Query, Request

from app.routers.deps import authorize
from app.schemas.dashboard import (
    AlertEvent,
    Detail,
    Listing,
    ManualAlertDraft,
    PatientSummary,
)
from app.services import query as service

router = APIRouter(dependencies=[Depends(authorize)])


@router.get("/patients", response_model=Listing[PatientSummary])
def patients(
    request: Request,
    query: str = Query(default="", max_length=128),
    risk_level: Literal["high", "warning", "low", "unavailable"] | None = None,
    assignee: str = Query(default="", max_length=128),
    unacknowledged_only: bool = False,
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=100, ge=1, le=100),
):
    return service.list_patients(
        request.app.state.store,
        {
            "query": query,
            "risk_level": risk_level,
            "assignee": assignee,
            "unacknowledged_only": unacknowledged_only,
        },
        cursor,
        limit,
    )


@router.get("/patients/{patient_id}", response_model=Detail)
def patient(patient_id: str, request: Request):
    return service.patient_detail(request.app.state.store, patient_id)


@router.post(
    "/patients/{patient_id}/alerts", response_model=AlertEvent, status_code=201
)
def create_alert(
    patient_id: str,
    draft: ManualAlertDraft,
    request: Request,
    key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=128)],
):
    return service.create_manual_alert(
        request.app.state.store,
        patient_id,
        request.app.state.settings.actor,
        key,
        draft,
    )
