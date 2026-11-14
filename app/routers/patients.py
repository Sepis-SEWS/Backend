"""Stored encounter listing and raw observations, never request-time scoring."""

from typing import Literal

from fastapi import APIRouter, Depends, Query, Request

from app.routers.deps import authorize
from app.schemas.dashboard import Detail, Listing, PatientSummary
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
