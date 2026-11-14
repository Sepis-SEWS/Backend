"""Pure inference: these endpoints never write dashboard state."""

from fastapi import APIRouter, Depends, Request

from app.errors import ServiceError
from app.routers.deps import authorize
from app.schemas.serving import BatchRequest, PredictionRequest, PredictionResponse
from app.services.inference import predict_state

router = APIRouter(dependencies=[Depends(authorize)])


def _predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
    result = predict_state(request.app.state.inference, payload.observations)
    return PredictionResponse(patient_id=payload.patient_id, **result)


@router.post("/predict", response_model=PredictionResponse)
def predict_one(payload: PredictionRequest, request: Request):
    return _predict(request, payload)


@router.post("/predict_batch", response_model=list[PredictionResponse])
def predict_batch(payload: BatchRequest, request: Request):
    identifiers = [patient.patient_id for patient in payload.patients]
    if len(set(identifiers)) != len(identifiers):
        raise ServiceError(422, "duplicate_patient_id", "Duplicate patient_id in batch")
    return [_predict(request, patient) for patient in payload.patients]
