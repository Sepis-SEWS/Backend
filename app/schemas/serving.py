"""Existing stateless inference contract."""

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    patient_id: str = Field(min_length=1, max_length=128)
    observations: list[dict[str, float | None]] = Field(min_length=1, max_length=10000)


class PredictionResponse(BaseModel):
    patient_id: str
    status: str
    risk_score: float | None
    alert: bool
    score_threshold: float
    records_received: int
    minimum_records: int
    calibrated_probability: float | None


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patients: list[PredictionRequest] = Field(min_length=1, max_length=32)
