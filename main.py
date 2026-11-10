"""SEWS prototype FastAPI server: serves the locked AI-9 LSTM (no retraining).

Set SEWS_AI_REPO_ROOT to the checked-out AI repository, e.g. C:\\AI.
Requires the local locked model, preprocessing stats and policy at that root.
Run: python -m uvicorn main:app --host 127.0.0.1 --port 8000

Research prototype only. Raw risk_score is not calibrated probability.
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str = Field(min_length=1, max_length=128)
    # Complete history up to now, not merely the last 24 rows:
    # age-since-observation features require the earlier observations.
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    ai_root = Path(os.environ.get("SEWS_AI_REPO_ROOT", "")).expanduser()
    if not str(os.environ.get("SEWS_AI_REPO_ROOT", "")).strip():
        raise RuntimeError("Set SEWS_AI_REPO_ROOT to the checked-out AI repository")
    if not (ai_root / "src" / "inference" / "predict_lstm.py").is_file():
        raise RuntimeError(f"AI inference module not found under {ai_root}")
    sys.path.insert(0, str(ai_root.resolve()))
    from src.inference.predict_lstm import load_frozen_model, predict
    model, stats = load_frozen_model(root=ai_root)
    app.state.inference = (model, stats, predict)
    yield
    app.state.inference = None


app = FastAPI(
    title="SEWS Locked LSTM Inference API",
    version="0.1.0",
    description="Research prototype. Risk score is not a calibrated sepsis probability.",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    ready = request.app.state.inference is not None
    return {"status": "ok" if ready else "not_ready", "model_loaded": ready,
            "model": "lstm_full_pw10", "score_calibrated": False}


def _predict_one(request: Request, payload: PredictionRequest) -> PredictionResponse:
    state = request.app.state.inference
    if state is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    model, stats, predict = state
    try:
        result = predict(payload.observations, model, stats)
    except (ValueError, TypeError) as error:
        # No clinical data or patient identifiers in error logging.
        raise HTTPException(status_code=422, detail=str(error)) from error
    return PredictionResponse(patient_id=payload.patient_id, **result)


@app.post("/predict", response_model=PredictionResponse)
def predict_one(payload: PredictionRequest, request: Request) -> PredictionResponse:
    return _predict_one(request, payload)


@app.post("/predict_batch", response_model=list[PredictionResponse])
def predict_batch(payload: BatchRequest, request: Request) -> list[PredictionResponse]:
    identifiers = [patient.patient_id for patient in payload.patients]
    if len(set(identifiers)) != len(identifiers):
        raise HTTPException(status_code=422, detail="Duplicate patient_id in batch")
    return [_predict_one(request, patient) for patient in payload.patients]
