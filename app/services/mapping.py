"""Frozen AI results projected onto the dashboard, not clinical bands."""

from datetime import datetime

from app.schemas.dashboard import Prediction


def prediction(row, observed_at: str) -> Prediction:
    import json

    result = json.loads(row["result"])
    score = result["risk_score"]
    return Prediction(
        risk_score=score,
        risk_level="unavailable"
        if score is None
        else "high"
        if result["alert"]
        else "low",
        predicted_at=None if score is None else datetime.fromisoformat(observed_at),
        model_version=row["model_version"],
        threshold_version=row["threshold_version"],
        prediction_id=row["id"],
        unavailable_reason="insufficient_history" if score is None else None,
    )
