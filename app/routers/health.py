"""Independent model and live storage readiness, without patient data."""

import sqlite3
from datetime import UTC, datetime

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    ready = request.app.state.inference is not None
    store_ready = True
    try:
        with request.app.state.store.connect() as connection:
            for table in (
                "encounters",
                "observations",
                "predictions",
                "alerts",
                "records",
            ):
                connection.execute(f"SELECT 1 FROM {table} LIMIT 0")
    except sqlite3.Error:
        # Keep model readiness independent and never reflect database error text.
        store_ready = False
    return {
        "status": "unavailable" if not store_ready else "ok" if ready else "degraded",
        "model_ready": ready,
        "store_ready": store_ready,
        "checked_at": datetime.now(UTC),
        "model_loaded": ready,
        "model": "lstm_full_pw10",
        "score_calibrated": False,
    }
