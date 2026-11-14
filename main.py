"""SEWS local research API. Run uvicorn main:app --host 127.0.0.1 --workers 1."""

import sqlite3
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.config import Settings
from app.errors import ServiceError
from app.lifespan import lifespan
from app.routers import alerts, health, patients, serving
from app.services.inference import load_bundle
from app.store.sqlite import Store


def create_app(settings: Settings | None = None, model_loader=load_bundle) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="SEWS Locked LSTM and Dashboard API",
        version="0.2.0",
        description="Deidentified local research prototype. Scores are not calibrated sepsis probabilities.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.store = Store(settings.database)
    app.state.model_loader = model_loader
    app.state.inference = None

    def error_response(request: Request, status: int, code: str, message: str):
        return JSONResponse(
            status_code=status,
            content={
                "code": code,
                "message": message,
                "request_id": request.state.request_id,
            },
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = str(uuid4())
        try:
            response = await call_next(request)
        except (
            ArithmeticError,
            LookupError,
            RuntimeError,
            TypeError,
            ValueError,
            AttributeError,
            OSError,
        ):
            # Do not log exception values: they may contain source patient data.
            response = error_response(
                request, 500, "internal_error", "Request could not be completed"
            )
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, error: ServiceError):
        return error_response(request, error.status, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        return error_response(
            request, 422, "invalid_request", "Request does not match the API contract"
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException):
        return error_response(
            request, error.status_code, "http_error", "Request could not be completed"
        )

    @app.exception_handler(sqlite3.Error)
    async def database_error(request: Request, error: sqlite3.Error):
        return error_response(
            request, 503, "store_unavailable", "Stored data is temporarily unavailable"
        )

    for router in (health.router, serving.router, patients.router, alerts.router):
        app.include_router(router)
    return app


app = create_app()
