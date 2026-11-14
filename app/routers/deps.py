"""A local-development shared-token guard, not clinical user identity."""

import secrets

from fastapi import Request

from app.errors import ServiceError


def authorize(request: Request) -> None:
    token = request.app.state.settings.token
    if token:
        supplied = request.headers.get("Authorization", "")
        if not secrets.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
            raise ServiceError(401, "unauthorized", "A valid bearer token is required")
