"""FastAPI entry point for the MarinFold CoreWeave inference service."""

from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from marinfold_server.auth import require_bearer_token
from marinfold_server.config import Settings, get_settings


class HealthResponse(BaseModel):
    """Health/readiness response payload."""

    ok: bool
    service: str
    environment: str


class AuthResponse(BaseModel):
    """Response for the protected auth smoke-test endpoint."""

    ok: bool
    authenticated: bool


def create_app() -> FastAPI:
    """Create the ASGI application."""

    app = FastAPI(
        title="MarinFold inference server",
        version="0.1.0",
        summary="Thin control-plane API for CoreWeave-hosted MarinFold inference runs.",
    )

    @app.get("/healthz", response_model=HealthResponse)
    def healthz(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
        return HealthResponse(
            ok=True,
            service=settings.service_name,
            environment=settings.environment,
        )

    @app.get("/readyz", response_model=HealthResponse)
    def readyz(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
        return HealthResponse(
            ok=True,
            service=settings.service_name,
            environment=settings.environment,
        )

    @app.get("/v1/auth-check", response_model=AuthResponse)
    def auth_check(_auth: Annotated[None, Depends(require_bearer_token)]) -> AuthResponse:
        return AuthResponse(ok=True, authenticated=True)

    return app


app = create_app()
