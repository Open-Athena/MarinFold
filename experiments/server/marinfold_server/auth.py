"""Bearer-token authentication helpers."""

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from marinfold_server.config import Settings, get_settings


def _extract_bearer_token(authorization: str | None) -> str | None:
    if authorization is None:
        return None
    scheme, sep, token = authorization.partition(" ")
    if not sep or scheme.lower() != "bearer" or not token:
        return None
    return token


def require_bearer_token(
    authorization: Annotated[str | None, Header()] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,
) -> None:
    """Require ``Authorization: Bearer <token>`` when a server token is set."""

    expected = settings.token if settings is not None else None
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MARINFOLD_SERVER_TOKEN is not configured",
        )
    supplied = _extract_bearer_token(authorization)
    if supplied is None or not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
