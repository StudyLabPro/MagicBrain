"""Validation of identifiers that end up in filesystem paths.

``model_id`` arrives from the request body or the URL and is concatenated into
``MODEL_STORAGE_PATH / f"{model_id}.npz"``. Without a check, a caller can walk
out of the storage directory (``../../etc/x``) and make the service create,
read or delete files elsewhere. The extension is fixed, so this is not code
execution, but it is still writing to arbitrary locations with the service's
privileges.

The rule is deliberately narrow: identifiers the API itself produces are UUIDs
or run ids like ``train-20260908-185554-ac9708``, and both fit.
"""

from __future__ import annotations

import re

MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class InvalidModelId(ValueError):
    """Raised when a model identifier cannot be used as a file name."""


def validate_model_id(model_id: str) -> str:
    """Return ``model_id`` unchanged or raise :class:`InvalidModelId`."""
    if not isinstance(model_id, str) or not MODEL_ID_PATTERN.match(model_id):
        raise InvalidModelId(
            "model_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63} "
            "(no path separators, no leading dot)"
        )
    if ".." in model_id:
        raise InvalidModelId("model_id must not contain '..'")
    return model_id


def require_model_id(model_id: str) -> str:
    """FastAPI-facing wrapper: invalid identifier becomes HTTP 422."""
    from fastapi import HTTPException, status

    try:
        return validate_model_id(model_id)
    except InvalidModelId as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
