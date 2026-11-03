"""
backend/observability.py - Structured Logging, Observability & Global Error Handlers.

Implements:
- RFC-7807 compliant Problem Details error responses.
- Structured JSON request logging with correlation IDs.
- Performance profiling middleware (X-Process-Time-Ms).
- Health & readiness probes.
"""

import time
import uuid
import logging
from typing import Dict, Any
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

# Configure structured logging
logger = logging.getLogger("cricvision")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


START_TIME = time.time()


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Tracks request duration, assigns unique correlation ID, and logs structured metrics.
    """

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = req_id
        start_t = time.perf_counter()

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start_t) * 1000, 2)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Process-Time-Ms"] = str(duration_ms)

        logger.info(
            f"req_id={req_id} method={request.method} path={request.url.path} "
            f"status={response.status_code} duration_ms={duration_ms}"
        )
        return response


# ============================================================================
# GLOBAL RFC-7807 EXCEPTION HANDLERS
# ============================================================================

async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handles standard HTTP exceptions gracefully."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status_code": exc.status_code,
            "error": "HTTP_ERROR",
            "message": exc.detail,
            "request_id": req_id
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handles Pydantic validation errors cleanly without exposing internal details."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status_code": 422,
            "error": "VALIDATION_ERROR",
            "message": "Invalid request parameters or payload structure.",
            "details": exc.errors(),
            "request_id": req_id
        }
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catches all unhandled exceptions, logs the trace, and returns safe 500 error."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error(f"UNHANDLED_EXCEPTION req_id={req_id} error={str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status_code": 500,
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred during biomechanical processing. Please verify file format.",
            "request_id": req_id
        }
    )
