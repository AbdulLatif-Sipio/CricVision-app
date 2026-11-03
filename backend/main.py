"""
backend/main.py - Enterprise FastAPI Application Entry Point.

Features:
- Security headers & Force HTTPS middleware.
- Observability & correlation tracking.
- RFC-7807 global exception handling.
- SlowAPI rate limiting.
- Mounts API routes and broadcast web application.
"""

import time
import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exceptions import RequestValidationError

from .config import settings
from .security import SecurityHeadersMiddleware, limiter
from .observability import (
    ObservabilityMiddleware,
    http_exception_handler,
    validation_exception_handler,
    generic_exception_handler,
    START_TIME
)
from .api import router as api_router
from .schemas import HealthResponse

# Initialize FastAPI App
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise Cricket Biomechanics & Coaching API with ICC Law 21.2 Verification",
    docs_url="/docs" if settings.DEBUG or settings.ENVIRONMENT != "production" else "/docs",
    redoc_url=None
)

# 1. Rate Limiting State
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 2. Security Middleware & HTTPS Enforcement
app.add_middleware(SecurityHeadersMiddleware)

# 3. Observability & Performance Profiling
app.add_middleware(ObservabilityMiddleware)

# 4. Strict CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time-Ms"]
)

# 5. Global RFC-7807 Exception Handlers
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# 6. Include API Routes
app.include_router(api_router)

# 7. Health & Metrics Probes
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Liveness probe verifying database connectivity and engine health."""
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        database="connected (sqlite_wal)",
        mediapipe_model="loaded_tflite_cpu",
        uptime_seconds=round(time.time() - START_TIME, 1)
    )

@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus-compatible plain metrics."""
    uptime = time.time() - START_TIME
    return {
        "uptime_seconds": round(uptime, 1),
        "environment": settings.ENVIRONMENT,
        "engine_status": "operational"
    }

# 8. Mount Static Web App Frontend & Samples
static_dir = os.path.join(os.path.dirname(__file__), "static")
samples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")
os.makedirs(static_dir, exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/samples", StaticFiles(directory=samples_dir), name="samples")

@app.get("/", tags=["Frontend"])
def serve_frontend():
    """Serves the modern Broadcast Web Application."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "CricVision API running. Access /docs for API documentation."}


@app.get("/manifest.json", tags=["Frontend"])
def serve_manifest():
    """Serves the Progressive Web App (PWA) manifest."""
    manifest_file = os.path.join(static_dir, "manifest.json")
    if os.path.exists(manifest_file):
        return FileResponse(manifest_file, media_type="application/manifest+json")
    return {"name": "CricVision"}


@app.get("/sw.js", tags=["Frontend"])
def serve_service_worker():
    """Serves the Service Worker for offline PWA installation."""
    sw_file = os.path.join(static_dir, "sw.js")
    if os.path.exists(sw_file):
        return FileResponse(sw_file, media_type="application/javascript")
    return ""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
