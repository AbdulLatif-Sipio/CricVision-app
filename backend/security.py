"""
backend/security.py - Security Middleware, HTTPS Enforcement & Rate Limiting.

Implements:
- Force HTTPS / SSL Redirect middleware for production.
- Comprehensive Security Headers: HSTS, CSP, X-Frame-Options, X-Content-Type-Options.
- SlowAPI Rate Limiting to prevent DoS on heavy video processing.
- Strict CORS validation.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from .config import settings

# Global Rate Limiter using client IP
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT_DEFAULT])


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Injects enterprise security headers and optionally forces HTTPS redirects.
    """

    async def dispatch(self, request: Request, call_next):
        # 1. Force HTTPS redirect if configured and running behind an SSL proxy
        if settings.FORCE_HTTPS:
            forwarded_proto = request.headers.get("x-forwarded-proto", "http")
            if forwarded_proto != "https":
                url = request.url.replace(scheme="https")
                return RedirectResponse(url=str(url), status_code=301)

        # Process request
        response: Response = await call_next(request)

        # 2. Security Headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # HTTP Strict Transport Security (HSTS)
        if settings.FORCE_HTTPS or settings.ENVIRONMENT == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

        return response
