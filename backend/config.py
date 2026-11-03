"""
backend/config.py - Centralized Enterprise Application Configuration.

Implements:
- Strict Pydantic Settings (Pydantic v2).
- Zero secrets on client-side: GEMINI_API_KEY, SECRET_KEY, DATABASE_URL.
- Security flags: FORCE_HTTPS, CORS_ORIGINS, RATE_LIMIT.
"""

import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Identity
    APP_NAME: str = "CricVision & CricTutor AI"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = "production"
    DEBUG: bool = False

    # Server & Port
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Security & Secrets (Server-Side Only - Never Sent to Frontend)
    SECRET_KEY: str = "cricvision-production-secret-key-change-in-env-982347192"
    GEMINI_API_KEY: str = ""
    FORCE_HTTPS: bool = False  # Set to True in production to enforce SSL/TLS redirect

    # CORS Whitelist
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://cricvision.ai",
        "https://cricvision.vercel.app"
    ]

    # Database
    DATABASE_URL: str = "sqlite:///./cricvision.db"

    # Rate Limiting
    RATE_LIMIT_DEFAULT: str = "60/minute"
    RATE_LIMIT_UPLOAD: str = "20/minute"

    # Storage & Upload Limits
    MAX_UPLOAD_SIZE_MB: int = 50
    UPLOAD_TEMP_DIR: str = "./temp_uploads"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
os.makedirs(settings.UPLOAD_TEMP_DIR, exist_ok=True)
