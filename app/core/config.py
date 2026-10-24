from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os

class Settings(BaseSettings):
    # ── Groq LLM (free tier) ────────────────────────────────────────────────
    # Get your free key at: https://console.groq.com
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")

    # Primary model (works on new personal Groq keys — best quality)
    GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

    # Fallback chain: tries each in order if previous fails
    # Covers both new keys (Llama) and restricted keys (Qwen/Allam)
    GROQ_MODEL_FALLBACKS: list[str] = [
        "llama-3.3-70b-versatile",   # Best — new Groq keys
        "llama-3.1-8b-instant",      # Fast + free on new keys
        "qwen/qwen3.8-27b",          # Works on restricted keys
        "allam-2-7b",                # Last resort fallback
    ]

    # ── Embeddings (free, local, no API key needed) ─────────────────────────
    # BGE-small outperforms MiniLM on retrieval benchmarks, same size (384d)
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384

    # ── Qdrant Vector DB ─────────────────────────────────────────────────────
    # Local path (default — free, no account needed)
    QDRANT_PATH: str = "data/qdrant_db"
    # Cloud override — set these env vars to switch to Qdrant Cloud (free 1GB)
    # Sign up free at: https://cloud.qdrant.io
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL", None)
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)

    # ── Application ──────────────────────────────────────────────────────────
    APP_ENV: str = "development"
    DEBUG: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
