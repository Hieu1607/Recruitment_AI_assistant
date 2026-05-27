import os
from pathlib import Path
from typing import List, Union
from urllib.parse import urlsplit, urlunsplit

from pydantic import AnyHttpUrl, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


def _load_env_files() -> None:
    backend_dir = Path(__file__).resolve().parents[2]
    repo_root = backend_dir.parent
    candidates = (
        repo_root / ".env",
        repo_root / "env",
        backend_dir / ".env",
    )
    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)


_load_env_files()


def _is_running_in_docker() -> bool:
    return Path("/.dockerenv").exists()


def _normalize_database_url_for_runtime(
    database_url: str,
    *,
    in_docker: bool | None = None,
) -> str:
    if not database_url:
        return database_url

    runtime_in_docker = _is_running_in_docker() if in_docker is None else in_docker
    parts = urlsplit(database_url)
    if runtime_in_docker or parts.hostname != "db":
        return database_url

    auth = ""
    if parts.username:
        auth = parts.username
        if parts.password is not None:
            auth = f"{auth}:{parts.password}"
        auth = f"{auth}@"

    port = f":{parts.port}" if parts.port is not None else ""
    netloc = f"{auth}localhost{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


class Settings(BaseSettings):
    PROJECT_NAME: str = "Recruitment AI Assistant"

    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Database
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "recruitment_db")
    DATABASE_URL: str = _normalize_database_url_for_runtime(
        os.getenv(
            "DATABASE_URL",
            f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:5432/{POSTGRES_DB}",
        )
    )

    # JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super-secret-key")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480

    # LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    LLM_TIMEOUT_SECONDS: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
    GROQ_VISION_MODEL_NAME: str = os.getenv(
        "GROQ_VISION_MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct"
    )

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL_NAME: str = os.getenv("OLLAMA_MODEL_NAME", "llama3.1:8b")
    OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "5m")

    # Voice / realtime interview providers
    VOICE_PROVIDER_MODE: str = os.getenv("VOICE_PROVIDER_MODE", "browser")
    STT_PROVIDER: str = os.getenv("STT_PROVIDER", "browser")
    TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "browser")

    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
    DEEPGRAM_STT_MODEL: str = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
    DEEPGRAM_TTS_MODEL: str = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")
    DEEPGRAM_BASE_URL: str = os.getenv("DEEPGRAM_BASE_URL", "https://api.deepgram.com")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_REALTIME_MODEL: str = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
    OPENAI_TRANSCRIPTION_MODEL: str = os.getenv(
        "OPENAI_TRANSCRIPTION_MODEL",
        "gpt-4o-mini-transcribe",
    )
    OPENAI_TTS_MODEL: str = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")

    HF_OCR_BASE_URL: str = os.getenv(
        "HF_OCR_BASE_URL", "https://hieuailearning-resume-ocr-tesseract.hf.space"
    )
    HF_OCR_POLL_INTERVAL: int = int(os.getenv("HF_OCR_POLL_INTERVAL", "2"))
    HF_OCR_POLL_TIMEOUT: int = int(os.getenv("HF_OCR_POLL_TIMEOUT", "300"))

    # Object storage
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    MINIO_REGION: str = os.getenv("MINIO_REGION", "us-east-1")
    MINIO_RESUME_BUCKET: str = os.getenv("MINIO_RESUME_BUCKET", "resumes")
    MINIO_PRESIGNED_GET_EXPIRY_SECONDS: int = int(
        os.getenv("MINIO_PRESIGNED_GET_EXPIRY_SECONDS", "3600")
    )

    # Google OAuth2
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    GOOGLE_REDIRECT_URI: str = os.getenv(
        "GOOGLE_REDIRECT_URI",
        "http://localhost:8000/api/v1/auth/google/callback",
    )
    FRONTEND_BASE_URL: str = os.getenv("FRONTEND_BASE_URL", "http://localhost:5173")
    OAUTH_STATE_TTL_SECONDS: int = 600  # 10 minutes

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
