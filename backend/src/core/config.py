import os
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator
from typing import List, Union

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
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:5432/{POSTGRES_DB}"
    )

    # JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super-secret-key")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
