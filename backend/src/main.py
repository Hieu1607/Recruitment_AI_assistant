from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.v1.api import api_router
from src.core.config import settings

app = FastAPI(
    title="Recruitment AI Assistant API",
    version="1.0.0",
    openapi_version="3.0.3",
)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin).rstrip("/") for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/")
def read_root():
    return {"message": "Welcome to Recruitment AI Assistant API"}


app.include_router(api_router, prefix="/api/v1")
