from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import (
    candidates_router,
    interview_questions_router,
    matching_router,
    outreach_router,
    query_router,
    resumes_router,
    shortlists_router,
)
from src.api.errors import register_error_handlers
from src.api.security.hardening import validate_runtime_security
from src.services.observability.audit_logger import audit_log
from src.services.observability.metrics import metrics_registry


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(level=logging.INFO)
    validate_runtime_security()
    yield


app = FastAPI(title="Recruitment AI Assistant API", version="0.1.0", lifespan=lifespan)
register_error_handlers(app)


def _cors_origins() -> list[str]:
    configured = os.getenv("CORS_ALLOW_ORIGINS", "")
    origins = [item.strip() for item in configured.split(",") if item.strip()]
    app_base_url = os.getenv("APP_BASE_URL", "").strip()
    if app_base_url and app_base_url not in origins:
        origins.append(app_base_url)
    if not origins:
        origins = ["http://localhost:5173"]
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Response-Time-Ms"],
)


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    metrics_registry.record_http_request(
        path=request.url.path,
        method=request.method,
        status_code=response.status_code,
        elapsed_ms=elapsed_ms,
    )
    audit_log(
        "http_request",
        {
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "elapsed_ms": elapsed_ms,
        },
    )
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
    return response


health_router = APIRouter(prefix="/health", tags=["health"])


@health_router.get("", response_class=JSONResponse)
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics", response_class=JSONResponse, tags=["health"])
def metrics_snapshot() -> dict:
    return metrics_registry.snapshot()


app.include_router(health_router)
app.include_router(resumes_router)
app.include_router(candidates_router)
app.include_router(matching_router)
app.include_router(query_router)
app.include_router(shortlists_router)
app.include_router(outreach_router)
app.include_router(interview_questions_router)
