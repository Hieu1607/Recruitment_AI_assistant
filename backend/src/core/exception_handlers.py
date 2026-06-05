import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from src.services.llm_service import LLMProviderLimitError

logger = logging.getLogger(__name__)


async def llm_provider_limit_exception_handler(
    request: Request,
    exc: LLMProviderLimitError,
) -> JSONResponse:
    logger.error(
        "LLM provider quota or rate limit reached at API boundary. path=%s error=%s",
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=429,
        content={
            "detail": (
                "LLM provider quota or rate limit has been reached. "
                "Please retry later or check provider billing/quota."
            )
        },
    )
