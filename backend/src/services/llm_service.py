import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.core.config import settings

logger = logging.getLogger(__name__)

PROVIDER_LIMIT_ERROR_MARKERS = (
    "429",
    "quota",
    "rate limit",
    "rate_limit_exceeded",
    "tokens per day",
    "requests per day",
    "too many requests",
)
_MAX_RETRY_BACKOFF_MULTIPLIER = 4
_RETRY_AFTER_SECONDS_RE = re.compile(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)


class LLMProviderError(Exception):
    pass


class LLMProviderLimitError(LLMProviderError):
    """Raised when an upstream LLM provider is blocked by quota or rate limiting."""


class LLMConfigurationError(LLMProviderError):
    pass


def _flatten_exception_messages(exc: BaseException) -> str:
    parts: List[str] = []
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).strip()
        if message:
            parts.append(message.lower())
        current = current.__cause__ or current.__context__
    return " | ".join(parts)


def is_provider_limit_error(exc: BaseException) -> bool:
    if isinstance(exc, LLMProviderLimitError):
        return True
    flattened = _flatten_exception_messages(exc)
    return any(marker in flattened for marker in PROVIDER_LIMIT_ERROR_MARKERS)


def _raise_provider_limit_error(
    *,
    provider: str,
    model: str,
    operation: str,
    exc: BaseException,
) -> None:
    logger.error(
        "LLM provider quota or rate limit reached. provider=%s model=%s operation=%s error=%s",
        provider,
        model,
        operation,
        exc,
    )
    raise LLMProviderLimitError(
        f"{provider} {operation} hit quota or rate limit for model {model}: {exc}"
    ) from exc


def _retry_after_seconds_from_error(exc: BaseException) -> Optional[float]:
    match = _RETRY_AFTER_SECONDS_RE.search(str(exc))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _retry_backoff_seconds(attempt: int, exc: Optional[BaseException] = None) -> float:
    multiplier = min(2**attempt, _MAX_RETRY_BACKOFF_MULTIPLIER)
    retry_after = _retry_after_seconds_from_error(exc) if exc is not None else None
    if retry_after is not None:
        return retry_after * multiplier
    return float(multiplier)


class ProviderType(str, Enum):
    SHOPAIKEY = "shopaikey"
    OLLAMA = "ollama"


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None


def _extract_finish_reason(raw: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(raw, dict):
        return None

    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            finish_reason = first_choice.get("finish_reason")
            if finish_reason:
                return str(finish_reason)

    done_reason = raw.get("done_reason")
    if done_reason:
        return str(done_reason)

    finish_reason = raw.get("finish_reason")
    if finish_reason:
        return str(finish_reason)

    return None


def _log_length_finish_reason(
    *,
    provider: str,
    model: str,
    max_tokens: int,
    usage: Optional[Dict[str, Any]],
    raw: Optional[Dict[str, Any]],
) -> None:
    finish_reason = _extract_finish_reason(raw)
    if (finish_reason or "").strip().lower() != "length":
        return

    logger.warning(
        "LLM response stopped because output token limit was reached. provider=%s model=%s finish_reason=%s max_tokens=%s usage=%s",
        provider,
        model,
        finish_reason,
        max_tokens,
        usage,
    )


class _BaseAdapter:
    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        max_retries: int,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        raise NotImplementedError


class _ShopAIKeyAdapter(_BaseAdapter):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        max_retries: int,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        if not api_key:
            raise LLMConfigurationError(
                "SHOPAIKEY_API_KEY is required for ShopAIKey fallback"
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                req = Request(
                    url=f"{self.base_url}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    method="POST",
                )
                with urlopen(req, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                    parsed = json.loads(body)

                choices = parsed.get("choices") or []
                message = (choices[0].get("message") if choices else {}) or {}
                content = (message.get("content") or "").strip()
                usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else None
                _log_length_finish_reason(
                    provider=ProviderType.SHOPAIKEY.value,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    usage=usage,
                    raw=parsed,
                )
                return LLMResponse(
                    text=content,
                    provider=ProviderType.SHOPAIKEY.value,
                    model=self.model,
                    usage=usage,
                    raw=parsed,
                )
            except HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8")
                except Exception:
                    body = ""
                enriched_error = RuntimeError(
                    f"ShopAIKey HTTP {exc.code}: {body or exc.reason or str(exc)}"
                )
                last_error = enriched_error
                if is_provider_limit_error(enriched_error):
                    if attempt < self.max_retries:
                        time.sleep(_retry_backoff_seconds(attempt, enriched_error))
                        continue
                    _raise_provider_limit_error(
                        provider=ProviderType.SHOPAIKEY.value,
                        model=self.model,
                        operation="chat request",
                        exc=enriched_error,
                    )
                if attempt < self.max_retries:
                    time.sleep(_retry_backoff_seconds(attempt, enriched_error))
                    continue
                break
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if is_provider_limit_error(exc):
                    if attempt < self.max_retries:
                        time.sleep(_retry_backoff_seconds(attempt, exc))
                        continue
                    _raise_provider_limit_error(
                        provider=ProviderType.SHOPAIKEY.value,
                        model=self.model,
                        operation="chat request",
                        exc=exc,
                    )
                if attempt < self.max_retries:
                    time.sleep(_retry_backoff_seconds(attempt, exc))
                    continue
                break
            except Exception as exc:
                last_error = exc
                if is_provider_limit_error(exc):
                    if attempt < self.max_retries:
                        time.sleep(_retry_backoff_seconds(attempt, exc))
                        continue
                    _raise_provider_limit_error(
                        provider=ProviderType.SHOPAIKEY.value,
                        model=self.model,
                        operation="chat request",
                        exc=exc,
                    )
                if attempt < self.max_retries:
                    time.sleep(_retry_backoff_seconds(attempt, exc))
                    continue
                break
        raise LLMProviderError(f"ShopAIKey request failed: {last_error}") from last_error


class _OllamaAdapter(_BaseAdapter):
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        max_retries: int,
        keep_alive: str,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive

    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                req = Request(
                    url=f"{self.base_url}/api/chat",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(req, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                    parsed = json.loads(body)

                message = parsed.get("message", {}) or {}
                content = (message.get("content") or "").strip()
                usage = {
                    "prompt_eval_count": parsed.get("prompt_eval_count"),
                    "eval_count": parsed.get("eval_count"),
                }
                _log_length_finish_reason(
                    provider=ProviderType.OLLAMA.value,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    usage=usage,
                    raw=parsed,
                )
                return LLMResponse(
                    text=content,
                    provider=ProviderType.OLLAMA.value,
                    model=self.model,
                    usage=usage,
                    raw=parsed,
                )
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if is_provider_limit_error(exc):
                    if attempt < self.max_retries:
                        time.sleep(_retry_backoff_seconds(attempt, exc))
                        continue
                    _raise_provider_limit_error(
                        provider=ProviderType.OLLAMA.value,
                        model=self.model,
                        operation="chat request",
                        exc=exc,
                    )
                if attempt < self.max_retries:
                    time.sleep(_retry_backoff_seconds(attempt, exc))
                    continue
                break
            except Exception as exc:
                last_error = exc
                if is_provider_limit_error(exc):
                    if attempt < self.max_retries:
                        time.sleep(_retry_backoff_seconds(attempt, exc))
                        continue
                    _raise_provider_limit_error(
                        provider=ProviderType.OLLAMA.value,
                        model=self.model,
                        operation="chat request",
                        exc=exc,
                    )
                break
        raise LLMProviderError(f"Ollama request failed: {last_error}") from last_error


class LLMProvider:
    """Unified LLM provider wrapper for ShopAIKey and Ollama.

    Configuration is loaded from src.core.config.settings.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
        allow_fallback: bool = True,
    ):
        selected_provider = (provider or settings.LLM_PROVIDER or ProviderType.SHOPAIKEY.value).lower()
        if selected_provider not in {member.value for member in ProviderType}:
            logger.warning(
                "Unsupported LLM provider '%s'; defaulting to %s",
                selected_provider,
                ProviderType.SHOPAIKEY.value,
            )
            selected_provider = ProviderType.SHOPAIKEY.value
        self.provider = ProviderType(selected_provider)

        temperature = settings.LLM_TEMPERATURE if temperature is None else temperature
        max_tokens = settings.LLM_MAX_TOKENS if max_tokens is None else max_tokens
        timeout_seconds = (
            settings.LLM_TIMEOUT_SECONDS
            if timeout_seconds is None
            else timeout_seconds
        )
        max_retries = settings.LLM_MAX_RETRIES if max_retries is None else max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.model_name = model_name
        self.allow_fallback = allow_fallback

        if self.provider == ProviderType.SHOPAIKEY:
            selected_model = model_name or settings.SHOPAIKEY_MODEL_NAME
            self.model_name = selected_model
            self._adapter = _ShopAIKeyAdapter(
                api_key=settings.SHOPAIKEY_API_KEY,
                base_url=settings.SHOPAIKEY_BASE_URL,
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
        else:
            selected_model = model_name or settings.OLLAMA_MODEL_NAME
            self.model_name = selected_model
            self._adapter = _OllamaAdapter(
                base_url=settings.OLLAMA_BASE_URL,
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                keep_alive=settings.OLLAMA_KEEP_ALIVE,
            )

    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        return self._adapter.chat(messages)

    async def achat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        return await asyncio.to_thread(self._adapter.chat, messages)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")

        messages: List[Dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})
        return self.chat(messages)

    def generate_with_images(self, prompt: str, images: List[bytes]) -> LLMResponse:
        raise LLMProviderError("Vision is not supported by the configured LLM provider")

    async def agenerate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")

        messages: List[Dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})
        return await self.achat(messages)

    def clone_with_model(
        self,
        *,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        allow_fallback: Optional[bool] = None,
    ) -> "LLMProvider":
        return LLMProvider(
            provider=provider or self.provider.value,
            model_name=model_name or self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            allow_fallback=self.allow_fallback if allow_fallback is None else allow_fallback,
        )
