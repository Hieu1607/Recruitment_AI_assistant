import asyncio
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.core.config import settings


class LLMProviderError(Exception):
    pass


class LLMConfigurationError(LLMProviderError):
    pass


class ProviderType(str, Enum):
    GROQ = "groq"
    OLLAMA = "ollama"


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None


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


class _GroqAdapter(_BaseAdapter):
    def __init__(
        self,
        api_key: str,
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
            raise LLMConfigurationError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        try:
            from groq import Groq
        except Exception as exc:
            raise LLMConfigurationError(
                "groq package is not available. Install it in backend requirements."
            ) from exc
        self._client = Groq(api_key=api_key)

    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = (completion.choices[0].message.content or "").strip()
                usage = None
                if getattr(completion, "usage", None) is not None:
                    usage = {
                        "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(completion.usage, "completion_tokens", None),
                        "total_tokens": getattr(completion.usage, "total_tokens", None),
                    }
                return LLMResponse(
                    text=content,
                    provider=ProviderType.GROQ.value,
                    model=self.model,
                    usage=usage,
                    raw=completion.model_dump() if hasattr(completion, "model_dump") else None,
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 3))
                    continue
                break
        raise LLMProviderError(f"Groq request failed: {last_error}") from last_error

    def generate_with_images(
        self, prompt: str, images: List[bytes], vision_model: str
    ) -> LLMResponse:
        import base64

        content: List[Any] = [{"type": "text", "text": prompt}]
        for img_bytes in images:
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            )
        messages = [{"role": "user", "content": content}]
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                completion = self._client.chat.completions.create(
                    model=vision_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max(self.max_tokens, 2048),
                )
                text = (completion.choices[0].message.content or "").strip()
                return LLMResponse(text=text, provider=ProviderType.GROQ.value, model=vision_model)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 3))
                    continue
                break
        raise LLMProviderError(f"Groq vision request failed: {last_error}") from last_error


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
                return LLMResponse(
                    text=content,
                    provider=ProviderType.OLLAMA.value,
                    model=self.model,
                    usage=usage,
                    raw=parsed,
                )
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 3))
                    continue
                break
            except Exception as exc:
                last_error = exc
                break
        raise LLMProviderError(f"Ollama request failed: {last_error}") from last_error


class LLMProvider:
    """Unified LLM provider wrapper for Groq and Ollama.

    Configuration is loaded from src.core.config.settings.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        selected_provider = (provider or settings.LLM_PROVIDER or ProviderType.GROQ.value).lower()
        self.provider = ProviderType(selected_provider)

        temperature = settings.LLM_TEMPERATURE if temperature is None else temperature
        max_tokens = settings.LLM_MAX_TOKENS if max_tokens is None else max_tokens
        timeout_seconds = (
            settings.LLM_TIMEOUT_SECONDS
            if timeout_seconds is None
            else timeout_seconds
        )
        max_retries = settings.LLM_MAX_RETRIES if max_retries is None else max_retries

        if self.provider == ProviderType.GROQ:
            self._adapter = _GroqAdapter(
                api_key=settings.GROQ_API_KEY,
                model=settings.GROQ_MODEL_NAME,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
        else:
            self._adapter = _OllamaAdapter(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL_NAME,
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
        if self.provider != ProviderType.GROQ:
            raise LLMProviderError("Vision is only supported for the Groq provider")
        vision_model = settings.GROQ_VISION_MODEL_NAME
        return self._adapter.generate_with_images(prompt, images, vision_model=vision_model)  # type: ignore[attr-defined]

    async def agenerate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")

        messages: List[Dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})
        return await self.achat(messages)