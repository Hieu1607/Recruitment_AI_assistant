import asyncio
import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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

    Configuration is loaded from environment variables by default:
    - LLM_PROVIDER=groq|ollama
    - LLM_TEMPERATURE
    - LLM_MAX_TOKENS
    - LLM_TIMEOUT_SECONDS
    - LLM_MAX_RETRIES
    - GROQ_API_KEY, GROQ_MODEL_NAME
    - OLLAMA_BASE_URL, OLLAMA_MODEL_NAME, OLLAMA_KEEP_ALIVE
    """

    _CV_ANALYSIS_SYSTEM_PROMPT = (
        "You are an expert technical recruiter and CV analyst. "
        "Analyze CV content accurately and return concise, evidence-based output. "
        "Avoid hallucinations and only use information present in the CV text. "
        "If job description is provided, assess candidate-job fit with clear rationale. "
        "Always return valid JSON with keys: summary, core_skills, experience_years_estimate, "
        "education, strengths, gaps, recommendations, fit_score_0_100, confidence_0_100."
    )

    def __init__(
        self,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        selected_provider = (provider or os.getenv("LLM_PROVIDER", ProviderType.GROQ.value)).lower()
        self.provider = ProviderType(selected_provider)

        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2")) if temperature is None else temperature
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024")) if max_tokens is None else max_tokens
        timeout_seconds = (
            int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
            if timeout_seconds is None
            else timeout_seconds
        )
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "2")) if max_retries is None else max_retries

        if self.provider == ProviderType.GROQ:
            self._adapter = _GroqAdapter(
                api_key=os.getenv("GROQ_API_KEY", ""),
                model=os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b"),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
        else:
            self._adapter = _OllamaAdapter(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL_NAME", "llama3.1:8b"),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "5m"),
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

    async def agenerate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")

        messages: List[Dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})
        return await self.achat(messages)

    # def _truncate_cv_text(self, text: str, max_chars: int = 24000) -> str:
    #     cleaned = (text or "").strip()
    #     if len(cleaned) <= max_chars:
    #         return cleaned

    #     head = cleaned[: int(max_chars * 0.7)]
    #     tail = cleaned[-int(max_chars * 0.3):]
    #     return f"{head}\n\n...[TRUNCATED FOR LENGTH]...\n\n{tail}"

    # def _build_cv_analysis_prompt(
    #     self,
    #     cv_text: str,
    #     job_description: Optional[str] = None,
    #     focus_areas: Optional[List[str]] = None,
    # ) -> str:
    #     cv_text = self._truncate_cv_text(cv_text)
    #     focus = ", ".join(focus_areas) if focus_areas else "technical skills, achievements, role relevance"

    #     prompt_parts = [
    #         "Task: Analyze the candidate CV and provide structured JSON output.",
    #         f"Focus areas: {focus}",
    #         "Output requirements:",
    #         "- summary: concise overview",
    #         "- core_skills: list of strongest skills",
    #         "- experience_years_estimate: numeric estimate",
    #         "- education: highest/most relevant education",
    #         "- strengths: list",
    #         "- gaps: list",
    #         "- recommendations: list",
    #         "- fit_score_0_100: integer",
    #         "- confidence_0_100: integer",
    #         "CV:",
    #         cv_text,
    #     ]

    #     if job_description and job_description.strip():
    #         prompt_parts.extend(["Job description:", job_description.strip()])

    #     prompt_parts.append("Return JSON only. No markdown, no extra text.")
    #     return "\n".join(prompt_parts)

    # def analyze_cv(
    #     self,
    #     cv_text: str,
    #     job_description: Optional[str] = None,
    #     focus_areas: Optional[List[str]] = None,
    # ) -> LLMResponse:
    #     if not cv_text or not cv_text.strip():
    #         raise ValueError("cv_text must not be empty")

    #     prompt = self._build_cv_analysis_prompt(
    #         cv_text=cv_text,
    #         job_description=job_description,
    #         focus_areas=focus_areas,
    #     )
    #     return self.generate(prompt=prompt, system_prompt=self._CV_ANALYSIS_SYSTEM_PROMPT)

    # async def aanalyze_cv(
    #     self,
    #     cv_text: str,
    #     job_description: Optional[str] = None,
    #     focus_areas: Optional[List[str]] = None,
    # ) -> LLMResponse:
    #     if not cv_text or not cv_text.strip():
    #         raise ValueError("cv_text must not be empty")

    #     prompt = self._build_cv_analysis_prompt(
    #         cv_text=cv_text,
    #         job_description=job_description,
    #         focus_areas=focus_areas,
    #     )
    #     return await self.agenerate(prompt=prompt, system_prompt=self._CV_ANALYSIS_SYSTEM_PROMPT)
