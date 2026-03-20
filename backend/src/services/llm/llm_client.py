from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx


class LLMClientError(RuntimeError):
    pass


@dataclass
class LLMRequest:
    prompt: str
    system_prompt: str | None = None
    temperature: float = 0.2


class BaseLLMClient:
    def generate(self, request: LLMRequest) -> str:
        raise NotImplementedError


class GroqLLMClient(BaseLLMClient):
    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise LLMClientError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        self.api_key = api_key
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def generate(self, request: LLMRequest) -> str:
        from groq import Groq

        client = Groq(api_key=self.api_key)
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=request.temperature,
            )
        except Exception as exc:
            raise LLMClientError("Groq request failed") from exc
        return response.choices[0].message.content or ""


class OllamaLLMClient(BaseLLMClient):
    def __init__(self) -> None:
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    def generate(self, request: LLMRequest) -> str:
        full_prompt = request.prompt
        if request.system_prompt:
            full_prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}"

        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.model, "prompt": full_prompt, "stream": False},
                )
                response.raise_for_status()
                payload = response.json()
                return payload.get("response", "")
        except Exception as exc:
            raise LLMClientError("Ollama request failed") from exc


class LLMClientFactory:
    @staticmethod
    def create() -> BaseLLMClient:
        provider = os.getenv("LLM_PROVIDER", "groq").lower()
        if provider == "groq":
            return GroqLLMClient()
        if provider == "ollama":
            return OllamaLLMClient()
        raise LLMClientError(f"Unsupported LLM provider: {provider}")


def generate_json(request: LLMRequest) -> dict[str, Any]:
    raw = LLMClientFactory.create().generate(request)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMClientError("LLM response is not valid JSON") from exc
