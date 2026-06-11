from __future__ import annotations

import asyncio
import json
import logging
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.core.config import settings

try:
    import edge_tts
except Exception:  # pragma: no cover
    edge_tts = None


logger = logging.getLogger(__name__)


class TTSProviderError(Exception):
    pass


def _normalize_language_code(language_code: str | None) -> str:
    candidate = str(language_code or "").strip()
    return candidate or "en-US"


def _edge_voice_for_language(language_code: str | None) -> str:
    normalized = _normalize_language_code(language_code).lower()
    if normalized.startswith("vi"):
        return settings.EDGE_TTS_VOICE_VI
    return settings.EDGE_TTS_VOICE_EN


async def _synthesize_with_edge_tts(text: str, *, language_code: str) -> bytes:
    if edge_tts is None:
        raise TTSProviderError("edge-tts package is not available")

    communicate = edge_tts.Communicate(
        text=text,
        voice=_edge_voice_for_language(language_code),
        rate=settings.EDGE_TTS_RATE,
        volume=settings.EDGE_TTS_VOLUME,
    )
    audio_chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            data = chunk.get("data")
            if isinstance(data, bytes):
                audio_chunks.append(data)

    if not audio_chunks:
        raise TTSProviderError("edge-tts returned no audio")
    return b"".join(audio_chunks)


def _synthesize_with_shopaikey_openai_tts(text: str, *, language_code: str) -> bytes:
    api_key = settings.SHOPAIKEY_API_KEY
    if not api_key:
        raise TTSProviderError("ShopAIKey API key is not configured for TTS fallback")

    payload = {
        "model": settings.OPENAI_TTS_MODEL,
        "voice": settings.OPENAI_TTS_VOICE,
        "input": text,
        "response_format": "mp3",
    }
    if _normalize_language_code(language_code).lower().startswith("vi"):
        payload["instructions"] = "Speak naturally in Vietnamese with clear pronunciation."
    else:
        payload["instructions"] = "Speak naturally in English with clear pronunciation."

    request = Request(
        url=f"{settings.SHOPAIKEY_BASE_URL.rstrip('/')}/audio/speech",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=settings.TTS_TIMEOUT_SECONDS) as response:
            audio = response.read()
    except HTTPError as exc:  # pragma: no cover
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""
        raise TTSProviderError(f"ShopAIKey TTS HTTP {exc.code}: {body or exc.reason}") from exc
    except (URLError, TimeoutError) as exc:  # pragma: no cover
        raise TTSProviderError(f"ShopAIKey TTS request failed: {exc}") from exc

    if not audio:
        raise TTSProviderError("ShopAIKey TTS returned no audio")
    return audio


def synthesize_speech(text: str, *, language_code: str) -> bytes:
    candidate = text.strip()
    if not candidate:
        raise ValueError("text must not be blank")

    normalized_language = _normalize_language_code(language_code)
    try:
        return asyncio.run(_synthesize_with_edge_tts(candidate, language_code=normalized_language))
    except Exception as exc:
        logger.warning(
            "Edge TTS failed; falling back to ShopAIKey OpenAI TTS. language=%s error=%s",
            normalized_language,
            exc,
        )
        return _synthesize_with_shopaikey_openai_tts(candidate, language_code=normalized_language)
