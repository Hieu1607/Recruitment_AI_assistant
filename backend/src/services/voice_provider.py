from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class NormalizedTranscriptEvent:
    speaker_role: str
    transcript_text: str
    time_offset_ms: int | None
    turn_index: int
    question_key: str | None = None
    payload: dict[str, Any] | None = None


class VoiceProvider:
    name: str

    def normalize_events(self, events: list[dict[str, Any]]) -> list[NormalizedTranscriptEvent]:
        raise NotImplementedError


class UnsupportedVoiceProviderError(ValueError):
    pass


class FakeVoiceProvider(VoiceProvider):
    name = "fake"
    _speaker_aliases = {
        "agent": "assistant",
        "assistant": "assistant",
        "ai": "assistant",
        "system": "assistant",
        "user": "candidate",
        "candidate": "candidate",
        "human": "candidate",
    }

    def normalize_events(self, events: list[dict[str, Any]]) -> list[NormalizedTranscriptEvent]:
        normalized: list[NormalizedTranscriptEvent] = []
        for index, event in enumerate(events):
            speaker = str(event["speaker"]).strip().lower()
            text = str(event["text"]).strip()
            offset_ms = event.get("offset_ms")
            normalized.append(
                NormalizedTranscriptEvent(
                    speaker_role=self._speaker_aliases.get(speaker, speaker),
                    transcript_text=text,
                    time_offset_ms=int(offset_ms) if offset_ms is not None else None,
                    turn_index=index,
                    question_key=self._normalize_optional_string(event.get("question_key")),
                    payload=event.get("payload"),
                )
            )
        return normalized

    @staticmethod
    def _normalize_optional_string(value: Any) -> str | None:
        if value is None:
            return None
        candidate = str(value).strip()
        return candidate or None


def get_voice_provider(provider_name: str | None) -> VoiceProvider:
    candidate = (provider_name or "fake").strip().lower()
    if candidate in {"fake", "mock", "test"}:
        return FakeVoiceProvider()
    raise UnsupportedVoiceProviderError(f"Unsupported voice provider: {provider_name}")
