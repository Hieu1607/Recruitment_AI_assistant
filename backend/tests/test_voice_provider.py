from pathlib import Path
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from src.services.voice_provider import get_voice_provider  # noqa: E402


def test_fake_voice_provider_normalizes_aliases_and_transcript_events():
    provider = get_voice_provider(" Fake ")

    assert provider.name == "fake"

    normalized = provider.normalize_events(
        [
            {
                "speaker": "Agent",
                "text": "Welcome to the interview.",
                "offset_ms": "12",
                "question_key": "intro",
            },
            {
                "speaker": "USER",
                "text": "Thank you.",
                "offset_ms": 48,
            },
        ]
    )

    assert [event.speaker_role for event in normalized] == ["assistant", "candidate"]
    assert [event.turn_index for event in normalized] == [0, 1]
    assert normalized[0].transcript_text == "Welcome to the interview."
    assert normalized[0].time_offset_ms == 12
    assert normalized[0].question_key == "intro"
    assert normalized[1].transcript_text == "Thank you."
    assert normalized[1].time_offset_ms == 48
