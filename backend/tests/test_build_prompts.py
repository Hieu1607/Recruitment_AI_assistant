import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prompts.build_prompts import BuildPrompts  # noqa: E402


def test_cv_parsing_prompt_requires_exhaustive_extraction():
    prompt = BuildPrompts().build_cv_parsing_prompt("Example CV text")

    assert "Keep extracted text concise" not in prompt
    assert "Preserve as much source detail as possible" in prompt
    assert "Do not summarize, shorten, paraphrase, or normalize away specifics." in prompt
    assert "If text does not clearly fit an earlier field, put it in other instead of dropping it." in prompt
    assert "Keep bullet points, lists, metrics, technologies, dates, organizations, titles, and outcomes whenever present." in prompt
    assert 'For "projects", include full project entries' in prompt
    assert 'For "experience", include full role entries' in prompt
    assert 'For "education", include full education entries' in prompt
    assert 'For "skills", preserve grouped skill categories' in prompt


def test_cv_vision_prompt_requires_exhaustive_extraction():
    prompt = BuildPrompts().build_cv_vision_prompt()

    assert "Keep extracted text concise" not in prompt
    assert "Preserve as much source detail as possible" in prompt
    assert "Do not summarize, shorten, paraphrase, or normalize away specifics." in prompt
    assert "If text does not clearly fit an earlier field, put it in other instead of dropping it." in prompt
    assert "The CV may be in Vietnamese — extract text exactly as written." in prompt
    assert 'For "projects", include full project entries' in prompt
    assert 'For "experience", include full role entries' in prompt
