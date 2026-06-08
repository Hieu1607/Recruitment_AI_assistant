import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_settings_default_llm_max_tokens_is_2048(monkeypatch):
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)

    config_module = importlib.import_module("src.core.config")
    config_module = importlib.reload(config_module)
    settings = config_module.Settings(_env_file=None)

    assert settings.LLM_MAX_TOKENS == 2048
