import importlib


def test_shopaikey_api_key_reads_single_env_var(monkeypatch):
    monkeypatch.setenv("SHOPAIKEY_API_KEY", "shop-key")

    import src.core.config as config_module

    reloaded = importlib.reload(config_module)

    try:
        assert reloaded.Settings().SHOPAIKEY_API_KEY == "shop-key"
    finally:
        importlib.reload(config_module)
