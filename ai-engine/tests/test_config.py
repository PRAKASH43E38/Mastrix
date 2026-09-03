from app.config import load_settings


def test_config_has_provider_order_and_no_required_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = load_settings()

    assert settings.llm.primary_provider == "gemini"
    assert settings.llm.fallback_providers == ["groq", "openrouter", "mistral"]
    assert settings.llm.gemini_api_key is None

