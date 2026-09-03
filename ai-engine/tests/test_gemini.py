import json

import pytest
from urllib.error import HTTPError

from app.config import load_settings
from app.llm.gemini import GeminiProvider
from app.llm.gateway import LLMGateway


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class FakeProvider:
    name = "groq"

    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate_response(self, prompt, *, system_prompt=None, **kwargs):
        self.calls += 1
        return self.response


def test_gemini_successful_response(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeHTTPResponse(
            {"candidates": [{"content": {"parts": [{"text": "Gemini answer"}]}}]}
        )

    monkeypatch.setattr("app.llm.http.urlopen", fake_urlopen)
    provider = GeminiProvider("test-key", model="configured-model", timeout=7)

    result = provider.generate_response("Explain trees", system_prompt="Be concise")

    assert result == "Gemini answer"
    assert captured["timeout"] == 7
    assert "configured-model" in captured["request"].full_url
    assert "test-key" in captured["request"].full_url


def test_gemini_api_failure_raises_provider_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 503, "unavailable", {}, None)

    monkeypatch.setattr("app.llm.http.urlopen", fake_urlopen)
    provider = GeminiProvider("test-key")

    with pytest.raises(RuntimeError, match="Gemini request failed"):
        provider.generate_response("hello")


def test_gemini_invalid_and_empty_responses_are_rejected(monkeypatch):
    monkeypatch.setattr("app.llm.http.urlopen", lambda request, timeout: FakeHTTPResponse({}))
    provider = GeminiProvider("test-key")

    with pytest.raises(RuntimeError, match="invalid Gemini response"):
        provider.generate_response("hello")

    gateway = LLMGateway(
        load_settings(),
        {"gemini": FakeProvider(""), "groq": FakeProvider("fallback")},
    )
    assert gateway.generate_response("hello") == "fallback"


def test_gateway_falls_back_after_gemini_failure(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 500, "failed", {}, None)

    monkeypatch.setattr("app.llm.http.urlopen", fake_urlopen)
    gemini = GeminiProvider("test-key")
    groq = FakeProvider("fallback answer")
    gateway = LLMGateway(load_settings(), {"gemini": gemini, "groq": groq})

    assert gateway.generate_response("hello") == "fallback answer"
    assert groq.calls == 1


def test_gemini_key_and_model_are_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "environment-key")
    monkeypatch.setenv("GEMINI_MODEL", "environment-model")
    settings = load_settings()
    gateway = LLMGateway(settings)

    provider = gateway.providers["gemini"]
    assert provider.model == "environment-model"
    assert provider._api_key == "environment-key"

