import json

import pytest
from urllib.error import HTTPError

from app.config import AppSettings, LLMSettings, load_settings
from app.llm.groq import GroqProvider
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
    name = "gemini"

    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate_response(self, prompt, *, system_prompt=None, **kwargs):
        self.calls += 1
        return self.response


def test_groq_successful_response(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        return FakeHTTPResponse(
            {"choices": [{"message": {"content": "Groq answer"}}]}
        )

    monkeypatch.setattr("app.llm.http.urlopen", fake_urlopen)
    provider = GroqProvider("test-key", model="configured-groq", timeout=9)

    result = provider.generate_response("Explain trees")

    assert result == "Groq answer"
    assert captured["request"].full_url == provider.endpoint
    assert captured["request"].headers["Authorization"] == "Bearer test-key"
    assert json.loads(captured["request"].data)["model"] == "configured-groq"


def test_groq_api_failure_raises_provider_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 503, "unavailable", {}, None)

    monkeypatch.setattr("app.llm.http.urlopen", fake_urlopen)
    provider = GroqProvider("test-key")

    with pytest.raises(RuntimeError, match="Groq request failed"):
        provider.generate_response("hello")


def test_groq_invalid_response_and_gateway_empty_response(monkeypatch):
    monkeypatch.setattr("app.llm.http.urlopen", lambda request, timeout: FakeHTTPResponse({}))
    provider = GroqProvider("test-key")

    with pytest.raises(RuntimeError, match="invalid groq response"):
        provider.generate_response("hello")

    gateway = LLMGateway(
        load_settings(),
        {"gemini": FakeProvider("gemini fallback"), "groq": FakeProvider("")},
    )
    assert gateway.generate_response("hello") == "gemini fallback"


def test_gemini_failure_falls_back_to_groq():
    gemini = FakeProvider(None)

    def failing_gemini(prompt, *, system_prompt=None, **kwargs):
        gemini.calls += 1
        raise TimeoutError("Gemini unavailable")

    gemini.generate_response = failing_gemini
    groq = FakeProvider("Groq fallback")
    gateway = LLMGateway(
        AppSettings(llm=LLMSettings(max_retries=0)),
        {"gemini": gemini, "groq": groq},
    )

    assert gateway.generate_response("hello") == "Groq fallback"
    assert gemini.calls == 1
    assert groq.calls == 1


def test_groq_key_and_model_are_loaded_from_environment(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "environment-groq-key")
    monkeypatch.setenv("GROQ_MODEL", "environment-groq-model")

    gateway = LLMGateway(load_settings())
    provider = gateway.providers["groq"]

    assert provider.model == "environment-groq-model"
    assert provider._api_key == "environment-groq-key"
    assert "openrouter" not in gateway.providers
    assert "mistral" not in gateway.providers
