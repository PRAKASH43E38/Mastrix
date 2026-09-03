import logging

import pytest
from pydantic import BaseModel

from app.config import AppSettings, LLMSettings
from app.llm.gateway import LLMGateway, LLMGatewayError


class Answer(BaseModel):
    topic: str


class FakeProvider:
    def __init__(self, name, responses=None, error=None):
        self.name = name
        self.responses = list(responses or [])
        self.error = error
        self.calls = 0

    def generate_response(self, prompt, *, system_prompt=None, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.responses.pop(0)


def settings(retries=0):
    return AppSettings(
        llm=LLMSettings(
            max_retries=retries,
            gemini_api_key=None,
            groq_api_key=None,
            openrouter_api_key=None,
            mistral_api_key=None,
        )
    )


def test_gateway_uses_required_fallback_order():
    gemini = FakeProvider("gemini", error=TimeoutError())
    groq = FakeProvider("groq", error=RuntimeError("down"))
    openrouter = FakeProvider("openrouter", responses=[" from openrouter "])
    mistral = FakeProvider("mistral", responses=["unused"])
    gateway = LLMGateway(
        settings(),
        {"gemini": gemini, "groq": groq, "openrouter": openrouter, "mistral": mistral},
    )

    assert gateway.generate_response("hello") == "from openrouter"
    assert [gemini.calls, groq.calls, openrouter.calls, mistral.calls] == [1, 1, 1, 0]


def test_gateway_retries_before_fallback():
    gemini = FakeProvider("gemini", error=TimeoutError())
    groq = FakeProvider("groq", responses=["ok"])
    gateway = LLMGateway(settings(retries=2), {"gemini": gemini, "groq": groq})

    assert gateway.generate_response("hello") == "ok"
    assert gemini.calls == 3


def test_structured_output_is_validated_and_invalid_output_falls_back():
    gemini = FakeProvider("gemini", responses=['{"wrong": "field"}'])
    groq = FakeProvider("groq", responses=['{"topic": "Decision Trees"}'])
    gateway = LLMGateway(settings(), {"gemini": gemini, "groq": groq})

    result = gateway.generate_structured("learn trees", Answer)

    assert result.topic == "Decision Trees"
    assert gemini.calls == 1
    assert groq.calls == 1


def test_all_provider_failures_are_controlled_and_logs_do_not_contain_key(caplog):
    secret = "do-not-log-this-key"
    gateway = LLMGateway(
        settings(),
        {"gemini": FakeProvider("gemini", error=RuntimeError(secret))},
    )

    with caplog.at_level(logging.WARNING), pytest.raises(LLMGatewayError):
        gateway.generate_response("hello")

    assert secret not in caplog.text
    assert "gemini" in caplog.text

