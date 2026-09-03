import pytest
from pydantic import ValidationError

from app.config import AppSettings, LLMSettings
from app.llm.gateway import LLMGateway, LLMGatewayError
from app.orchestrator.state import StructuredTask
from app.orchestrator.task_understanding import TaskUnderstanding


class FakeGateway:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate_structured(self, prompt, response_model, *, system_prompt=None, **kwargs):
        self.calls.append(
            {"prompt": prompt, "response_model": response_model, "system_prompt": system_prompt}
        )
        return response_model.model_validate(self.payload)


def understood_task():
    return {
        "topic": "Decision Trees",
        "intent": "learn",
        "objective": "understand the concept",
        "requested_outcome": None,
        "constraints": [],
        "difficulty": None,
        "ambiguity": None,
        "needs_clarification": False,
        "clarification_question": None,
    }


def test_successful_task_understanding_uses_dedicated_prompt():
    gateway = FakeGateway(understood_task())
    result = TaskUnderstanding(gateway).understand("I want to learn Decision Trees")

    assert isinstance(result, StructuredTask)
    assert gateway.calls[0]["prompt"] == "I want to learn Decision Trees"
    assert "task-understanding component" in gateway.calls[0]["system_prompt"]


def test_task_understanding_extracts_topic():
    result = TaskUnderstanding(FakeGateway(understood_task())).understand("learn trees")
    assert result.topic == "Decision Trees"


def test_task_understanding_extracts_intent():
    payload = understood_task()
    payload["intent"] = "practice"
    result = TaskUnderstanding(FakeGateway(payload)).understand("Give me practice problems")
    assert result.intent == "practice"


def test_task_understanding_extracts_objective():
    result = TaskUnderstanding(FakeGateway(understood_task())).understand("learn trees")
    assert result.objective == "understand the concept"


def test_task_understanding_extracts_difficulty():
    payload = understood_task()
    payload["difficulty"] = "beginner"
    result = TaskUnderstanding(FakeGateway(payload)).understand("teach me trees as a beginner")
    assert result.difficulty == "beginner"


def test_task_understanding_extracts_constraints():
    payload = understood_task()
    payload["constraints"] = ["use Python", "keep it concise"]
    result = TaskUnderstanding(FakeGateway(payload)).understand("teach trees in Python")
    assert result.constraints == ["use Python", "keep it concise"]


def test_ambiguous_task_returns_structured_clarification():
    payload = {
        "topic": None,
        "intent": None,
        "objective": None,
        "constraints": [],
        "difficulty": None,
        "ambiguity": "The subject and desired learning activity are unclear.",
        "needs_clarification": True,
        "clarification_question": "What topic would you like to study?",
    }
    result = TaskUnderstanding(FakeGateway(payload)).understand("Help me")

    assert result.needs_clarification is True
    assert result.topic is None
    assert result.clarification_question


def test_malformed_llm_output_is_rejected():
    with pytest.raises(ValidationError):
        TaskUnderstanding(FakeGateway({"needs_clarification": False})).understand("learn")


def test_llm_provider_failure_is_controlled():
    gateway = LLMGateway(
        AppSettings(llm=LLMSettings(max_retries=0)),
        providers={},
    )

    with pytest.raises(LLMGatewayError):
        TaskUnderstanding(gateway).understand("I want to learn Decision Trees")

