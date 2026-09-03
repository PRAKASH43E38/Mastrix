import json

import pytest
from pydantic import ValidationError

from app.config import AppSettings, LLMSettings
from app.llm.gateway import LLMGateway, LLMGatewayError
from app.orchestrator.planner import LearningPlanner
from app.orchestrator.state import (
    LearningPlan,
    LearningStage,
    LearnerState,
    MasteryStatus,
    NextAction,
    StructuredTask,
)


class FakeGateway:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate_structured(self, prompt, response_model, *, system_prompt=None, **kwargs):
        self.calls.append(
            {"prompt": prompt, "response_model": response_model, "system_prompt": system_prompt}
        )
        return response_model.model_validate(self.payload)


def task(difficulty=None):
    return StructuredTask(
        topic="Decision Trees",
        intent="learn",
        objective="understand the concept",
        difficulty=difficulty,
    )


def learner(**kwargs):
    return LearnerState(student_id="S001", topic="Decision Trees", **kwargs)


def plan_payload(**overrides):
    payload = {
        "objective": "Understand Decision Trees",
        "current_stage": "learning",
        "steps": ["Review prerequisites", "Study core concepts"],
        "stages": ["prerequisites", "core concepts", "practice"],
        "concepts": ["nodes", "splits", "leaves"],
        "recommended_action": "teach",
        "target_difficulty": "intermediate",
        "required_intervention": None,
        "rationale": "Begin with foundations before practice.",
        "decision_summary": "Start with the core concepts.",
    }
    payload.update(overrides)
    return payload


def test_normal_learning_plan_is_structured_and_uses_planner_prompt():
    gateway = FakeGateway(plan_payload())
    result = LearningPlanner(gateway).plan(task(), learner())

    assert isinstance(result, LearningPlan)
    assert result.concepts == ["nodes", "splits", "leaves"]
    assert result.recommended_action == "teach"
    assert "learning planner" in gateway.calls[0]["system_prompt"]


def test_beginner_difficulty_is_preserved_in_plan_context_and_output():
    gateway = FakeGateway(plan_payload(target_difficulty="beginner"))
    LearningPlanner(gateway).plan(task("beginner"), learner())

    request = json.loads(gateway.calls[0]["prompt"])
    assert request["structured_task"]["difficulty"] == "beginner"
    assert gateway.payload["target_difficulty"] == "beginner"


def test_advanced_difficulty_is_preserved_in_plan_context_and_output():
    gateway = FakeGateway(plan_payload(target_difficulty="advanced"))
    LearningPlanner(gateway).plan(task("advanced"), learner())

    request = json.loads(gateway.calls[0]["prompt"])
    assert request["structured_task"]["difficulty"] == "advanced"
    assert gateway.payload["target_difficulty"] == "advanced"


def test_weak_concepts_force_a_remediation_plan():
    gateway = FakeGateway(plan_payload(concepts=["new material"], recommended_action="teach"))
    result = LearningPlanner(gateway).plan(
        task(), learner(weak_concepts=["entropy", "information gain"])
    )

    assert result.current_stage == LearningStage.REMEDIATION
    assert result.recommended_action == NextAction.REMEDIATE
    assert result.concepts[:2] == ["entropy", "information gain"]


def test_strong_performance_allows_advancement():
    gateway = FakeGateway(plan_payload(recommended_action="continue"))
    result = LearningPlanner(gateway).plan(
        task(),
        learner(
            concept_scores={"definition": 0.95, "structure": 0.9},
            mastery_status=MasteryStatus.MASTERED,
        ),
    )

    assert result.recommended_action == NextAction.ADVANCE


def test_previous_learner_state_influences_planning_input():
    gateway = FakeGateway(plan_payload())
    state = learner(
        current_stage=LearningStage.PRACTICE,
        concept_scores={"entropy": 0.55},
        last_agent="assessment",
        last_result={"score": 0.55, "feedback": "Review entropy"},
    )
    LearningPlanner(gateway).plan(task(), state)

    request = json.loads(gateway.calls[0]["prompt"])
    assert request["learner_state"]["current_stage"] == "practice"
    assert request["learner_state"]["last_agent"] == "assessment"
    assert request["learner_state"]["last_result"]["score"] == 0.55


def test_malformed_llm_output_is_rejected():
    with pytest.raises(ValidationError):
        LearningPlanner(FakeGateway({"objective": "incomplete"})).plan(task(), learner())


def test_llm_provider_failure_is_controlled():
    gateway = LLMGateway(AppSettings(llm=LLMSettings(max_retries=0)), providers={})

    with pytest.raises(LLMGatewayError):
        LearningPlanner(gateway).plan(task(), learner())

