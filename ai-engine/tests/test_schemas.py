import json

import pytest
from pydantic import ValidationError

from app.agents.interfaces import AgentRequest, AgentResponse, AgentType
from app.orchestrator.state import (
    LearnerState,
    MasteryStatus,
    NextAction,
    OrchestratorState,
    StructuredTask,
)


def test_orchestrator_state_has_required_defaults_and_is_serializable():
    state = OrchestratorState(
        student_id="S001",
        raw_task="I want to learn Decision Trees.",
        learner_state=LearnerState(student_id="S001"),
    )

    payload = json.loads(state.model_dump_json())
    assert payload["raw_task"].startswith("I want to learn")
    assert payload["mastery_status"] == MasteryStatus.IN_PROGRESS
    assert "updated_at" in payload


def test_structured_task_validates_and_preserves_learning_fields():
    task = StructuredTask(
        topic="Decision Trees",
        intent="learn",
        objective="understand the concept",
        needs_clarification=False,
    )
    assert task.topic == "Decision Trees"
    assert task.constraints == []


def test_agent_request_and_response_round_trip():
    request = AgentRequest(
        session_id="SESSION001",
        student_id="S001",
        agent_type=AgentType.TEACHING,
        task="Explain the fundamentals of Decision Trees",
    )
    response = AgentResponse(
        request_id=request.request_id,
        session_id=request.session_id,
        agent_type=request.agent_type,
        status="success",
        result={"text": "A decision tree splits data using features."},
        next_action=NextAction.CONTINUE,
    )

    assert response.model_validate_json(response.model_dump_json()).request_id == request.request_id


def test_invalid_score_and_empty_task_are_rejected():
    with pytest.raises(ValidationError):
        LearnerState(student_id="S001", concept_scores={"entropy": 1.5})
    with pytest.raises(ValidationError):
        AgentRequest(
            session_id="SESSION001",
            student_id="S001",
            agent_type=AgentType.TEACHING,
            task="",
        )

