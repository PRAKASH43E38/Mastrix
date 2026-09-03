import json

import pytest
from pydantic import ValidationError

from app.agents.interfaces import AgentResponse, AgentType
from app.orchestrator.state import (
    LearningStage,
    LearnerState,
    MasteryStatus,
    NextAction,
    OrchestratorState,
)
from app.orchestrator.state_manager import StateManager


def state(**kwargs):
    return LearnerState(student_id="S001", topic="Decision Trees", **kwargs)


def response(**kwargs):
    values = {
        "request_id": "REQ001",
        "session_id": "SESSION001",
        "agent_type": AgentType.TEACHING,
        "status": "success",
    }
    values.update(kwargs)
    return AgentResponse(**values)


def test_initial_state_update_stores_result_and_history():
    updated = StateManager().update_learner_state(
        state(), response(result={"text": "intro"}, summary="Introduction provided")
    )

    assert updated.last_agent == AgentType.TEACHING.value
    assert updated.last_result == {"text": "intro"}
    assert len(updated.progress_history) == 1
    assert updated.progress_history[0].summary == "Introduction provided"


def test_concept_score_update_is_validated_and_merged():
    updated = StateManager().update_learner_state(
        state(concept_scores={"definition": 0.7}),
        response(concept_scores={"entropy": 0.6}),
    )
    assert updated.concept_scores == {"definition": 0.7, "entropy": 0.6}


def test_low_concept_score_creates_weakness():
    updated = StateManager().update_learner_state(
        state(), response(concept_scores={"information_gain": 0.4})
    )
    assert updated.weak_concepts == ["information_gain"]
    assert updated.next_action == NextAction.REMEDIATE


def test_improved_concept_resolves_only_that_weakness():
    updated = StateManager().update_learner_state(
        state(weak_concepts=["entropy", "information_gain"]),
        response(concept_scores={"entropy": 0.9}),
    )
    assert updated.weak_concepts == ["information_gain"]
    assert updated.concept_scores["entropy"] == 0.9


def test_last_agent_and_result_are_replaced_by_new_response():
    updated = StateManager().update_learner_state(
        state(last_agent="research", last_result={"old": True}),
        response(agent_type=AgentType.ASSESSMENT, result={"score": 0.8}),
    )
    assert updated.last_agent == AgentType.ASSESSMENT.value
    assert updated.last_result == {"score": 0.8}


def test_explicit_stage_transition_is_applied():
    updated = StateManager().update_learner_state(
        state(current_stage=LearningStage.LEARNING),
        response(metadata={"next_stage": "assessment"}),
    )
    assert updated.current_stage == LearningStage.ASSESSMENT


def test_explicit_next_action_is_applied():
    updated = StateManager().update_learner_state(
        state(), response(next_action=NextAction.ADVANCE)
    )
    assert updated.next_action == NextAction.ADVANCE


def test_unrelated_existing_scores_are_preserved():
    updated = StateManager().update_learner_state(
        state(concept_scores={"definition": 0.9, "entropy": 0.5}),
        response(concept_scores={"splits": 0.8}),
    )
    assert updated.concept_scores["definition"] == 0.9
    assert updated.concept_scores["entropy"] == 0.5
    assert updated.concept_scores["splits"] == 0.8


def test_repeated_remediation_preserves_history_and_unrelated_progress():
    manager = StateManager()
    first = manager.update_learner_state(
        state(concept_scores={"definition": 0.9}),
        response(concept_scores={"entropy": 0.3}, next_action=NextAction.REMEDIATE),
    )
    second = manager.update_learner_state(
        first,
        response(concept_scores={"entropy": 0.75}, next_action=NextAction.REPEAT),
    )
    assert len(second.progress_history) == 2
    assert second.concept_scores["definition"] == 0.9
    assert second.weak_concepts == ["entropy"]


def test_mastery_status_is_preserved_without_explicit_agent_status():
    updated = StateManager().update_learner_state(
        state(mastery_status=MasteryStatus.IN_PROGRESS),
        response(concept_scores={"definition": 1.0}),
    )
    assert updated.mastery_status == MasteryStatus.IN_PROGRESS


def test_explicit_mastery_status_is_recorded_without_inference_from_score():
    updated = StateManager().update_learner_state(
        state(), response(mastery_status=MasteryStatus.MASTERED)
    )
    assert updated.mastery_status == MasteryStatus.MASTERED


def test_invalid_agent_result_is_rejected():
    with pytest.raises(ValidationError):
        response(concept_scores={"entropy": 1.5})


def test_missing_required_state_is_rejected():
    manager = StateManager()
    with pytest.raises(ValueError, match="required"):
        manager.update_learner_state(None, response())  # type: ignore[arg-type]


def test_orchestrator_state_is_updated_and_remains_serializable():
    orchestrator_state = OrchestratorState(
        session_id="SESSION001",
        student_id="S001",
        raw_task="Learn Decision Trees",
        learner_state=state(),
    )
    updated = StateManager().update_orchestrator_state(
        orchestrator_state,
        response(concept_scores={"definition": 0.8}, result={"score": 0.8}),
    )

    payload = json.loads(updated.model_dump_json())
    assert payload["concept_scores"] == {"definition": 0.8}
    assert payload["learner_state"]["last_agent"] == "teaching"

