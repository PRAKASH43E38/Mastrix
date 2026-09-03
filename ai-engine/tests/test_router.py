import pytest

from app.agents.interfaces import AgentType
from app.orchestrator.router import AgentRouter, RoutingDecision
from app.orchestrator.state import LearningPlan, LearningStage, LearnerState, StructuredTask


def task(intent="learn"):
    return StructuredTask(
        topic="Decision Trees", intent=intent, objective="understand the concept"
    )


def plan(action="continue", concepts=None):
    return LearningPlan(
        objective="Understand Decision Trees",
        recommended_action=action,
        concepts=concepts or ["nodes", "splits"],
        stages=["foundations", "core concepts"],
    )


def state(**kwargs):
    return LearnerState(student_id="S001", topic="Decision Trees", **kwargs)


def test_research_routing():
    result = AgentRouter().route(task("research"), plan("research foundations"), state())
    assert result.selected_agent == AgentType.RESEARCH


def test_teaching_routing():
    result = AgentRouter().route(task("learn"), plan("explain fundamentals"), state())
    assert result.selected_agent == AgentType.TEACHING


def test_critical_thinking_routing():
    result = AgentRouter().route(task("reasoning"), plan("validate reasoning"), state())
    assert result.selected_agent == AgentType.CRITICAL_THINKING


def test_practical_application_routing():
    result = AgentRouter().route(task("practice"), plan("practice implementation"), state())
    assert result.selected_agent == AgentType.PRACTICAL_APPLICATION


def test_assessment_routing():
    result = AgentRouter().route(task("test"), plan("assess progress"), state())
    assert result.selected_agent == AgentType.ASSESSMENT


def test_conceptual_clarity_routing():
    result = AgentRouter().route(
        task(), plan("continue"), state(last_result={"conflicting_evidence": True})
    )
    assert result.selected_agent == AgentType.CONCEPTUAL_CLARITY_FEEDBACK


def test_weakness_based_routing_focuses_agent_task():
    result = AgentRouter().route(
        task(), plan("continue"), state(weak_concepts=["entropy", "information gain"])
    )
    assert result.selected_agent == AgentType.TEACHING
    assert "entropy" in result.agent_task
    assert "information gain" in result.agent_task


def test_plan_based_routing_is_a_strong_signal():
    result = AgentRouter().route(task("learn"), plan("practice exercises"), state())
    assert result.selected_agent == AgentType.PRACTICAL_APPLICATION
    assert result.priority == 2


def test_previous_results_enable_non_linear_routing():
    result = AgentRouter().route(
        task("learn"),
        plan("continue"),
        state(
            current_stage=LearningStage.PRACTICE,
            last_agent="application",
            last_result={"application_score": 0.4},
        ),
    )
    assert result.selected_agent == AgentType.PRACTICAL_APPLICATION
    assert "application" in result.reason.lower()


def test_router_result_and_agent_request_are_validated():
    router = AgentRouter()
    decision = router.route(task(), plan("explain fundamentals"), state())
    request = router.build_request(
        decision, task(), state(), session_id="SESSION001"
    )

    assert isinstance(decision, RoutingDecision)
    assert request.agent_type == AgentType.TEACHING
    assert request.student_id == "S001"
    assert request.context["stage"] == LearningStage.UNDERSTANDING.value


def test_invalid_or_missing_state_is_rejected():
    router = AgentRouter()
    with pytest.raises(ValueError, match="clarification"):
        router.route(
            StructuredTask(
                topic=None,
                intent=None,
                objective=None,
                needs_clarification=True,
            ),
            plan(),
            state(),
        )
    with pytest.raises(ValueError, match="required"):
        router.route(task(), plan(), None)  # type: ignore[arg-type]
