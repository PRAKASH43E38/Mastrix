import pytest
from pydantic import ValidationError

from app.agents.interfaces import AgentResponse, AgentType
from app.config import AppSettings, LLMSettings
from app.llm.gateway import LLMGatewayError
from app.orchestrator.graph import MainOrchestrator
from app.orchestrator.planner import LearningPlanner
from app.orchestrator.router import AgentRouter
from app.orchestrator.state import (
    LearningPlan,
    LearnerState,
    MasteryStatus,
    NextAction,
    OrchestratorState,
    StructuredTask,
)
from app.orchestrator.state_manager import StateManager
from app.orchestrator.task_understanding import TaskUnderstanding


class FakeGateway:
    def __init__(self, task_payload, plans):
        self.task_payload = task_payload
        self.plans = list(plans)
        self.calls = []

    def generate_structured(self, prompt, response_model, *, system_prompt=None, **kwargs):
        self.calls.append((prompt, response_model, system_prompt))
        if response_model is StructuredTask:
            return response_model.model_validate(self.task_payload)
        return response_model.model_validate(self.plans.pop(0))


class MockExecutor:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return response.model_copy(update={"request_id": request.request_id, "session_id": request.session_id})


def task_payload():
    return {
        "topic": "Decision Trees",
        "intent": "learn",
        "objective": "understand the concept",
        "constraints": [],
        "needs_clarification": False,
    }


def plan_payload(action="continue"):
    return {
        "objective": "Understand Decision Trees",
        "current_stage": "learning",
        "steps": ["Study foundations"],
        "stages": ["foundations", "practice"],
        "concepts": ["nodes", "splits"],
        "recommended_action": action,
        "target_difficulty": "beginner",
        "decision_summary": "Start with the foundations.",
    }


def agent_response(
    agent=AgentType.TEACHING,
    *,
    result=None,
    scores=None,
    weaknesses=None,
    next_action=None,
    mastery=None,
    status="success",
    error=None,
):
    return AgentResponse(
        request_id="placeholder",
        session_id="placeholder",
        agent_type=agent,
        status=status,
        result=result or {"summary": "done"},
        concept_scores=scores or {},
        weaknesses=weaknesses or [],
        next_action=next_action,
        mastery_status=mastery,
        error=error,
    )


def make_orchestrator(responses, plans=None):
    gateway = FakeGateway(task_payload(), plans or [plan_payload()])
    return MainOrchestrator(
        TaskUnderstanding(gateway),
        LearningPlanner(gateway),
        AgentRouter(),
        StateManager(),
        MockExecutor(responses),
    )


def initial_state(**kwargs):
    values = {
        "session_id": "SESSION001",
        "student_id": "S001",
        "raw_task": "I want to learn Decision Trees",
        "learner_state": LearnerState(student_id="S001"),
    }
    values.update(kwargs)
    return OrchestratorState(**values)


def test_graph_construction_contains_compilable_workflow():
    orchestrator = make_orchestrator([agent_response(mastery=MasteryStatus.MASTERED)])
    assert orchestrator.build() is not None


def test_normal_learning_flow_executes_end_to_end():
    orchestrator = make_orchestrator(
        [agent_response(next_action=NextAction.COMPLETE, mastery=MasteryStatus.MASTERED)]
    )
    result = orchestrator.invoke(initial_state())
    assert result.final_response["topic"] == "Decision Trees"
    assert result.final_response["status"] == "mastered"
    assert result.iteration == 1


def test_task_understanding_flows_into_planning():
    orchestrator = make_orchestrator([agent_response(mastery=MasteryStatus.MASTERED)])
    understood = orchestrator.understand_task(initial_state().model_dump(mode="json"))
    planned = orchestrator.create_plan(understood)
    assert planned["structured_task"]["topic"] == "Decision Trees"
    assert planned["current_plan"]["objective"] == "Understand Decision Trees"


def test_planning_flows_into_routing():
    orchestrator = make_orchestrator([agent_response(mastery=MasteryStatus.MASTERED)])
    state = orchestrator.create_plan(
        orchestrator.understand_task(initial_state().model_dump(mode="json"))
    )
    routed = orchestrator.route_agent(orchestrator.decide_next_action(state))
    assert routed["selected_agent"] == AgentType.TEACHING.value
    assert routed["pending_agent_request"]["agent_type"] == AgentType.TEACHING.value


def test_routing_reaches_mock_agent_and_state_manager():
    orchestrator = make_orchestrator(
        [agent_response(scores={"definition": 0.8}, mastery=MasteryStatus.MASTERED)]
    )
    result = orchestrator.invoke(initial_state())
    assert result.learner_state.last_agent == AgentType.TEACHING.value
    assert result.learner_state.concept_scores["definition"] == 0.8
    assert len(result.learner_state.progress_history) == 1


def test_remediation_loop_routes_weak_concept_then_terminates_on_mastery():
    orchestrator = make_orchestrator(
        [
            agent_response(scores={"entropy": 0.3}),
            agent_response(scores={"entropy": 0.9}, mastery=MasteryStatus.MASTERED),
        ],
        plans=[plan_payload(), plan_payload()],
    )
    result = orchestrator.invoke(initial_state())
    executor = orchestrator.agent_executor
    assert [request.agent_type for request in executor.requests] == [
        AgentType.TEACHING,
        AgentType.TEACHING,
    ]
    assert "entropy" in executor.requests[1].task
    assert result.final_response["status"] == "mastered"


def test_advancement_path_replans_and_routes_to_assessment():
    learner = LearnerState(
        student_id="S001",
        topic="Decision Trees",
        concept_scores={"definition": 0.95},
    )
    orchestrator = make_orchestrator(
        [agent_response(AgentType.ASSESSMENT, mastery=MasteryStatus.MASTERED)],
        plans=[plan_payload()],
    )
    result = orchestrator.invoke(initial_state(learner_state=learner))
    assert orchestrator.agent_executor.requests[0].agent_type == AgentType.ASSESSMENT
    assert result.final_response["status"] == "mastered"


def test_repeated_attempt_follows_repeat_action():
    orchestrator = make_orchestrator(
        [
            agent_response(next_action=NextAction.REPEAT),
            agent_response(next_action=NextAction.COMPLETE, mastery=MasteryStatus.MASTERED),
        ],
        plans=[plan_payload(), plan_payload()],
    )
    result = orchestrator.invoke(initial_state())
    assert result.iteration == 2
    assert len(orchestrator.agent_executor.requests) == 2


def test_weakness_driven_routing_is_preserved_in_graph():
    learner = LearnerState(
        student_id="S001", topic="Decision Trees", weak_concepts=["entropy"]
    )
    orchestrator = make_orchestrator(
        [agent_response(mastery=MasteryStatus.MASTERED)], plans=[plan_payload()]
    )
    result = orchestrator.invoke(initial_state(learner_state=learner))
    request = orchestrator.agent_executor.requests[0]
    assert request.agent_type == AgentType.TEACHING
    assert "entropy" in request.task
    assert result.learner_state.weak_concepts == ["entropy"]


def test_invalid_task_is_rejected_before_graph_execution():
    with pytest.raises(ValidationError):
        initial_state(raw_task="")


def test_task_understanding_failure_generates_controlled_response():
    class FailingUnderstanding:
        def understand(self, raw_task):
            raise LLMGatewayError("provider failure")

    orchestrator = make_orchestrator([agent_response()])
    orchestrator.task_understanding = FailingUnderstanding()
    result = orchestrator.invoke(initial_state())
    assert result.final_response["status"] == "error"
    assert "task understanding failed" in result.errors[0]


def test_agent_failure_generates_controlled_response():
    orchestrator = make_orchestrator([RuntimeError("agent down")])
    result = orchestrator.invoke(initial_state())
    assert result.final_response["status"] == "error"
    assert any("agent execution failed" in error for error in result.errors)


def test_safe_iteration_limit_stops_repeating_graph():
    orchestrator = make_orchestrator(
        [agent_response(next_action=NextAction.REPEAT)], plans=[plan_payload()] * 5
    )
    result = orchestrator.invoke(initial_state(max_iterations=2))
    assert result.iteration == 2
    assert result.final_response["status"] == "error"
    assert any("iteration limit" in error for error in result.errors)


def test_final_response_is_concise_and_structured():
    orchestrator = make_orchestrator(
        [agent_response(mastery=MasteryStatus.MASTERED)]
    )
    result = orchestrator.invoke(initial_state())
    assert set(result.final_response) == {
        "session_id", "topic", "current_stage", "next_action", "selected_agent",
        "mastery_status", "status", "message",
    }


def test_ambiguous_task_terminates_with_clarification_instead_of_looping():
    gateway = FakeGateway(
        {
            "topic": None,
            "intent": None,
            "objective": None,
            "needs_clarification": True,
            "ambiguity": "Topic is missing.",
            "clarification_question": "What topic would you like to study?",
        },
        [],
    )
    orchestrator = MainOrchestrator(
        TaskUnderstanding(gateway),
        LearningPlanner(gateway),
        AgentRouter(),
        StateManager(),
        MockExecutor([agent_response()]),
    )

    result = orchestrator.invoke(initial_state())

    assert result.iteration == 0
    assert result.final_response["status"] == "needs_clarification"
    assert result.final_response["message"] == "What topic would you like to study?"
