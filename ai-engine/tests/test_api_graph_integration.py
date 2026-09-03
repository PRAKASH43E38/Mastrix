import os

os.environ["MASTRIX_USE_API_COMPAT"] = "1"

from app.agents.interfaces import AgentResponse, AgentType
from app.api.app import create_app
from app.orchestrator.graph import MainOrchestrator
from app.orchestrator.planner import LearningPlanner
from app.orchestrator.router import AgentRouter
from app.orchestrator.state import MasteryStatus, NextAction, StructuredTask
from app.orchestrator.state_manager import StateManager
from app.orchestrator.task_understanding import TaskUnderstanding
from test_api import ASGIClient


class Gateway:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate_structured(self, prompt, response_model, *, system_prompt=None, **kwargs):
        return response_model.model_validate(self.responses.pop(0))


class Executor:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return response.model_copy(update={"request_id": request.request_id, "session_id": request.session_id})


def task_payload():
    return {"topic": "Decision Trees", "intent": "learn", "objective": "understand the concept"}


def plan_payload():
    return {
        "objective": "Understand Decision Trees",
        "current_stage": "learning",
        "steps": ["Study foundations"],
        "stages": ["foundations", "practice"],
        "concepts": ["nodes", "splits"],
        "recommended_action": "explain fundamentals",
    }


def response(*, score=None, mastery=None):
    return AgentResponse(
        request_id="placeholder",
        session_id="placeholder",
        agent_type=AgentType.TEACHING,
        status="success",
        concept_scores={"entropy": score} if score is not None else {},
        next_action=NextAction.COMPLETE if mastery else None,
        mastery_status=mastery,
    )


def build(responses, plans=2):
    gateway = Gateway([task_payload(), *[plan_payload() for _ in range(plans)]])
    executor = Executor(responses)
    orchestrator = MainOrchestrator(
        TaskUnderstanding(gateway),
        LearningPlanner(gateway),
        AgentRouter(),
        StateManager(),
        executor,
    )
    return create_app(orchestrator), executor


def submit(app):
    return ASGIClient(app).post(
        "/orchestrator/task",
        json={
            "student_id": "student-001",
            "session_id": "session-001",
            "task": "I want to learn Decision Trees",
        },
    )


def test_api_runs_complete_orchestrator_flow_with_mock_agent():
    app, executor = build([response(mastery=MasteryStatus.MASTERED)], plans=1)
    result = submit(app)

    assert result.status_code == 200
    assert result.json()["selected_agent"] == AgentType.TEACHING.value
    assert result.json()["mastery_status"] == MasteryStatus.MASTERED.value
    assert len(executor.requests) == 1


def test_api_runs_remediation_loop_before_termination():
    app, executor = build(
        [response(score=0.3), response(score=0.9, mastery=MasteryStatus.MASTERED)],
        plans=2,
    )
    result = submit(app)

    assert result.status_code == 200
    assert result.json()["status"] == "mastered"
    assert len(executor.requests) == 2
    assert "entropy" in executor.requests[1].task

