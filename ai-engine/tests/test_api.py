import asyncio
import json
import os

# The managed runtime ships an incompatible Starlette TestClient stack. Use
# the app's local ASGI compatibility transport for deterministic unit tests.
os.environ["MASTRIX_USE_API_COMPAT"] = "1"

from app.agents.executor import DevelopmentAgentExecutor
from app.agents.interfaces import AgentRequest, AgentType
from app.api.app import create_app
from app.api.schemas import TaskResponse
from app.orchestrator.state import LearningStage, LearnerState, MasteryStatus, NextAction, OrchestratorState



class FakeOrchestrator:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.received = None

    def invoke(self, state):
        self.received = state
        if self.error:
            raise self.error
        return self.result


def result_state():
    return OrchestratorState(
        session_id="session-001",
        student_id="student-001",
        raw_task="I want to learn Decision Trees",
        topic="Decision Trees",
        current_stage=LearningStage.LEARNING,
        learner_state=LearnerState(
            student_id="student-001",
            topic="Decision Trees",
            current_stage=LearningStage.LEARNING,
            next_action=NextAction.CONTINUE,
            mastery_status=MasteryStatus.IN_PROGRESS,
        ),
        selected_agent="teaching",
        next_action=NextAction.CONTINUE,
        mastery_status=MasteryStatus.IN_PROGRESS,
        final_response={"status": "in_progress"},
    )


class ASGIResponse:
    def __init__(self, status_code, headers, body):
        self.status_code = status_code
        self.headers = {
            key.decode().lower(): value.decode() for key, value in headers
        }
        self.text = body.decode()

    def json(self):
        return json.loads(self.text)


class ASGIClient:
    """Small synchronous HTTP harness for this environment's TestClient issue."""

    def __init__(self, app):
        self.app = app

    def request(self, method, path, *, json_body=None, content=None, headers=None):
        body = content.encode() if isinstance(content, str) else content
        body = body if body is not None else (
            json.dumps(json_body).encode() if json_body is not None else b""
        )
        request_headers = [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]
        if json_body is not None:
            request_headers.append((b"content-type", b"application/json"))
        return asyncio.run(self._request(method, path, body, request_headers))

    async def _request(self, method, path, body, headers):
        messages = iter([{"type": "http.request", "body": body, "more_body": False}])
        output = []

        async def receive():
            return next(messages)

        async def send(message):
            output.append(message)

        await self.app(
            {
                "type": "http",
                "method": method,
                "path": path,
                "query_string": b"",
                "headers": headers,
                "scheme": "http",
                "server": ("test", 80),
                "http_version": "1.1",
            },
            receive,
            send,
        )
        start = output[0]
        body = b"".join(message.get("body", b"") for message in output[1:])
        return ASGIResponse(start["status"], start.get("headers", []), body)

    def get(self, path, *, headers=None):
        return self.request("GET", path, headers=headers)

    def post(self, path, *, json=None, content=None, headers=None):
        return self.request("POST", path, json_body=json, content=content, headers=headers)


def client(orchestrator=None, settings=None):
    return ASGIClient(create_app(orchestrator, settings=settings))


def test_health_endpoint():
    response = client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_valid_task_request_calls_existing_orchestrator():
    orchestrator = FakeOrchestrator(result_state())
    response = client(orchestrator).post(
        "/orchestrator/task",
        json={
            "student_id": "student-001",
            "session_id": "session-001",
            "task": "I want to learn Decision Trees",
        },
    )
    assert response.status_code == 200
    assert response.json()["topic"] == "Decision Trees"
    assert orchestrator.received.raw_task == "I want to learn Decision Trees"


def test_missing_task_returns_4xx():
    response = client(FakeOrchestrator(result_state())).post(
        "/orchestrator/task",
        json={"student_id": "student-001", "session_id": "session-001"},
    )
    assert 400 <= response.status_code < 500


def test_invalid_request_body_returns_4xx():
    response = client(FakeOrchestrator(result_state())).post(
        "/orchestrator/task", content="not-json", headers={"content-type": "application/json"}
    )
    assert 400 <= response.status_code < 500


def test_orchestrator_failure_returns_generic_5xx():
    response = client(FakeOrchestrator(error=RuntimeError("secret internal detail"))).post(
        "/orchestrator/task",
        json={"student_id": "student-001", "session_id": "session-001", "task": "Learn trees"},
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "orchestrator unavailable"}
    assert "secret" not in response.text


def test_response_conforms_to_public_schema():
    response = client(FakeOrchestrator(result_state())).post(
        "/orchestrator/task",
        json={"student_id": "student-001", "session_id": "session-001", "task": "Learn trees"},
    )
    parsed = TaskResponse.model_validate(response.json())
    assert parsed.session_id == "session-001"
    assert parsed.mastery_status == "in_progress"
    assert parsed.message == "Task processed."


def test_cors_allows_configured_local_frontend_only():
    response = client().get("/health", headers={"origin": "http://localhost:3000"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

    response = client().get("/health", headers={"origin": "https://untrusted.example"})
    assert "access-control-allow-origin" not in response.headers


def test_default_create_app_initializes_main_orchestrator_and_mock_executor():
    app = create_app()
    assert app.state.orchestrator is not None
    assert isinstance(app.state.orchestrator.agent_executor, DevelopmentAgentExecutor)


def test_development_agent_executor_contract():
    executor = DevelopmentAgentExecutor()
    request = AgentRequest(
        session_id="session-001",
        student_id="student-001",
        agent_type=AgentType.TEACHING,
        task="Explain Decision Trees",
        topic="Decision Trees",
    )
    response = executor.execute(request)
    assert response.request_id == request.request_id
    assert response.session_id == request.session_id
    assert response.agent_type == AgentType.TEACHING
    assert response.status == "success"
    assert response.result["execution_mode"] == "development_mock"
    assert response.result["executor"] == "DevelopmentAgentExecutor"
    assert "[Development/Mock Executor]" in response.summary
    assert response.concept_scores["Decision Trees"] == 0.85
    assert response.metadata["is_mock"] is True
    assert response.metadata["execution_mode"] == "development_mock"

