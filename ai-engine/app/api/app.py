"""Thin FastAPI transport layer for the Main Orchestrator."""

import os

try:  # pragma: no cover - real FastAPI is used in normal installations
    if os.getenv("MASTRIX_USE_API_COMPAT") == "1":
        raise ImportError
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:  # pragma: no cover - restricted local environments only
    from ._fastapi_compat import CORSMiddleware, FastAPI, HTTPException, Request

from app.agents.executor import DevelopmentAgentExecutor
from app.api.factory import build_orchestrator
from app.config import load_settings
from app.orchestrator.state import LearnerState, OrchestratorState

from .schemas import HealthResponse, TaskRequest, TaskResponse


def create_app(orchestrator=None, *, settings=None) -> FastAPI:
    """Create an app with an injectable orchestrator for production/tests."""

    config = settings or load_settings()
    api = FastAPI(title="MastriX Main Orchestrator", version="1.0.0")
    if orchestrator is None:
        executor = DevelopmentAgentExecutor()
        orchestrator = build_orchestrator(executor, settings=config)
    api.state.orchestrator = orchestrator
    api.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @api.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="healthy")

    @api.post("/orchestrator/task", response_model=TaskResponse)
    def submit_task(payload: TaskRequest, request: Request) -> TaskResponse:
        controller = request.app.state.orchestrator
        if controller is None:
            raise HTTPException(status_code=503, detail="orchestrator unavailable")

        state = OrchestratorState(
            session_id=payload.session_id,
            student_id=payload.student_id,
            raw_task=payload.task,
            learner_state=LearnerState(student_id=payload.student_id),
        )
        try:
            result = controller.invoke(state)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="orchestrator unavailable") from exc

        if result.errors:
            raise HTTPException(status_code=500, detail="orchestrator unavailable")

        final = result.final_response
        return TaskResponse(
            session_id=result.session_id,
            topic=result.topic,
            current_stage=result.current_stage.value,
            selected_agent=result.selected_agent,
            next_action=result.next_action.value,
            mastery_status=result.mastery_status.value,
            status=str(final.get("status", "in_progress")),
            message="Task processed.",
        )

    return api


app = create_app()
