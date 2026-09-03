"""Execution boundary for future downstream multi-agents."""

from typing import Protocol

from app.agents.interfaces import AgentRequest, AgentResponse
from app.orchestrator.state import MasteryStatus, NextAction


class AgentExecutor(Protocol):
    """A caller-supplied adapter that executes one agent request."""

    def execute(self, request: AgentRequest) -> AgentResponse:
        """Return a validated response; no agent implementation is provided."""


class DevelopmentAgentExecutor:
    """Minimal Development/Mock AgentExecutor implementation.

    Provides a deterministic mock response for development and testing when no real
    downstream multi-agent system is connected.
    """

    def execute(self, request: AgentRequest) -> AgentResponse:
        """Return a validated AgentResponse clearly marked as Development/Mock."""

        topic = request.topic or "core_concept"
        return AgentResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            agent_type=request.agent_type,
            status="success",
            result={
                "execution_mode": "development_mock",
                "executor": "DevelopmentAgentExecutor",
                "message": f"Development/Mock execution for task: {request.task}",
                "agent_type": request.agent_type.value,
            },
            summary=f"[Development/Mock Executor] Executed task for agent '{request.agent_type.value}': {request.task}",
            concept_scores={topic: 0.85},
            weaknesses=[],
            next_action=NextAction.COMPLETE,
            mastery_status=MasteryStatus.MASTERED,
            metadata={
                "execution_mode": "development_mock",
                "is_mock": True,
                "executor_class": "DevelopmentAgentExecutor",
                "note": "Development/Mock response, not a real downstream multi-agent execution.",
            },
        )


MockAgentExecutor = DevelopmentAgentExecutor


