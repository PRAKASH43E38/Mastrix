"""Dependency factory for wiring the API to MainOrchestrator."""

from collections.abc import Callable

from app.agents.executor import AgentExecutor
from app.agents.interfaces import AgentRequest, AgentResponse
from app.config import AppSettings, load_settings
from app.llm.gateway import LLMGateway
from app.orchestrator.graph import MainOrchestrator
from app.orchestrator.planner import LearningPlanner
from app.orchestrator.router import AgentRouter
from app.orchestrator.state_manager import StateManager
from app.orchestrator.task_understanding import TaskUnderstanding


def build_orchestrator(
    agent_executor: AgentExecutor | Callable[[AgentRequest], AgentResponse],
    settings: AppSettings | None = None,
) -> MainOrchestrator:
    """Build the controller; downstream execution must be supplied by the caller."""

    gateway = LLMGateway(settings or load_settings())
    return MainOrchestrator(
        TaskUnderstanding(gateway),
        LearningPlanner(gateway),
        AgentRouter(),
        StateManager(),
        agent_executor,
    )

