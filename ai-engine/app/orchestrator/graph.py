"""LangGraph Main Orchestrator integration."""

from collections.abc import Callable
from typing import Any

try:  # pragma: no cover - fallback is used only when dependency is absent
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover
    from app.orchestrator._graph_compat import END, START, StateGraph

from app.agents.executor import AgentExecutor
from app.agents.interfaces import AgentRequest, AgentResponse
from app.orchestrator.planner import LearningPlanner
from app.orchestrator.router import AgentRouter
from app.orchestrator.state import (
    LearningStage,
    MasteryStatus,
    NextAction,
    OrchestratorState,
)
from app.orchestrator.state_manager import StateManager
from app.orchestrator.task_understanding import TaskUnderstanding


class MainOrchestrator:
    """Build and run the graph from existing Phase 1–8 components."""

    def __init__(
        self,
        task_understanding: TaskUnderstanding,
        planner: LearningPlanner,
        router: AgentRouter,
        state_manager: StateManager,
        agent_executor: AgentExecutor | Callable[[AgentRequest], AgentResponse],
    ) -> None:
        self.task_understanding = task_understanding
        self.planner = planner
        self.router = router
        self.state_manager = state_manager
        self.agent_executor = agent_executor

    def build(self):
        graph = StateGraph(OrchestratorState)
        graph.add_node("receive_task", self.receive_task)
        graph.add_node("understand_task", self.understand_task)
        graph.add_node("create_plan", self.create_plan)
        graph.add_node("decide_next_action", self.decide_next_action)
        graph.add_node("route_agent", self.route_agent)
        graph.add_node("receive_agent_result", self.receive_agent_result)
        graph.add_node("update_learner_state", self.update_learner_state)
        graph.add_node("evaluate_progress", self.evaluate_progress)
        graph.add_node("generate_response", self.generate_response)
        graph.add_edge(START, "receive_task")
        graph.add_edge("receive_task", "understand_task")
        graph.add_edge("understand_task", "create_plan")
        graph.add_edge("create_plan", "decide_next_action")
        graph.add_edge("decide_next_action", "route_agent")
        graph.add_edge("route_agent", "receive_agent_result")
        graph.add_edge("receive_agent_result", "update_learner_state")
        graph.add_edge("update_learner_state", "evaluate_progress")
        graph.add_conditional_edges(
            "evaluate_progress",
            self._next_branch,
            {
                "remediation": "create_plan",
                "advance": "create_plan",
                "repeat": "route_agent",
                "complete": "generate_response",
            },
        )
        graph.add_edge("generate_response", END)
        return graph.compile()

    def invoke(self, state: OrchestratorState) -> OrchestratorState:
        if state is None:
            raise ValueError("orchestrator state is required")
        result = self.build().invoke(state.model_dump(mode="json"))
        return OrchestratorState.model_validate(result)

    @staticmethod
    def _state(value: dict[str, Any] | OrchestratorState) -> OrchestratorState:
        return value if isinstance(value, OrchestratorState) else OrchestratorState.model_validate(value)

    def receive_task(self, value):
        return self._state(value).model_dump(mode="json")

    def understand_task(self, value):
        state = self._state(value)
        if state.structured_task is not None:
            return state.model_dump(mode="json")
        try:
            structured = self.task_understanding.understand(state.raw_task)
            state = state.model_copy(
                update={
                    "structured_task": structured,
                    "topic": structured.topic,
                    "learning_objective": structured.objective,
                    "learner_state": state.learner_state.model_copy(
                        update={"topic": structured.topic}
                    ),
                }
            )
        except Exception as exc:
            state = self._record_error(state, "task understanding failed", exc)
            state = state.model_copy(update={"next_action": NextAction.REQUEST_CLARIFICATION})
        return state.model_dump(mode="json")

    def create_plan(self, value):
        state = self._state(value)
        if state.structured_task is None or state.structured_task.needs_clarification:
            return state.model_dump(mode="json")
        try:
            plan = self.planner.plan(state.structured_task, state.learner_state)
            state = state.model_copy(update={"current_plan": plan})
        except Exception as exc:
            state = self._record_error(state, "planning failed", exc)
        return state.model_dump(mode="json")

    def decide_next_action(self, value):
        state = self._state(value)
        if state.errors or state.current_plan is None:
            return state.model_dump(mode="json")
        action = state.current_plan.recommended_action.lower()
        mapping = (
            ("remediat", NextAction.REMEDIATE),
            ("repeat", NextAction.REPEAT),
            ("advance", NextAction.ADVANCE),
            ("complete", NextAction.COMPLETE),
        )
        for marker, next_action in mapping:
            if marker in action:
                state = state.model_copy(update={"next_action": next_action})
                break
        return state.model_dump(mode="json")

    def route_agent(self, value):
        state = self._state(value)
        if state.errors or state.structured_task is None or state.current_plan is None:
            return state.model_dump(mode="json")
        try:
            decision = self.router.route(
                state.structured_task, state.current_plan, state.learner_state
            )
            request = self.router.build_request(
                decision, state.structured_task, state.learner_state,
                session_id=state.session_id,
            )
            state = state.model_copy(update={
                "selected_agent": decision.selected_agent.value,
                "agent_task": decision.agent_task,
                "routing_decision": decision.model_dump(mode="json"),
                "pending_agent_request": request.model_dump(mode="json"),
            })
        except Exception as exc:
            state = self._record_error(state, "routing failed", exc)
        return state.model_dump(mode="json")

    def receive_agent_result(self, value):
        state = self._state(value)
        if state.errors or state.pending_agent_request is None:
            return state.model_dump(mode="json")
        try:
            request = AgentRequest.model_validate(state.pending_agent_request)
            if hasattr(self.agent_executor, "execute"):
                response = self.agent_executor.execute(request)
            else:
                response = self.agent_executor(request)
            response = AgentResponse.model_validate(response)
        except Exception as exc:
            response = AgentResponse(
                request_id=state.pending_agent_request["request_id"],
                session_id=state.session_id,
                agent_type=state.pending_agent_request["agent_type"],
                status="failure",
                error="mock agent execution failed",
            )
            state = self._record_error(state, "agent execution failed", exc)
        state = state.model_copy(update={
            "agent_result": response.model_dump(mode="json"),
            "iteration": state.iteration + 1,
        })
        return state.model_dump(mode="json")

    def update_learner_state(self, value):
        state = self._state(value)
        if not state.agent_result:
            return state.model_dump(mode="json")
        try:
            response = AgentResponse.model_validate(state.agent_result)
            state = self.state_manager.update_orchestrator_state(state, response)
        except Exception as exc:
            state = self._record_error(state, "state update failed", exc)
        return state.model_dump(mode="json")

    def evaluate_progress(self, value):
        state = self._state(value)
        if state.structured_task is not None and state.structured_task.needs_clarification:
            state = state.model_copy(update={"next_action": NextAction.REQUEST_CLARIFICATION})
        if state.current_stage is LearningStage.COMPLETE or state.mastery_status is MasteryStatus.MASTERED:
            state = state.model_copy(update={"next_action": NextAction.COMPLETE})
        if state.iteration >= state.max_iterations and state.next_action is not NextAction.COMPLETE:
            state = self._record_error(state, "safe iteration limit reached")
        return state.model_dump(mode="json")

    def generate_response(self, value):
        state = self._state(value)
        status = "error" if state.errors else (
            "needs_clarification" if state.structured_task and state.structured_task.needs_clarification else (
            "mastered" if state.mastery_status is MasteryStatus.MASTERED else "in_progress"
            )
        )
        message = "Unable to process task." if state.errors else (
            state.structured_task.clarification_question
            if state.structured_task and state.structured_task.needs_clarification
            else "Task processed."
        )
        final_response = {
            "session_id": state.session_id,
            "topic": state.topic,
            "current_stage": state.current_stage.value,
            "next_action": state.next_action.value,
            "selected_agent": state.selected_agent,
            "mastery_status": state.mastery_status.value,
            "status": status,
            "message": message,
        }
        return state.model_copy(update={"final_response": final_response}).model_dump(mode="json")

    @staticmethod
    def _next_branch(value):
        state = value if isinstance(value, OrchestratorState) else OrchestratorState.model_validate(value)
        if (
            state.errors
            or (state.structured_task is not None and state.structured_task.needs_clarification)
            or state.next_action is NextAction.COMPLETE
            or state.mastery_status is MasteryStatus.MASTERED
        ):
            return "complete"
        if state.next_action is NextAction.REMEDIATE:
            return "remediation"
        if state.next_action is NextAction.ADVANCE:
            return "advance"
        return "repeat"

    @staticmethod
    def _record_error(state: OrchestratorState, message: str, exc: Exception | None = None):
        detail = f": {type(exc).__name__}" if exc else ""
        return state.model_copy(update={"errors": [*state.errors, message + detail]})
