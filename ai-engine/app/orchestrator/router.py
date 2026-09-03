"""Deterministic, state-based routing to downstream agent contracts."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agents.interfaces import AgentRequest, AgentType
from app.orchestrator.state import LearningPlan, LearningStage, LearnerState, StructuredTask


class RoutingDecision(BaseModel):
    """A concise routing decision; it contains no hidden reasoning."""

    model_config = ConfigDict(extra="forbid")

    selected_agent: AgentType
    reason: str = Field(min_length=1, max_length=240)
    agent_task: str = Field(min_length=1)
    stage: LearningStage
    priority: int = Field(ge=1, le=5)


class AgentRouter:
    """Select an agent and formulate its bounded task, without executing it."""

    def route(
        self,
        task: StructuredTask,
        plan: LearningPlan,
        learner_state: LearnerState,
    ) -> RoutingDecision:
        """Route using explicit state signals and non-linear evidence."""

        if task is None or plan is None or learner_state is None:
            raise ValueError("task, plan, and learner state are required for routing")
        if task.needs_clarification:
            raise ValueError("cannot route a task that needs clarification")
        if not task.topic or not task.objective:
            raise ValueError("task topic and objective are required for routing")

        agent, reason, priority = self._select_agent(task, plan, learner_state)
        return RoutingDecision(
            selected_agent=agent,
            reason=reason,
            agent_task=self._build_agent_task(agent, task, plan, learner_state),
            stage=learner_state.current_stage,
            priority=priority,
        )

    def build_request(
        self,
        decision: RoutingDecision,
        task: StructuredTask,
        learner_state: LearnerState,
        *,
        session_id: str,
    ) -> AgentRequest:
        """Create the existing downstream request contract from a decision."""

        return AgentRequest(
            session_id=session_id,
            student_id=learner_state.student_id,
            agent_type=decision.selected_agent,
            task=decision.agent_task,
            topic=task.topic,
            learning_objective=task.objective,
            learner_state=learner_state,
            context={
                "stage": decision.stage.value,
                "priority": decision.priority,
                "routing_reason": decision.reason,
            },
        )

    def _select_agent(
        self,
        task: StructuredTask,
        plan: LearningPlan,
        learner_state: LearnerState,
    ) -> tuple[AgentType, str, int]:
        evidence = learner_state.last_result
        # Conflicting or repeated evidence needs interpretation before another
        # content step is selected.
        if self._has_signal(evidence, "conflicting", "repeated", "inconsistent"):
            return (
                AgentType.CONCEPTUAL_CLARITY_FEEDBACK,
                "Resolve repeated or conflicting learning evidence",
                1,
            )

        if learner_state.weak_concepts:
            weakness_text = " ".join(learner_state.weak_concepts).lower()
            if self._contains(weakness_text, "reason", "logic", "inference"):
                return AgentType.CRITICAL_THINKING, "Address reasoning weaknesses", 1
            if self._contains(weakness_text, "application", "practice", "implementation", "coding"):
                return AgentType.PRACTICAL_APPLICATION, "Address application weaknesses", 1
            if self._contains(weakness_text, "assessment", "performance", "score"):
                return AgentType.ASSESSMENT, "Measure the identified performance weakness", 1
            return AgentType.TEACHING, "Remediate identified conceptual weaknesses", 1

        # An explicit plan action is a strong signal, but remains a routing
        # instruction rather than content generation.
        action = plan.recommended_action.lower()
        if self._contains(action, "clarif"):
            return AgentType.CONCEPTUAL_CLARITY_FEEDBACK, "Resolve the plan's clarity requirement", 1
        if self._contains(action, "research", "foundational", "prerequisite"):
            return AgentType.RESEARCH, "Prepare the plan's foundational material", 2
        if self._contains(action, "explain", "teach", "learn", "concept", "remediat"):
            return AgentType.TEACHING, "Deliver the plan's conceptual learning step", 2
        if self._contains(action, "reason", "think", "logic"):
            return AgentType.CRITICAL_THINKING, "Validate the plan's reasoning objective", 2
        if self._contains(action, "practice", "apply", "implement"):
            return AgentType.PRACTICAL_APPLICATION, "Run the plan's practical application step", 2
        if self._contains(action, "assess", "test", "measure", "advance"):
            return AgentType.ASSESSMENT, "Measure progress for the planned decision", 2

        # Previous evidence can move the learner to a different branch instead
        # of repeating a fixed stage sequence.
        score = evidence.get("score", evidence.get("application_score"))
        if isinstance(score, (int, float)) and score < 0.7:
            if learner_state.last_agent in {AgentType.APPLICATION.value, "application"}:
                return AgentType.PRACTICAL_APPLICATION, "Strengthen low application performance", 1
            return AgentType.TEACHING, "Remediate low performance evidence", 1
        if learner_state.last_agent == AgentType.TEACHING.value and self._has_signal(
            evidence, "needs_practice", "practice_needed"
        ):
            return AgentType.PRACTICAL_APPLICATION, "Apply concepts after the previous teaching step", 2

        intent = (task.intent or "").lower()
        if self._contains(intent, "research", "discover"):
            return AgentType.RESEARCH, "Gather foundations for the requested topic", 2
        if self._contains(intent, "explain", "learn", "understand"):
            return AgentType.TEACHING, "Build understanding for the requested objective", 2
        if self._contains(intent, "reason", "think"):
            return AgentType.CRITICAL_THINKING, "Validate the requested reasoning objective", 2
        if self._contains(intent, "practice", "apply", "implement"):
            return AgentType.PRACTICAL_APPLICATION, "Practice the requested skill", 2
        if self._contains(intent, "test", "assess", "quiz"):
            return AgentType.ASSESSMENT, "Measure the requested learning performance", 2

        if learner_state.current_stage is LearningStage.ASSESSMENT:
            return AgentType.ASSESSMENT, "Measure progress at the current stage", 2
        if learner_state.current_stage in {LearningStage.UNDERSTANDING, LearningStage.REMEDIATION}:
            return AgentType.TEACHING, "Establish or repair foundational understanding", 2
        return AgentType.TEACHING, "Start the next learning step for the objective", 3

    @staticmethod
    def _build_agent_task(
        agent: AgentType,
        task: StructuredTask,
        plan: LearningPlan,
        learner_state: LearnerState,
    ) -> str:
        topic = task.topic
        concepts = learner_state.weak_concepts or plan.concepts
        focus = f" Focus on: {', '.join(concepts)}." if concepts else ""
        labels = {
            AgentType.RESEARCH: f"Research foundational material for {topic}.{focus}",
            AgentType.TEACHING: f"Explain the relevant concepts for {topic}.{focus}",
            AgentType.CRITICAL_THINKING: f"Validate reasoning about {topic}.{focus}",
            AgentType.REASONING: f"Validate reasoning about {topic}.{focus}",
            AgentType.APPLICATION: f"Create an application task for {topic}.{focus}",
            AgentType.PRACTICAL_APPLICATION: f"Create practical work for {topic}.{focus}",
            AgentType.ASSESSMENT: f"Measure learning progress on {topic}.{focus}",
            AgentType.CONCEPTUAL_CLARITY_FEEDBACK: f"Analyze conceptual clarity for {topic}.{focus}",
        }
        return labels.get(agent, f"Handle the next learning task for {topic}.")

    @staticmethod
    def _contains(value: str, *terms: str) -> bool:
        return any(term in value for term in terms)

    @staticmethod
    def _evidence_text(evidence: dict[str, Any]) -> str:
        return " ".join(
            [str(key).lower() for key in evidence]
            + [str(value).lower() for value in evidence.values()]
        )

    @classmethod
    def _has_signal(cls, evidence: dict[str, Any], *terms: str) -> bool:
        return cls._contains(cls._evidence_text(evidence), *terms)
