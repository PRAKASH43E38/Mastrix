"""Deterministic learner-state updates from structured agent responses."""

from datetime import datetime, timezone

from app.agents.interfaces import AgentResponse, AgentType
from app.orchestrator.state import (
    LearningStage,
    LearnerState,
    MasteryStatus,
    NextAction,
    OrchestratorState,
    ProgressRecord,
)


class StateManager:
    """Update state only; it does not route, teach, grade, or call an LLM."""

    def update_learner_state(
        self, learner_state: LearnerState, agent_response: AgentResponse
    ) -> LearnerState:
        """Merge one validated agent result into learner state."""

        if learner_state is None or agent_response is None:
            raise ValueError("learner state and agent response are required")

        scores = dict(learner_state.concept_scores)
        scores.update(agent_response.concept_scores)
        weaknesses = list(dict.fromkeys([
            *learner_state.weak_concepts,
            *agent_response.weaknesses,
        ]))

        # Low scores create weaknesses; demonstrated improvement resolves only
        # the corresponding weakness and leaves unrelated progress untouched.
        for concept, score in agent_response.concept_scores.items():
            if score < 0.7 and concept not in weaknesses:
                weaknesses.append(concept)
            elif score >= 0.8 and concept not in agent_response.weaknesses:
                weaknesses = [item for item in weaknesses if item != concept]

        stage = self._next_stage(learner_state, agent_response, weaknesses)
        next_action = self._next_action(learner_state, agent_response, weaknesses)
        mastery_status = agent_response.mastery_status or learner_state.mastery_status
        record = ProgressRecord(
            agent=agent_response.agent_type.value,
            stage=stage,
            concept_scores=dict(agent_response.concept_scores),
            weaknesses=list(weaknesses),
            next_action=next_action,
            mastery_status=mastery_status,
            summary=agent_response.summary,
            occurred_at=agent_response.responded_at,
        )

        return learner_state.model_copy(
            update={
                "topic": learner_state.topic,
                "current_stage": stage,
                "concept_scores": scores,
                "weak_concepts": weaknesses,
                "last_agent": agent_response.agent_type.value,
                "last_result": dict(agent_response.result),
                "next_action": next_action,
                "mastery_status": mastery_status,
                "progress_history": [*learner_state.progress_history, record],
            }
        )

    def update_orchestrator_state(
        self, state: OrchestratorState, agent_response: AgentResponse
    ) -> OrchestratorState:
        """Update nested learner state and synchronized top-level fields."""

        if state is None or agent_response is None:
            raise ValueError("orchestrator state and agent response are required")
        learner_state = self.update_learner_state(state.learner_state, agent_response)
        return state.model_copy(
            update={
                "learner_state": learner_state,
                "topic": learner_state.topic,
                "current_stage": learner_state.current_stage,
                "concept_scores": learner_state.concept_scores,
                "weaknesses": learner_state.weak_concepts,
                "next_action": learner_state.next_action,
                "mastery_status": learner_state.mastery_status,
                "agent_result": learner_state.last_result,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    @staticmethod
    def _next_action(
        learner_state: LearnerState,
        response: AgentResponse,
        weaknesses: list[str],
    ) -> NextAction:
        if response.status.lower() not in {"success", "succeeded", "complete"} or response.error:
            return NextAction.REPEAT
        if response.next_action is not None:
            return response.next_action
        if weaknesses:
            return NextAction.REMEDIATE
        return learner_state.next_action

    @classmethod
    def _next_stage(
        cls,
        learner_state: LearnerState,
        response: AgentResponse,
        weaknesses: list[str],
    ) -> LearningStage:
        metadata = response.metadata
        requested_stage = metadata.get("next_stage", metadata.get("stage"))
        if requested_stage:
            try:
                return LearningStage(requested_stage)
            except ValueError:
                pass
        if response.next_action is NextAction.REMEDIATE or weaknesses:
            return LearningStage.REMEDIATION
        if response.agent_type is AgentType.ASSESSMENT:
            return LearningStage.ASSESSMENT
        if response.agent_type in {
            AgentType.APPLICATION,
            AgentType.PRACTICAL_APPLICATION,
            AgentType.CRITICAL_THINKING,
        }:
            return LearningStage.PRACTICE
        if response.agent_type in {AgentType.RESEARCH, AgentType.TEACHING} and learner_state.current_stage in {
            LearningStage.UNDERSTANDING,
            LearningStage.PLANNING,
        }:
            return LearningStage.LEARNING
        return learner_state.current_stage
