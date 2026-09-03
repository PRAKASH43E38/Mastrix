"""Adaptive learning-plan generation for the Main Orchestrator."""

import json

from app.llm.gateway import LLMGateway
from app.llm.prompts.learning_planner import LEARNING_PLANNER_SYSTEM_PROMPT
from app.orchestrator.state import (
    LearningPlan,
    LearningStage,
    LearnerState,
    MasteryStatus,
    NextAction,
    StructuredTask,
)


class LearningPlanner:
    """Create plans from task and state; never executes the plan."""

    def __init__(self, gateway: LLMGateway) -> None:
        self.gateway = gateway

    def plan(self, task: StructuredTask, learner_state: LearnerState) -> LearningPlan:
        """Generate and validate an adaptive plan through the central gateway."""

        planning_input = {
            "structured_task": task.model_dump(mode="json"),
            "learner_state": learner_state.model_dump(mode="json"),
        }
        plan = self.gateway.generate_structured(
            json.dumps(planning_input),
            LearningPlan,
            system_prompt=LEARNING_PLANNER_SYSTEM_PROMPT,
        )
        return self._apply_state_guardrails(plan, learner_state)

    @staticmethod
    def _apply_state_guardrails(
        plan: LearningPlan, learner_state: LearnerState
    ) -> LearningPlan:
        """Keep critical state decisions deterministic after LLM validation."""

        if learner_state.weak_concepts:
            concepts = list(dict.fromkeys([*learner_state.weak_concepts, *plan.concepts]))
            return plan.model_copy(
                update={
                    "current_stage": LearningStage.REMEDIATION,
                    "concepts": concepts,
                    "recommended_action": NextAction.REMEDIATE.value,
                    "required_intervention": "Targeted remediation of weak concepts",
                }
            )

        scores_are_strong = bool(learner_state.concept_scores) and all(
            score >= 0.8 for score in learner_state.concept_scores.values()
        )
        if learner_state.mastery_status is MasteryStatus.MASTERED or scores_are_strong:
            return plan.model_copy(
                update={"recommended_action": NextAction.ADVANCE.value}
            )

        return plan
