"""Pydantic models for the serializable orchestrator state."""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


ConceptScores = dict[str, Annotated[float, Field(ge=0.0, le=1.0)]]


class LearningStage(StrEnum):
    """Stages are data used for routing, not a fixed execution pipeline."""

    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    LEARNING = "learning"
    PRACTICE = "practice"
    ASSESSMENT = "assessment"
    REMEDIATION = "remediation"
    COMPLETE = "complete"


class NextAction(StrEnum):
    CONTINUE = "continue"
    REMEDIATE = "remediate"
    REPEAT = "repeat"
    INCREASE_DIFFICULTY = "increase_difficulty"
    DECREASE_DIFFICULTY = "decrease_difficulty"
    ADVANCE = "advance"
    REQUEST_CLARIFICATION = "request_clarification"
    COMPLETE = "complete"


class MasteryStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    MASTERED = "mastered"
    BLOCKED = "blocked"


class ProgressRecord(BaseModel):
    """Small observable state snapshot retained after an agent result."""

    model_config = ConfigDict(extra="forbid")

    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent: str
    stage: LearningStage
    concept_scores: ConceptScores = Field(default_factory=dict)
    weaknesses: list[str] = Field(default_factory=list)
    next_action: NextAction
    mastery_status: MasteryStatus
    summary: str | None = None


class StructuredTask(BaseModel):
    """Natural-language task normalized for planning and routing."""

    model_config = ConfigDict(extra="forbid")

    topic: str | None = Field(default=None, min_length=1)
    intent: str | None = Field(default=None, min_length=1)
    objective: str | None = Field(default=None, min_length=1)
    requested_outcome: str | None = None
    constraints: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    ambiguity: str | None = None
    needs_clarification: bool = False
    clarification_question: str | None = None

    @model_validator(mode="after")
    def require_task_fields_when_unambiguous(self) -> "StructuredTask":
        if not self.needs_clarification and not all(
            (self.topic, self.intent, self.objective)
        ):
            raise ValueError(
                "topic, intent, and objective are required for an unambiguous task"
            )
        return self


class LearningPlan(BaseModel):
    """Planner output; it does not execute, teach, or grade any agent task."""

    model_config = ConfigDict(extra="forbid")

    objective: str = Field(min_length=1)
    current_stage: LearningStage = LearningStage.LEARNING
    steps: list[str] = Field(default_factory=list)
    stages: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    recommended_action: str = Field(min_length=1)
    target_difficulty: str | None = None
    required_intervention: str | None = None
    rationale: str | None = None
    decision_summary: str | None = None


class LearnerState(BaseModel):
    """Lightweight learner progress state carried between decisions."""

    model_config = ConfigDict(extra="forbid")

    student_id: str = Field(min_length=1)
    topic: str | None = None
    current_stage: LearningStage = LearningStage.UNDERSTANDING
    concept_scores: ConceptScores = Field(default_factory=dict)
    weak_concepts: list[str] = Field(default_factory=list)
    last_agent: str | None = None
    last_result: dict[str, Any] = Field(default_factory=dict)
    next_action: NextAction = NextAction.CONTINUE
    mastery_status: MasteryStatus = MasteryStatus.IN_PROGRESS
    progress_history: list[ProgressRecord] = Field(default_factory=list)


class OrchestratorState(BaseModel):
    """Complete state passed through the future StateGraph."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    student_id: str = Field(min_length=1)
    raw_task: str = Field(min_length=1)
    structured_task: StructuredTask | None = None
    topic: str | None = None
    learning_objective: str | None = None
    current_stage: LearningStage = LearningStage.UNDERSTANDING
    learner_state: LearnerState
    current_plan: LearningPlan | None = None
    selected_agent: str | None = None
    agent_task: str | None = None
    agent_result: dict[str, Any] = Field(default_factory=dict)
    concept_scores: ConceptScores = Field(default_factory=dict)
    weaknesses: list[str] = Field(default_factory=list)
    next_action: NextAction = NextAction.CONTINUE
    mastery_status: MasteryStatus = MasteryStatus.IN_PROGRESS
    errors: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    routing_decision: dict[str, Any] | None = None
    pending_agent_request: dict[str, Any] | None = None
    iteration: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=5, ge=1)
    final_response: dict[str, Any] = Field(default_factory=dict)
