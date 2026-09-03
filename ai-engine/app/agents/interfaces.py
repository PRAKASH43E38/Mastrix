"""Stable contracts between the orchestrator and downstream agents."""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.orchestrator.state import (
    ConceptScores,
    LearnerState,
    MasteryStatus,
    NextAction,
)


class AgentType(StrEnum):
    """Routing identifiers only; no agent implementations live here."""

    RESEARCH = "research"
    TEACHING = "teaching"
    CRITICAL_THINKING = "critical_thinking"
    REASONING = "critical_thinking"
    APPLICATION = "application"
    PRACTICAL_APPLICATION = "practical_application"
    ASSESSMENT = "assessment"
    CONCEPTUAL_CLARITY_FEEDBACK = "conceptual_clarity_feedback"
    UNSPECIFIED = "unspecified"


class AgentRequest(BaseModel):
    """Task sent by the orchestrator to one downstream agent."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    session_id: str = Field(min_length=1)
    student_id: str = Field(min_length=1)
    agent_type: AgentType
    task: str = Field(min_length=1)
    topic: str | None = None
    learning_objective: str | None = None
    learner_state: LearnerState | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Validated response returned by a downstream agent adapter."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    agent_type: AgentType
    status: str = Field(min_length=1)
    result: dict[str, Any] = Field(default_factory=dict)
    summary: str | None = None
    concept_scores: ConceptScores = Field(default_factory=dict)
    weaknesses: list[str] = Field(default_factory=list)
    next_action: NextAction | None = None
    mastery_status: MasteryStatus | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    responded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
