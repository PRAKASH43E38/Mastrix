"""Public HTTP request and response contracts."""

from pydantic import BaseModel, ConfigDict, Field


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    student_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    task: str = Field(min_length=1)


class TaskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    topic: str | None = None
    current_stage: str
    selected_agent: str | None = None
    next_action: str
    mastery_status: str
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str

