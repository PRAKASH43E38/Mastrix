"""Convert a student's raw task into a validated structured task."""

from app.llm.gateway import LLMGateway
from app.orchestrator.state import StructuredTask
from app.llm.prompts.task_understanding import TASK_UNDERSTANDING_SYSTEM_PROMPT


class TaskUnderstanding:
    """Task-understanding component; not a downstream learning agent."""

    def __init__(self, gateway: LLMGateway) -> None:
        self.gateway = gateway

    def understand(self, raw_task: str) -> StructuredTask:
        """Ask the centralized gateway for structured task understanding."""

        if not raw_task.strip():
            raise ValueError("raw_task must not be empty")

        return self.gateway.generate_structured(
            raw_task,
            StructuredTask,
            system_prompt=TASK_UNDERSTANDING_SYSTEM_PROMPT,
        )
