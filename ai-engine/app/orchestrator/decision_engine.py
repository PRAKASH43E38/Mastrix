"""Deterministic decision helpers backed by CLARIO's CALA rules."""

from app.orchestrator.cala import CALA


class DecisionEngine:
    """Expose CALA decisions to the LangGraph orchestrator."""

    def __init__(self, cala: CALA | None = None) -> None:
        self.cala = cala or CALA()

    def next_concept(self, concept_scores: dict[str, float], goal_relevance: dict[str, float] | None = None) -> str | None:
        return self.cala.select_priority(concept_scores, goal_relevance)

    def difficulty_action(self, current_difficulty: int, current_score: float, previous_score: float | None = None) -> str:
        return self.cala.difficulty_action(current_difficulty, current_score, previous_score)

    def quiz_time_limit(self, mode: str) -> int:
        return self.cala.quiz_seconds(mode)

    def question_is_fresh(self, question: str) -> bool:
        return self.cala.is_fresh_question(question)

    def remember_question(self, question: str) -> None:
        self.cala.remember_question(question)
