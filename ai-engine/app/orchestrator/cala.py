"""CLARIO Adaptive Learning Algorithm (CALA) v1."""

from __future__ import annotations

from collections import defaultdict, deque
import heapq
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Iterable


QUIZ_MODES = {"easy": 30, "medium": 60, "hard": 90}


@dataclass(order=True)
class ConceptPriority:
    priority: float
    concept: str = field(compare=False)


class KnowledgeGraph:
    """Directed prerequisite graph: prerequisite -> dependent concept."""

    def __init__(self) -> None:
        self.edges: dict[str, set[str]] = defaultdict(set)

    def add_dependency(self, prerequisite: str, concept: str) -> None:
        self.edges[prerequisite].add(concept)
        self.edges.setdefault(concept, set())

    def prerequisites(self, concept: str) -> list[str]:
        return [node for node, children in self.edges.items() if concept in children]


class CALA:
    """Deterministic DSA primitives used by the Main Orchestrator."""

    def __init__(self) -> None:
        self.agent_queue: deque[str] = deque()
        self.seen_questions: set[str] = set()

    @staticmethod
    def question_fingerprint(question: str) -> str:
        normalized = " ".join(question.lower().split())
        return sha256(normalized.encode("utf-8")).hexdigest()

    def is_fresh_question(self, question: str) -> bool:
        return self.question_fingerprint(question) not in self.seen_questions

    def remember_question(self, question: str) -> None:
        self.seen_questions.add(self.question_fingerprint(question))

    @staticmethod
    def select_priority(concept_scores: dict[str, float], goal_relevance: dict[str, float] | None = None) -> str | None:
        if not concept_scores:
            return None
        relevance = goal_relevance or {}
        heap: list[ConceptPriority] = []
        for concept, score in concept_scores.items():
            priority = (1.0 - score) + relevance.get(concept, 0.0)
            heapq.heappush(heap, ConceptPriority(-priority, concept))
        return heapq.heappop(heap).concept

    @staticmethod
    def difficulty_action(current: int, current_score: float, previous_score: float | None) -> str:
        if current_score >= 0.85 and (previous_score is None or current_score >= previous_score):
            return "increase_gradually"
        if current_score < 0.60:
            return "decrease_or_remediate"
        return "maintain"

    @staticmethod
    def learner_signal_window(signals: Iterable[float], size: int = 5) -> list[float]:
        window: deque[float] = deque(maxlen=size)
        for signal in signals:
            window.append(signal)
        return list(window)

    @staticmethod
    def quiz_seconds(mode: str) -> int:
        try:
            return QUIZ_MODES[mode.lower()]
        except KeyError as exc:
            raise ValueError(f"unsupported quiz mode: {mode}") from exc
