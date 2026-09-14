# CLARIO Adaptive Learning Algorithm (CALA) v1

CALA (Context-Aware Learning Adaptation Algorithm) is the project-specific algorithmic layer used by the Main Orchestrator.

## Inputs

- Task
- Goal
- Interest / Preference
- Learner State

## Flow

```text
4 Inputs
  -> Learner Profile
  -> Topic Knowledge Graph
  -> Prerequisite Detection
  -> Starting Concept
  -> Master Roadmap
  -> Initial Difficulty
  -> Adaptive Agent Loop

Nova -> Mira -> Ayan -> Kira -> Zayn -> Elara
                    ^                    |
                    |                    v
              targeted support <- Evaluation
                    |
                    +---- updated learner state
                                   |
                                   v
                              next action
```

The sequence is adaptive, not a mandatory fixed pipeline. The Main Orchestrator may skip, repeat or revisit agents based on learner state.

## DSA Primitives

- Graph: concept prerequisites and roadmap.
- Hash Map: concept-level learner state and scores.
- Priority Queue: weakest / most goal-relevant concept selection.
- Queue: pending agent workflow events.
- Set: fresh-question protection using question fingerprints.
- Sliding Window: recent learner-signal analysis.

## Agent Rules

### Nova — Research
Research current and relevant knowledge/resources from appropriate external sources. It supplies a knowledge package; it does not teach, grade or route.

### Mira — Teaching
Teach from the learner's correct starting level. Begin with context and foundations when the learner is new, then introduce technical terminology progressively. If a learner struggles with a real-world problem, Mira provides prerequisite knowledge, hints or a simpler explanation, but must not reveal the direct answer to the active problem.

### Ayan — Critical Thinking
Use reasoning-focused questions: why, how, what-if, compare, predict and justify. Prefer learner-generated answers over answer-selection when the goal is reasoning. If the learner is stuck, provide a hint/support path rather than the final answer.

### Kira — Real-World Application
Connect the learned concept to the learner's actual domain. Ask the learner to solve a practical problem, scenario, case or coding task independently. If application fails because a prerequisite concept is weak, route through targeted support before reassessment.

### Zayn — Quiz
Generate fresh, objectively gradable assessments based on taught concepts and current difficulty. Never intentionally repeat a previously used question. Fixed time modes:

- Easy: 30 seconds/question
- Medium: 60 seconds/question
- Hard: 90 seconds/question

Zayn reports performance; it does not make the final mastery/routing decision.

### Elara — Evaluation
Compare current performance with previous performance across understanding, reasoning, application and assessment. Identify strengths, weaknesses and concept gaps. Recommend increase, decrease or maintain difficulty and the next learning action. The Main Orchestrator owns the final decision and persistent learner-state update.

## Adaptive Difficulty

Difficulty must change gradually. Strong performance can move the learner from 10% -> 20% -> 30% -> 40% rather than jumping directly to a high level. If performance drops after an increase, step back to an intermediate level and rebuild gradually.

Keeping the same difficulty level does not permit repeating the same questions. Question wording, context and reasoning angle should remain fresh.

## Learner Interaction Signals

CALA may use response correctness, response delay, repeated mistakes, abandonment, explicit doubt and like/dislike feedback as observable signals. A single signal must never be treated as proof of an emotion. The system infers possible struggle/engagement and adapts presentation style across all agents.

Possible style adaptation:

- formal <-> conversational
- technical <-> beginner-friendly
- concise <-> detailed
- text <-> diagram/visual
- direct explanation <-> analogy/story

## Core Decision Loop

```text
Evaluate result
  -> update learner state
  -> update dashboard
  -> prioritize next concept
  -> compare current vs previous performance
  -> adapt difficulty
  -> select next agent/action
  -> execute fresh learning activity
  -> evaluate again
```

The objective is not to force every learner through all six agents every time. The objective is to select the smallest effective sequence that produces a meaningful conceptual clarity gain.
