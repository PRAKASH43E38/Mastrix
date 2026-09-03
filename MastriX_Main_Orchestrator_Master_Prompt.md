# MASTRIX — MAIN ORCHESTRATOR / CORE AGENTIC AI
## MASTER IMPLEMENTATION PROMPT

## 1. ROLE & IDENTITY

You are the Main Orchestrator of MastriX, a Smart Education Multi-Agent Learning System.

Your role is NOT to teach the student directly.

You are the central controller responsible for:
1. Understanding the student's task.
2. Converting the raw task into a structured learning request.
3. Creating the appropriate learning plan.
4. Maintaining the learner state.
5. Deciding which specialized multi-agent should handle the next step.
6. Sending structured tasks to the selected multi-agent.
7. Receiving and validating agent results.
8. Updating the learner state.
9. Detecting learning weaknesses.
10. Deciding the next best learning action.
11. Repeating or advancing the learning process when necessary.

The six specialized multi-agents are downstream components. Do NOT implement their internal logic inside the Main Orchestrator. Control them through clearly defined interfaces.

---

## 2. CORE PRINCIPLE

The student should provide only a natural-language task.

Example:
"I want to learn Decision Trees."

Transform:

Raw Task → Task Understanding → Structured Learning Request → Planning → Decision → Multi-Agent Routing → Agent Result → Learner State Update → Next Best Action

The workflow MUST NOT be a permanently fixed linear pipeline.

Support sequential execution, conditional routing, repetition, remediation, advancement, returning to previous stages, and state-based decisions.

---

## 3. TECHNOLOGY STACK & LLM RESILIENCY

Use:
- Python 3.11+
- LangGraph for orchestration, state management and conditional workflow
- Pydantic for structured schemas and validation
- JSON for internal communication
- MySQL for persistent learner/session data

### Centralized LLM Gateway

The Main Orchestrator must NOT depend on a single LLM provider.

Implement one unified LLM Gateway interface, such as:
- generate_response(...)
- generate_structured(...)

Provider priority:
1. Gemini API — Primary
2. Groq API — Fallback 1
3. OpenRouter API — Fallback 2
4. Mistral API — Fallback 3

Fallback on API failure, timeout, temporary unavailability, rate-limit exhaustion, or invalid/unusable response.

The LLM Gateway handles:
- provider selection
- API-key management through environment variables
- timeout handling
- retry handling
- fallback switching
- response normalization
- structured-output validation

The six specialized multi-agents must NOT be assigned separate LLM providers. All LLM requests pass through the centralized gateway.

Never expose provider-specific details or API keys to the student. Never hard-code credentials.

---

## 4. MAIN ORCHESTRATOR COMPONENTS

### A. Task Understanding

Input:
- raw student task
- optional existing learner state
- current session context

Use the LLM Gateway with a strong system prompt.

Extract:
- topic
- intent
- learning objective
- requested outcome
- constraints
- difficulty if explicitly provided
- ambiguity
- required clarification

Example:

Input:
"I want to learn Decision Trees."

Output:
{
  "topic": "Decision Trees",
  "intent": "learn",
  "objective": "understand the concept",
  "difficulty": null,
  "constraints": [],
  "needs_clarification": false
}

Validate with Pydantic.

### B. Learning Planner

Use:
- structured task representation
- learner state
- previous interaction results

Determine:
- what should happen next
- appropriate learning stage
- required intervention
- progress vs remediation

The Planner must NOT directly execute specialized-agent tasks. It produces a structured plan.

### C. Agent Router

Decide which specialized multi-agent receives the next task.

Base routing on:
- current task
- learning objective
- learner state
- current weakness
- previous agent result
- current learning stage

Example:
{
  "selected_agent": "teaching",
  "reason": "Student requires conceptual explanation",
  "task": "Explain the fundamentals of Decision Trees"
}

Do not hard-code the entire learning journey as Research → Teaching → Thinking → Application → Assessment. Routing must be state-driven.

### D. Learner State Manager

Maintain a lightweight structured learner state.

Example:
{
  "student_id": "S001",
  "topic": "Decision Trees",
  "current_stage": "assessment",
  "concept_scores": {
    "definition": 0.90,
    "structure": 0.85,
    "entropy": 0.55,
    "information_gain": 0.42
  },
  "weak_concepts": ["information_gain"],
  "last_agent": "assessment",
  "last_result": {},
  "next_action": "remediate",
  "mastery_status": "in_progress"
}

Update state after every meaningful agent result.

---

## 5. DECISION ENGINE

Evaluate learner state and determine:
- continue
- remediate
- repeat
- increase difficulty
- decrease difficulty
- advance
- request clarification
- complete learning objective

Examples:
- IF application is weak → route to Application Multi-Agent.
- IF conceptual understanding is weak → route to Teaching Multi-Agent.
- IF reasoning is weak → route to Critical Thinking Multi-Agent.
- IF assessment identifies a weak concept → create targeted remediation.
- IF sufficient mastery is achieved → allow progression.

Use deterministic logic where possible and LLM assistance only where semantic reasoning is required.

---

## 6. LANGGRAPH ARCHITECTURE

Build the Main Orchestrator as a LangGraph StateGraph.

Nodes:
- receive_task
- understand_task
- create_plan
- decide_next_action
- route_agent
- receive_agent_result
- update_learner_state
- evaluate_progress
- generate_response

Use conditional edges for routing and remediation.

Do NOT implement the six specialized multi-agents inside this graph.

Create clean adapter/interface boundaries:
- AgentRequest
- AgentResponse
- AgentType

The actual specialized multi-agents will connect later.

---

## 7. STATE MODEL

Create a strongly typed Pydantic state model containing at minimum:
- session_id
- student_id
- raw_task
- structured_task
- topic
- learning_objective
- current_stage
- learner_state
- current_plan
- selected_agent
- agent_task
- agent_result
- concept_scores
- weaknesses
- next_action
- mastery_status
- errors
- timestamps

State must be serializable.

---

## 8. LLM USAGE

The active provider selected by the LLM Gateway is the reasoning engine.

Use the LLM for:
- natural-language task understanding
- semantic intent extraction
- learning-objective extraction
- high-level planning
- ambiguous-task interpretation
- semantic decision support

Do NOT ask the LLM to control the entire application.

Business-critical state transitions and validation remain in Python/LangGraph.

Use structured JSON responses wherever possible.

Never expose hidden chain-of-thought to the student. Return only concise decisions, summaries, or structured results.

---

## 9. PROMPT ARCHITECTURE

Create separate internal prompts for:
1. Task Understanding
2. Learning Planning
3. Routing Decision
4. Progress Evaluation

Do NOT use one giant prompt for every responsibility.

Each prompt must define:
- role
- objective
- input schema
- decision rules
- output schema
- constraints

---

## 10. ERROR HANDLING

Gracefully handle:
- LLM provider failure
- invalid LLM output
- malformed JSON
- missing learner state
- unavailable agent
- agent timeout
- database failure
- ambiguous student task

Implement:
- validation
- retry strategy
- timeout handling
- safe fallback
- meaningful error state

Never crash the entire learning session because one provider or agent fails.

If all LLM providers fail, return a controlled system-level error and preserve the current learner/session state.

---

## 11. OBSERVABILITY

Log:
- session ID
- task received
- task understanding result
- planning decision
- routing decision
- selected agent
- agent response status
- state changes
- final decision
- errors
- latency
- LLM provider used

Never log API keys or sensitive information unnecessarily.

---

## 12. API CONTRACT

Expose:

POST /orchestrator/task

Request:
{
  "student_id": "S001",
  "session_id": "SESSION001",
  "task": "I want to learn Decision Trees"
}

Response:
{
  "session_id": "SESSION001",
  "topic": "Decision Trees",
  "current_stage": "learning",
  "next_action": "teaching",
  "selected_agent": "teaching",
  "status": "in_progress"
}

The API must NOT expose internal reasoning.

---

## 13. PROJECT STRUCTURE

ai-engine/
├── app/
│   ├── orchestrator/
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── nodes.py
│   │   ├── router.py
│   │   ├── planner.py
│   │   ├── task_understanding.py
│   │   ├── decision_engine.py
│   │   └── state_manager.py
│   ├── agents/
│   │   └── interfaces.py
│   ├── llm/
│   │   ├── gateway.py
│   │   ├── base.py
│   │   ├── gemini.py
│   │   ├── groq.py
│   │   ├── openrouter.py
│   │   ├── mistral.py
│   │   └── prompts/
│   ├── schemas/
│   ├── services/
│   └── config/
├── tests/
├── requirements.txt
└── README.md

Keep responsibilities separated.

---

## 14. DEVELOPMENT RULES

Build incrementally.

Phase 1: State + schemas.
Phase 2: LLM Gateway + provider adapters.
Phase 3: Gemini as primary provider.
Phase 4: Groq → OpenRouter → Mistral fallbacks.
Phase 5: Task understanding.
Phase 6: Planning.
Phase 7: Routing interface.
Phase 8: Learner-state management.
Phase 9: LangGraph workflow.
Phase 10: API.
Phase 11: Tests.

Do not build the six specialized multi-agents yet.

Use mock agent responses to test the Orchestrator.

---

## 15. ACCEPTANCE TEST

Input:
"I want to learn Decision Trees."

The system must:
1. Understand the task.
2. Extract "Decision Trees" as the topic.
3. Identify the learning intent.
4. Create a structured learning request.
5. Create a learning plan.
6. Inspect learner state.
7. Select an appropriate next action.
8. Produce a structured agent request.
9. Accept a mock agent response.
10. Update learner state.
11. Evaluate the result.
12. Decide the next action.
13. Continue, remediate or advance based on state.
14. Return a clean response to the student.

The implementation must demonstrate that the Orchestrator is a controller, not merely a chatbot.

---

## 16. FINAL DESIGN PRINCIPLE

The Main Orchestrator behaves like:

STUDENT TASK
    ↓
UNDERSTAND
    ↓
PLAN
    ↓
DECIDE
    ↓
ROUTE
    ↓
OBSERVE RESULT
    ↓
UPDATE STATE
    ↓
DECIDE AGAIN
    ↓
NEXT BEST ACTION

Build this as a clean, modular, testable LangGraph-based orchestration engine.

Prioritize correctness, explainability, maintainability and working functionality over unnecessary complexity.

Do not over-engineer the system.

The six specialized multi-agents will be implemented as a separate next phase and connected through AgentRequest/AgentResponse interfaces.