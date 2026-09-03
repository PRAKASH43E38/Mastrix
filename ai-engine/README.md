# MastriX AI Engine

Phase 11 foundation for the MastriX Main Orchestrator.

This phase contains the project boundaries, Pydantic state and agent contracts,
and a centralized LLM gateway with adaptive task understanding, learning
planning, deterministic agent routing, learner-state orchestration, a
LangGraph main workflow, and a thin FastAPI transport layer. Specialized
agents, persistence, and UI are intentionally deferred to later phases.

The Phase 1–11 foundation is ready for integration with real downstream agent
implementations through the `AgentExecutor` and `AgentRequest`/
`AgentResponse` contracts.

## Development

From this directory, install the runtime dependency and run the tests with:

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest
```

## Local API keys

Create a local environment file from the template:

```bash
cp .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` and, optionally, `GROQ_API_KEY`. The
application reads these values from the process environment. Without adding a
dotenv dependency, load them before running the application or tests:

```bash
set -a
source .env
set +a
```

`.env` is ignored by Git and must never contain a committed or shared secret.
