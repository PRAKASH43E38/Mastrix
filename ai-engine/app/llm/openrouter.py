"""OpenRouter adapter using its OpenAI-compatible HTTP API."""

from app.llm.openai_compatible import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

