"""Groq adapter using its OpenAI-compatible HTTP API."""

from app.llm.openai_compatible import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    name = "groq"
    endpoint = "https://api.groq.com/openai/v1/chat/completions"

