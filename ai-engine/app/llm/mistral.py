"""Mistral adapter using its OpenAI-compatible HTTP API."""

from app.llm.openai_compatible import OpenAICompatibleProvider


class MistralProvider(OpenAICompatibleProvider):
    name = "mistral"
    endpoint = "https://api.mistral.ai/v1/chat/completions"

