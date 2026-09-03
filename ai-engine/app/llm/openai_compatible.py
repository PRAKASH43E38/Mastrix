"""Shared payload handling for Groq, OpenRouter, and Mistral adapters."""

from typing import Any

from app.llm.base import ProviderError, build_messages
from app.llm.http import JSONHTTPProvider


class OpenAICompatibleProvider(JSONHTTPProvider):
    endpoint = ""
    name = "openai-compatible"

    def __init__(self, api_key: str, *, model: str = "default", timeout: float = 30.0):
        super().__init__(api_key, timeout=timeout)
        self.model = model

    def request(self, prompt: str, system_prompt: str | None, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        return self.endpoint, {
            "model": kwargs.get("model", self.model),
            "messages": build_messages(prompt, system_prompt),
        }

    def extract_text(self, payload: dict[str, Any]) -> str:
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"invalid {self.name} response") from exc

