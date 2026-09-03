"""Small Gemini HTTP adapter; provider details stay outside the gateway."""

from typing import Any

from app.llm.base import ProviderError, build_messages
from app.llm.http import JSONHTTPProvider


class GeminiProvider(JSONHTTPProvider):
    name = "gemini"
    auth_in_query = True

    def __init__(self, api_key: str, *, model: str = "gemini-2.0-flash", timeout: float = 30.0):
        super().__init__(api_key, timeout=timeout)
        self.model = model

    def request(self, prompt: str, system_prompt: str | None, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        return url, body

    def extract_text(self, payload: dict[str, Any]) -> str:
        try:
            return payload["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("invalid Gemini response") from exc
