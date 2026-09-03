"""Provider-agnostic contracts used by the centralized gateway."""

from collections.abc import Mapping
from typing import Any, Protocol


class ProviderError(RuntimeError):
    """An expected provider/API failure that can be handled by fallback."""


class LLMProvider(Protocol):
    """Minimal interface every provider adapter must implement."""

    name: str

    def generate_response(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate normalized text from a provider."""


def build_messages(prompt: str, system_prompt: str | None) -> list[Mapping[str, str]]:
    """Build the common chat message shape used by compatible providers."""

    messages: list[Mapping[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages
