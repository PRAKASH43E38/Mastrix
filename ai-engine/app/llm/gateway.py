"""Centralized provider selection, retries, fallback, and validation."""

import json
import logging
from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import AppSettings, load_settings
from app.llm.base import LLMProvider, ProviderError
from app.llm.gemini import GeminiProvider
from app.llm.groq import GroqProvider

logger = logging.getLogger(__name__)
ModelT = TypeVar("ModelT", bound=BaseModel)
PROVIDER_ORDER = ("gemini", "groq", "openrouter", "mistral")


class LLMGatewayError(RuntimeError):
    """Raised when every configured provider fails safely."""


class LLMGateway:
    """One LLM entry point for the entire MastriX application."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        providers: Mapping[str, LLMProvider] | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.providers = dict(providers) if providers is not None else self._build_providers()

    def _build_providers(self) -> dict[str, LLMProvider]:
        """Construct only adapters whose API key is configured."""

        config = self.settings.llm
        timeout = config.timeout_seconds
        result: dict[str, LLMProvider] = {}
        if config.gemini_api_key:
            result["gemini"] = GeminiProvider(
                config.gemini_api_key, model=config.gemini_model, timeout=timeout
            )
        if config.groq_api_key:
            result["groq"] = GroqProvider(
                config.groq_api_key, model=config.groq_model, timeout=timeout
            )
        return result

    @property
    def provider_order(self) -> tuple[str, ...]:
        """Return the fixed MastriX provider priority."""

        return PROVIDER_ORDER

    def generate_response(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text, retrying a provider before moving to the next one."""

        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        failures: list[str] = []
        for provider_name in self.provider_order:
            provider = self.providers.get(provider_name)
            if provider is None:
                failures.append(f"{provider_name}: unavailable")
                continue

            for attempt in range(self.settings.llm.max_retries + 1):
                try:
                    response = provider.generate_response(
                        prompt, system_prompt=system_prompt, **kwargs
                    )
                    if not isinstance(response, str) or not response.strip():
                        raise ProviderError("empty response")
                    return response.strip()
                except Exception as exc:  # provider boundaries must not crash a session
                    if attempt == self.settings.llm.max_retries:
                        failures.append(f"{provider_name}: {type(exc).__name__}")
                        logger.warning(
                            "LLM provider failed; trying fallback provider=%s error_type=%s",
                            provider_name,
                            type(exc).__name__,
                        )

        raise LLMGatewayError(
            "All configured LLM providers failed: " + ", ".join(failures)
        )

    def generate_structured(
        self,
        prompt: str,
        response_model: type[ModelT],
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ModelT:
        """Generate JSON and validate it, falling back on invalid output too."""

        failures: list[str] = []
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        for provider_name in self.provider_order:
            provider = self.providers.get(provider_name)
            if provider is None:
                failures.append(f"{provider_name}: unavailable")
                continue
            for attempt in range(self.settings.llm.max_retries + 1):
                try:
                    raw = provider.generate_response(
                        prompt, system_prompt=system_prompt, **kwargs
                    )
                    if not isinstance(raw, str) or not raw.strip():
                        raise ProviderError("empty response")
                    value = json.loads(raw)
                    return response_model.model_validate(value)
                except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                    if attempt == self.settings.llm.max_retries:
                        failures.append(f"{provider_name}: {type(exc).__name__}")
                        logger.warning(
                            "LLM provider returned unusable structured output; provider=%s error_type=%s",
                            provider_name,
                            type(exc).__name__,
                        )
                except Exception as exc:
                    if attempt == self.settings.llm.max_retries:
                        failures.append(f"{provider_name}: {type(exc).__name__}")
                        logger.warning(
                            "LLM provider failed; trying fallback provider=%s error_type=%s",
                            provider_name,
                            type(exc).__name__,
                        )

        raise LLMGatewayError(
            "All configured LLM providers failed: " + ", ".join(failures)
        )
