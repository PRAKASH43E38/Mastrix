"""Tiny standard-library HTTP base for provider adapters."""

import json
from abc import ABC, abstractmethod
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.llm.base import ProviderError


class JSONHTTPProvider(ABC):
    """Shared transport only; request/response shapes belong to adapters."""

    auth_in_query = False
    name = "provider"

    def __init__(self, api_key: str, *, timeout: float = 30.0) -> None:
        if not api_key:
            raise ValueError("provider API key is required")
        self._api_key = api_key
        self.timeout = timeout

    @abstractmethod
    def request(
        self, prompt: str, system_prompt: str | None, **kwargs: Any
    ) -> tuple[str, dict[str, Any]]:
        pass

    @abstractmethod
    def extract_text(self, payload: dict[str, Any]) -> str:
        pass

    def generate_response(
        self, prompt: str, *, system_prompt: str | None = None, **kwargs: Any
    ) -> str:
        url, body = self.request(prompt, system_prompt, **kwargs)
        headers = {"Content-Type": "application/json"}
        if self.auth_in_query:
            url = f"{url}?{urlencode({'key': self._api_key})}"
        else:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            request = Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise ProviderError(f"{self.name.capitalize()} request failed") from exc

        if not isinstance(payload, dict):
            raise ProviderError(f"invalid {self.name} response")
        return self.extract_text(payload)
