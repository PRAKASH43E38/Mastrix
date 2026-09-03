"""Environment-backed application and LLM configuration.

Only environment variables are read. API keys are optional and never given
hard-coded values.
"""

import os

from pydantic import BaseModel, ConfigDict, Field


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_provider: str = "gemini"
    fallback_providers: list[str] = Field(
        default_factory=lambda: ["groq", "openrouter", "mistral"]
    )
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    gemini_model: str = "gemini-2.0-flash"
    groq_model: str = "llama-3.3-70b-versatile"
    openrouter_model: str = "openai/gpt-4o-mini"
    mistral_model: str = "mistral-small-latest"
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    openrouter_api_key: str | None = None
    mistral_api_key: str | None = None


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment: str = "development"
    log_level: str = "INFO"
    cors_allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"]
    )
    llm: LLMSettings = Field(default_factory=LLMSettings)


def load_settings() -> AppSettings:
    """Load safe defaults and explicitly supported environment variables."""

    return AppSettings(
        environment=os.getenv("MASTRIX_ENVIRONMENT", "development"),
        log_level=os.getenv("MASTRIX_LOG_LEVEL", "INFO"),
        cors_allowed_origins=[
            origin.strip()
            for origin in os.getenv(
                "MASTRIX_CORS_ORIGINS",
                "http://localhost:3000,http://localhost:5173",
            ).split(",")
            if origin.strip()
        ],
        llm=LLMSettings(
            primary_provider=os.getenv("MASTRIX_LLM_PRIMARY", "gemini"),
            timeout_seconds=float(os.getenv("MASTRIX_LLM_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.getenv("MASTRIX_LLM_MAX_RETRIES", "2")),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            openrouter_model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            mistral_model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        ),
    )
