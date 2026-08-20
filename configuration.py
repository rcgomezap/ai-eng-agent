import logging
import os
from dataclasses import dataclass
from pathlib import Path

from injector import Binder, Module, provider, singleton
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from graph import AgentFactory
from hotel_client import MockHotelClient
from interfaces import HotelClient


class ConfigurationError(Exception):
    """Raised when required application configuration is invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    openai_api_key: str
    openai_model: str
    openai_base_url: str | None
    database_path: Path
    log_level: str

    @classmethod
    def from_environment(cls) -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is required.")

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        if not model:
            raise ConfigurationError("OPENAI_MODEL cannot be empty.")

        database_value = os.getenv("AGENT_DB_PATH", ".data/agent-memory.sqlite3").strip()
        if not database_value:
            raise ConfigurationError("AGENT_DB_PATH cannot be empty.")

        log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
        if log_level not in logging.getLevelNamesMapping():
            raise ConfigurationError(f"Unsupported LOG_LEVEL: {log_level}")

        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        return cls(api_key, model, base_url, Path(database_value), log_level)


class ApplicationModule(Module):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def configure(self, binder: Binder) -> None:
        binder.bind(Settings, to=self._settings, scope=singleton)
        binder.bind(HotelClient, to=MockHotelClient, scope=singleton)
        binder.bind(AgentFactory, scope=singleton)

    @provider
    @singleton
    def provide_chat_model(self, settings: Settings) -> BaseChatModel:
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=SecretStr(settings.openai_api_key),
            base_url=settings.openai_base_url,
            temperature=0,
            max_retries=2,
        )
