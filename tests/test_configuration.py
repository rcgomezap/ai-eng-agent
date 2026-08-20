from pathlib import Path

from injector import Injector

from configuration import ApplicationModule, Settings
from graph import AgentFactory


def test_injector_resolves_agent_factory() -> None:
    settings = Settings(
        openai_api_key="test-key",
        openai_model="test-model",
        openai_base_url=None,
        database_path=Path("memory.sqlite3"),
        log_level="INFO",
    )

    injector = Injector([ApplicationModule(settings)])

    assert isinstance(injector.get(AgentFactory), AgentFactory)
