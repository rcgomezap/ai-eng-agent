import pytest
from typer.testing import CliRunner

from main import app

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--thread-id" in result.stdout


def test_missing_api_key_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def empty_environment(key: str, default: str = "") -> str:
        del key
        return default

    monkeypatch.setattr("configuration.os.getenv", empty_environment)

    result = runner.invoke(app, ["hello"])

    assert result.exit_code == 1
    assert "OPENAI_API_KEY is required" in result.output
