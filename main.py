import asyncio
import logging
import os
import sqlite3
from typing import Annotated

import typer
from dotenv import load_dotenv
from injector import Injector
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from configuration import ApplicationModule, ConfigurationError, Settings
from graph import AgentError, AgentFactory, HotelAgent

app = typer.Typer(
    help="Autonomous hotel agent with persistent LangGraph memory.",
    no_args_is_help=False,
)
logger = logging.getLogger(__name__)


def _message_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, str):
            parts.append(block)
        elif block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip() or "The agent returned no textual response."


async def _ask(agent: HotelAgent, prompt: str, thread_id: str) -> None:
    response = await agent.invoke(prompt, thread_id)
    typer.echo(f"Assistant: {_message_text(response)}")


async def _interactive_chat(agent: HotelAgent, thread_id: str) -> None:
    typer.echo(f"Thread: {thread_id}. Type 'exit' to finish.")
    while True:
        try:
            prompt = await asyncio.to_thread(typer.prompt, "You")
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nSession closed.")
            return
        if prompt.strip().casefold() in {"exit", "quit"}:
            return
        try:
            await _ask(agent, prompt, thread_id)
        except AgentError as exc:
            typer.secho(f"Agent error: {exc}", fg=typer.colors.RED, err=True)


async def _run(prompt: str | None, thread_id: str, settings: Settings) -> None:
    try:
        settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigurationError(
            f"Cannot create the database directory: {settings.database_path.parent}"
        ) from exc

    injector = Injector([ApplicationModule(settings)])
    factory = injector.get(AgentFactory)
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

    try:
        async with AsyncSqliteSaver.from_conn_string(str(settings.database_path)) as checkpointer:
            agent = factory.create(checkpointer)
            if prompt is None:
                await _interactive_chat(agent, thread_id)
            else:
                await _ask(agent, prompt, thread_id)
    except sqlite3.Error as exc:
        raise AgentError("Persistent memory is unavailable.") from exc


@app.command()
def chat(
    prompt: Annotated[
        str | None,
        typer.Argument(help="One prompt. Omit it to start an interactive session."),
    ] = None,
    thread_id: Annotated[
        str,
        typer.Option("--thread-id", "-t", help="Persistent conversation identifier."),
    ] = "default",
) -> None:
    """Chat with the agent while retaining memory under THREAD_ID."""
    try:
        load_dotenv()
        settings = Settings.from_environment()
        logging.basicConfig(level=settings.log_level)
        asyncio.run(_run(prompt, thread_id, settings))
    except (ConfigurationError, AgentError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt as exc:
        typer.echo("\nSession interrupted.", err=True)
        raise typer.Exit(code=130) from exc
    except Exception as exc:
        logger.exception("Unexpected application failure")
        typer.secho("Unexpected internal error. Check the logs.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
