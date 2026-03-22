"""
CLI: Start the voice server.

Usage:
    python scripts/run_voice_server.py --agent v1
    python scripts/run_voice_server.py --agent v1 --host 0.0.0.0 --port 8765
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console

from src.agent.config_loader import load_agent_config_by_version
from src.voice.server import run_server

console = Console()


@click.command()
@click.option("--agent", default="v1", show_default=True, help="Agent version (e.g. v1)")
@click.option("--host", default="localhost", show_default=True, help="Server host")
@click.option("--port", default=8765, show_default=True, help="Server port")
def main(agent: str, host: str, port: int):
    """Start the WebSocket voice server for the debt collection agent."""
    config = load_agent_config_by_version(agent)

    console.print(f"[bold green]Voice Server[/bold green]")
    console.print(f"  Agent  : [cyan]{config.version}[/cyan] — {config.description}")
    console.print(f"  LLM    : [cyan]{config.llm.model}[/cyan]")
    console.print(f"  STT    : [cyan]{config.stt.provider} / {config.stt.model}[/cyan]")
    console.print(f"  TTS    : [cyan]{config.tts.provider} / {config.tts.voice_id}[/cyan]")
    console.print(f"  Listen : [bold]ws://{host}:{port}[/bold]")
    console.print("\n[dim]Connect a browser WebSocket client to start a conversation.[/dim]\n")

    try:
        asyncio.run(run_server(config, host=host, port=port))
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")


if __name__ == "__main__":
    main()
