"""
CLI: Run text-based simulations.

Usage:
    python scripts/run_simulation.py --agent v1 --runs 5
    python scripts/run_simulation.py --agent v1 --runs 25 --persona angry_defaulter
"""

import asyncio
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from src.agent.config_loader import load_agent_config_by_version, load_all_personas, load_persona_config
from src.simulation.conversation_runner import run_conversation

console = Console()


async def _run_batch(agent_version: str, runs: int, persona_id: str | None):
    agent_config = load_agent_config_by_version(agent_version)

    if persona_id:
        persona_path = Path("configs/personas") / f"{persona_id}.yaml"
        personas = [load_persona_config(persona_path)]
    else:
        personas = load_all_personas()

    results = []
    run_count = 0

    with console.status("[bold green]Running simulations...") as status:
        for i in range(runs):
            persona = personas[i % len(personas)]
            status.update(f"[bold green]Run {i+1}/{runs} — persona: {persona.persona_id}")

            record = await run_conversation(agent_config, persona)
            path = record.save()
            results.append(record)
            run_count += 1

            rprint(
                f"  [cyan]{record.session_id}[/cyan] "
                f"[yellow]{persona.persona_id}[/yellow] → "
                f"[{'green' if record.outcome == 'agreement' else 'red'}]{record.outcome}[/] "
                f"({len(record.turns)} turns) → saved to {path}"
            )

    # Summary table
    table = Table(title=f"Simulation Summary — agent {agent_version} ({run_count} runs)")
    table.add_column("Outcome", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Rate", justify="right")

    outcomes = {}
    for r in results:
        outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1

    for outcome, count in sorted(outcomes.items()):
        rate = f"{count / run_count * 100:.0f}%"
        color = "green" if outcome == "agreement" else "red" if outcome == "hangup" else "yellow"
        table.add_row(f"[{color}]{outcome}[/{color}]", str(count), rate)

    console.print(table)
    console.print(f"\n[bold]Results saved to:[/bold] results/{agent_version}/")


@click.command()
@click.option("--agent", default="v1", show_default=True, help="Agent version (e.g. v1)")
@click.option("--runs", default=5, show_default=True, help="Number of simulations to run")
@click.option("--persona", default=None, help="Specific persona ID (default: cycle through all)")
def main(agent: str, runs: int, persona: str | None):
    """Run text-based debt collection simulations."""
    asyncio.run(_run_batch(agent, runs, persona))


if __name__ == "__main__":
    main()
