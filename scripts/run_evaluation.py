"""
CLI: Score saved simulation transcripts and print evaluation report.

Usage:
    python scripts/run_evaluation.py --agent v1
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console

from src.agent.config_loader import load_rubric
from src.simulation.session_recorder import ConversationRecord
from src.evaluation.evaluator import score_batch
from src.evaluation.report_generator import print_report, save_report

console = Console()


async def _run(agent_version: str):
    results_dir = Path("results") / agent_version
    if not results_dir.exists():
        console.print(f"[red]No results found for agent {agent_version}. Run simulations first.[/red]")
        return

    json_files = [f for f in results_dir.glob("*.json") if f.name != "evaluation_report.json"]
    if not json_files:
        console.print(f"[red]No session files found in {results_dir}[/red]")
        return

    console.print(f"Loading {len(json_files)} sessions from {results_dir}...")
    records = []
    for f in json_files:
        with open(f) as fh:
            data = json.load(fh)
        records.append(ConversationRecord(**data))

    rubric = load_rubric()

    with console.status("[bold green]Evaluating sessions with Gemini 2.5 Pro..."):
        batch = await score_batch(records, rubric, agent_version)

    print_report(batch)
    save_report(batch)

    # Write fitness score back to agent config
    agent_config_path = Path("configs/agents") / f"agent_{agent_version}.yaml"
    if agent_config_path.exists():
        import yaml
        with open(agent_config_path) as f:
            config = yaml.safe_load(f)
        config["fitness_score"] = round(batch.fitness_score, 4)
        with open(agent_config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        console.print(f"[dim]fitness_score written back to {agent_config_path}[/dim]")


@click.command()
@click.option("--agent", default="v1", show_default=True, help="Agent version to evaluate")
def main(agent: str):
    """Evaluate saved simulation sessions and print a scored report."""
    asyncio.run(_run(agent))


if __name__ == "__main__":
    main()
