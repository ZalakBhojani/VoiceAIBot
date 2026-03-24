"""
CLI: Darwin Gödel evolution loop.

Iteratively improves an agent by:
  1. Extracting weak sessions from the current best agent
  2. Prompting Gemini to propose targeted prompt mutations
  3. Simulating and evaluating the mutated agent
  4. Promoting if fitness improves, logging as failed otherwise

Usage:
    python scripts/run_evolution.py
    python scripts/run_evolution.py --start-version v1 --max-generations 10
    python scripts/run_evolution.py --success-threshold 4.9 --plateau-patience 3
    python scripts/run_evolution.py --sim-runs 15 --allow-llm-param-mutation
    python scripts/run_evolution.py --dry-run
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich import box

from src.agent.config_loader import (
    load_agent_config_by_version,
    load_all_personas,
    load_persona_config,
    load_rubric,
    AgentConfig,
)
from src.simulation.conversation_runner import run_conversation
from src.simulation.session_recorder import ConversationRecord
from src.evaluation.evaluator import score_batch
from src.evaluation.metrics import BatchEvaluationResult
from src.evaluation.report_generator import print_report, save_report
from src.evolution.archive import ArchiveManager
from src.evolution.mutator import MutationEngine
from src.evolution.failure_analyzer import FailureAnalyzer

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_agent_config(config: AgentConfig, configs_dir: Path) -> Path:
    """Serialize AgentConfig to YAML at configs/agents/agent_{version}.yaml."""
    path = configs_dir / "agents" / f"agent_{config.version}.yaml"
    data = config.model_dump()
    # Remove None values except parent_version (keep explicit null for root)
    cleaned = {k: v for k, v in data.items() if v is not None or k == "parent_version"}
    with open(path, "w") as f:
        yaml.dump(cleaned, f, default_flow_style=False, allow_unicode=True)
    return path


def load_existing_batch_result(
    agent_version: str, results_dir: Path
) -> BatchEvaluationResult | None:
    report_path = results_dir / agent_version / "evaluation_report.json"
    if not report_path.exists():
        return None
    with open(report_path) as f:
        data = json.load(f)
    return BatchEvaluationResult(**data)


async def _run_simulations(
    agent_version: str,
    runs: int,
    configs_dir: Path,
    results_dir: Path,
) -> list[ConversationRecord]:
    """Run N simulations cycling through all personas."""
    agent_config = load_agent_config_by_version(agent_version, configs_dir)
    personas = load_all_personas(configs_dir)
    records = []

    with console.status(f"[bold green]Running {runs} simulations for {agent_version}...") as status:
        for i in range(runs):
            persona = personas[i % len(personas)]
            status.update(
                f"[bold green]Simulation {i+1}/{runs} — persona: {persona.persona_id}"
            )
            record = await run_conversation(agent_config, persona)
            record.save(results_dir)
            records.append(record)

    console.print(f"  [dim]{len(records)} sessions saved to {results_dir / agent_version}/[/dim]")
    return records


async def _run_evaluation(
    agent_version: str,
    configs_dir: Path,
    results_dir: Path,
) -> BatchEvaluationResult:
    """Evaluate all sessions for an agent version and write fitness back to YAML."""
    version_dir = results_dir / agent_version
    json_files = [
        f for f in version_dir.glob("*.json")
        if f.name != "evaluation_report.json"
    ]

    records = []
    for f in json_files:
        with open(f) as fh:
            data = json.load(fh)
        records.append(ConversationRecord(**data))

    rubric = load_rubric(configs_dir / "evaluation" / "rubric_v1.yaml")

    with console.status("[bold green]Evaluating sessions..."):
        batch = await score_batch(records, rubric, agent_version)

    save_report(batch, str(results_dir))

    # Write fitness score back to agent YAML
    agent_config_path = configs_dir / "agents" / f"agent_{agent_version}.yaml"
    if agent_config_path.exists():
        with open(agent_config_path) as f:
            config_data = yaml.safe_load(f)
        config_data["fitness_score"] = round(batch.fitness_score, 4)
        with open(agent_config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

    return batch


def print_generation_summary(
    archive: ArchiveManager,
    generation: int,
    parent_fitness: float,
    new_version: str,
    new_fitness: float,
) -> None:
    delta = new_fitness - parent_fitness
    color = "green" if delta > 0 else "red"
    sign = "+" if delta >= 0 else ""
    console.print(
        f"\n[bold]Generation {generation} result:[/bold] "
        f"{new_version} fitness = [{color}]{new_fitness:.4f}[/{color}] "
        f"([{color}]{sign}{delta:.4f}[/{color}])"
    )
    best = archive.get_best_agent()
    if best:
        console.print(f"[dim]Best so far: {best.version} ({best.fitness_score:.4f})[/dim]")


def print_evolution_report(archive: ArchiveManager) -> None:
    console.rule("[bold blue]Evolution Report[/bold blue]")

    entries = archive._archive.entries
    table = Table(box=box.ROUNDED, title="All Agent Versions")
    table.add_column("Version", style="cyan")
    table.add_column("Gen", justify="right")
    table.add_column("Parent")
    table.add_column("Fitness", justify="right")
    table.add_column("Status")
    table.add_column("Mutations")

    for e in sorted(entries, key=lambda x: x.generation):
        status_color = (
            "green" if e.status == "promoted"
            else "red" if e.status == "failed"
            else "yellow"
        )
        fitness_str = f"{e.fitness_score:.4f}" if e.fitness_score is not None else "—"
        table.add_row(
            e.version,
            str(e.generation),
            e.parent_version or "—",
            fitness_str,
            f"[{status_color}]{e.status}[/{status_color}]",
            ", ".join(e.mutations_applied) or "—",
        )

    console.print(table)

    # Lineage of best agent
    best = archive.get_best_agent()
    if best:
        lineage = archive.get_lineage(best.version)
        fitness_path = " → ".join(
            f"{e.version}({e.fitness_score:.4f})" if e.fitness_score else e.version
            for e in lineage
        )
        console.print(f"\n[bold]Best lineage:[/bold] {fitness_path}")
        console.print(f"[bold green]Final best agent: {best.version} | Fitness: {best.fitness_score:.4f}[/bold green]")


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

@click.command()
@click.option("--start-version", default="v1", show_default=True, help="Seed agent version")
@click.option("--max-generations", default=10, show_default=True)
@click.option("--success-threshold", default=4.9, show_default=True, help="Stop when fitness >= this")
@click.option("--plateau-patience", default=3, show_default=True, help="Stop after N consecutive non-improving generations")
@click.option("--sim-runs", default=15, show_default=True, help="Simulation runs per candidate (15 = 3 per persona)")
@click.option("--configs-dir", default="configs", show_default=True)
@click.option("--results-dir", default="results", show_default=True)
@click.option("--allow-llm-param-mutation", is_flag=True, default=False, help="Also evolve temperature/max_tokens")
@click.option("--dry-run", is_flag=True, default=False, help="Propose mutation only, do not simulate or evaluate")
def main(
    start_version: str,
    max_generations: int,
    success_threshold: float,
    plateau_patience: int,
    sim_runs: int,
    configs_dir: str,
    results_dir: str,
    allow_llm_param_mutation: bool,
    dry_run: bool,
):
    """Darwin Gödel evolution loop: iteratively improve agent via LLM-guided mutation."""
    asyncio.run(_evolve(
        start_version=start_version,
        max_generations=max_generations,
        success_threshold=success_threshold,
        plateau_patience=plateau_patience,
        sim_runs=sim_runs,
        configs_dir=Path(configs_dir),
        results_dir=Path(results_dir),
        allow_llm_param_mutation=allow_llm_param_mutation,
        dry_run=dry_run,
    ))


async def _evolve(
    start_version: str,
    max_generations: int,
    success_threshold: float,
    plateau_patience: int,
    sim_runs: int,
    configs_dir: Path,
    results_dir: Path,
    allow_llm_param_mutation: bool,
    dry_run: bool,
):
    archive = ArchiveManager(configs_dir / "agents" / "archive.json")
    rubric = load_rubric(configs_dir / "evaluation" / "rubric_v1.yaml")

    # --- Seed archive with start_version if not already present ---
    if not archive.get_entry(start_version):
        seed_config = load_agent_config_by_version(start_version, configs_dir)
        seed_batch = load_existing_batch_result(start_version, results_dir)
        archive.record_version(seed_config, seed_batch, status="promoted", generation=0)
        archive.save()
        console.print(f"[dim]Seeded archive with {start_version} (fitness: {seed_config.fitness_score})[/dim]")

    mutator = MutationEngine(
        allow_llm_param_mutation=allow_llm_param_mutation,
    )
    analyzer = FailureAnalyzer(results_dir)
    plateau_counter = 0

    for gen_idx in range(max_generations):
        current_generation = gen_idx + 1
        console.rule(f"[bold]Generation {current_generation} / {max_generations}[/bold]")

        parent_entry = archive.get_best_agent()
        if parent_entry is None:
            console.print("[red]No promoted agent found in archive. Aborting.[/red]")
            break

        parent_config = load_agent_config_by_version(parent_entry.version, configs_dir)
        parent_fitness = parent_entry.fitness_score or 0.0

        console.print(
            f"Parent: [cyan]{parent_entry.version}[/cyan] | "
            f"Fitness: [bold]{parent_fitness:.4f}[/bold] | "
            f"Generation: {parent_entry.generation}"
        )

        # --- Extract weak sessions ---
        try:
            weak_sessions = analyzer.get_weak_sessions(
                agent_version=parent_entry.version,
                bottom_fraction=0.25,
                min_sessions=2,
            )
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            break

        console.print(
            f"Weak sessions: {[s.session_id for s in weak_sessions]} "
            f"(scores: {[round(s.weighted_total, 2) for s in weak_sessions]})"
        )

        # --- Propose mutation ---
        with console.status("[bold green]Generating mutation proposal via Gemini..."):
            proposal = await mutator.propose_mutation(
                parent_config=parent_config,
                weak_sessions=weak_sessions,
                rubric=rubric,
                generation=current_generation,
            )

        console.print(f"Proposed sections: [yellow]{proposal.sections_to_mutate}[/yellow]")
        console.print(f"Rationale: {proposal.rationale}")
        console.print(f"Failure addressed: {proposal.failure_addressed}")
        if proposal.llm_param_changes:
            console.print(f"LLM param changes: {proposal.llm_param_changes}")

        # --- Determine new version string ---
        all_versions = archive.get_all_versions()
        version_numbers = []
        for v in all_versions:
            try:
                version_numbers.append(int(v.lstrip("v")))
            except ValueError:
                pass
        next_num = max(version_numbers, default=1) + 1
        new_version = f"v{next_num}"

        # --- Apply mutation and save YAML ---
        new_config = mutator.apply_mutation(parent_config, proposal, new_version)
        yaml_path = save_agent_config(new_config, configs_dir)
        console.print(f"[dim]New agent config saved: {yaml_path}[/dim]")

        if dry_run:
            console.print("\n[bold yellow]--dry-run: skipping simulation and evaluation.[/bold yellow]")
            console.print(f"Inspect proposed config at: {yaml_path}")
            break

        # --- Record as pending ---
        archive.record_version(new_config, batch_result=None, status="pending",
                               generation=current_generation)
        archive.save()

        # --- Simulate ---
        await _run_simulations(new_version, sim_runs, configs_dir, results_dir)

        # --- Evaluate ---
        batch = await _run_evaluation(new_version, configs_dir, results_dir)
        print_report(batch)

        # Reload config to get fitness_score written back by evaluator
        new_config_evaluated = load_agent_config_by_version(new_version, configs_dir)
        new_fitness = new_config_evaluated.fitness_score or 0.0

        # --- Update archive with evaluation results ---
        archive.record_version(
            new_config_evaluated, batch,
            status="pending",
            generation=current_generation,
        )

        # --- Promotion decision ---
        if new_fitness > parent_fitness:
            archive.mark_promoted(new_version)
            plateau_counter = 0
            console.print(
                f"[bold green]PROMOTED {new_version}: {new_fitness:.4f} "
                f"(+{new_fitness - parent_fitness:.4f} vs {parent_entry.version})[/bold green]"
            )
        else:
            archive.mark_failed(new_version)
            plateau_counter += 1
            console.print(
                f"[bold red]FAILED {new_version}: {new_fitness:.4f} "
                f"({new_fitness - parent_fitness:.4f} vs {parent_entry.version})[/bold red]"
            )

        archive.save()
        print_generation_summary(archive, current_generation, parent_fitness, new_version, new_fitness)

        # --- Termination checks ---
        best = archive.get_best_agent()
        if best and best.fitness_score is not None and best.fitness_score >= success_threshold:
            console.print(
                f"\n[bold green]SUCCESS: Reached fitness {best.fitness_score:.4f} "
                f">= threshold {success_threshold}[/bold green]"
            )
            break

        if plateau_counter >= plateau_patience:
            console.print(
                f"\n[bold yellow]PLATEAU: No improvement for {plateau_patience} "
                f"consecutive generations. Stopping.[/bold yellow]"
            )
            break

    # --- Final report ---
    print_evolution_report(archive)


if __name__ == "__main__":
    main()
