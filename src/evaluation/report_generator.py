"""
Prints and saves evaluation reports using Rich tables.
"""

from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box
from src.evaluation.metrics import BatchEvaluationResult

console = Console()


def print_report(batch: BatchEvaluationResult) -> None:
    console.rule(f"[bold blue]Evaluation Report — Agent {batch.agent_version}[/bold blue]")

    # Per-session table
    session_table = Table(box=box.ROUNDED, title="Per-Session Scores")
    session_table.add_column("Session", style="cyan")
    session_table.add_column("Persona")
    session_table.add_column("Outcome")
    session_table.add_column("Goal (40%)", justify="right")
    session_table.add_column("Quality (35%)", justify="right")
    session_table.add_column("Compliance (25%)", justify="right")
    session_table.add_column("Weighted Total", justify="right", style="bold")

    for r in batch.results:
        scores = {m.name: m.score for m in r.metrics}
        outcome_color = "green" if r.outcome == "agreement" else "red" if r.outcome == "hangup" else "yellow"
        flag = " ⚠" if r.compliance_regex_violation else ""
        session_table.add_row(
            r.session_id,
            r.persona_id,
            f"[{outcome_color}]{r.outcome}[/{outcome_color}]",
            f"{scores.get('goal_completion', 0):.1f}",
            f"{scores.get('conversational_quality', 0):.1f}",
            f"{scores.get('compliance', 0):.1f}{flag}",
            f"[bold]{r.weighted_total:.2f}[/bold]",
        )

    console.print(session_table)

    # Per-persona summary
    persona_scores: dict[str, list[float]] = {}
    for r in batch.results:
        persona_scores.setdefault(r.persona_id, []).append(r.weighted_total)

    persona_table = Table(box=box.ROUNDED, title="Per-Persona Summary")
    persona_table.add_column("Persona")
    persona_table.add_column("Sessions", justify="right")
    persona_table.add_column("Avg Score", justify="right")

    for persona_id, scores in sorted(persona_scores.items()):
        avg = sum(scores) / len(scores)
        persona_table.add_row(persona_id, str(len(scores)), f"{avg:.2f}")

    console.print(persona_table)

    # Fitness score
    fitness_color = "green" if batch.fitness_score >= 3.0 else "yellow" if batch.fitness_score >= 2.0 else "red"
    console.print(
        f"\n[bold]FITNESS SCORE:[/bold] [{fitness_color}]{batch.fitness_score:.3f}[/{fitness_color}] "
        f"(mean weighted total across {batch.session_count} sessions)\n"
    )


def save_report(batch: BatchEvaluationResult, results_dir: str = "results") -> Path:
    import json
    out_dir = Path(results_dir) / batch.agent_version
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "evaluation_report.json"
    with open(path, "w") as f:
        json.dump(batch.model_dump(), f, indent=2)
    console.print(f"[dim]Report saved to {path}[/dim]")
    return path
