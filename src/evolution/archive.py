"""
Persistent archive of agent versions and their evolution history.

archive.json is a flat list of ArchiveEntry objects. The genealogy tree is
reconstructed on demand by walking parent_version pointers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.agent.config_loader import AgentConfig
from src.evaluation.metrics import BatchEvaluationResult


class ArchiveEntry(BaseModel):
    version: str
    parent_version: Optional[str] = None
    fitness_score: Optional[float] = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: Literal["promoted", "failed", "pending"] = "pending"
    mutation_rationale: Optional[str] = None
    failure_addressed: Optional[str] = None
    mutations_applied: list[str] = Field(default_factory=list)
    per_metric_scores: dict[str, float] = Field(default_factory=dict)
    per_persona_scores: dict[str, float] = Field(default_factory=dict)
    simulation_runs: int = 0
    generation: int = 0


class EvolutionArchive(BaseModel):
    schema_version: str = "1"
    entries: list[ArchiveEntry] = Field(default_factory=list)


class ArchiveManager:
    def __init__(self, archive_path: str | Path = "configs/agents/archive.json"):
        self.archive_path = Path(archive_path)
        self._archive: EvolutionArchive = self._load()

    def _load(self) -> EvolutionArchive:
        if self.archive_path.exists():
            with open(self.archive_path) as f:
                return EvolutionArchive(**json.load(f))
        return EvolutionArchive()

    def save(self) -> None:
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.archive_path, "w") as f:
            json.dump(self._archive.model_dump(), f, indent=2)

    def record_version(
        self,
        agent_config: AgentConfig,
        batch_result: Optional[BatchEvaluationResult] = None,
        status: str = "pending",
        generation: int = 0,
    ) -> ArchiveEntry:
        """Create or update an archive entry for the given agent version."""
        per_metric_scores: dict[str, float] = {}
        per_persona_scores: dict[str, float] = {}
        simulation_runs = 0

        if batch_result is not None:
            simulation_runs = batch_result.session_count

            # Aggregate per-metric averages
            metric_totals: dict[str, list[float]] = {}
            persona_totals: dict[str, list[float]] = {}
            for result in batch_result.results:
                for m in result.metrics:
                    metric_totals.setdefault(m.name, []).append(m.score)
                persona_totals.setdefault(result.persona_id, []).append(
                    result.weighted_total
                )
            per_metric_scores = {
                k: round(sum(v) / len(v), 4) for k, v in metric_totals.items()
            }
            per_persona_scores = {
                k: round(sum(v) / len(v), 4) for k, v in persona_totals.items()
            }

        entry = ArchiveEntry(
            version=agent_config.version,
            parent_version=agent_config.parent_version,
            fitness_score=agent_config.fitness_score,
            status=status,
            mutation_rationale=agent_config.mutation_rationale,
            failure_addressed=agent_config.failure_addressed,
            mutations_applied=agent_config.mutations_applied,
            per_metric_scores=per_metric_scores,
            per_persona_scores=per_persona_scores,
            simulation_runs=simulation_runs,
            generation=generation,
        )

        # Replace existing entry for this version or append new one
        for i, existing in enumerate(self._archive.entries):
            if existing.version == agent_config.version:
                self._archive.entries[i] = entry
                return entry
        self._archive.entries.append(entry)
        return entry

    def get_best_agent(self) -> Optional[ArchiveEntry]:
        """Return the promoted entry with the highest fitness score."""
        promoted = [e for e in self._archive.entries if e.status == "promoted"]
        if not promoted:
            return None
        return max(promoted, key=lambda e: e.fitness_score or 0.0)

    def get_lineage(self, version: str) -> list[ArchiveEntry]:
        """Walk parent_version chain from root to the given version."""
        index = {e.version: e for e in self._archive.entries}
        chain: list[ArchiveEntry] = []
        current = index.get(version)
        while current is not None:
            chain.append(current)
            current = index.get(current.parent_version) if current.parent_version else None
        chain.reverse()
        return chain

    def get_latest_generation(self) -> int:
        if not self._archive.entries:
            return 0
        return max(e.generation for e in self._archive.entries)

    def mark_promoted(self, version: str) -> None:
        self._update_status(version, "promoted")

    def mark_failed(self, version: str) -> None:
        self._update_status(version, "failed")

    def _update_status(self, version: str, status: str) -> None:
        for entry in self._archive.entries:
            if entry.version == version:
                entry.status = status
                return

    def get_entry(self, version: str) -> Optional[ArchiveEntry]:
        for entry in self._archive.entries:
            if entry.version == version:
                return entry
        return None

    def get_entries_by_status(self, status: str) -> list[ArchiveEntry]:
        return [e for e in self._archive.entries if e.status == status]

    def get_all_versions(self) -> list[str]:
        return [e.version for e in self._archive.entries]
