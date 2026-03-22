"""
ConversationRecord model and persistence.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class Turn(BaseModel):
    speaker: str  # "agent" | "persona"
    text: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConversationRecord(BaseModel):
    session_id: str
    agent_version: str
    persona_id: str
    borrower_name: str
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ended_at: Optional[str] = None
    turns: list[Turn] = Field(default_factory=list)
    outcome: str = "incomplete"  # "agreement" | "hangup" | "max_turns" | "incomplete"
    fitness_score: Optional[float] = None  # filled in by evaluator

    def add_turn(self, speaker: str, text: str) -> None:
        self.turns.append(Turn(speaker=speaker, text=text))

    def finalize(self, outcome: str) -> None:
        self.outcome = outcome
        self.ended_at = datetime.now(timezone.utc).isoformat()

    def as_transcript(self) -> str:
        lines = []
        for turn in self.turns:
            label = "AGENT" if turn.speaker == "agent" else "BORROWER"
            lines.append(f"{label}: {turn.text}")
        return "\n".join(lines)

    def save(self, results_dir: str | Path = "results") -> Path:
        out_dir = Path(results_dir) / self.agent_version
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.session_id}.json"
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)
        return path
