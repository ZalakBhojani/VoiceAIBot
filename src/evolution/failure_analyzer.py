"""
Extracts and summarizes weak sessions from evaluation results.

Reads results/{version}/evaluation_report.json + individual session JSONs
to build WeakSession objects that feed the mutation prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from simulation.session_recorder import ConversationRecord


@dataclass
class WeakSession:
    session_id: str
    persona_id: str
    outcome: str
    weighted_total: float
    metric_scores: dict[str, float]
    metric_reasonings: dict[str, str]
    transcript: str
    failure_summary: str


class FailureAnalyzer:
    def __init__(self, results_dir: str | Path = "results"):
        self.results_dir = Path(results_dir)

    def get_weak_sessions(
        self,
        agent_version: str,
        bottom_fraction: float = 0.25,
        min_sessions: int = 2,
    ) -> list[WeakSession]:
        """Return bottom N sessions by weighted_total score."""
        version_dir = self.results_dir / agent_version
        report_path = version_dir / "evaluation_report.json"

        if not report_path.exists():
            raise FileNotFoundError(
                f"No evaluation report found at {report_path}. "
                "Run run_evaluation.py first."
            )

        with open(report_path) as f:
            report = json.load(f)

        # Build a score map from the report
        session_scores: dict[str, dict] = {}
        for result in report["results"]:
            session_scores[result["session_id"]] = result

        # Load session transcripts
        session_files = [
            f for f in version_dir.glob("*.json")
            if f.name != "evaluation_report.json"
        ]

        sessions: list[WeakSession] = []
        for session_file in session_files:
            sid = session_file.stem
            if sid not in session_scores:
                continue

            result = session_scores[sid]

            with open(session_file) as f:
                session_data = json.load(f)

            transcript = ConversationRecord(**session_data).as_transcript()

            metric_scores = {m["name"]: m["score"] for m in result["metrics"]}
            metric_reasonings = {m["name"]: m["reasoning"] for m in result["metrics"]}

            failure_summary = self._build_failure_summary(
                sid, result["persona_id"], result["outcome"],
                result["weighted_total"], metric_scores, metric_reasonings,
            )

            sessions.append(WeakSession(
                session_id=sid,
                persona_id=result["persona_id"],
                outcome=result["outcome"],
                weighted_total=result["weighted_total"],
                metric_scores=metric_scores,
                metric_reasonings=metric_reasonings,
                transcript=transcript,
                failure_summary=failure_summary,
            ))

        # Sort by score ascending and return bottom fraction
        sessions.sort(key=lambda s: s.weighted_total)
        n = max(min_sessions, int(len(sessions) * bottom_fraction))
        return sessions[:n]

    def summarize_failures(self, weak_sessions: list[WeakSession]) -> str:
        """Build a multi-line failure pattern summary for the mutation prompt."""
        lines = [f"Total weak sessions analyzed: {len(weak_sessions)}", ""]
        for s in weak_sessions:
            lines.append(f"• {s.failure_summary}")
        lines.append("")

        # Identify which metrics are consistently low
        if weak_sessions:
            metric_names = list(weak_sessions[0].metric_scores.keys())
            for metric in metric_names:
                scores = [s.metric_scores.get(metric, 0) for s in weak_sessions]
                avg = sum(scores) / len(scores)
                if avg < 4.0:
                    lines.append(f"⚠ Low average {metric}: {avg:.2f}/5.0")

        return "\n".join(lines)

    @staticmethod
    def _build_failure_summary(
        session_id: str,
        persona_id: str,
        outcome: str,
        weighted_total: float,
        metric_scores: dict[str, float],
        metric_reasonings: dict[str, str],
    ) -> str:
        low_metrics = [
            f"{name}={score:.1f}"
            for name, score in metric_scores.items()
            if score < 4.0
        ]
        parts = [
            f"[{session_id}] {persona_id} | outcome={outcome} | score={weighted_total:.2f}",
        ]
        if low_metrics:
            parts.append(f"  Low metrics: {', '.join(low_metrics)}")
        # Include the reasoning for the lowest-scoring metric
        if metric_scores:
            worst_metric = min(metric_scores, key=lambda k: metric_scores[k])
            worst_score = metric_scores[worst_metric]
            if worst_score < 4.0:
                reasoning = metric_reasonings.get(worst_metric, "")
                parts.append(f"  Evaluator note ({worst_metric}): \"{reasoning}\"")
        return "\n".join(parts)
