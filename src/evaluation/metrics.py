"""
Pydantic models for evaluation results.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class MetricScore(BaseModel):
    name: str
    score: float          # 1-5
    weight: float         # 0.0-1.0
    reasoning: str
    weighted_score: float = 0.0

    def model_post_init(self, __context):
        self.weighted_score = self.score * self.weight


class EvaluationResult(BaseModel):
    session_id: str
    persona_id: str
    outcome: str
    metrics: list[MetricScore]
    compliance_regex_violation: bool = False
    weighted_total: float = 0.0  # sum of weighted scores

    def model_post_init(self, __context):
        self.weighted_total = sum(m.weighted_score for m in self.metrics)


class BatchEvaluationResult(BaseModel):
    agent_version: str
    session_count: int
    results: list[EvaluationResult]
    fitness_score: float = 0.0  # mean weighted_total — Part 2 optimization target

    def model_post_init(self, __context):
        if self.results:
            self.fitness_score = sum(r.weighted_total for r in self.results) / len(self.results)
