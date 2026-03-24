"""
LLM-based scoring engine.

For each ConversationRecord:
1. Regex pre-check for forbidden phrases (auto-caps compliance at 2)
2. Calls Gemini 2.5 Pro concurrently for all 3 metrics in JSON mode
3. Returns EvaluationResult
"""

import asyncio
import json
import re
from src.agent.config_loader import EvaluationRubric, MetricConfig
from src.evaluation.metrics import MetricScore, EvaluationResult, BatchEvaluationResult
from src.simulation.session_recorder import ConversationRecord
from src.utils import llm_client
from src.utils.logger import logger


def _has_forbidden_phrase(transcript: str, forbidden: list[str]) -> bool:
    transcript_lower = transcript.lower()
    return any(phrase.lower() in transcript_lower for phrase in forbidden)


def _build_metric_prompt(metric: MetricConfig, transcript: str, outcome: str) -> str:
    return f"""You are an objective evaluator of debt collection calls. Score the following
conversation transcript on ONE specific metric and return a JSON object.

METRIC: {metric.name}
DESCRIPTION: {metric.description}
SCALE: {metric.scale_min} (worst) to {metric.scale_max} (best)
CALL OUTCOME: {outcome}

TRANSCRIPT:
{transcript}

Return ONLY a valid JSON object with this exact structure:
{{"score": <integer {metric.scale_min}-{metric.scale_max}>, "reasoning": "<1-2 sentence explanation>"}}"""


async def _score_metric(
    metric: MetricConfig,
    transcript: str,
    outcome: str,
    evaluator_model: str,
    compliance_capped: bool = False,
) -> MetricScore:
    prompt = _build_metric_prompt(metric, transcript, outcome)

    raw = await llm_client.complete(
        model=evaluator_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000,
        json_mode=True,
    )

    logger.info(raw)

    try:
        data = json.loads(raw)
        score = float(data["score"])
        reasoning = data.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Failed to parse evaluator response for {metric.name}: {e}. Raw: {raw}")
        score = 3.0
        reasoning = "Parse error — defaulted to 3"

    # Cap compliance at 2 if regex violation found
    if metric.name == "compliance" and compliance_capped:
        score = min(score, 2.0)
        reasoning = f"[AUTO-CAPPED: forbidden phrase detected] {reasoning}"

    score = max(float(metric.scale_min), min(float(metric.scale_max), score))

    return MetricScore(
        name=metric.name,
        score=score,
        weight=metric.weight,
        reasoning=reasoning,
    )


async def evaluate_session(
    record: ConversationRecord,
    rubric: EvaluationRubric,
) -> EvaluationResult:
    transcript = record.as_transcript()
    compliance_violation = _has_forbidden_phrase(transcript, rubric.forbidden_phrases)

    if compliance_violation:
        logger.warning(f"[{record.session_id}] Forbidden phrase detected in transcript")

    # Score all metrics concurrently
    tasks = [
        _score_metric(
            metric=metric,
            transcript=transcript,
            outcome=record.outcome,
            evaluator_model=rubric.evaluator.model,
            compliance_capped=(compliance_violation and metric.name == "compliance"),
        )
        for metric in rubric.metrics
    ]
    metric_scores = await asyncio.gather(*tasks)

    return EvaluationResult(
        session_id=record.session_id,
        persona_id=record.persona_id,
        outcome=record.outcome,
        metrics=list(metric_scores),
        compliance_regex_violation=compliance_violation,
    )


async def score_batch(
    records: list[ConversationRecord],
    rubric: EvaluationRubric,
    agent_version: str,
) -> BatchEvaluationResult:
    results = list(
        await asyncio.gather(*[evaluate_session(record, rubric) for record in records])
    )
    for result in results:
        logger.info(
            f"[{result.session_id}] Scored — weighted_total: {result.weighted_total:.2f}"
        )

    return BatchEvaluationResult(
        agent_version=agent_version,
        session_count=len(results),
        results=results,
    )
