"""
LLM-driven mutation engine.

Given a parent AgentConfig and a list of weak sessions, proposes targeted
prompt mutations and produces a new AgentConfig with provenance metadata.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

from src.agent.config_loader import AgentConfig, EvaluationRubric
from src.evolution.failure_analyzer import WeakSession
from src.utils import llm_client


MUTATION_PROMPT_TEMPLATE = """\
You are an expert prompt engineer specializing in debt collection voice AI optimization.
Your task is to improve a debt collection agent's behavior by mutating its prompt configuration.

## Current Agent Configuration (Parent: {parent_version}, Generation: {generation})

### PERSONA HEADER
{persona_header}

### GOAL STATEMENT
{goal_statement}

### BEHAVIORAL GUIDELINES
{behavioral_guidelines}

### CONVERSATION STYLE
{conversation_style}

### OPENING SCRIPT
{opening_script}

### COMPLIANCE RULES (LOCKED — DO NOT MODIFY)
{compliance_rules}

---

## Evaluation Rubric

The agent is scored on three metrics:
1. goal_completion (weight: 40%) — Did the agent secure a repayment commitment?
   Score 5 = clear commitment, 3 = partial engagement, 1 = no progress.
2. conversational_quality (weight: 35%) — How natural and effective was communication?
   Score 5 = human-like, 3 = adequate but mechanical, 1 = robotic/looping.
3. compliance (weight: 25%) — Did the agent follow all rules?
   Score 5 = fully compliant, 1 = serious violations.

Fitness score = mean(0.4*goal + 0.35*quality + 0.25*compliance) across all sessions.
Current parent fitness: {parent_fitness:.4f} / 5.0

---

## Weak Sessions Requiring Improvement

The following {num_weak} sessions scored in the bottom 25%:

{weak_session_details}

---

## Failure Pattern Analysis

{failure_summary}

---

## Your Task

Identify 1-3 prompt sections that, if rewritten, would most directly address the observed
failures. Then provide improved versions of those sections only.

CRITICAL CONSTRAINTS:
- Do NOT modify compliance_rules under any circumstances
- Keep the agent's name (Alex) and company (FinCorp Financial Services)
- Preserve the core goal (arrange repayment commitments)
- Mutations must address specific observed failures, not be generic improvements
- Keep each section concise (under 200 words)
- For conversation_style mutations targeting goodbye/closing loops: add an explicit
  call-termination instruction (e.g., "Once the borrower says goodbye or the call reaches
  a natural conclusion, deliver one final closing sentence and do not continue the conversation")
{llm_param_section}
Return ONLY a valid JSON object with this exact structure:
{{
  "sections_to_mutate": ["section_name1"],
  "new_prompt_values": {{
    "section_name1": "full new text for this section"
  }},
  "llm_param_changes": {{}},
  "rationale": "1-3 sentence explanation of why these changes address the failures",
  "failure_addressed": "brief description of the primary failure being fixed",
  "confidence": 0.75
}}

Valid section names: persona_header, goal_statement, behavioral_guidelines,
conversation_style, opening_script
"""

WEAK_SESSION_TEMPLATE = """\
SESSION {session_id} | Persona: {persona_id} | Outcome: {outcome} | Score: {score:.2f}
Metric breakdown: goal={goal:.1f}, quality={quality:.1f}, compliance={compliance:.1f}
Evaluator reasoning ({worst_metric}): "{worst_reasoning}"

TRANSCRIPT:
{transcript}
---"""


@dataclass
class MutationProposal:
    sections_to_mutate: list[str]
    new_prompt_values: dict[str, str]
    llm_param_changes: dict[str, float]
    rationale: str
    failure_addressed: str
    confidence: float


class MutationEngine:
    MUTABLE_SECTIONS = [
        "persona_header",
        "goal_statement",
        "behavioral_guidelines",
        "conversation_style",
        "opening_script",
    ]
    LOCKED_SECTIONS = ["compliance_rules"]

    TEMP_MIN, TEMP_MAX = 0.1, 0.7
    TOKENS_MIN, TOKENS_MAX = 1000, 4000

    def __init__(
        self,
        allow_compliance_mutation: bool = False,
        allow_llm_param_mutation: bool = False,
        mutator_model: str = "gemini-2.5-pro",
        mutator_temperature: float = 0.4,
        mutator_max_tokens: int = 4000,
    ):
        self.allow_compliance_mutation = allow_compliance_mutation
        self.allow_llm_param_mutation = allow_llm_param_mutation
        self.mutator_model = mutator_model
        self.mutator_temperature = mutator_temperature
        self.mutator_max_tokens = mutator_max_tokens

    async def propose_mutation(
        self,
        parent_config: AgentConfig,
        weak_sessions: list[WeakSession],
        rubric: EvaluationRubric,
        generation: int,
    ) -> MutationProposal:
        prompt = self._build_mutation_prompt(parent_config, weak_sessions, rubric, generation)

        raw = await llm_client.complete(
            model=self.mutator_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.mutator_temperature,
            max_tokens=self.mutator_max_tokens,
            json_mode=True,
        )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Mutator LLM returned invalid JSON: {e}\nRaw: {raw[:500]}")

        proposal = MutationProposal(
            sections_to_mutate=data.get("sections_to_mutate", []),
            new_prompt_values=data.get("new_prompt_values", {}),
            llm_param_changes=data.get("llm_param_changes", {}),
            rationale=data.get("rationale", ""),
            failure_addressed=data.get("failure_addressed", ""),
            confidence=float(data.get("confidence", 0.5)),
        )
        return self._validate_proposal(proposal)

    def apply_mutation(
        self,
        parent_config: AgentConfig,
        proposal: MutationProposal,
        new_version: str,
    ) -> AgentConfig:
        """Return a new AgentConfig with mutated sections and provenance metadata."""
        # Deep copy to avoid mutating parent
        data = parent_config.model_dump()
        new_data = copy.deepcopy(data)

        # Apply prompt section mutations
        for section, new_text in proposal.new_prompt_values.items():
            if section in self.MUTABLE_SECTIONS:
                new_data["prompt"][section] = new_text

        # Apply LLM param mutations
        if "temperature" in proposal.llm_param_changes:
            new_data["llm"]["temperature"] = proposal.llm_param_changes["temperature"]
        if "max_tokens" in proposal.llm_param_changes:
            new_data["llm"]["max_tokens"] = int(proposal.llm_param_changes["max_tokens"])

        # Set provenance metadata
        new_data["version"] = new_version
        new_data["description"] = f"Evolved from {parent_config.version}: {proposal.failure_addressed}"
        new_data["parent_version"] = parent_config.version
        new_data["fitness_score"] = None  # will be filled by evaluator
        new_data["mutation_rationale"] = proposal.rationale
        new_data["failure_addressed"] = proposal.failure_addressed
        new_data["mutations_applied"] = proposal.sections_to_mutate + list(proposal.llm_param_changes.keys())
        new_data["generation"] = parent_config.generation + 1

        return AgentConfig(**new_data)

    def _validate_proposal(self, proposal: MutationProposal) -> MutationProposal:
        """Strip locked sections and clamp LLM params to safe ranges."""
        # Remove locked sections unless explicitly allowed
        if not self.allow_compliance_mutation:
            proposal.sections_to_mutate = [
                s for s in proposal.sections_to_mutate if s not in self.LOCKED_SECTIONS
            ]
            for locked in self.LOCKED_SECTIONS:
                proposal.new_prompt_values.pop(locked, None)

        # Only keep valid mutable section names
        valid = set(self.MUTABLE_SECTIONS if not self.allow_compliance_mutation
                    else self.MUTABLE_SECTIONS + self.LOCKED_SECTIONS)
        proposal.sections_to_mutate = [s for s in proposal.sections_to_mutate if s in valid]
        proposal.new_prompt_values = {
            k: v for k, v in proposal.new_prompt_values.items() if k in valid
        }

        # Strip LLM param mutations if not enabled
        if not self.allow_llm_param_mutation:
            proposal.llm_param_changes = {}
        else:
            # Clamp to safe ranges
            if "temperature" in proposal.llm_param_changes:
                proposal.llm_param_changes["temperature"] = max(
                    self.TEMP_MIN, min(self.TEMP_MAX, float(proposal.llm_param_changes["temperature"]))
                )
            if "max_tokens" in proposal.llm_param_changes:
                proposal.llm_param_changes["max_tokens"] = max(
                    self.TOKENS_MIN, min(self.TOKENS_MAX, int(proposal.llm_param_changes["max_tokens"]))
                )

        return proposal

    def _build_mutation_prompt(
        self,
        parent_config: AgentConfig,
        weak_sessions: list[WeakSession],
        rubric: EvaluationRubric,
        generation: int,
    ) -> str:
        weak_details = "\n\n".join(
            self._format_weak_session(s) for s in weak_sessions
        )
        failure_summary = self._build_failure_summary(weak_sessions)

        llm_param_section = ""
        if self.allow_llm_param_mutation:
            llm_param_section = (
                f"- You may also suggest LLM parameter changes in 'llm_param_changes':\n"
                f"  - 'temperature': float in [{self.TEMP_MIN}, {self.TEMP_MAX}] "
                f"(current: {parent_config.llm.temperature})\n"
                f"  - 'max_tokens': int in [{self.TOKENS_MIN}, {self.TOKENS_MAX}] "
                f"(current: {parent_config.llm.max_tokens})\n"
                f"  Only suggest these if the failure pattern suggests over/under generation.\n"
            )

        return MUTATION_PROMPT_TEMPLATE.format(
            parent_version=parent_config.version,
            generation=generation,
            persona_header=parent_config.prompt.persona_header,
            goal_statement=parent_config.prompt.goal_statement,
            behavioral_guidelines=parent_config.prompt.behavioral_guidelines,
            conversation_style=parent_config.prompt.conversation_style,
            opening_script=parent_config.prompt.opening_script,
            compliance_rules=parent_config.prompt.compliance_rules,
            parent_fitness=parent_config.fitness_score or 0.0,
            num_weak=len(weak_sessions),
            weak_session_details=weak_details,
            failure_summary=failure_summary,
            llm_param_section=llm_param_section,
        )

    @staticmethod
    def _format_weak_session(session: WeakSession) -> str:
        goal = session.metric_scores.get("goal_completion", 0)
        quality = session.metric_scores.get("conversational_quality", 0)
        compliance = session.metric_scores.get("compliance", 0)

        worst_metric = min(session.metric_scores, key=lambda k: session.metric_scores[k])
        worst_reasoning = session.metric_reasonings.get(worst_metric, "")

        return WEAK_SESSION_TEMPLATE.format(
            session_id=session.session_id,
            persona_id=session.persona_id,
            outcome=session.outcome,
            score=session.weighted_total,
            goal=goal,
            quality=quality,
            compliance=compliance,
            worst_metric=worst_metric,
            worst_reasoning=worst_reasoning,
            transcript=session.transcript,
        )

    @staticmethod
    def _build_failure_summary(weak_sessions: list[WeakSession]) -> str:
        lines = []
        persona_counts: dict[str, int] = {}
        low_metrics: dict[str, list[float]] = {}

        for s in weak_sessions:
            persona_counts[s.persona_id] = persona_counts.get(s.persona_id, 0) + 1
            for metric, score in s.metric_scores.items():
                low_metrics.setdefault(metric, []).append(score)

        if persona_counts:
            lines.append("Personas with weak sessions: " +
                         ", ".join(f"{p} ({c}x)" for p, c in persona_counts.items()))

        for metric, scores in low_metrics.items():
            avg = sum(scores) / len(scores)
            if avg < 4.0:
                lines.append(f"Average {metric}: {avg:.2f}/5.0 (needs improvement)")

        return "\n".join(lines) if lines else "No clear pattern detected."
