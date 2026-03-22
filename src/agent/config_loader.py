"""
Pydantic models for loading and validating YAML configs.
"""

from __future__ import annotations
from typing import Literal, Optional
from pathlib import Path
import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    provider: Literal["vertex-gemini"]
    model: str
    temperature: float = 0.3
    max_tokens: int = 300


class TTSConfig(BaseModel):
    provider: Literal["cartesia"]
    voice_id: str


class STTConfig(BaseModel):
    provider: Literal["deepgram"]
    model: str = "nova-2"
    language: str = "en-US"


class PromptConfig(BaseModel):
    persona_header: str
    goal_statement: str
    behavioral_guidelines: str
    compliance_rules: str
    conversation_style: str
    opening_script: str


class AgentConfig(BaseModel):
    version: str
    description: str
    parent_version: Optional[str] = None
    fitness_score: Optional[float] = None
    llm: LLMConfig
    tts: TTSConfig
    stt: STTConfig
    prompt: PromptConfig


def load_agent_config(path: str | Path) -> AgentConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return AgentConfig(**data)


def load_agent_config_by_version(version: str, configs_dir: str | Path = "configs") -> AgentConfig:
    path = Path(configs_dir) / "agents" / f"agent_{version}.yaml"
    return load_agent_config(path)


# ---------------------------------------------------------------------------
# Persona config
# ---------------------------------------------------------------------------

class PersonaLLMConfig(BaseModel):
    provider: Literal["vertex-gemini"]
    model: str
    temperature: float = 0.8


class PersonaContext(BaseModel):
    name: str
    loan_amount: float
    months_overdue: int
    reason: str


class PersonaConfig(BaseModel):
    persona_id: str
    llm: PersonaLLMConfig
    context: PersonaContext
    personality_traits: list[str]
    system_prompt: str
    resolution_keywords: list[str] = Field(default_factory=list)
    hangup_keywords: list[str] = Field(default_factory=list)
    agreement_probability: float = 0.5


def load_persona_config(path: str | Path) -> PersonaConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return PersonaConfig(**data)


def load_all_personas(configs_dir: str | Path = "configs") -> list[PersonaConfig]:
    personas_dir = Path(configs_dir) / "personas"
    configs = []
    for yaml_file in sorted(personas_dir.glob("*.yaml")):
        configs.append(load_persona_config(yaml_file))
    return configs


# ---------------------------------------------------------------------------
# Evaluation rubric config
# ---------------------------------------------------------------------------

class MetricConfig(BaseModel):
    name: str
    weight: float
    description: str
    scale_min: int = 1
    scale_max: int = 5


class EvaluatorLLMConfig(BaseModel):
    provider: Literal["vertex-gemini"]
    model: str


class EvaluationRubric(BaseModel):
    version: str
    evaluator: EvaluatorLLMConfig
    metrics: list[MetricConfig]
    forbidden_phrases: list[str] = Field(default_factory=list)


def load_rubric(path: str | Path = "configs/evaluation/rubric_v1.yaml") -> EvaluationRubric:
    with open(path) as f:
        data = yaml.safe_load(f)
    return EvaluationRubric(**data)
