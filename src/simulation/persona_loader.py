"""
Loads persona configs from YAML files.
Re-exports load_persona_config and load_all_personas for convenience.
"""

from src.agent.config_loader import (
    PersonaConfig,
    load_persona_config,
    load_all_personas,
)

__all__ = ["PersonaConfig", "load_persona_config", "load_all_personas"]
