"""
Re-exports load_rubric for convenience.
"""

from src.agent.config_loader import EvaluationRubric, load_rubric

__all__ = ["EvaluationRubric", "load_rubric"]
