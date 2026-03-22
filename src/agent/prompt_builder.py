"""
Assembles the agent system prompt from AgentConfig sections.
Each section is independently mutable — Part 2's mutation operator targets them individually.
"""

from src.agent.config_loader import AgentConfig


def build_system_prompt(config: AgentConfig, borrower_name: str = "the borrower") -> str:
    """Concatenate prompt sections and substitute borrower context."""
    p = config.prompt
    sections = [
        p.persona_header,
        f"GOAL: {p.goal_statement}",
        f"BEHAVIORAL GUIDELINES:\n{p.behavioral_guidelines}",
        f"COMPLIANCE RULES:\n{p.compliance_rules}",
        f"CONVERSATION STYLE:\n{p.conversation_style}",
    ]
    prompt = "\n\n".join(sections)
    prompt = prompt.replace("[BORROWER_NAME]", borrower_name)
    return prompt


def get_opening_message(config: AgentConfig, borrower_name: str = "the borrower") -> str:
    return config.prompt.opening_script.replace("[BORROWER_NAME]", borrower_name)
