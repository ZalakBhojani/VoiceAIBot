"""
LLM-backed persona turn generator.
Given a PersonaConfig and the conversation history so far, generates the next persona response.
"""

from src.agent.config_loader import PersonaConfig
from src.utils import llm_client


async def get_persona_response(
    persona: PersonaConfig,
    history: list[dict],
) -> str:
    """
    Generate the next persona response given conversation history.

    Args:
        persona: Loaded PersonaConfig
        history: List of {"role": "user"/"assistant", "content": "..."} dicts
                 from the persona's perspective (agent turns = "user")

    Returns:
        Persona's next response as plain text
    """
    return await llm_client.complete(
        model=persona.llm.model,
        messages=history,
        system_prompt=persona.system_prompt,
        temperature=persona.llm.temperature,
        max_tokens=200,
    )
