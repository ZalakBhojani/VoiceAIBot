"""
Text-only agent<->persona conversation loop.

The agent uses the same system prompt it would use in the voice pipeline,
ensuring simulation fidelity.
"""

import uuid
from src.agent.config_loader import AgentConfig, PersonaConfig
from src.agent.prompt_builder import build_system_prompt, get_opening_message
from src.simulation.persona_simulator import get_persona_response
from src.simulation.session_recorder import ConversationRecord
from src.utils import llm_client
from src.utils.logger import logger


MAX_TURNS = 20


def _check_keywords(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


async def run_conversation(
    agent_config: AgentConfig,
    persona: PersonaConfig,
    max_turns: int = MAX_TURNS,
) -> ConversationRecord:
    """
    Run a full simulated conversation between the agent LLM and a persona LLM.

    Returns a ConversationRecord with the full transcript and outcome.
    """
    session_id = str(uuid.uuid4())[:8]
    record = ConversationRecord(
        session_id=session_id,
        agent_version=agent_config.version,
        persona_id=persona.persona_id,
        borrower_name=persona.context.name,
    )

    agent_system_prompt = build_system_prompt(agent_config, persona.context.name)
    opening = get_opening_message(agent_config, persona.context.name)

    # Conversation histories — each side sees the other as "user"
    # agent_history: from agent's POV (persona responses = "user", agent responses = "assistant")
    # persona_history: from persona's POV (agent responses = "user", persona responses = "assistant")
    agent_history: list[dict] = []
    persona_history: list[dict] = []

    logger.info(f"[{session_id}] Starting conversation — persona: {persona.persona_id}")

    # Agent sends opening message
    record.add_turn("agent", opening)
    persona_history.append({"role": "user", "content": opening})
    logger.debug(f"AGENT: {opening}")

    outcome = "max_turns"

    for turn_num in range(max_turns):
        # --- Persona responds ---
        persona_response = await get_persona_response(persona, persona_history)
        record.add_turn("persona", persona_response)
        persona_history.append({"role": "assistant", "content": persona_response})
        agent_history.append({"role": "user", "content": persona_response})
        logger.debug(f"PERSONA: {persona_response}")

        # Check for hangup
        if _check_keywords(persona_response, persona.hangup_keywords):
            outcome = "hangup"
            logger.info(f"[{session_id}] Persona hung up after {turn_num + 1} turns")
            break

        # Check for resolution
        if _check_keywords(persona_response, persona.resolution_keywords):
            outcome = "agreement"
            logger.info(f"[{session_id}] Agreement reached after {turn_num + 1} turns")
            break

        # --- Agent responds ---
        agent_response = await llm_client.complete(
            model=agent_config.llm.model,
            messages=agent_history,
            system_prompt=agent_system_prompt,
            temperature=agent_config.llm.temperature,
            max_tokens=agent_config.llm.max_tokens,
        )
        record.add_turn("agent", agent_response)
        agent_history.append({"role": "assistant", "content": agent_response})
        persona_history.append({"role": "user", "content": agent_response})
        logger.debug(f"AGENT: {agent_response}")

    record.finalize(outcome)
    logger.info(f"[{session_id}] Conversation ended — outcome: {outcome}, turns: {len(record.turns)}")
    return record
