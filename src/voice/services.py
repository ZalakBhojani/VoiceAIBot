"""
Factory functions for Pipecat STT / LLM / TTS services.
All services are configured from AgentConfig; credentials come from env vars.
"""

import os

from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.vertex.llm import GoogleVertexLLMService

from src.agent.config_loader import AgentConfig


def build_llm_service(config: AgentConfig, system_prompt: str) -> GoogleVertexLLMService:
    """Return a GoogleVertexLLMService configured from AgentConfig.

    Uses Application Default Credentials (ADC) — no explicit key needed.
    """
    return GoogleVertexLLMService(
        project_id=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ["GOOGLE_CLOUD_LOCATION"],
        settings=GoogleVertexLLMService.Settings(
            model=config.llm.model,
            system_instruction=system_prompt,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        ),
    )


def build_stt_service(config: AgentConfig) -> DeepgramSTTService:
    """Return a DeepgramSTTService configured from AgentConfig."""
    return DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramSTTService.Settings(
            model=config.stt.model,
            language=config.stt.language,
            smart_format=True,
            punctuate=True,
        ),
    )


def build_tts_service(config: AgentConfig) -> CartesiaTTSService:
    """Return a CartesiaTTSService configured from AgentConfig."""
    return CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice=config.tts.voice_id,
        ),
    )
