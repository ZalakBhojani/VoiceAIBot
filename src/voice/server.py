"""
WebSocket voice server entry point.

Starts the Pipecat pipeline and handles client lifecycle:
- On connect: injects the opening message to trigger the agent's greeting.
- On disconnect: logs the event.
"""

import asyncio

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.runner import PipelineRunner

from src.agent.config_loader import AgentConfig, load_agent_config_by_version
from src.agent.prompt_builder import get_opening_message
from src.voice.pipeline_factory import build_pipeline

load_dotenv()


async def run_server(
    agent_config: AgentConfig,
    host: str = "localhost",
    port: int = 8765,
) -> None:
    """Run the voice server until interrupted."""
    transport, task, context_aggregator = build_pipeline(agent_config, host=host, port=port)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, websocket):
        logger.info(f"Client connected: {websocket.remote_address}")
        # Inject the scripted opening message as a user turn so the LLM
        # generates the agent greeting and TTS speaks it.
        opening = get_opening_message(agent_config)
        await task.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "begin"}],
                    run_llm=True,
                )
            ]
        )
        logger.info(f"Opening script: {opening!r}")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, websocket):
        logger.info(f"Client disconnected: {websocket.remote_address}")
        await task.cancel()

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    import sys

    version = sys.argv[1] if len(sys.argv) > 1 else "v1"
    config = load_agent_config_by_version(version)
    asyncio.run(run_server(config))
