"""
Builds the Pipecat 0.0.106 voice pipeline from an AgentConfig.

Pipeline chain:
    transport.input()
        -> DeepgramSTTService
        -> context_aggregator.user()
        -> GoogleVertexLLMService
        -> CartesiaTTSService
        -> context_aggregator.assistant()
        -> transport.output()
"""

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.transports.websocket.server import WebsocketServerParams, WebsocketServerTransport

from src.agent.config_loader import AgentConfig
from src.agent.prompt_builder import build_system_prompt
from src.voice.end_call_detector import HangupPhraseDetector
from src.voice.services import build_llm_service, build_stt_service, build_tts_service


def build_pipeline(
    agent_config: AgentConfig,
    host: str = "localhost",
    port: int = 8765,
) -> tuple[WebsocketServerTransport, PipelineTask, LLMContextAggregatorPair]:
    """Construct the voice pipeline.

    Returns:
        transport: WebSocket server transport — attach event handlers to this.
        task: PipelineTask ready to pass to PipelineRunner.
        context_aggregator: Use to inject frames (e.g. opening trigger on connect).
    """
    system_prompt = build_system_prompt(agent_config)

    # Transport
    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            vad_analyzer=SileroVADAnalyzer(),
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            # session_timeout=60 * 3,  # 3 minutes
        ),
        host=host,
        port=port,
    )

    # Services
    stt = build_stt_service(agent_config)
    llm = build_llm_service(agent_config, system_prompt)
    tts = build_tts_service(agent_config)
    hangup_detector = HangupPhraseDetector(hangup_phrases=agent_config.hangup_phrases)

    # Context — system instruction lives in LLM settings; context starts empty.
    # The opening message is injected via LLMMessagesAppendFrame on client connect.
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    # Pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            hangup_detector,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            # audio_in_sample_rate=16000,
            # audio_out_sample_rate=16000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    return transport, task, context_aggregator
