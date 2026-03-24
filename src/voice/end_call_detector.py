"""
HangupPhraseDetector — terminates the pipeline after the agent finishes speaking
a goodbye phrase.

Placed between the LLM and TTS in the pipeline:

  llm → [HangupPhraseDetector] → tts → transport.output()

Flow:
1. A TextFrame containing a hangup phrase arrives (downstream, from LLM).
   The frame is passed through normally so TTS speaks it. A flag is set.
2. After TTS finishes, the transport emits BotStoppedSpeakingFrame upstream.
3. When BotStoppedSpeakingFrame reaches this processor with the flag set,
   EndFrame is pushed downstream — closing the WebSocket cleanly after the
   goodbye audio has already been sent.
"""

from pipecat.frames.frames import BotStoppedSpeakingFrame, EndTaskFrame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from loguru import logger


class HangupPhraseDetector(FrameProcessor):
    """Shuts down the pipeline after the agent finishes saying a hangup phrase."""

    def __init__(self, hangup_phrases: list[str], **kwargs):
        super().__init__(**kwargs)
        self._phrases = [p.lower() for p in hangup_phrases]
        self._triggered = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        # Step 1: detect terminal phrase in LLM output, pass it through for TTS
        if isinstance(frame, TextFrame) and not self._triggered:
            if any(phrase in frame.text.lower() for phrase in self._phrases):
                self._triggered = True
                logger.info(
                    f"[HangupDetector] Terminal phrase detected: {frame.text!r} — "
                    "will close after bot finishes speaking"
                )

        # Step 2: once TTS has fully played, close the pipeline
        elif isinstance(frame, BotStoppedSpeakingFrame) and self._triggered:
            logger.info("[HangupDetector] Bot finished speaking — sending EndFrame")
            await self.push_frame(frame, direction)
            await self.push_frame(EndTaskFrame("Bot came up with the hangup_phase"), FrameDirection.UPSTREAM)
            return

        await self.push_frame(frame, direction)
