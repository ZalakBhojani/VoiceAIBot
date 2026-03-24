"""
VoiceBotSerializer — extends ProtobufFrameSerializer to handle control frames
that the transport tries to send but the base serializer rejects.

InterruptionFrame, CancelFrame, and EndFrame are converted to
OutputTransportMessageFrame with a JSON payload, which the base serializer
already knows how to encode. pipecat-client-js surfaces these via onMessage.
"""

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    EndTaskFrame,
    Frame,
    InterruptionFrame,
    OutputTransportMessageFrame,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer


class VoiceBotSerializer(ProtobufFrameSerializer):
    """Protobuf serializer that also handles pipeline control frames."""

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, InterruptionFrame):
            frame = OutputTransportMessageFrame(message={"type": "interruption"})
        elif isinstance(frame, CancelFrame):
            frame = OutputTransportMessageFrame(message={"type": "cancel"})
        elif isinstance(frame, EndFrame):
            frame = OutputTransportMessageFrame(message={"type": "end"})
        elif isinstance(frame, EndTaskFrame):
            frame = OutputTransportMessageFrame(message={"type": "end"})
        return await super().serialize(frame)
