import numpy as np

from shared.messages import (
    MotionChunk,
    MotionCommandRequest,
    command_from_arrow,
    command_to_arrow,
    motion_from_arrow,
    motion_to_arrow,
)


def test_command_request_arrow_round_trip() -> None:
    command = MotionCommandRequest.from_text("walk left 0.8")
    assert command.text == "walk left 0.8"
    value, metadata = command_to_arrow(command)
    assert command_from_arrow(value, metadata) == command


def test_motion_arrow_round_trip() -> None:
    chunk = MotionChunk("abc", np.arange(72, dtype=np.float32).reshape(2, 36))
    value, metadata = motion_to_arrow(chunk)
    restored = motion_from_arrow(value, metadata)
    assert restored.command_id == "abc"
    np.testing.assert_array_equal(restored.qpos, chunk.qpos)
