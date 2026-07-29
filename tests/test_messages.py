import numpy as np

from shared.messages import (
    MotionChunk,
    PlannerCommand,
    command_from_arrow,
    command_to_arrow,
    motion_from_arrow,
    motion_to_arrow,
)


def test_command_parser_and_arrow_round_trip() -> None:
    command = PlannerCommand.parse("walk left 0.8")
    assert command.mode == 2
    assert command.movement_direction == (0.0, 1.0, 0.0)
    assert command.target_vel == 0.8
    assert command_from_arrow(command_to_arrow(command)) == command


def test_motion_arrow_round_trip() -> None:
    chunk = MotionChunk("abc", np.arange(72, dtype=np.float32).reshape(2, 36))
    value, metadata = motion_to_arrow(chunk)
    restored = motion_from_arrow(value, metadata)
    assert restored.command_id == "abc"
    assert restored.fps == 50
    np.testing.assert_array_equal(restored.qpos, chunk.qpos)
