import numpy as np

from shared.messages import (
    AgentCommand,
    MotionChunk,
    PipelineError,
    VisualObservation,
    agent_command_from_arrow,
    agent_command_to_arrow,
    motion_from_arrow,
    motion_to_arrow,
    observation_from_arrow,
    observation_to_arrow,
    pipeline_error_from_arrow,
    pipeline_error_to_arrow,
)


def test_motion_arrow_round_trip() -> None:
    chunk = MotionChunk(
        7,
        "walk forward 0.4",
        np.arange(72, dtype=np.float32).reshape(2, 36),
    )
    value, metadata = motion_to_arrow(chunk)
    restored = motion_from_arrow(value, metadata)
    assert restored.observation_id == 7
    assert restored.command == "walk forward 0.4"
    np.testing.assert_array_equal(restored.qpos, chunk.qpos)


def test_agent_command_arrow_round_trip() -> None:
    command = AgentCommand(3, "walk left 0.3 facing=forward")
    value, metadata = agent_command_to_arrow(command)
    assert agent_command_from_arrow(value, metadata) == command


def test_observation_arrow_round_trip() -> None:
    observation = VisualObservation(4, "stand", b"jpeg")
    value, metadata = observation_to_arrow(observation)
    assert observation_from_arrow(value, metadata) == observation


def test_pipeline_error_arrow_round_trip() -> None:
    error = PipelineError("motion-gen", 2, "bad command")
    assert pipeline_error_from_arrow(pipeline_error_to_arrow(error)) == error
