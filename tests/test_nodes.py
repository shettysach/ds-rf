from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pytest
import torch

import nodes.motion_gen as motion_gen_node
import sonic.runtime as sonic_runtime
from shared.messages import (
    AgentCommand,
    MotionChunk,
    agent_command_to_arrow,
    motion_from_arrow,
    motion_to_arrow,
    observation_from_arrow,
    pipeline_error_from_arrow,
)


class _Node:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.events = iter(events)
        self.outputs: list[tuple[str, object, dict[str, object]]] = []
        self.logs: list[tuple[str, str, dict[str, object]]] = []

    def __iter__(self):
        return self.events

    def send_output(self, output_id, value, **kwargs) -> None:
        self.outputs.append((output_id, value, kwargs))

    def log(self, level, message, **kwargs) -> None:
        self.logs.append((level, message, kwargs))


def _command_event(observation_id: int, text: str) -> dict[str, object]:
    value, metadata = agent_command_to_arrow(AgentCommand(observation_id, text))
    return {
        "type": "INPUT",
        "id": "command",
        "value": value,
        "metadata": metadata,
    }


def _run_motion_gen(
    monkeypatch,
    events,
    generate,
    *,
    generator_name="planner_sonic",
):
    node = _Node(events)
    generator = SimpleNamespace(generate=generate, fps=30)
    config = SimpleNamespace(
        generator=generator_name,
        device="cpu",
        planner_onnx=Path("planner.onnx"),
    )
    monkeypatch.setattr(motion_gen_node.MotionGenConfig, "from_env", lambda: config)
    monkeypatch.setattr(motion_gen_node, "Node", lambda: node)
    monkeypatch.setattr(motion_gen_node, "_create_generator", lambda cfg: generator)
    motion_gen_node.main()
    return node


def _planner_motion() -> np.ndarray:
    qpos = np.zeros((2, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    return qpos


def test_motion_gen_generates_one_segment_per_command(monkeypatch) -> None:
    generated: list[str] = []

    def generate(text):
        generated.append(text)
        return _planner_motion()

    node = _run_motion_gen(
        monkeypatch,
        [_command_event(4, "walk forward 0.4")],
        generate,
    )

    motions = [output for output in node.outputs if output[0] == "motion"]
    assert generated == ["walk forward 0.4"]
    assert len(motions) == 1
    _, value, kwargs = motions[0]
    chunk = motion_from_arrow(value, kwargs["metadata"])
    assert chunk.observation_id == 4
    assert chunk.command == "walk forward 0.4"
    assert any("motion generated" in message for _, message, _ in node.logs)


def test_motion_gen_preserves_ardy_root_z(monkeypatch) -> None:
    node = _run_motion_gen(
        monkeypatch,
        [_command_event(4, "ardy smoke test")],
        lambda text: _planner_motion(),
        generator_name="ardy",
    )

    motion = next(output for output in node.outputs if output[0] == "motion")
    chunk = motion_from_arrow(motion[1], motion[2]["metadata"])
    assert chunk.preserve_root_z


def test_motion_gen_reports_invalid_raw_vlm_response(monkeypatch) -> None:
    def generate(text):
        raise ValueError(f"Unknown planner_sonic mode {text.split()[0].lower()!r}")

    node = _run_motion_gen(
        monkeypatch,
        [_command_event(5, "I think the robot should walk")],
        generate,
    )

    errors = [output for output in node.outputs if output[0] == "error"]
    assert len(errors) == 1
    error = pipeline_error_from_arrow(errors[0][1])
    assert error.source == "motion-gen"
    assert error.observation_id == 5
    assert "Unknown planner_sonic mode" in error.detail


def test_motion_gen_does_not_swallow_planner_errors(monkeypatch) -> None:
    def generate(command):
        raise KeyError("unexpected")

    with pytest.raises(KeyError, match="unexpected"):
        _run_motion_gen(
            monkeypatch,
            [_command_event(0, "walk forward")],
            generate,
        )


class _Simulation:
    device = "cpu"

    def __init__(self) -> None:
        self.steps = 0

    def compute_context(self):
        return nullcontext()

    def robot_state(self):
        return SimpleNamespace(
            root_pos_w=torch.zeros(3),
            root_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )

    def step(self, action) -> None:
        del action
        self.steps += 1


class _Policy:
    def __init__(self) -> None:
        self.calls = 0
        self.loaded: MotionChunk | None = None

    def load_motion(self, chunk, root_pos_w, root_quat_w) -> None:
        del root_pos_w, root_quat_w
        self.loaded = chunk

    def infer(self, state):
        del state
        self.calls += 1
        return torch.zeros((1, 29)), self.calls == 2


class _Renderer:
    def __init__(self, simulation: _Simulation) -> None:
        self.simulation = simulation
        self.capture_steps: list[int] = []

    def capture_jpeg(self) -> bytes:
        self.capture_steps.append(self.simulation.steps)
        return f"jpeg-{self.simulation.steps}".encode()


class _Viewer:
    def __init__(self, simulation: _Simulation) -> None:
        self.simulation = simulation
        self.sync_steps: list[int] = []

    def sync(self) -> None:
        self.sync_steps.append(self.simulation.steps)

    def close(self) -> None:
        pass


def _motion_event(chunk: MotionChunk) -> dict[str, object]:
    value, metadata = motion_to_arrow(chunk)
    return {
        "type": "INPUT",
        "id": "motion",
        "value": value,
        "metadata": metadata,
    }


def test_sonic_steps_final_action_before_capture(monkeypatch) -> None:
    qpos = np.zeros((2, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    chunk = MotionChunk(0, "walk forward", qpos)
    node = _Node([_motion_event(chunk), {"type": "STOP"}])
    simulation = _Simulation()
    policy = _Policy()
    renderer = _Renderer(simulation)
    viewer = _Viewer(simulation)
    monkeypatch.setattr(sonic_runtime.time, "sleep", lambda delay: None)

    runtime = sonic_runtime.SonicRuntime(
        cast(Any, node),
        cast(Any, simulation),
        cast(Any, policy),
        cast(Any, renderer),
        cast(Any, viewer),
    )
    runtime.run()

    assert simulation.steps == 2
    assert viewer.sync_steps == [1, 2]
    assert renderer.capture_steps == [0, 2]
    observations = [output for output in node.outputs if output[0] == "observation"]
    first = observation_from_arrow(
        observations[0][1], cast(Any, observations[0][2]["metadata"])
    )
    second = observation_from_arrow(
        observations[1][1], cast(Any, observations[1][2]["metadata"])
    )
    assert first.observation_id == 0
    assert first.completed_command is None
    assert second.observation_id == 1
    assert second.completed_command == "walk forward"
    assert any("[OBS 0->1] motion complete" in message for _, message, _ in node.logs)


def test_sonic_rejects_motion_for_stale_observation() -> None:
    node = _Node([])
    simulation = _Simulation()
    runtime = sonic_runtime.SonicRuntime(
        cast(Any, node),
        cast(Any, simulation),
        cast(Any, _Policy()),
        cast(Any, _Renderer(simulation)),
    )
    runtime._accept_motion(
        {
            "type": "INPUT",
            "id": "motion",
            "value": pa.array(np.zeros(72), type=pa.float32()),
            "metadata": {"observation_id": "3", "command": "stand"},
        }
    )

    error = pipeline_error_from_arrow(node.outputs[-1][1])
    assert error.observation_id == 0
    assert "Expected motion for observation 0, got 3" in error.detail
