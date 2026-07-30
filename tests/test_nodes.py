from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

import motion_gen.node as motion_gen_node
import sonic.node as sonic_node
from shared.messages import MotionCommandRequest, RuntimeStatus, status_from_arrow


class _Node:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.events = iter(events)
        self.outputs: list[tuple[str, object, dict[str, object]]] = []

    def __iter__(self):
        return self.events

    def try_recv(self):
        return next(self.events, None)

    def send_output(self, output_id, value, **kwargs) -> None:
        self.outputs.append((output_id, value, kwargs))


def _run_motion_gen(monkeypatch, events, generate):
    node = _Node(events)
    generator = SimpleNamespace(generate=generate)
    config = SimpleNamespace(
        device="cpu",
        planner_onnx=Path("planner.onnx"),
        validate_motion_gen=lambda: None,
    )
    monkeypatch.setattr(motion_gen_node.RuntimeConfig, "from_env", lambda: config)
    monkeypatch.setattr(motion_gen_node, "validate_onnx_device", lambda _: None)
    monkeypatch.setattr(motion_gen_node, "Node", lambda: node)
    monkeypatch.setattr(
        motion_gen_node, "PlannerSonic", lambda *args, **kwargs: generator
    )
    monkeypatch.setattr(
        motion_gen_node, "command_from_arrow", lambda value, metadata: value
    )
    monkeypatch.setattr(motion_gen_node, "status_from_arrow", lambda value: value)
    monkeypatch.setattr(
        motion_gen_node, "resample_motion", lambda motion, **kwargs: motion
    )
    monkeypatch.setattr(motion_gen_node, "motion_to_arrow", lambda motion: (motion, {}))
    motion_gen_node.main()
    return node


def test_sonic_error_releases_pending_motion(monkeypatch) -> None:
    first = MotionCommandRequest("first", "walk")
    second = MotionCommandRequest("second", "run")
    generated: list[int] = []

    def generate(command):
        generated.append(command.mode)
        return object()

    _run_motion_gen(
        monkeypatch,
        [
            {"type": "INPUT", "id": "command", "value": first},
            {"type": "INPUT", "id": "command", "value": second},
            {
                "type": "INPUT",
                "id": "sonic_status",
                "value": RuntimeStatus("sonic", "error", "first"),
            },
        ],
        generate,
    )

    assert generated == [2, 3]


def test_motion_gen_does_not_swallow_unexpected_errors(monkeypatch) -> None:
    request = MotionCommandRequest("command", "walk")

    def generate(command):
        raise KeyError("unexpected")

    with pytest.raises(KeyError, match="unexpected"):
        _run_motion_gen(
            monkeypatch,
            [{"type": "INPUT", "id": "command", "value": request}],
            generate,
        )


def test_sonic_error_preserves_motion_command_id(monkeypatch) -> None:
    node = _Node(
        [
            {
                "type": "INPUT",
                "id": "motion",
                "value": pa.array([], type=pa.float32()),
                "metadata": {"command_id": "command"},
            }
        ]
    )
    simulation = SimpleNamespace(device="cpu", cuda_stream_ptr=None)
    controller = sonic_node.SonicController(node, simulation, SimpleNamespace())
    monkeypatch.setattr(
        sonic_node,
        "motion_from_arrow",
        lambda value, metadata: _raise(ValueError("bad")),
    )

    controller.poll()

    status = status_from_arrow(node.outputs[-1][1])
    assert status == RuntimeStatus("sonic", "error", "command", "bad")


def _raise(error: Exception):
    raise error
