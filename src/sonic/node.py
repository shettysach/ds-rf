from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, cast

import torch
from dora import Node

from shared.config import RuntimeConfig
from shared.messages import (
    RuntimeStatus,
    motion_from_arrow,
    status_to_arrow,
)
from sonic.mjlab_env import SonicMjlabEnv
from sonic.policy import SonicPolicy


class SonicController:
    def __init__(
        self,
        node: Node,
        simulation: SonicMjlabEnv,
        policy: SonicPolicy,
    ) -> None:
        self.node = node
        self.simulation = simulation
        self.policy = policy
        self.stopped = False
        self.on_stop: Callable[[], None] | None = None
        self.node.send_output(
            "status", status_to_arrow(RuntimeStatus("sonic", "ready"))
        )

    def __call__(self, obs: Any) -> torch.Tensor:
        del obs
        self.poll()
        state = self.simulation.robot_state()
        action, completed = self.policy.infer(state)
        if completed is not None:
            self.node.send_output(
                "status",
                status_to_arrow(RuntimeStatus("sonic", "done", completed)),
            )
        return torch.from_numpy(action).to(self.simulation.env.device).unsqueeze(0)

    def reset(self) -> None:
        self.policy.reset()

    def poll(self) -> None:
        while True:
            event = cast(Any, self.node).try_recv()
            if event is None:
                return
            if event["type"] == "STOP":
                self.stopped = True
                if self.on_stop is not None:
                    self.on_stop()
                return
            if event["type"] != "INPUT" or event["id"] != "motion":
                continue
            try:
                chunk = motion_from_arrow(
                    event["value"], dict(event.get("metadata") or {})
                )
                state = self.simulation.robot_state()
                self.policy.load_motion(chunk, state.root_quat_w)
                self.node.send_output(
                    "status",
                    status_to_arrow(
                        RuntimeStatus("sonic", "playing", chunk.command_id)
                    ),
                )
            except Exception as exc:
                self.node.send_output(
                    "status",
                    status_to_arrow(RuntimeStatus("sonic", "error", detail=str(exc))),
                )


def main() -> None:
    cfg = RuntimeConfig.from_env()
    cfg.validate_sonic()
    if cfg.onnx_provider != "cpu" or cfg.device != "cpu":
        raise NotImplementedError(
            "This implementation is the CPU device; cu128 support comes later"
        )

    node = Node()
    simulation = SonicMjlabEnv(device=cfg.device)
    try:
        policy = SonicPolicy(cfg.sonic_dir, provider=cfg.onnx_provider)
        controller = SonicController(node, simulation, policy)
        if cfg.viewer == "native":
            _run_native(simulation, controller)
        else:
            _run_headless(simulation, controller)
    finally:
        simulation.close()


def _run_native(simulation: SonicMjlabEnv, controller: SonicController) -> None:
    from mjlab.viewer.native import NativeMujocoViewer

    viewer = NativeMujocoViewer(
        cast(Any, simulation.env), cast(Any, controller), frame_rate=60.0
    )
    controller.on_stop = lambda: setattr(viewer, "_interrupted", True)
    viewer.run()


def _run_headless(simulation: SonicMjlabEnv, controller: SonicController) -> None:
    control_period = simulation.env.step_dt
    while not controller.stopped:
        started = time.perf_counter()
        simulation.env.step(controller(None))
        remaining = control_period - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)


if __name__ == "__main__":
    main()
