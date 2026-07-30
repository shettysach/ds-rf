from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, cast

import torch
from dora import Node

from shared.config import RuntimeConfig
from shared.messages import (
    RuntimeStatus,
    StatusState,
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
        stream = self.simulation.cuda_stream_ptr
        stream_detail = "none" if stream is None else hex(stream)
        self._report(
            StatusState.READY,
            detail=f"device={simulation.device}, stream={stream_detail}",
        )

    def __call__(self, obs: object) -> torch.Tensor:
        del obs
        with self.simulation.compute_context():
            self.poll()
            state = self.simulation.robot_state()
            action, completed = self.policy.infer(state)
        if completed is not None:
            self._report(StatusState.DONE, completed)
        return action

    def reset(self) -> None:
        with self.simulation.compute_context():
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
            metadata = dict(event.get("metadata") or {})
            command_id_value = metadata.get("command_id")
            command_id = str(command_id_value) if command_id_value is not None else None
            try:
                chunk = motion_from_arrow(event["value"], metadata)
            except (KeyError, TypeError, ValueError) as exc:
                self._report(StatusState.ERROR, command_id, str(exc))
                continue

            state = self.simulation.robot_state()
            try:
                self.policy.load_motion(chunk, state.root_quat_w)
            except ValueError as exc:
                self._report(StatusState.ERROR, chunk.command_id, str(exc))
                continue

            self._report(StatusState.PLAYING, chunk.command_id)

    def _report(
        self,
        state: StatusState,
        command_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        status = RuntimeStatus("sonic", state, command_id, detail)
        self.node.send_output("status", status_to_arrow(status))


def main() -> None:
    cfg = RuntimeConfig.from_env()
    cfg.validate_sonic()

    node = Node()
    simulation = SonicMjlabEnv(device=cfg.device)
    try:
        with simulation.compute_context():
            policy = SonicPolicy(
                cfg.sonic_dir,
                device=cfg.device,
                cuda_stream=simulation.cuda_stream,
            )
        controller = SonicController(node, simulation, policy)
        if cfg.viewer == "native":
            _run_native(simulation, controller)
        else:
            _run_headless(simulation, controller)
    finally:
        simulation.close()


def _run_native(simulation: SonicMjlabEnv, controller: SonicController) -> None:
    from mjlab.viewer.native import NativeMujocoViewer

    viewer = NativeMujocoViewer(simulation, controller, frame_rate=60.0)
    # NOTE: MJLab has no public deferred-stop API; close() is unsafe mid-viewer tick.
    controller.on_stop = lambda: setattr(viewer, "_interrupted", True)
    viewer.run()


def _run_headless(simulation: SonicMjlabEnv, controller: SonicController) -> None:
    control_period = simulation.step_dt
    while not controller.stopped:
        started = time.perf_counter()
        simulation.step(controller(None))
        remaining = control_period - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)


if __name__ == "__main__":
    main()
