from __future__ import annotations

import time
from typing import Any

import torch
from dora import Node

from shared.messages import (
    SONIC_FPS,
    PipelineError,
    VisualObservation,
    motion_from_arrow,
    observation_to_arrow,
    pipeline_error_to_arrow,
)
from sonic.mjlab_env import SonicMjlabEnv
from sonic.policy import SonicPolicy
from sonic.renderer import SonicRenderer

CONTROL_PERIOD = 1.0 / SONIC_FPS


class SonicRuntime:
    def __init__(
        self,
        node: Node,
        simulation: SonicMjlabEnv,
        policy: SonicPolicy,
        renderer: SonicRenderer,
    ) -> None:
        self.node = node
        self.simulation = simulation
        self.policy = policy
        self.renderer = renderer
        self.observation_id = 0

    def run(self) -> None:
        self._publish_observation(completed_command=None)
        for event in self.node:
            if event["type"] == "STOP":
                return
            if event["type"] != "INPUT" or event["id"] != "motion":
                continue
            self._accept_motion(event)

    def _accept_motion(self, event: dict[str, Any]) -> None:
        metadata = dict(event.get("metadata") or {})
        try:
            chunk = motion_from_arrow(event["value"], metadata)
        except (KeyError, TypeError, ValueError) as exc:
            self._report_error(str(exc))
            return
        if chunk.observation_id != self.observation_id:
            self._report_error(
                f"Expected motion for observation {self.observation_id}, got "
                f"{chunk.observation_id}"
            )
            return

        with self.simulation.compute_context():
            state = self.simulation.robot_state()
            try:
                self.policy.load_motion(chunk, state.root_quat_w)
            except ValueError as exc:
                self._report_error(str(exc))
                return

        self._execute()
        self.observation_id += 1
        self._publish_observation(completed_command=chunk.command)

    def _execute(self) -> None:
        next_step = time.perf_counter()
        with torch.no_grad():
            while True:
                delay = next_step - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)

                with self.simulation.compute_context():
                    state = self.simulation.robot_state()
                    action, completed = self.policy.infer(state)
                self.simulation.step(action)

                # Completion is detected while producing the last reference
                # action. Capture only after that action's physics step.
                if completed:
                    return

                next_step += CONTROL_PERIOD
                now = time.perf_counter()
                if next_step < now:
                    # Do not execute burst catch-up steps after an overrun.
                    next_step = now

    def _publish_observation(self, *, completed_command: str | None) -> None:
        observation = VisualObservation(
            observation_id=self.observation_id,
            completed_command=completed_command,
            jpeg=self.renderer.capture_jpeg(),
        )
        data, metadata = observation_to_arrow(observation)
        self.node.send_output("observation", data, metadata=metadata)

    def _report_error(self, detail: str) -> None:
        error = PipelineError("sonic", self.observation_id, detail)
        self.node.send_output("error", pipeline_error_to_arrow(error))
