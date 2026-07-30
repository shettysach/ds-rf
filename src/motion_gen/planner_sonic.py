from __future__ import annotations

from pathlib import Path

import numpy as np

from motion_gen.planner_sonic_command import PlannerSonicCommand
from shared.g1 import G1_QPOS_SIZE, standing_qpos
from shared.onnx import create_onnx_session

PLANNER_CONTEXT_FRAMES = 4
PLANNER_MIN_FRAMES = 24
PLANNER_MAX_FRAMES = 64


class PlannerSonicOutputError(RuntimeError):
    """The planner returned output that violates its model contract."""


class PlannerSonic:
    """ONNX Runtime wrapper for NVIDIA's G1 kinematic planner."""

    def __init__(self, model_path: Path, *, device: str = "cpu") -> None:
        self.session = create_onnx_session(model_path, device=device)
        self._validate_signature()
        initial = standing_qpos()
        self._context = np.tile(initial, (1, PLANNER_CONTEXT_FRAMES, 1))

    def generate(self, command: PlannerSonicCommand) -> np.ndarray:
        outputs = self.session.run(
            None,
            {
                "context_mujoco_qpos": self._context,
                "target_vel": np.array([command.target_vel], dtype=np.float32),
                "mode": np.array([command.mode], dtype=np.int64),
                "movement_direction": np.array(
                    [command.movement_direction], dtype=np.float32
                ),
                "facing_direction": np.array(
                    [command.facing_direction], dtype=np.float32
                ),
                "random_seed": np.array([command.random_seed], dtype=np.int64),
                "has_specific_target": np.zeros((1, 1), dtype=np.int64),
                "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
                "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
                # Allow planner_sonic to select its learned 6-16 token horizon.
                "allowed_pred_num_tokens": np.ones((1, 11), dtype=np.int64),
                "height": np.array([command.height], dtype=np.float32),
            },
        )
        padded_qpos = np.asarray(outputs[0], dtype=np.float32)
        frame_count = int(np.asarray(outputs[1]).reshape(-1)[0])
        if not PLANNER_MIN_FRAMES <= frame_count <= PLANNER_MAX_FRAMES:
            raise PlannerSonicOutputError(
                f"Planner returned invalid frame count: {frame_count}"
            )
        qpos = np.ascontiguousarray(padded_qpos[0, :frame_count])
        if qpos.shape != (frame_count, G1_QPOS_SIZE):
            raise PlannerSonicOutputError(
                f"Planner returned unexpected qpos shape: {qpos.shape}"
            )
        if not np.isfinite(qpos).all():
            raise PlannerSonicOutputError(
                "Planner output contains NaN or infinite values"
            )
        quat_norm = np.linalg.norm(qpos[:, 3:7], axis=-1)
        if not np.allclose(quat_norm, 1.0, atol=2e-3):
            raise PlannerSonicOutputError(
                "Planner output contains non-unit root quaternions"
            )

        self._context = qpos[-PLANNER_CONTEXT_FRAMES:][None].copy()
        return qpos

    def _validate_signature(self) -> None:
        expected_inputs = {
            "context_mujoco_qpos",
            "target_vel",
            "mode",
            "movement_direction",
            "facing_direction",
            "random_seed",
            "has_specific_target",
            "specific_target_positions",
            "specific_target_headings",
            "allowed_pred_num_tokens",
            "height",
        }
        actual_inputs = {value.name for value in self.session.get_inputs()}
        if actual_inputs != expected_inputs:
            raise RuntimeError(
                "Unexpected planner inputs: "
                f"missing={sorted(expected_inputs - actual_inputs)}, "
                f"extra={sorted(actual_inputs - expected_inputs)}"
            )
        outputs = self.session.get_outputs()
        if len(outputs) != 2 or outputs[0].name != "mujoco_qpos":
            raise RuntimeError("Unexpected planner output signature")
