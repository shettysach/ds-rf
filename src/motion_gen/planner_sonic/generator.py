from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from motion_gen.planner_sonic.parser import planner_mode
from shared.g1 import standing_qpos
from shared.onnx import create_onnx_session

PLANNER_CONTEXT_FRAMES = 4


class PlannerSonic:
    """Text-to-motion wrapper for NVIDIA's G1 kinematic planner."""

    fps = 30

    def __init__(self, model_path: Path, *, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.session = create_onnx_session(model_path, device=self.device)
        initial = standing_qpos()
        self._context = np.tile(initial, (1, PLANNER_CONTEXT_FRAMES, 1))

    def generate(
        self,
        motion: str,
        target_xy: tuple[float, float] | None,
        direction: str | None = None,
    ) -> np.ndarray:
        mode = planner_mode(motion)
        if motion == "stand" and (target_xy is not None or direction is not None):
            raise ValueError("stand requires no target")
        if motion == "walk" and (target_xy is None) == (direction is None):
            raise ValueError("walk requires exactly one target")

        root = self._context[0, -1]
        root_position = root[:3].astype(np.float32)
        yaw = _quaternion_yaw(root[3:7])
        facing = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
        movement = np.zeros(3, dtype=np.float32)
        has_target = np.zeros((1, 1), dtype=np.int64)
        positions = np.zeros((1, PLANNER_CONTEXT_FRAMES, 3), dtype=np.float32)
        headings = np.zeros((1, PLANNER_CONTEXT_FRAMES), dtype=np.float32)

        if direction is not None:
            movement = _direction_vector(direction)
        elif target_xy is not None:
            forward, left = target_xy
            world_delta = np.array(
                [
                    np.cos(yaw) * forward - np.sin(yaw) * left,
                    np.sin(yaw) * forward + np.cos(yaw) * left,
                    0.0,
                ],
                dtype=np.float32,
            )
            distance = float(np.linalg.norm(world_delta[:2]))
            if distance <= 1e-6:
                raise ValueError("walk target_xy must be non-zero")
            movement = world_delta / distance
            endpoint = root_position + world_delta
            positions[:] = endpoint
            headings[:] = yaw
            has_target[:] = 1

        outputs = self.session.run(
            None,
            {
                "context_mujoco_qpos": self._context,
                "target_vel": np.array([-1.0], dtype=np.float32),
                "mode": np.array([mode], dtype=np.int64),
                "movement_direction": movement[None],
                "facing_direction": facing[None],
                "random_seed": np.array([1234], dtype=np.int64),
                "has_specific_target": has_target,
                "specific_target_positions": positions,
                "specific_target_headings": headings,
                # Allow planner_sonic to select its learned 6-16 token horizon.
                "allowed_pred_num_tokens": np.ones((1, 11), dtype=np.int64),
                "height": np.array([-1.0], dtype=np.float32),
            },
        )
        padded_qpos = np.asarray(outputs[0], dtype=np.float32)
        frame_count = int(np.asarray(outputs[1]).reshape(-1)[0])
        qpos = np.ascontiguousarray(padded_qpos[0, :frame_count])
        self._context = qpos[-PLANNER_CONTEXT_FRAMES:][None].copy()
        return qpos


def _quaternion_yaw(quaternion_wxyz: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _direction_vector(direction: str) -> np.ndarray:
    """Return the fixed world-frame vector for a planner direction."""
    vectors = {
        "forward": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "backward": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "left": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "right": np.array([0.0, -1.0, 0.0], dtype=np.float32),
    }
    try:
        return np.asarray(vectors[direction], dtype=np.float32)
    except KeyError as exc:
        raise ValueError(f"Unsupported direction {direction!r}") from exc
