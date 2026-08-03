from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from motion_gen.planner_sonic.parser import PlannerSonicInput, parse_motion_command
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

    def generate(self, text: str) -> np.ndarray:
        return self._generate(parse_motion_command(text))

    def _generate(self, command: PlannerSonicInput) -> np.ndarray:
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
        qpos = np.ascontiguousarray(padded_qpos[0, :frame_count])
        self._context = qpos[-PLANNER_CONTEXT_FRAMES:][None].copy()
        return qpos
