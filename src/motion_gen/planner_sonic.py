from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import torch

from shared.g1 import standing_qpos
from shared.onnx import create_onnx_session

PLANNER_CONTEXT_FRAMES = 4
Vector3 = tuple[float, float, float]


class PlannerMode(IntEnum):
    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    SQUAT = 4
    KNEEL_TWO_LEG = 5
    KNEEL_ONE_LEG = 6
    LYING_FACEDOWN = 7
    HAND_CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_JAB = 11
    RIGHT_JAB = 12
    RANDOM_PUNCHES = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    HAPPY = 17
    STEALTH = 18
    INJURED = 19
    CAREFUL = 20
    OBJECT_CARRYING = 21
    CROUCH = 22
    HAPPY_DANCE = 23
    ZOMBIE = 24
    POINT = 25
    SCARED = 26


@dataclass(frozen=True)
class PlannerSonicInput:
    mode: PlannerMode
    movement_direction: Vector3
    facing_direction: Vector3
    target_vel: float = -1.0
    height: float = -1.0
    random_seed: int = 1234


class PlannerSonic:
    """ONNX Runtime wrapper for NVIDIA's G1 kinematic planner."""

    def __init__(self, model_path: Path, *, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.session = create_onnx_session(model_path, device=self.device)
        initial = standing_qpos()
        self._context = np.tile(initial, (1, PLANNER_CONTEXT_FRAMES, 1))

    def generate(self, command: PlannerSonicInput) -> np.ndarray:
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
