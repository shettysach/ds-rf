from __future__ import annotations

import math

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_slerp

from shared.messages import SONIC_FPS, MotionChunk

PLANNER_FPS = 30


def resample_motion(
    planner_qpos: np.ndarray,
    *,
    command_id: str,
) -> MotionChunk:
    """Resample planner_sonic output to SONIC's control frequency."""

    qpos = torch.as_tensor(planner_qpos, dtype=torch.float32)
    output_frames = math.floor(qpos.shape[0] * SONIC_FPS / PLANNER_FPS)
    output = torch.empty((output_frames, qpos.shape[1]), dtype=torch.float32)

    for output_index in range(output_frames):
        source_position = output_index * PLANNER_FPS / SONIC_FPS
        index_0 = min(math.floor(source_position), qpos.shape[0] - 1)
        index_1 = min(index_0 + 1, qpos.shape[0] - 1)
        blend = float(source_position - index_0)
        output[output_index] = torch.lerp(qpos[index_0], qpos[index_1], blend)
        # NOTE: MJLab's quat_slerp may negate q2 in place for the shortest path.
        output[output_index, 3:7] = quat_slerp(
            qpos[index_0, 3:7],
            qpos[index_1, 3:7].clone(),
            blend,
        )

    return MotionChunk(command_id=command_id, qpos=output.numpy())
