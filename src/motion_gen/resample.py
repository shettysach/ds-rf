from __future__ import annotations

import math

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_slerp

from shared.messages import SONIC_FPS, MotionChunk


def resample_motion(
    source_qpos: np.ndarray,
    *,
    source_fps: float,
    observation_id: int,
    command: str,
    reasoning: str | None = None,
) -> MotionChunk:
    """Resample backend qpos output to SONIC's control frequency."""

    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError("Source FPS must be positive and finite")

    qpos = torch.as_tensor(source_qpos, dtype=torch.float32)
    output_frames = math.floor(qpos.shape[0] * SONIC_FPS / source_fps)
    output = torch.empty((output_frames, qpos.shape[1]), dtype=torch.float32)

    for output_index in range(output_frames):
        source_position = output_index * source_fps / SONIC_FPS
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

    return MotionChunk(
        observation_id=observation_id,
        command=command,
        qpos=output.numpy(),
        reasoning=reasoning,
    )
