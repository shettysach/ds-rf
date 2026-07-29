from __future__ import annotations

import math

import torch
from mjlab.utils.lab_api.math import quat_slerp

from motion_gen.backend import NativeMotion
from shared.messages import MotionChunk


def resample_motion(
    motion: NativeMotion,
    *,
    command_id: str,
    output_fps: int = 50,
) -> MotionChunk:
    """Match the official planner's position lerp and quaternion slerp."""

    qpos = torch.as_tensor(motion.qpos, dtype=torch.float32)
    output_frames = math.floor(qpos.shape[0] * output_fps / motion.fps)
    output = torch.empty((output_frames, qpos.shape[1]), dtype=torch.float32)

    for output_index in range(output_frames):
        source_position = output_index * motion.fps / output_fps
        index_0 = min(math.floor(source_position), qpos.shape[0] - 1)
        index_1 = min(index_0 + 1, qpos.shape[0] - 1)
        blend = float(source_position - index_0)
        output[output_index] = torch.lerp(qpos[index_0], qpos[index_1], blend)
        # NOTE: MJLab's quat_slerp may negate q2 in place for the shortest path.
        # Clone both source views so resampling cannot mutate the stored trajectory.
        output[output_index, 3:7] = quat_slerp(
            qpos[index_0, 3:7].clone(),
            qpos[index_1, 3:7].clone(),
            blend,
        )

    return MotionChunk(command_id=command_id, qpos=output.numpy(), fps=output_fps)
