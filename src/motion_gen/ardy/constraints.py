from __future__ import annotations

import torch

from motion_gen.ardy.parser import Vector2

VELOCITY_TRANSITION_S = 2.0
VELOCITY_CONSTRAINT_INTERVAL = 10


def build_velocity_constraints(
    motion_rep,
    root_history: torch.Tensor,
    target_velocity: Vector2,
    *,
    generated_frames: int,
    history_frames: int,
    fps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert an x/z target velocity into ARDY root-position constraints."""
    if root_history.ndim != 2 or root_history.shape[0] < 2 or root_history.shape[1] != 3:
        raise ValueError(
            "ARDY root history must have shape [T >= 2, 3], "
            f"got {tuple(root_history.shape)}"
        )

    current_position = root_history[-1]
    current_velocity = (root_history[-1] - root_history[-2]) * fps
    target = torch.tensor(
        [target_velocity[0], 0.0, target_velocity[1]],
        dtype=torch.float32,
        device=device,
    )
    transition_frames = min(int(fps * VELOCITY_TRANSITION_S), generated_frames)
    future_positions: list[torch.Tensor] = []
    position = current_position
    for frame in range(generated_frames):
        alpha = min((frame + 1) / transition_frames, 1.0)
        velocity = (1.0 - alpha) * current_velocity + alpha * target
        position = position + velocity / fps
        future_positions.append(position)

    future = torch.stack(future_positions)
    relative_indices = torch.arange(
        VELOCITY_CONSTRAINT_INTERVAL,
        generated_frames + 1,
        VELOCITY_CONSTRAINT_INTERVAL,
        device=device,
    )
    frame_indices = relative_indices + history_frames - 1
    root_2d = future[relative_indices - 1][:, [0, 2]]
    observed_motion, motion_mask = motion_rep.create_conditions(
        {"root_2d": [frame_indices]},
        {"root_2d": [root_2d]},
        generated_frames + history_frames,
        True,
        device,
    )
    return motion_mask.unsqueeze(0), observed_motion.unsqueeze(0)
