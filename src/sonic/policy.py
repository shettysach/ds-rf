from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_conjugate,
    quat_mul,
    yaw_quat,
)

from shared.g1 import (
    DEFAULT_JOINT_POS_MJLAB,
    MJLAB_FROM_SONIC,
    SONIC_FROM_MJLAB,
    standing_qpos,
)
from shared.messages import MotionChunk
from sonic.observations import ObservationLayout
from sonic.onnx_runner import FixedShapeOnnxModel

HISTORY_FRAMES = 10
TensorLike = torch.Tensor | np.ndarray


@dataclass(frozen=True)
class RobotState:
    root_quat_w: TensorLike
    root_ang_vel_b: TensorLike
    projected_gravity_b: TensorLike
    joint_pos: TensorLike
    joint_vel: TensorLike

    def __post_init__(self) -> None:
        expected = {
            "root_quat_w": (4,),
            "root_ang_vel_b": (3,),
            "projected_gravity_b": (3,),
            "joint_pos": (29,),
            "joint_vel": (29,),
        }
        for name, shape in expected.items():
            actual = tuple(getattr(self, name).shape)
            if actual != shape:
                raise ValueError(f"{name} has shape {actual}; expected {shape}")


@dataclass(frozen=True)
class _HistoryState:
    angular_velocity: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    last_action: torch.Tensor
    gravity: torch.Tensor


class MotionReference:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._qpos = torch.as_tensor(
            standing_qpos()[None], dtype=torch.float32, device=device
        )
        self._joint_vel = torch.zeros((1, 29), dtype=torch.float32, device=device)
        self._heading_delta = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device
        )
        self._sonic_from_mjlab = torch.as_tensor(
            SONIC_FROM_MJLAB, dtype=torch.long, device=device
        )
        self._future_offsets = torch.arange(
            HISTORY_FRAMES, dtype=torch.long, device=device
        )
        self._frame = 0
        self._command_id: str | None = None

    def load(self, chunk: MotionChunk, robot_quat_w: TensorLike) -> None:
        if chunk.fps != 50:
            raise ValueError(f"SONIC requires a 50 Hz reference, got {chunk.fps} Hz")
        self._qpos = torch.as_tensor(
            chunk.qpos, dtype=torch.float32, device=self.device
        ).contiguous()
        if len(self._qpos) < 2:
            raise ValueError("SONIC requires at least two reference frames")
        natural_positions = self._qpos[:, 7:]
        velocities = torch.empty_like(natural_positions)
        velocities[:-1] = torch.diff(natural_positions, dim=0) * chunk.fps
        velocities[-1] = velocities[-2]
        self._joint_vel = velocities

        robot_quat = _as_tensor(robot_quat_w, self.device)
        reference_quat = self._qpos[0, 3:7]
        self._heading_delta = quat_mul(
            yaw_quat(robot_quat), quat_conjugate(yaw_quat(reference_quat))
        )
        self._frame = 0
        self._command_id = chunk.command_id

    def window(self, *, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.clamp(
            self._future_offsets * step + self._frame,
            max=len(self._qpos) - 1,
        )
        natural_positions = self._qpos.index_select(0, indices)[:, 7:]
        natural_velocities = self._joint_vel.index_select(0, indices)
        positions = natural_positions.index_select(1, self._sonic_from_mjlab)
        velocities = natural_velocities.index_select(1, self._sonic_from_mjlab)
        reference_quats = self._qpos.index_select(0, indices)[:, 3:7]
        aligned_quats = quat_mul(
            self._heading_delta.expand_as(reference_quats), reference_quats
        )
        return positions, velocities, aligned_quats

    def advance(self) -> str | None:
        if self._frame < len(self._qpos) - 1:
            self._frame += 1
            return None
        command_id, self._command_id = self._command_id, None
        return command_id


class SonicPolicy:
    """SONIC inference with CPU execution or zero-copy CUDA I/O binding."""

    def __init__(
        self,
        bundle_dir: Path,
        *,
        device: str = "cpu",
        cuda_stream: Any | None = None,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and cuda_stream is None:
            raise ValueError("CUDA SONIC requires MJLab's CUDA stream")
        if self.device.type == "cpu" and cuda_stream is not None:
            raise ValueError("A CUDA stream cannot be used with the CPU device")

        self.layout = ObservationLayout.load(bundle_dir / "observation_config.yaml")
        self.encoder = FixedShapeOnnxModel(
            bundle_dir / "model_encoder.onnx",
            input_shape=(1, self.layout.encoder_input_dimension),
            output_shape=(1, self.layout.encoder_dimension),
            device=self.device,
            cuda_stream=cuda_stream,
        )
        self.decoder = FixedShapeOnnxModel(
            bundle_dir / "model_decoder.onnx",
            input_shape=(1, self.layout.policy_input_dimension),
            output_shape=(1, 29),
            device=self.device,
            cuda_stream=cuda_stream,
        )
        self._encoder_slices = self.layout.encoder_slices
        self._policy_slices = self.layout.policy_slices
        self._default_joint_pos = torch.as_tensor(
            DEFAULT_JOINT_POS_MJLAB, dtype=torch.float32, device=self.device
        )
        self._sonic_from_mjlab = torch.as_tensor(
            SONIC_FROM_MJLAB, dtype=torch.long, device=self.device
        )
        self._mjlab_from_sonic = torch.as_tensor(
            MJLAB_FROM_SONIC, dtype=torch.long, device=self.device
        )
        self._g1_encoder_mode = torch.zeros(4, dtype=torch.float32, device=self.device)
        self.reference = MotionReference(self.device)
        self._history: deque[_HistoryState] = deque(maxlen=HISTORY_FRAMES)
        self._last_action = torch.zeros(29, dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        self.reference = MotionReference(self.device)
        self._history.clear()
        self._last_action.zero_()
        self.encoder.input.zero_()
        self.decoder.input.zero_()

    def load_motion(self, chunk: MotionChunk, robot_quat_w: TensorLike) -> None:
        self.reference.load(chunk, robot_quat_w)

    def infer(self, state: RobotState) -> tuple[torch.Tensor, str | None]:
        root_quat = _as_tensor(state.root_quat_w, self.device)
        history_state = _HistoryState(
            angular_velocity=_as_tensor(state.root_ang_vel_b, self.device).clone(),
            joint_position=(
                _as_tensor(state.joint_pos, self.device) - self._default_joint_pos
            ).index_select(0, self._sonic_from_mjlab),
            joint_velocity=_as_tensor(state.joint_vel, self.device)
            .index_select(0, self._sonic_from_mjlab)
            .clone(),
            last_action=self._last_action.clone(),
            gravity=_as_tensor(state.projected_gravity_b, self.device).clone(),
        )
        self._history.append(history_state)
        while len(self._history) < HISTORY_FRAMES:
            self._history.appendleft(self._zero_history())

        step = (
            5
            if "motion_joint_positions_10frame_step5" in self.layout.required_g1
            else 1
        )
        positions, velocities, reference_quats = self.reference.window(step=step)
        relative_quats = quat_mul(
            quat_conjugate(root_quat).expand_as(reference_quats), reference_quats
        )
        orientation_6d = matrix_from_quat(relative_quats)[..., :2]
        suffix = f"10frame_step{step}"
        self._copy_encoder("encoder_mode_4", self._g1_encoder_mode)
        self._copy_encoder(f"motion_joint_positions_{suffix}", positions)
        self._copy_encoder(f"motion_joint_velocities_{suffix}", velocities)
        self._copy_encoder(f"motion_anchor_orientation_{suffix}", orientation_6d)
        token = self.encoder.run()

        history = tuple(self._history)
        self._copy_policy("token_state", token)
        self._copy_policy(
            "his_base_angular_velocity_10frame_step1",
            torch.stack([item.angular_velocity for item in history]),
        )
        self._copy_policy(
            "his_body_joint_positions_10frame_step1",
            torch.stack([item.joint_position for item in history]),
        )
        self._copy_policy(
            "his_body_joint_velocities_10frame_step1",
            torch.stack([item.joint_velocity for item in history]),
        )
        self._copy_policy(
            "his_last_actions_10frame_step1",
            torch.stack([item.last_action for item in history]),
        )
        self._copy_policy(
            "his_gravity_dir_10frame_step1",
            torch.stack([item.gravity for item in history]),
        )
        action_sonic = self.decoder.run().reshape(29)
        if self.device.type == "cpu" and not bool(torch.isfinite(action_sonic).all()):
            raise RuntimeError("SONIC decoder returned NaN or infinite actions")
        self._last_action.copy_(action_sonic)
        completed = self.reference.advance()
        action_mjlab = action_sonic.index_select(0, self._mjlab_from_sonic)
        return action_mjlab.unsqueeze(0), completed

    def _copy_encoder(self, name: str, value: torch.Tensor) -> None:
        self.encoder.input[0, self._encoder_slices[name]].copy_(value.reshape(-1))

    def _copy_policy(self, name: str, value: torch.Tensor) -> None:
        self.decoder.input[0, self._policy_slices[name]].copy_(value.reshape(-1))

    def _zero_history(self) -> _HistoryState:
        return _HistoryState(
            angular_velocity=torch.zeros(3, device=self.device),
            joint_position=torch.zeros(29, device=self.device),
            joint_velocity=torch.zeros(29, device=self.device),
            last_action=torch.zeros(29, device=self.device),
            gravity=torch.zeros(3, device=self.device),
        )


def _as_tensor(value: TensorLike, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)
