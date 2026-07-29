from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
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

HISTORY_FRAMES = 10


@dataclass(frozen=True)
class RobotState:
    root_quat_w: np.ndarray
    root_ang_vel_b: np.ndarray
    projected_gravity_b: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray

    def __post_init__(self) -> None:
        expected = {
            "root_quat_w": (4,),
            "root_ang_vel_b": (3,),
            "projected_gravity_b": (3,),
            "joint_pos": (29,),
            "joint_vel": (29,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}; expected {shape}")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class _HistoryState:
    angular_velocity: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    last_action: np.ndarray
    gravity: np.ndarray


class MotionReference:
    def __init__(self) -> None:
        self._qpos = standing_qpos()[None]
        self._joint_vel = np.zeros((1, 29), dtype=np.float32)
        self._heading_delta = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self._frame = 0
        self._command_id: str | None = None

    def load(self, chunk: MotionChunk, robot_quat_w: np.ndarray) -> None:
        if chunk.fps != 50:
            raise ValueError(f"SONIC requires a 50 Hz reference, got {chunk.fps} Hz")
        self._qpos = chunk.qpos.copy()
        natural_positions = self._qpos[:, 7:]
        velocities = np.empty_like(natural_positions)
        if len(natural_positions) == 1:
            velocities[0] = 0.0
        else:
            velocities[:-1] = np.diff(natural_positions, axis=0) * chunk.fps
            velocities[-1] = velocities[-2]
        self._joint_vel = velocities

        robot_quat = torch.as_tensor(robot_quat_w, dtype=torch.float32)
        reference_quat = torch.as_tensor(self._qpos[0, 3:7], dtype=torch.float32)
        self._heading_delta = quat_mul(
            yaw_quat(robot_quat), quat_conjugate(yaw_quat(reference_quat))
        )
        self._frame = 0
        self._command_id = chunk.command_id

    def window(
        self, *, frames: int, step: int
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        indices = np.minimum(
            self._frame + np.arange(frames, dtype=np.int64) * step,
            len(self._qpos) - 1,
        )
        positions = self._qpos[indices, 7:][:, SONIC_FROM_MJLAB]
        velocities = self._joint_vel[indices][:, SONIC_FROM_MJLAB]
        reference_quats = torch.as_tensor(self._qpos[indices, 3:7], dtype=torch.float32)
        heading = self._heading_delta.expand_as(reference_quats)
        aligned_quats = quat_mul(heading, reference_quats)
        return positions, velocities, aligned_quats

    def advance(self) -> str | None:
        if self._frame < len(self._qpos) - 1:
            self._frame += 1
            return None
        command_id, self._command_id = self._command_id, None
        return command_id


class SonicPolicy:
    """CPU ONNX Runtime implementation of the SONIC G1 deployment policy."""

    def __init__(self, bundle_dir: Path, *, provider: str = "cpu") -> None:
        if provider != "cpu":
            raise NotImplementedError(
                "CUDA SONIC execution is reserved for the cu128 implementation"
            )
        self.layout = ObservationLayout.load(bundle_dir / "observation_config.yaml")
        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(
            str(bundle_dir / "model_encoder.onnx"), providers=providers
        )
        self.decoder = ort.InferenceSession(
            str(bundle_dir / "model_decoder.onnx"), providers=providers
        )
        self._encoder_input = self.encoder.get_inputs()[0].name
        self._decoder_input = self.decoder.get_inputs()[0].name
        self._validate_signatures()
        self.reference = MotionReference()
        self._history: deque[_HistoryState] = deque(maxlen=HISTORY_FRAMES)
        self._last_action = np.zeros(29, dtype=np.float32)

    def reset(self) -> None:
        self.reference = MotionReference()
        self._history.clear()
        self._last_action.fill(0.0)

    def load_motion(self, chunk: MotionChunk, robot_quat_w: np.ndarray) -> None:
        self.reference.load(chunk, robot_quat_w)

    def infer(self, state: RobotState) -> tuple[np.ndarray, str | None]:
        history_state = _HistoryState(
            angular_velocity=state.root_ang_vel_b,
            joint_position=(state.joint_pos - DEFAULT_JOINT_POS_MJLAB)[
                SONIC_FROM_MJLAB
            ],
            joint_velocity=state.joint_vel[SONIC_FROM_MJLAB],
            last_action=self._last_action.copy(),
            gravity=state.projected_gravity_b,
        )
        self._history.append(history_state)
        while len(self._history) < HISTORY_FRAMES:
            self._history.appendleft(
                _HistoryState(
                    angular_velocity=np.zeros(3, dtype=np.float32),
                    joint_position=np.zeros(29, dtype=np.float32),
                    joint_velocity=np.zeros(29, dtype=np.float32),
                    last_action=np.zeros(29, dtype=np.float32),
                    gravity=np.zeros(3, dtype=np.float32),
                )
            )

        step = (
            5
            if "motion_joint_positions_10frame_step5" in self.layout.required_g1
            else 1
        )
        positions, velocities, reference_quats = self.reference.window(
            frames=HISTORY_FRAMES, step=step
        )
        robot_quat = torch.as_tensor(state.root_quat_w, dtype=torch.float32)
        relative_quats = quat_mul(
            quat_conjugate(robot_quat).expand_as(reference_quats), reference_quats
        )
        orientation_6d = matrix_from_quat(relative_quats)[..., :2].reshape(-1).numpy()
        suffix = f"10frame_step{step}"
        encoder_values = {
            "encoder_mode_4": np.array([0.0, 0.0, 0.0, 0.0]),
            f"motion_joint_positions_{suffix}": positions,
            f"motion_joint_velocities_{suffix}": velocities,
            f"motion_anchor_orientation_{suffix}": orientation_6d,
        }
        encoder_obs = self.layout.pack_encoder(encoder_values)
        token = np.asarray(
            self.encoder.run(None, {self._encoder_input: encoder_obs})[0],
            dtype=np.float32,
        )

        history = tuple(self._history)
        policy_values = {
            "token_state": token,
            "his_base_angular_velocity_10frame_step1": np.stack(
                [item.angular_velocity for item in history]
            ),
            "his_body_joint_positions_10frame_step1": np.stack(
                [item.joint_position for item in history]
            ),
            "his_body_joint_velocities_10frame_step1": np.stack(
                [item.joint_velocity for item in history]
            ),
            "his_last_actions_10frame_step1": np.stack(
                [item.last_action for item in history]
            ),
            "his_gravity_dir_10frame_step1": np.stack(
                [item.gravity for item in history]
            ),
        }
        policy_obs = self.layout.pack_policy(policy_values)
        action_sonic = np.asarray(
            self.decoder.run(None, {self._decoder_input: policy_obs})[0],
            dtype=np.float32,
        ).reshape(29)
        if not np.isfinite(action_sonic).all():
            raise RuntimeError("SONIC decoder returned NaN or infinite actions")
        self._last_action = action_sonic
        completed = self.reference.advance()
        return action_sonic[MJLAB_FROM_SONIC], completed

    def _validate_signatures(self) -> None:
        encoder_input = self.encoder.get_inputs()[0]
        encoder_output = self.encoder.get_outputs()[0]
        decoder_input = self.decoder.get_inputs()[0]
        decoder_output = self.decoder.get_outputs()[0]
        expected = (
            (encoder_input, self.layout.encoder_input_dimension),
            (encoder_output, self.layout.encoder_dimension),
            (decoder_input, self.layout.policy_input_dimension),
            (decoder_output, 29),
        )
        for tensor, final_dimension in expected:
            if len(tensor.shape) != 2 or tensor.shape[-1] != final_dimension:
                raise RuntimeError(
                    f"Unexpected ONNX tensor {tensor.name} shape: {tensor.shape}"
                )
