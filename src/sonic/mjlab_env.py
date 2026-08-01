from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import euler_xyz_from_quat
from mjlab.viewer.debug_visualizer import DebugVisualizer

from sonic.mjlab_config import make_sonic_env_cfg

if TYPE_CHECKING:
    from mjlab.envs.types import VecEnvObs, VecEnvStepReturn
    from mjlab.sim import Simulation


@dataclass(frozen=True)
class RobotState:
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_ang_vel_b: torch.Tensor
    projected_gravity_b: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


class SonicManagerBasedRlEnv(ManagerBasedRlEnv):
    """MJLab environment with composable application debug visualizers."""

    def __init__(self, *args, **kwargs) -> None:
        self._application_visualizers: list[Callable[[DebugVisualizer], None]] = []
        super().__init__(*args, **kwargs)

    def add_debug_visualizer(self, callback: Callable[[DebugVisualizer], None]) -> None:
        self._application_visualizers.append(callback)

    def update_visualizers(self, visualizer: DebugVisualizer) -> None:
        super().update_visualizers(visualizer)
        for callback in self._application_visualizers:
            callback(visualizer)


class SonicMjlabEnv:
    def __init__(
        self,
        *,
        device: str = "cpu",
        image_width: int = 640,
        image_height: int = 480,
        task: str | None = None,
    ) -> None:
        torch_device = torch.device(device)
        self._env = SonicManagerBasedRlEnv(
            cfg=make_sonic_env_cfg(
                image_width=image_width,
                image_height=image_height,
                task=task,
            ),
            device=str(torch_device),
            render_mode="rgb_array",
        )
        self.num_envs = self._env.num_envs
        self.cfg = self._env.cfg
        self.device = self._env.device
        self.unwrapped = self._env

        self.cuda_stream = (
            connect_torch_to_mjlab(self._env.sim, torch_device)
            if torch_device.type == "cuda"
            else None
        )

        with self.compute_context():
            self._env.reset()

    def compute_context(self) -> AbstractContextManager[None]:
        return stream_context(self.cuda_stream)

    def robot_state(self) -> RobotState:
        data = self._env.scene["robot"].data
        return RobotState(
            root_pos_w=data.root_link_pos_w[0],
            root_quat_w=data.root_link_quat_w[0],
            root_ang_vel_b=data.root_link_ang_vel_b[0],
            projected_gravity_b=data.projected_gravity_b[0],
            joint_pos=data.joint_pos[0],
            joint_vel=data.joint_vel[0],
        )

    def get_observations(self) -> VecEnvObs:
        with self.compute_context():
            return self._env.get_observations()

    def step(self, actions: torch.Tensor) -> VecEnvStepReturn:
        with self.compute_context():
            return self._env.step(actions)

    def reset(self) -> tuple[VecEnvObs, dict[str, object]]:
        with self.compute_context():
            return self._env.reset()

    def add_debug_visualizer(self, callback: Callable[[DebugVisualizer], None]) -> None:
        self._env.add_debug_visualizer(callback)

    def render(self) -> np.ndarray:
        with self.compute_context():
            self._align_camera_with_robot()
            image = self._env.render()
        if image is None:
            raise RuntimeError("MJLab offscreen renderer returned no image")
        return image

    def _align_camera_with_robot(self) -> None:
        """Keep the tracking camera behind the robot as its heading changes."""
        renderer = self._env._offline_renderer
        if renderer is None:
            raise RuntimeError("MJLab offscreen renderer is not initialized")

        camera_cfg = self.cfg.viewer
        if camera_cfg.entity_name is None or camera_cfg.body_name is None:
            raise RuntimeError("Third-person camera requires an entity and body")

        robot = self._env.scene[camera_cfg.entity_name]
        body_index = robot.body_names.index(camera_cfg.body_name)
        body_quat_w = robot.data.body_link_quat_w[camera_cfg.env_idx, body_index]
        renderer._cam.azimuth = camera_cfg.azimuth + _yaw_degrees(body_quat_w)

    def close(self) -> None:
        self._env.close()


def _yaw_degrees(quat_w: torch.Tensor) -> float:
    _, _, yaw = euler_xyz_from_quat(quat_w.reshape(1, 4))
    return float(torch.rad2deg(yaw).item())


# CUDA


def connect_torch_to_mjlab(
    simulation: Simulation,
    device: torch.device,
) -> torch.cuda.Stream:
    import warp as wp

    torch.cuda.synchronize(device)
    wp.synchronize_device(simulation.wp_device)
    return wp.stream_to_torch(simulation.wp_device)


def stream_context(stream: torch.cuda.Stream | None) -> AbstractContextManager[None]:
    return torch.cuda.stream(stream) if stream is not None else nullcontext()
