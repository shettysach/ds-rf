from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from sonic.mjlab_config import make_sonic_env_cfg

if TYPE_CHECKING:
    from mjlab.envs.types import VecEnvObs, VecEnvStepReturn
    from mjlab.sim import Simulation
    from mjlab.viewer import EnvProtocol


@dataclass(frozen=True)
class RobotState:
    root_quat_w: torch.Tensor
    root_ang_vel_b: torch.Tensor
    projected_gravity_b: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


class SonicMjlabEnv:
    def __init__(
        self,
        *,
        device: str = "cpu",
        image_width: int = 640,
        image_height: int = 480,
        task: str | None = None,
        show_viewer: bool = False,
    ) -> None:
        from mjlab.envs import ManagerBasedRlEnv

        torch_device = torch.device(device)
        self._env = ManagerBasedRlEnv(
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
        self._viewer: Any | None = None

        self.cuda_stream = (
            connect_torch_to_mjlab(self._env.sim, torch_device)
            if torch_device.type == "cuda"
            else None
        )

        with self.compute_context():
            self._env.reset()

        if show_viewer:
            from mjlab.viewer import NativeMujocoViewer

            # We only use the passive viewer's state-copy/render path. Calling
            # viewer.run() or viewer.tick() would give it control of physics.
            self._viewer = NativeMujocoViewer(
                cast("EnvProtocol", self._env),
                _ViewerOnlyPolicy(),
                frame_rate=50.0,
                enable_perturbations=False,
            )
            self._viewer.setup()
            self._sync_viewer()

    def compute_context(self) -> AbstractContextManager[None]:
        return stream_context(self.cuda_stream)

    def robot_state(self) -> RobotState:
        data = self._env.scene["robot"].data
        return RobotState(
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
            result = self._env.step(actions)
            self._sync_viewer()
            return result

    def reset(self) -> tuple[VecEnvObs, dict[str, object]]:
        with self.compute_context():
            result = self._env.reset()
            self._sync_viewer()
            return result

    def render(self) -> np.ndarray:
        with self.compute_context():
            image = self._env.render()
        if image is None:
            raise RuntimeError("MJLab offscreen renderer returned no image")
        return image

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._env.close()

    def _sync_viewer(self) -> None:
        if self._viewer is not None:
            self._viewer.sync_env_to_viewer()


class _ViewerOnlyPolicy:
    """Sentinel policy: the passive display must never advance simulation."""

    def __call__(self, obs: object) -> torch.Tensor:
        del obs
        raise RuntimeError("The SONIC passive viewer cannot drive physics")


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
