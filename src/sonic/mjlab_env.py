from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import torch

from shared.g1 import G1_JOINT_NAMES_MJLAB
from sonic.policy import RobotState


def make_sonic_env_cfg():
    """Build a minimal 50 Hz MJLab environment matching SONIC's G1 motors."""

    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
        G1_ACTUATOR_4010,
        G1_ACTUATOR_5020,
        G1_ACTUATOR_7520_14,
        G1_ACTUATOR_7520_22,
        G1_ACTUATOR_ANKLE,
        G1_ACTUATOR_WAIST,
        get_g1_robot_cfg,
    )
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.viewer import ViewerConfig

    actuator_7520_14 = replace(
        G1_ACTUATOR_7520_14,
        target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint"),
    )
    actuator_7520_22 = replace(
        G1_ACTUATOR_7520_22,
        target_names_expr=(
            ".*_hip_pitch_joint",
            ".*_hip_roll_joint",
            ".*_knee_joint",
        ),
    )
    actuators = (
        G1_ACTUATOR_5020,
        actuator_7520_14,
        actuator_7520_22,
        G1_ACTUATOR_4010,
        G1_ACTUATOR_WAIST,
        G1_ACTUATOR_ANKLE,
    )
    robot_cfg = get_g1_robot_cfg()
    robot_cfg.articulation = EntityArticulationInfoCfg(
        actuators=actuators,
        soft_joint_pos_limit_factor=0.9,
    )
    action_scale: dict[str, float] = {}
    for actuator in actuators:
        assert actuator.effort_limit is not None
        for pattern in actuator.target_names_expr:
            action_scale[pattern] = (
                0.25 * float(actuator.effort_limit) / actuator.stiffness
            )

    return ManagerBasedRlEnvCfg(
        decimation=4,
        scene=SceneCfg(
            num_envs=1,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot_cfg},
        ),
        actions={
            "joint_position": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=action_scale,
                use_default_offset=True,
            )
        },
        sim=SimulationCfg(njmax=128, mujoco=MujocoCfg(timestep=0.005)),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="pelvis",
            distance=3.0,
            elevation=-15.0,
            azimuth=135.0,
        ),
        episode_length_s=0.0,
    )


class SonicMjlabEnv:
    def __init__(self, *, device: str = "cpu") -> None:
        from mjlab.envs import ManagerBasedRlEnv

        self._env = ManagerBasedRlEnv(cfg=make_sonic_env_cfg(), device=device)
        self._cuda_stream: torch.cuda.Stream | None = None
        if device.startswith("cuda:"):
            import warp as wp

            # MJLab creates some Torch buffers on Torch's default stream before its
            # Warp stream is exposed. Synchronize once, then keep every control-loop
            # operation on the shared Warp stream below.
            torch.cuda.synchronize(torch.device(device))
            wp.synchronize_device(self._env.sim.wp_device)
            warp_stream = wp.get_stream(self._env.sim.wp_device)
            self._cuda_stream = torch.cuda.ExternalStream(
                warp_stream.cuda_stream,
                device=torch.device(device),
            )
            if int(self._cuda_stream.cuda_stream) != int(warp_stream.cuda_stream):
                raise RuntimeError("Torch and MJLab did not adopt the same CUDA stream")
        with self.compute_context():
            self._env.reset()
        joint_names = tuple(self._env.scene["robot"].joint_names)
        if joint_names != G1_JOINT_NAMES_MJLAB:
            raise RuntimeError(
                f"MJLab G1 joint order changed; SONIC mapping is unsafe: {joint_names}"
            )

    @property
    def cuda_stream(self) -> torch.cuda.Stream | None:
        return self._cuda_stream

    @property
    def cuda_stream_ptr(self) -> int | None:
        if self._cuda_stream is None:
            return None
        return int(self._cuda_stream.cuda_stream)

    def compute_context(self):
        if self._cuda_stream is None:
            return nullcontext()
        return torch.cuda.stream(self._cuda_stream)

    def robot_state(self) -> RobotState:
        data = self._env.scene["robot"].data
        return RobotState(
            root_quat_w=data.root_link_quat_w[0],
            root_ang_vel_b=data.root_link_ang_vel_b[0],
            projected_gravity_b=data.projected_gravity_b[0],
            joint_pos=data.joint_pos[0],
            joint_vel=data.joint_vel[0],
        )

    @property
    def cfg(self):
        return self._env.cfg

    @property
    def device(self) -> str:
        return self._env.device

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def unwrapped(self):
        return self._env

    @property
    def step_dt(self) -> float:
        return self._env.step_dt

    def get_observations(self) -> Any:
        with self.compute_context():
            return self._env.get_observations()

    def step(self, action: torch.Tensor):
        with self.compute_context():
            return self._env.step(action)

    def reset(self):
        with self.compute_context():
            return self._env.reset()

    def close(self) -> None:
        self._env.close()
