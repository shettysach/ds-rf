from __future__ import annotations

from dataclasses import replace

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


def make_sonic_env_cfg(
    *,
    image_width: int = 640,
    image_height: int = 480,
) -> ManagerBasedRlEnvCfg:
    """Build a minimal 50 Hz MJLab environment matching SONIC's G1 motors."""

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
            # The camera follows translation but keeps a world-fixed heading.
            # At reset it sits behind the +x-facing robot and looks along +x.
            azimuth=0.0,
            width=image_width,
            height=image_height,
            max_extra_envs=0,
        ),
        episode_length_s=0.0,
    )
