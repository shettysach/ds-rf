from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

    from sonic.policy import MotionReference

REFERENCE_GHOST_COLOR = np.array((0.5, 0.7, 0.5, 0.5), dtype=np.float32)


class SonicReferenceGhost:
    """Draw SONIC's active reference through MJLab's debug visualizer."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        reference: MotionReference,
    ) -> None:
        self._env = env
        self._robot = env.scene["robot"]
        self._reference = reference
        self._ghost_model: Any | None = None

    def draw(self, visualizer: DebugVisualizer) -> None:
        pose = self._reference.visualization_pose()
        if pose is None:
            return

        root_pos_w, root_quat_w, joint_pos = pose
        indexing = self._robot.indexing
        free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
        joint_q_adr = indexing.joint_q_adr.cpu().numpy()
        if len(free_joint_q_adr) < 7:
            raise ValueError(
                "SONIC reference ghost requires a floating-base robot with "
                f"at least 7 root qpos addresses, got {len(free_joint_q_adr)}"
            )
        if len(joint_q_adr) != joint_pos.numel():
            raise ValueError(
                "SONIC reference joint count does not match robot qpos indexing: "
                f"{joint_pos.numel()} reference joints vs {len(joint_q_adr)} "
                "qpos addresses"
            )

        qpos = np.zeros(self._env.sim.mj_model.nq, dtype=np.float64)
        qpos[free_joint_q_adr[:3]] = root_pos_w.detach().cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = root_quat_w.detach().cpu().numpy()
        qpos[joint_q_adr] = joint_pos.detach().cpu().numpy()
        visualizer.add_ghost_mesh(
            qpos,
            model=self._get_ghost_model(),
            alpha=float(REFERENCE_GHOST_COLOR[3]),
            label="sonic_reference",
        )

    def _get_ghost_model(self) -> Any:
        if self._ghost_model is None:
            model = copy.deepcopy(self._env.sim.mj_model)
            for geom_id in range(model.ngeom):
                if (
                    model.geom_contype[geom_id] != 0
                    or model.geom_conaffinity[geom_id] != 0
                ):
                    model.geom_rgba[geom_id, 3] = 0.0
                else:
                    model.geom_rgba[geom_id] = REFERENCE_GHOST_COLOR
            self._ghost_model = model
        return self._ghost_model
