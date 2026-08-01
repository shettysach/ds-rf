from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import mujoco.viewer
import torch
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

from shared.messages import SONIC_FPS

if TYPE_CHECKING:
    from mjlab.viewer import EnvProtocol

    from sonic.mjlab_env import SonicMjlabEnv
    from sonic.policy import MotionReference

from sonic.reference_ghost import SonicReferenceGhost


class SonicViewer(Protocol):
    def sync(self) -> None: ...

    def close(self) -> None: ...


class NativeSonicViewer(NativeMujocoViewer):
    """Passive MJLab viewer that never owns simulation stepping."""

    def __init__(
        self,
        simulation: SonicMjlabEnv,
        reference: MotionReference | None = None,
    ) -> None:
        super().__init__(
            cast("EnvProtocol", simulation),
            _ViewerOnlyPolicy(),
            frame_rate=float(SONIC_FPS),
            enable_perturbations=False,
        )
        self._reference_ghost = (
            SonicReferenceGhost(simulation.unwrapped, reference)
            if reference is not None
            else None
        )
        self.setup()
        self.sync()

    def sync(self) -> None:
        self.sync_env_to_viewer()

    def _update_debug_visualizers(self, viewer: mujoco.viewer.Handle) -> None:
        super()._update_debug_visualizers(viewer)
        if self._reference_ghost is None or not self._show_debug_vis:
            return

        assert self.mjm is not None
        visualizer = MujocoNativeDebugVisualizer(
            viewer.user_scn,
            self.mjm,
            self.env_idx,
            self._show_all_envs,
        )
        self._reference_ghost.draw(visualizer)


class _ViewerOnlyPolicy:
    """Sentinel policy: the passive display must never advance simulation."""

    def __call__(self, obs: object) -> torch.Tensor:
        del obs
        raise RuntimeError("The SONIC passive viewer cannot drive physics")
