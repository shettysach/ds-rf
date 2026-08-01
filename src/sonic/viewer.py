from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import torch

from shared.messages import SONIC_FPS

if TYPE_CHECKING:
    from mjlab.viewer import EnvProtocol, NativeMujocoViewer

    from sonic.mjlab_env import SonicMjlabEnv
    from sonic.policy import MotionReference

from sonic.reference_ghost import SonicReferenceGhost


class SonicViewer(Protocol):
    def sync(self) -> None: ...

    def close(self) -> None: ...


class NativeSonicViewer:
    """Passive MJLab viewer that never owns simulation stepping."""

    def __init__(
        self,
        simulation: SonicMjlabEnv,
        reference: MotionReference | None = None,
    ) -> None:
        from mjlab.viewer import NativeMujocoViewer

        if reference is not None:
            reference_ghost = SonicReferenceGhost(simulation.unwrapped, reference)
            simulation.add_debug_visualizer(reference_ghost.draw)

        self._viewer: NativeMujocoViewer = NativeMujocoViewer(
            cast("EnvProtocol", simulation),
            _ViewerOnlyPolicy(),
            frame_rate=float(SONIC_FPS),
            enable_perturbations=False,
        )
        self._viewer.setup()
        self.sync()

    def sync(self) -> None:
        self._viewer.sync_env_to_viewer()

    def close(self) -> None:
        self._viewer.close()


class _ViewerOnlyPolicy:
    """Sentinel policy: the passive display must never advance simulation."""

    def __call__(self, obs: object) -> torch.Tensor:
        del obs
        raise RuntimeError("The SONIC passive viewer cannot drive physics")
