from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast
from unittest.mock import patch

import mujoco.viewer
import torch
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

from shared.messages import SONIC_FPS

if TYPE_CHECKING:
    from mjlab.viewer import EnvProtocol

    from sim.env import MjlabEnv
    from sim.sonic.policy import MotionReference

from sim.reference_ghost import ReferenceGhost


class SimViewer(Protocol):
    def sync(self) -> None: ...

    def close(self) -> None: ...

    def set_vlm_thinking(self, observation_id: int) -> None: ...

    def set_vlm_result(
        self, observation_id: int, reasoning: str | None, command: str
    ) -> None: ...


class NativeSimViewer(NativeMujocoViewer):
    """Passive MJLab viewer that never owns simulation stepping."""

    def __init__(
        self,
        simulation: MjlabEnv,
        reference: MotionReference | None = None,
    ) -> None:
        super().__init__(
            cast("EnvProtocol", simulation),
            _ViewerOnlyPolicy(),
            frame_rate=float(SONIC_FPS),
            enable_perturbations=False,
        )
        self._reference_ghost = (
            ReferenceGhost(simulation.unwrapped, reference)
            if reference is not None
            else None
        )
        self.setup()
        self.sync()

    def sync(self) -> None:
        self.sync_env_to_viewer()

    def set_vlm_thinking(self, observation_id: int) -> None:
        del observation_id

    def set_vlm_result(
        self, observation_id: int, reasoning: str | None, command: str
    ) -> None:
        del observation_id, reasoning, command

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


class ViserSimViewer(ViserPlayViewer):
    """Passive MJLab Viser display that never owns simulation stepping."""

    def __init__(
        self,
        simulation: MjlabEnv,
        reference: MotionReference | None = None,
    ) -> None:
        super().__init__(
            cast("EnvProtocol", simulation),
            _ViewerOnlyPolicy(),
            frame_rate=float(SONIC_FPS),
        )
        self._reference_ghost = (
            ReferenceGhost(simulation.unwrapped, reference)
            if reference is not None
            else None
        )
        self.setup()
        self.sync()

    def set_vlm_thinking(self, observation_id: int) -> None:
        self._vlm_panel.content = (
            "## VLM\n\n**Thinking…**\n\n"
            f"Observation #{observation_id}"
        )

    def set_vlm_result(
        self, observation_id: int, reasoning: str | None, command: str
    ) -> None:
        reasoning_text = reasoning or "(No reasoning returned)"
        self._vlm_panel.content = (
            "## VLM\n\n"
            "### Reasoning\n"
            f"{reasoning_text}\n\n"
            "### Decision\n"
            f"`{command}`\n\n"
            f"Observation #{observation_id}"
        )

    def setup(self) -> None:
        # Reuse MJLab's scene/control initialization, then discard its GUI
        # because the demo sidebar is intentionally VLM-only.
        tab_groups: list[Any] = []
        add_tab_group = self._server.gui.add_tab_group

        def capture_tab_group(*args: Any, **kwargs: Any) -> Any:
            group = add_tab_group(*args, **kwargs)
            tab_groups.append(group)
            return group

        with patch.object(self._server.gui, "add_tab_group", capture_tab_group):
            super().setup()
        assert len(tab_groups) == 1
        tab_groups[0].remove()

        with self._server.gui.add_folder("VLM"):
            self._vlm_panel = self._server.gui.add_markdown(
                "## VLM\n\n**Waiting for observation…**"
            )

    def sync(self) -> None:
        self.sync_env_to_viewer()

    def _queue_debug_visualizers(self) -> None:
        super()._queue_debug_visualizers()
        if (
            self._reference_ghost is not None
            and self._scene.debug_visualization_enabled
        ):
            self._reference_ghost.draw(self._scene)


class _ViewerOnlyPolicy:
    """Sentinel policy: the passive display must never advance simulation."""

    def __call__(self, obs: object) -> torch.Tensor:
        del obs
        raise RuntimeError("The passive simulation viewer cannot drive physics")
