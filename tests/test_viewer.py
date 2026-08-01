from types import SimpleNamespace
from typing import Any, cast

import pytest
from mjlab.viewer import NativeMujocoViewer

import sonic.viewer as viewer_module
from sonic.viewer import NativeSonicViewer


@pytest.mark.parametrize(
    ("debug_visualization_enabled", "expected"),
    [
        (True, ["mjlab", "visualizer", "ghost"]),
        (False, ["mjlab"]),
    ],
)
def test_reference_ghost_follows_mjlab_visualizers(
    monkeypatch: pytest.MonkeyPatch,
    debug_visualization_enabled: bool,
    expected: list[str],
) -> None:
    calls: list[str] = []
    debug_visualizer = object()

    monkeypatch.setattr(
        NativeMujocoViewer,
        "_update_debug_visualizers",
        lambda self, viewer: calls.append("mjlab"),
    )
    monkeypatch.setattr(
        viewer_module,
        "MujocoNativeDebugVisualizer",
        lambda *args: calls.append("visualizer") or debug_visualizer,
    )

    sonic_viewer = cast(Any, NativeSonicViewer.__new__(NativeSonicViewer))
    sonic_viewer._reference_ghost = SimpleNamespace(
        draw=lambda visualizer: calls.append("ghost")
    )
    sonic_viewer._show_debug_vis = debug_visualization_enabled
    sonic_viewer._show_all_envs = False
    sonic_viewer.env_idx = 0
    sonic_viewer.mjm = object()

    sonic_viewer._update_debug_visualizers(
        SimpleNamespace(user_scn=object())
    )

    assert calls == expected
