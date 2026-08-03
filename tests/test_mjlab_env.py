from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from sonic.mjlab_env import SonicMjlabEnv


def test_render_keeps_camera_azimuth_fixed() -> None:
    renderer = SimpleNamespace(_cam=SimpleNamespace(azimuth=10.0))
    simulation = cast(Any, SonicMjlabEnv.__new__(SonicMjlabEnv))
    simulation.cuda_stream = None
    simulation._env = SimpleNamespace(
        _offline_renderer=renderer,
        render=lambda: np.zeros((1, 1, 3), dtype=np.uint8),
    )

    simulation.render()

    assert renderer._cam.azimuth == 10.0
