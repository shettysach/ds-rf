from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from sonic.mjlab_env import SonicMjlabEnv, _yaw_degrees


@pytest.mark.parametrize(
    ("quat_w", "expected_degrees"),
    [
        ((1.0, 0.0, 0.0, 0.0), 0.0),
        ((2**-0.5, 0.0, 0.0, 2**-0.5), 90.0),
        ((2**-0.5, 0.0, 0.0, -(2**-0.5)), -90.0),
    ],
)
def test_yaw_degrees(quat_w, expected_degrees) -> None:
    assert _yaw_degrees(torch.tensor(quat_w)) == pytest.approx(expected_degrees)


def test_camera_azimuth_tracks_body_yaw_without_accumulating() -> None:
    camera = SimpleNamespace(
        entity_name="robot",
        body_name="torso_link",
        env_idx=0,
        azimuth=10.0,
    )
    renderer = SimpleNamespace(_cam=SimpleNamespace(azimuth=0.0))
    robot = SimpleNamespace(
        body_names=["torso_link"],
        data=SimpleNamespace(
            body_link_quat_w=torch.tensor(
                [[[2**-0.5, 0.0, 0.0, 2**-0.5]]]
            )
        ),
    )
    simulation = cast(Any, SonicMjlabEnv.__new__(SonicMjlabEnv))
    simulation.cfg = SimpleNamespace(viewer=camera)
    simulation._env = SimpleNamespace(
        _offline_renderer=renderer,
        scene={"robot": robot},
    )

    simulation._align_camera_with_robot()
    assert renderer._cam.azimuth == pytest.approx(100.0)

    robot.data.body_link_quat_w[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    simulation._align_camera_with_robot()
    assert renderer._cam.azimuth == pytest.approx(10.0)
