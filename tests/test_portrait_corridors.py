import mujoco
import pytest
from tasks.catalog import TASKS, get_task
from tasks.portrait_corridors import make_portrait_corridors_spec_fn

from sonic.mjlab_config import make_sonic_env_cfg


def test_catalog_contains_portrait_corridors() -> None:
    name, definition = get_task("portrait-corridors")

    assert name == "portrait-corridors"
    assert definition is TASKS[name]
    assert definition.objective == "Stand in front of the image of the cartoon."
    assert definition.camera_distance == 2.0


def test_catalog_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="Available: portrait-corridors"):
        get_task("unknown")


def test_sonic_config_applies_task_scene_and_camera_distance() -> None:
    task_cfg = make_sonic_env_cfg(task="portrait-corridors")
    plain_cfg = make_sonic_env_cfg(task=None)

    assert task_cfg.scene.spec_fn is not None
    assert task_cfg.viewer.distance == 2.0
    assert plain_cfg.scene.spec_fn is None
    assert plain_cfg.viewer.distance == 3.0


def test_portrait_corridors_spec_adds_portraits_walls_and_cameras() -> None:
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    make_portrait_corridors_spec_fn()(spec)

    assert {body.name for body in spec.bodies if body.name.endswith("_portrait")} == {
        "portrait_corridors_linus_portrait",
        "portrait_corridors_karpathy_portrait",
        "portrait_corridors_bugs_portrait",
    }
    assert {texture.name for texture in spec.textures} == {
        "portrait_corridors_linus_texture",
        "portrait_corridors_karpathy_texture",
        "portrait_corridors_bugs_texture",
    }
    assert len([body for body in spec.bodies if body.name.endswith("_wall")]) == 6

    cameras = {camera.name: camera for camera in spec.cameras}
    assert set(cameras) == {"corridor_left", "corridor_center", "corridor_right"}
    assert [tuple(camera.pos) for camera in cameras.values()] == pytest.approx(
        [(1.8, 2.0, 1.25), (1.8, 0.0, 1.25), (1.8, -2.0, 1.25)]
    )

    model = spec.compile()
    assert model.ntex == 3
    assert model.nmesh == 3
    assert model.ncam == 3
    assert model.mat_texid[:, 1].tolist() == [0, 1, 2]
    assert model.mesh_texcoordnum.tolist() == [4, 4, 4]
