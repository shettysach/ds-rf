import numpy as np

from motion_gen.resample import resample_motion


def test_resample_64_frames_to_106_without_mutating_input() -> None:
    qpos = np.zeros((64, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    qpos[1::2, 3] = -1.0
    qpos[:, 7] = np.arange(64)
    original = qpos.copy()

    chunk = resample_motion(qpos, command_id="command")

    assert chunk.qpos.shape == (106, 36)
    assert chunk.fps == 50
    np.testing.assert_array_equal(qpos, original)
    np.testing.assert_allclose(np.linalg.norm(chunk.qpos[:, 3:7], axis=1), 1.0)
