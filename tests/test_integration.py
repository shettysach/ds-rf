import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from motion_gen.planner_sonic import PlannerSonic
from motion_gen.resample import resample_motion
from shared.g1 import DEFAULT_JOINT_POS_MJLAB
from sonic.mjlab_env import RobotState, SonicMjlabEnv
from sonic.policy import SonicPolicy
from sonic.renderer import SonicRenderer

SONIC_DIR = Path("/tmp/GEAR-SONIC")
ARDY_CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", "/missing"))
ARDY_ENCODING = Path(os.environ.get("ENCODING", "/missing"))

pytestmark = pytest.mark.integration
CUDA_READY = torch.cuda.is_available() and "CUDAExecutionProvider" in (
    ort.get_available_providers()
)


@pytest.mark.skipif(
    not ARDY_CHECKPOINTS_DIR.is_dir() or not ARDY_ENCODING.is_file(),
    reason="ARDY checkpoint or fixed encoding is unavailable",
)
def test_real_ardy_checkpoint_generates_resampled_g1_qpos() -> None:
    from motion_gen.ardy.generator import Ardy

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    generator = Ardy(
        ARDY_CHECKPOINTS_DIR,
        ARDY_ENCODING,
        device=device,
    )

    qpos = generator.generate("ignored smoke-test text")
    chunk = resample_motion(
        qpos,
        source_fps=generator.fps,
        observation_id=0,
        command="ardy smoke test",
    )

    assert qpos.ndim == 2
    assert qpos.shape[1] == 36
    assert np.isfinite(qpos).all()
    assert chunk.qpos.shape == (qpos.shape[0] * 2, 36)


@pytest.mark.skipif(not SONIC_DIR.is_dir(), reason="SONIC bundle is unavailable")
def test_real_checkpoints_generate_action_and_motion() -> None:
    policy = SonicPolicy(SONIC_DIR)
    encoder_mode = policy.layout.encoder_slices["encoder_mode_4"]
    policy.encoder.input[0, encoder_mode].fill_(1.0)
    state = RobotState(
        root_pos_w=torch.zeros(3),
        root_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        root_ang_vel_b=torch.zeros(3),
        projected_gravity_b=torch.tensor([0.0, 0.0, -1.0]),
        joint_pos=torch.as_tensor(DEFAULT_JOINT_POS_MJLAB),
        joint_vel=torch.zeros(29),
    )
    action, completed = policy.infer(state)
    assert not bool(policy.encoder.input[0, encoder_mode].any())
    assert action.shape == (1, 29)
    assert bool(torch.isfinite(action).all())
    assert not completed

    planner = PlannerSonic(SONIC_DIR / "planner_sonic.onnx")
    planner_qpos = planner.generate("walk forward 0.5")
    chunk = resample_motion(
        planner_qpos,
        source_fps=planner.fps,
        observation_id=0,
        command="walk forward 0.5",
    )
    assert 24 <= planner_qpos.shape[0] <= 64
    assert planner_qpos.shape[0] % 4 == 0
    assert chunk.qpos.shape == (planner_qpos.shape[0] * 50 // 30, 36)


@pytest.mark.skipif(not SONIC_DIR.is_dir(), reason="SONIC bundle is unavailable")
def test_mjlab_cpu_control_step() -> None:
    simulation = SonicMjlabEnv(device="cpu")
    try:
        policy = SonicPolicy(SONIC_DIR)
        action, _ = policy.infer(simulation.robot_state())
        simulation.step(action)
        assert simulation.unwrapped.common_step_counter == 1
        assert simulation.cfg.sim.njmax == 128
    finally:
        simulation.close()


@pytest.mark.skipif(not SONIC_DIR.is_dir(), reason="SONIC bundle is unavailable")
def test_mjlab_offscreen_capture_is_jpeg() -> None:
    simulation = SonicMjlabEnv(
        device="cpu",
        image_width=160,
        image_height=120,
    )
    try:
        jpeg = SonicRenderer(simulation, jpeg_quality=80).capture_jpeg()
        assert jpeg.startswith(b"\xff\xd8")
        assert jpeg.endswith(b"\xff\xd9")
    finally:
        simulation.close()


@pytest.mark.skipif(not CUDA_READY, reason="CUDA Torch and ONNX Runtime are required")
@pytest.mark.skipif(not SONIC_DIR.is_dir(), reason="SONIC bundle is unavailable")
def test_mjlab_and_sonic_share_one_cuda_stream() -> None:
    simulation = SonicMjlabEnv(device="cuda:0")
    try:
        with simulation.compute_context():
            policy = SonicPolicy(
                SONIC_DIR,
                device="cuda:0",
                cuda_stream=simulation.cuda_stream,
            )
            action, _ = policy.infer(simulation.robot_state())

        assert simulation.cuda_stream is not None
        stream_ptr = int(simulation.cuda_stream.cuda_stream)
        assert policy.encoder.cuda_stream_ptr == stream_ptr
        assert policy.decoder.cuda_stream_ptr == stream_ptr
        assert action.device == torch.device("cuda:0")
        simulation.step(action)
        assert simulation.unwrapped.common_step_counter == 1
    finally:
        simulation.close()
