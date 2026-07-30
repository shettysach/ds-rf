from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from motion_gen.planner_sonic import PlannerSonic
from motion_gen.planner_sonic_command import PlannerSonicCommand
from motion_gen.resample import resample_motion
from shared.g1 import DEFAULT_JOINT_POS_MJLAB
from sonic.mjlab_env import SonicMjlabEnv
from sonic.policy import RobotState, SonicPolicy

SONIC_DIR = Path("/tmp/GEAR-SONIC")

pytestmark = pytest.mark.integration
CUDA_READY = torch.cuda.is_available() and "CUDAExecutionProvider" in (
    ort.get_available_providers()
)


@pytest.mark.skipif(not SONIC_DIR.is_dir(), reason="SONIC bundle is unavailable")
def test_real_checkpoints_generate_action_and_motion() -> None:
    policy = SonicPolicy(SONIC_DIR)
    encoder_mode = policy.layout.encoder_slices["encoder_mode_4"]
    policy.encoder.input[0, encoder_mode].fill_(1.0)
    state = RobotState(
        root_quat_w=np.array([1.0, 0.0, 0.0, 0.0]),
        root_ang_vel_b=np.zeros(3),
        projected_gravity_b=np.array([0.0, 0.0, -1.0]),
        joint_pos=DEFAULT_JOINT_POS_MJLAB,
        joint_vel=np.zeros(29),
    )
    action, completed = policy.infer(state)
    assert not bool(policy.encoder.input[0, encoder_mode].any())
    assert action.shape == (1, 29)
    assert bool(torch.isfinite(action).all())
    assert completed is None

    planner = PlannerSonic(SONIC_DIR / "planner_sonic.onnx")
    native = planner.generate(
        PlannerSonicCommand.parse("walk forward 0.5", command_id="integration")
    )
    chunk = resample_motion(native, command_id="integration")
    assert 24 <= native.qpos.shape[0] <= 64
    assert native.qpos.shape[0] % 4 == 0
    assert chunk.qpos.shape == (native.qpos.shape[0] * 50 // 30, 36)


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

        assert simulation.cuda_stream_ptr is not None
        assert policy.encoder.cuda_stream_ptr == simulation.cuda_stream_ptr
        assert policy.decoder.cuda_stream_ptr == simulation.cuda_stream_ptr
        assert action.device == torch.device("cuda:0")
        simulation.step(action)
        assert simulation.unwrapped.common_step_counter == 1
    finally:
        simulation.close()
