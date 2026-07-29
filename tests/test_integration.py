from pathlib import Path

import numpy as np
import pytest
import torch

from motion_gen.planner_sonic import PlannerSonic
from motion_gen.resample import resample_motion
from shared.g1 import DEFAULT_JOINT_POS_MJLAB
from shared.messages import PlannerCommand
from sonic.mjlab_env import SonicMjlabEnv
from sonic.policy import RobotState, SonicPolicy

SONIC_DIR = Path("/tmp/GEAR-SONIC")

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not SONIC_DIR.is_dir(), reason="SONIC bundle is unavailable")
def test_real_checkpoints_generate_action_and_motion() -> None:
    policy = SonicPolicy(SONIC_DIR)
    state = RobotState(
        root_quat_w=np.array([1.0, 0.0, 0.0, 0.0]),
        root_ang_vel_b=np.zeros(3),
        projected_gravity_b=np.array([0.0, 0.0, -1.0]),
        joint_pos=DEFAULT_JOINT_POS_MJLAB,
        joint_vel=np.zeros(29),
    )
    action, completed = policy.infer(state)
    assert action.shape == (29,)
    assert np.isfinite(action).all()
    assert completed is None

    planner = PlannerSonic(SONIC_DIR / "planner_sonic.onnx")
    native = planner.generate(PlannerCommand.parse("walk forward 0.5"))
    chunk = resample_motion(native, command_id="integration")
    assert native.qpos.shape == (64, 36)
    assert chunk.qpos.shape == (106, 36)


@pytest.mark.skipif(not SONIC_DIR.is_dir(), reason="SONIC bundle is unavailable")
def test_mjlab_cpu_control_step() -> None:
    simulation = SonicMjlabEnv(device="cpu")
    try:
        policy = SonicPolicy(SONIC_DIR)
        action, _ = policy.infer(simulation.robot_state())
        simulation.env.step(torch.from_numpy(action).unsqueeze(0))
        assert simulation.env.common_step_counter == 1
    finally:
        simulation.close()
