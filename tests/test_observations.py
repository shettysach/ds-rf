from pathlib import Path

from sonic.observations import ObservationLayout

SONIC_DIR = Path("/tmp/GEAR-SONIC")


def test_default_observation_layout() -> None:
    layout = ObservationLayout.load(SONIC_DIR / "observation_config.yaml")
    assert layout.encoder_input_dimension == 1762
    assert layout.policy_input_dimension == 994
    assert "motion_joint_positions_10frame_step5" in layout.required_g1


def test_low_latency_observation_layout() -> None:
    layout = ObservationLayout.load(
        SONIC_DIR / "low_latency" / "observation_config.yaml"
    )
    assert layout.encoder_input_dimension == 1247
    assert layout.policy_input_dimension == 994
    assert "motion_joint_positions_10frame_step1" in layout.required_g1
