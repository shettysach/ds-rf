from pathlib import Path

import pytest

from shared.config import (
    AgentConfig,
    ArdyConfig,
    MotionGenConfig,
    PlannerSonicConfig,
    SonicConfig,
    TextEncoderConfig,
)


def test_motion_gen_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DSRF_DEVICE", "cuda:0")
    monkeypatch.setenv("DSRF_MOTION_GENERATOR", "planner_sonic")
    monkeypatch.setenv("DSRF_PLANNER_ONNX", "/models/planner.onnx")

    assert MotionGenConfig.from_env() == MotionGenConfig(
        device="cuda:0",
        backend=PlannerSonicConfig(planner_onnx=Path("/models/planner.onnx")),
    )


def test_ardy_motion_gen_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DSRF_DEVICE", "cpu")
    monkeypatch.setenv("DSRF_MOTION_GENERATOR", "ardy")
    monkeypatch.setenv("CHECKPOINTS_DIR", "/models/ardy")

    assert MotionGenConfig.from_env() == MotionGenConfig(
        device="cpu",
        backend=ArdyConfig(
            checkpoints_dir=Path("/models/ardy"),
        ),
    )


def test_ardy_motion_gen_config_requires_no_fixed_conditioning(monkeypatch) -> None:
    monkeypatch.setenv("DSRF_DEVICE", "cuda:0")
    monkeypatch.setenv("DSRF_MOTION_GENERATOR", "ardy")
    monkeypatch.setenv("CHECKPOINTS_DIR", "/models/ardy")
    assert MotionGenConfig.from_env().backend == ArdyConfig(Path("/models/ardy"))


def test_text_encoder_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DSRF_DEVICE", "cuda:0")
    monkeypatch.setenv("DSRF_TEXT_ENCODER_MODEL", "/models/text-encoder")

    assert TextEncoderConfig.from_env() == TextEncoderConfig(
        model=Path("/models/text-encoder"),
        device="cuda:0",
    )


def test_motion_gen_config_rejects_unknown_backend(monkeypatch) -> None:
    monkeypatch.setenv("DSRF_DEVICE", "cpu")
    monkeypatch.setenv("DSRF_MOTION_GENERATOR", "unknown")

    with pytest.raises(ValueError, match="DSRF_MOTION_GENERATOR"):
        MotionGenConfig.from_env()


def test_sonic_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DSRF_DEVICE", "cpu")
    monkeypatch.setenv("DSRF_SONIC_DIR", "/models/sonic")
    monkeypatch.setenv("DSRF_TASK", "portrait-corridors")
    monkeypatch.setenv("DSRF_IMAGE_WIDTH", "640")
    monkeypatch.setenv("DSRF_IMAGE_HEIGHT", "480")
    monkeypatch.setenv("DSRF_JPEG_QUALITY", "85")
    monkeypatch.setenv("DSRF_VIEWER", "native")
    monkeypatch.setenv("DSRF_REFERENCE_GHOST", "true")

    assert SonicConfig.from_env() == SonicConfig(
        sonic_dir=Path("/models/sonic"),
        device="cpu",
        task="portrait-corridors",
        image_width=640,
        image_height=480,
        jpeg_quality=85,
        viewer="native",
        reference_ghost=True,
    )


def test_sonic_config_rejects_unknown_viewer(monkeypatch) -> None:
    monkeypatch.setenv("DSRF_DEVICE", "cpu")
    monkeypatch.setenv("DSRF_SONIC_DIR", "/models/sonic")
    monkeypatch.setenv("DSRF_TASK", "none")
    monkeypatch.setenv("DSRF_IMAGE_WIDTH", "640")
    monkeypatch.setenv("DSRF_IMAGE_HEIGHT", "480")
    monkeypatch.setenv("DSRF_JPEG_QUALITY", "85")
    monkeypatch.setenv("DSRF_VIEWER", "viser")
    monkeypatch.setenv("DSRF_REFERENCE_GHOST", "false")

    with pytest.raises(ValueError, match="DSRF_VIEWER"):
        SonicConfig.from_env()


def test_sonic_config_rejects_invalid_reference_ghost(monkeypatch) -> None:
    monkeypatch.setenv("DSRF_DEVICE", "cpu")
    monkeypatch.setenv("DSRF_SONIC_DIR", "/models/sonic")
    monkeypatch.setenv("DSRF_TASK", "none")
    monkeypatch.setenv("DSRF_IMAGE_WIDTH", "640")
    monkeypatch.setenv("DSRF_IMAGE_HEIGHT", "480")
    monkeypatch.setenv("DSRF_JPEG_QUALITY", "85")
    monkeypatch.setenv("DSRF_VIEWER", "native")
    monkeypatch.setenv("DSRF_REFERENCE_GHOST", "yes")

    with pytest.raises(ValueError, match="DSRF_REFERENCE_GHOST"):
        SonicConfig.from_env()


def test_agent_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DSRF_VLM_URL", "http://127.0.0.1:8080/")
    monkeypatch.setenv("DSRF_VLM_TIMEOUT", "12.5")
    monkeypatch.setenv("DSRF_VLM_SYSTEM_PROMPT", "/prompts/system.md")
    monkeypatch.setenv("DSRF_VLM_USER_PROMPT", "/prompts/user.md")

    assert AgentConfig.from_env() == AgentConfig(
        vlm_url="http://127.0.0.1:8080",
        vlm_timeout=12.5,
        system_prompt=Path("/prompts/system.md"),
        user_prompt=Path("/prompts/user.md"),
        waypoint_debug=False,
        command_mode="waypoint",
    )


def test_missing_runtime_value_fails(monkeypatch) -> None:
    monkeypatch.delenv("DSRF_PLANNER_ONNX", raising=False)
    monkeypatch.setenv("DSRF_MOTION_GENERATOR", "planner_sonic")
    monkeypatch.setenv("DSRF_DEVICE", "cpu")

    with pytest.raises(KeyError, match="DSRF_PLANNER_ONNX"):
        MotionGenConfig.from_env()
