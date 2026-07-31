from pathlib import Path

import pytest

from shared.config import AgentConfig, MotionGenConfig, SonicConfig


def test_motion_gen_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DSRF_DEVICE", "cuda:0")
    monkeypatch.setenv("DSRF_PLANNER_ONNX", "/models/planner.onnx")

    assert MotionGenConfig.from_env() == MotionGenConfig(
        planner_onnx=Path("/models/planner.onnx"),
        device="cuda:0",
    )


def test_sonic_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DSRF_DEVICE", "cpu")
    monkeypatch.setenv("DSRF_SONIC_DIR", "/models/sonic")
    monkeypatch.setenv("DSRF_TASK", "portrait-corridors")
    monkeypatch.setenv("DSRF_IMAGE_WIDTH", "640")
    monkeypatch.setenv("DSRF_IMAGE_HEIGHT", "480")
    monkeypatch.setenv("DSRF_JPEG_QUALITY", "85")

    assert SonicConfig.from_env() == SonicConfig(
        sonic_dir=Path("/models/sonic"),
        device="cpu",
        task="portrait-corridors",
        image_width=640,
        image_height=480,
        jpeg_quality=85,
    )


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
    )


def test_missing_runtime_value_fails(monkeypatch) -> None:
    monkeypatch.delenv("DSRF_PLANNER_ONNX", raising=False)
    monkeypatch.setenv("DSRF_DEVICE", "cpu")

    with pytest.raises(KeyError, match="DSRF_PLANNER_ONNX"):
        MotionGenConfig.from_env()
