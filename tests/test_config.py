from pathlib import Path

import pytest

from shared.config import MotionGenConfig, SonicConfig


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

    assert SonicConfig.from_env() == SonicConfig(
        sonic_dir=Path("/models/sonic"),
        device="cpu",
    )


def test_missing_runtime_value_fails(monkeypatch) -> None:
    monkeypatch.delenv("DSRF_PLANNER_ONNX", raising=False)
    monkeypatch.setenv("DSRF_DEVICE", "cpu")

    with pytest.raises(KeyError, match="DSRF_PLANNER_ONNX"):
        MotionGenConfig.from_env()
