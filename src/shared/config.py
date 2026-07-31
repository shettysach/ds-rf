from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MotionGenConfig:
    planner_onnx: Path
    device: str

    @classmethod
    def from_env(cls) -> "MotionGenConfig":
        return cls(
            planner_onnx=Path(os.environ["DSRF_PLANNER_ONNX"]),
            device=os.environ["DSRF_DEVICE"],
        )


@dataclass(frozen=True)
class SonicConfig:
    sonic_dir: Path
    device: str
    task: str | None
    image_width: int
    image_height: int
    jpeg_quality: int

    @classmethod
    def from_env(cls) -> "SonicConfig":
        return cls(
            sonic_dir=Path(os.environ["DSRF_SONIC_DIR"]),
            device=os.environ["DSRF_DEVICE"],
            task=_optional_name("DSRF_TASK"),
            image_width=_positive_int("DSRF_IMAGE_WIDTH"),
            image_height=_positive_int("DSRF_IMAGE_HEIGHT"),
            jpeg_quality=_bounded_int("DSRF_JPEG_QUALITY", minimum=1, maximum=100),
        )


@dataclass(frozen=True)
class AgentConfig:
    vlm_url: str
    vlm_timeout: float
    system_prompt: Path
    user_prompt: Path

    @classmethod
    def from_env(cls) -> "AgentConfig":
        url = os.environ["DSRF_VLM_URL"].strip().rstrip("/")
        if not url:
            raise ValueError("DSRF_VLM_URL must not be empty")
        timeout = float(os.environ["DSRF_VLM_TIMEOUT"])
        if timeout <= 0.0:
            raise ValueError("DSRF_VLM_TIMEOUT must be positive")
        return cls(
            vlm_url=url,
            vlm_timeout=timeout,
            system_prompt=Path(os.environ["DSRF_VLM_SYSTEM_PROMPT"]),
            user_prompt=Path(os.environ["DSRF_VLM_USER_PROMPT"]),
        )


def _positive_int(name: str) -> int:
    return _bounded_int(name, minimum=1)


def _bounded_int(name: str, *, minimum: int, maximum: int | None = None) -> int:
    value = int(os.environ[name])
    if value < minimum or (maximum is not None and value > maximum):
        expected = f">= {minimum}" if maximum is None else f"{minimum}..{maximum}"
        raise ValueError(f"{name} must be in {expected}, got {value}")
    return value


def _optional_name(name: str) -> str | None:
    value = os.environ[name].strip()
    return None if value.lower() == "none" or not value else value
