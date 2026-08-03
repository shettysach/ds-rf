from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

type ViewerMode = Literal["none", "native"]


@dataclass(frozen=True)
class MotionGenConfig:
    device: str
    backend: PlannerSonicConfig | ArdyConfig

    @classmethod
    def from_env(cls) -> "MotionGenConfig":
        generator = _motion_generator()
        if generator == "ardy":
            return cls(
                device=os.environ["DSRF_DEVICE"],
                backend=ArdyConfig(
                    checkpoints_dir=Path(os.environ["CHECKPOINTS_DIR"]),
                ),
            )
        return cls(
            device=os.environ["DSRF_DEVICE"],
            backend=PlannerSonicConfig(
                planner_onnx=Path(os.environ["DSRF_PLANNER_ONNX"]),
            ),
        )


@dataclass(frozen=True)
class PlannerSonicConfig:
    planner_onnx: Path


@dataclass(frozen=True)
class ArdyConfig:
    checkpoints_dir: Path


@dataclass(frozen=True)
class TextEncoderConfig:
    model: Path
    device: str

    @classmethod
    def from_env(cls) -> "TextEncoderConfig":
        return cls(
            model=Path(os.environ["DSRF_TEXT_ENCODER_MODEL"]),
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
    viewer: ViewerMode
    reference_ghost: bool

    @classmethod
    def from_env(cls) -> "SonicConfig":
        return cls(
            sonic_dir=Path(os.environ["DSRF_SONIC_DIR"]),
            device=os.environ["DSRF_DEVICE"],
            task=_optional_name("DSRF_TASK"),
            image_width=_positive_int("DSRF_IMAGE_WIDTH"),
            image_height=_positive_int("DSRF_IMAGE_HEIGHT"),
            jpeg_quality=_bounded_int("DSRF_JPEG_QUALITY", minimum=1, maximum=100),
            viewer=_viewer_mode(),
            reference_ghost=_boolean("DSRF_REFERENCE_GHOST"),
        )


@dataclass(frozen=True)
class AgentConfig:
    vlm_url: str
    vlm_timeout: float
    system_prompt: Path
    user_prompt: Path
    waypoint_debug: bool
    command_mode: Literal["waypoint", "direction"]

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
            waypoint_debug=_optional_boolean("DSRF_WAYPOINT_DEBUG", default=False),
            command_mode=(
                "direction"
                if os.environ.get("DSRF_MOTION_GENERATOR", "").strip().lower()
                == "planner_sonic"
                else "waypoint"
            ),
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


def _viewer_mode() -> ViewerMode:
    value = os.environ["DSRF_VIEWER"].strip().lower()
    if value not in {"none", "native"}:
        raise ValueError("DSRF_VIEWER must be 'none' or 'native'")
    return value


def _motion_generator() -> Literal["planner_sonic", "ardy"]:
    value = os.environ["DSRF_MOTION_GENERATOR"].strip().lower()
    if value not in {"planner_sonic", "ardy"}:
        raise ValueError("DSRF_MOTION_GENERATOR must be 'planner_sonic' or 'ardy'")
    return value


def _boolean(name: str) -> bool:
    value = os.environ[name].strip().lower()
    if value not in {"false", "true"}:
        raise ValueError(f"{name} must be 'false' or 'true'")
    return value == "true"


def _optional_boolean(name: str, *, default: bool) -> bool:
    if name not in os.environ:
        return default
    return _boolean(name)
