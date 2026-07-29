from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast
from uuid import uuid4

import numpy as np
import pyarrow as pa

SCHEMA_VERSION = 1

Direction = Literal["forward", "backward", "left", "right"]
StatusState = Literal["ready", "generating", "playing", "done", "error"]

_MODE_IDS = {"stand": 0, "slow-walk": 1, "walk": 2, "run": 3}
_DIRECTIONS: dict[Direction, tuple[float, float, float]] = {
    "forward": (1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
}


@dataclass(frozen=True)
class PlannerCommand:
    command_id: str
    mode: int
    movement_direction: tuple[float, float, float]
    facing_direction: tuple[float, float, float]
    target_vel: float = -1.0
    height: float = -1.0
    random_seed: int = 1234

    @classmethod
    def parse(cls, text: str) -> "PlannerCommand":
        fields = text.strip().lower().split()
        if not fields:
            raise ValueError("Command is empty")
        mode_name = fields.pop(0)
        if mode_name not in _MODE_IDS:
            choices = ", ".join(_MODE_IDS)
            raise ValueError(
                f"Unknown motion {mode_name!r}; expected one of: {choices}"
            )

        direction: Direction = "forward"
        if fields and fields[0] in _DIRECTIONS:
            direction = cast(Direction, fields.pop(0))
        target_vel = -1.0
        if fields:
            try:
                target_vel = float(fields.pop(0))
            except ValueError as exc:
                raise ValueError("Speed must be a number in meters per second") from exc
            if target_vel <= 0.0:
                raise ValueError("Speed must be positive")
        if fields:
            raise ValueError(f"Unexpected command fields: {' '.join(fields)}")

        movement = (0.0, 0.0, 0.0) if mode_name == "stand" else _DIRECTIONS[direction]
        # Keep the robot facing forward for lateral and backward motion. This exposes
        # strafing/backward behavior without silently rotating the world frame.
        facing = (1.0, 0.0, 0.0)
        return cls(
            command_id=uuid4().hex,
            mode=_MODE_IDS[mode_name],
            movement_direction=movement,
            facing_direction=facing,
            target_vel=target_vel,
        )


@dataclass(frozen=True)
class MotionChunk:
    command_id: str
    qpos: np.ndarray
    fps: int = 50

    def __post_init__(self) -> None:
        qpos = np.asarray(self.qpos, dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != 36:
            raise ValueError(f"Motion qpos must have shape [T, 36], got {qpos.shape}")
        if qpos.shape[0] == 0:
            raise ValueError("Motion chunk must contain at least one frame")
        if self.fps <= 0:
            raise ValueError("Motion fps must be positive")
        if not np.isfinite(qpos).all():
            raise ValueError("Motion chunk contains NaN or infinite values")
        object.__setattr__(self, "qpos", np.ascontiguousarray(qpos))


@dataclass(frozen=True)
class RuntimeStatus:
    source: str
    state: StatusState
    command_id: str | None = None
    detail: str | None = None


def command_to_arrow(command: PlannerCommand) -> pa.Array:
    return _json_to_arrow(asdict(command))


def command_from_arrow(value: pa.Array) -> PlannerCommand:
    data = _json_from_arrow(value)
    return PlannerCommand(
        command_id=str(data["command_id"]),
        mode=int(data["mode"]),
        movement_direction=_vec3(data["movement_direction"]),
        facing_direction=_vec3(data["facing_direction"]),
        target_vel=float(data["target_vel"]),
        height=float(data["height"]),
        random_seed=int(data["random_seed"]),
    )


def motion_to_arrow(chunk: MotionChunk) -> tuple[pa.Array, dict[str, str]]:
    metadata = {
        "schema_version": str(SCHEMA_VERSION),
        "command_id": chunk.command_id,
        "frames": str(chunk.qpos.shape[0]),
        "columns": "36",
        "fps": str(chunk.fps),
    }
    return pa.array(chunk.qpos.reshape(-1), type=pa.float32()), metadata


def motion_from_arrow(value: pa.Array, metadata: dict[str, Any]) -> MotionChunk:
    _validate_schema(metadata)
    frames = int(metadata["frames"])
    columns = int(metadata["columns"])
    if columns != 36:
        raise ValueError(f"Unsupported motion column count: {columns}")
    flat = np.asarray(value.to_numpy(zero_copy_only=False), dtype=np.float32)
    if flat.size != frames * columns:
        raise ValueError(
            f"Motion payload has {flat.size} values; expected {frames * columns}"
        )
    return MotionChunk(
        command_id=str(metadata["command_id"]),
        qpos=flat.reshape(frames, columns),
        fps=int(metadata["fps"]),
    )


def status_to_arrow(status: RuntimeStatus) -> pa.Array:
    return _json_to_arrow(asdict(status))


def status_from_arrow(value: pa.Array) -> RuntimeStatus:
    data = _json_from_arrow(value)
    return RuntimeStatus(
        source=str(data["source"]),
        state=cast(StatusState, str(data["state"])),
        command_id=data.get("command_id"),
        detail=data.get("detail"),
    )


def _json_to_arrow(value: dict[str, Any]) -> pa.Array:
    return pa.array([json.dumps(value, separators=(",", ":"))], type=pa.string())


def _json_from_arrow(value: pa.Array) -> dict[str, Any]:
    values = value.to_pylist()
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError("Expected one JSON string")
    decoded = json.loads(values[0])
    if not isinstance(decoded, dict):
        raise ValueError("Expected a JSON object")
    return decoded


def _vec3(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Expected a 3-vector, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


def _validate_schema(metadata: dict[str, Any]) -> None:
    version = int(metadata.get("schema_version", -1))
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported message schema version: {version}")
