from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any
from uuid import uuid4

import numpy as np
import pyarrow as pa

MOTION_COLUMNS = 36
SONIC_FPS = 50


class StatusState(StrEnum):
    READY = "ready"
    GENERATING = "generating"
    PLAYING = "playing"
    DONE = "done"
    ERROR = "error"


@dataclass(frozen=True)
class MotionCommandRequest:
    command_id: str
    text: str

    @classmethod
    def from_text(cls, text: str) -> "MotionCommandRequest":
        normalized = text.strip()
        if not normalized:
            raise ValueError("Command is empty")
        return cls(command_id=uuid4().hex, text=normalized)


@dataclass(frozen=True)
class MotionChunk:
    command_id: str
    qpos: np.ndarray

    def __post_init__(self) -> None:
        qpos = np.asarray(self.qpos, dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != MOTION_COLUMNS:
            raise ValueError(
                f"Motion qpos must have shape [T, {MOTION_COLUMNS}], got {qpos.shape}"
            )
        if qpos.shape[0] == 0:
            raise ValueError("Motion chunk must contain at least one frame")
        if not np.isfinite(qpos).all():
            raise ValueError("Motion chunk contains NaN or infinite values")
        object.__setattr__(self, "qpos", np.ascontiguousarray(qpos))


@dataclass(frozen=True)
class RuntimeStatus:
    source: str
    state: StatusState
    command_id: str | None = None
    detail: str | None = None


def command_to_arrow(command: MotionCommandRequest) -> tuple[pa.Array, dict[str, str]]:
    return pa.array([command.text], type=pa.string()), {
        "command_id": command.command_id
    }


def command_from_arrow(
    value: pa.Array, metadata: dict[str, Any]
) -> MotionCommandRequest:
    return MotionCommandRequest(
        command_id=str(metadata["command_id"]),
        text=_string_from_arrow(value),
    )


def motion_to_arrow(chunk: MotionChunk) -> tuple[pa.Array, dict[str, str]]:
    return pa.array(chunk.qpos.reshape(-1), type=pa.float32()), {
        "command_id": chunk.command_id
    }


def motion_from_arrow(value: pa.Array, metadata: dict[str, Any]) -> MotionChunk:
    flat = np.asarray(value.to_numpy(zero_copy_only=False), dtype=np.float32)
    if flat.size == 0 or flat.size % MOTION_COLUMNS:
        raise ValueError(
            f"Motion payload has {flat.size} values; expected complete "
            f"{MOTION_COLUMNS}-value frames"
        )
    return MotionChunk(
        command_id=str(metadata["command_id"]),
        qpos=flat.reshape(-1, MOTION_COLUMNS),
    )


def status_to_arrow(status: RuntimeStatus) -> pa.Array:
    return _json_to_arrow(asdict(status))


def status_from_arrow(value: pa.Array) -> RuntimeStatus:
    data = _json_from_arrow(value)
    return RuntimeStatus(
        source=str(data["source"]),
        state=StatusState(str(data["state"])),
        command_id=data.get("command_id"),
        detail=data.get("detail"),
    )


def _json_to_arrow(value: dict[str, Any]) -> pa.Array:
    return pa.array([json.dumps(value, separators=(",", ":"))], type=pa.string())


def _json_from_arrow(value: pa.Array) -> dict[str, Any]:
    decoded = json.loads(_string_from_arrow(value))
    if not isinstance(decoded, dict):
        raise ValueError("Expected a JSON object")
    return decoded


def _string_from_arrow(value: pa.Array) -> str:
    values = value.to_pylist()
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError("Expected one string")
    return values[0]
