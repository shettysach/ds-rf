from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pyarrow as pa

MOTION_COLUMNS = 36
SONIC_FPS = 50


@dataclass(frozen=True)
class MotionChunk:
    observation_id: int
    command: str
    qpos: np.ndarray
    preserve_root_z: bool = False

    def __post_init__(self) -> None:
        if self.observation_id < 0:
            raise ValueError("Observation ID must be non-negative")
        if not self.command.strip():
            raise ValueError("Motion command is empty")
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
class AgentCommand:
    observation_id: int
    text: str

    def __post_init__(self) -> None:
        if self.observation_id < 0:
            raise ValueError("Observation ID must be non-negative")
        normalized = self.text.strip()
        if not normalized:
            raise ValueError("Command is empty")
        object.__setattr__(self, "text", normalized)


@dataclass(frozen=True)
class VisualObservation:
    observation_id: int
    completed_command: str | None
    jpeg: bytes

    def __post_init__(self) -> None:
        if self.observation_id < 0:
            raise ValueError("Observation ID must be non-negative")
        if not self.jpeg:
            raise ValueError("Observation JPEG is empty")


@dataclass(frozen=True)
class PipelineError:
    source: str
    observation_id: int
    detail: str

    def __post_init__(self) -> None:
        if self.observation_id < 0:
            raise ValueError("Observation ID must be non-negative")
        if not self.source:
            raise ValueError("Error source is empty")
        if not self.detail:
            raise ValueError("Error detail is empty")


def motion_to_arrow(chunk: MotionChunk) -> tuple[pa.Array, dict[str, str]]:
    return pa.array(chunk.qpos.reshape(-1), type=pa.float32()), {
        "observation_id": str(chunk.observation_id),
        "command": chunk.command,
        "preserve_root_z": str(chunk.preserve_root_z).lower(),
    }


def motion_from_arrow(value: pa.Array, metadata: dict[str, Any]) -> MotionChunk:
    flat = np.asarray(value.to_numpy(zero_copy_only=False), dtype=np.float32)
    if flat.size == 0 or flat.size % MOTION_COLUMNS:
        raise ValueError(
            f"Motion payload has {flat.size} values; expected complete "
            f"{MOTION_COLUMNS}-value frames"
        )
    return MotionChunk(
        observation_id=_observation_id(metadata),
        command=str(metadata["command"]),
        qpos=flat.reshape(-1, MOTION_COLUMNS),
        preserve_root_z=metadata.get("preserve_root_z", "false") == "true",
    )


def agent_command_to_arrow(
    command: AgentCommand,
) -> tuple[pa.Array, dict[str, str]]:
    return pa.array([command.text], type=pa.string()), {
        "observation_id": str(command.observation_id)
    }


def agent_command_from_arrow(
    value: pa.Array, metadata: dict[str, Any]
) -> AgentCommand:
    return AgentCommand(
        observation_id=_observation_id(metadata),
        text=_string_from_arrow(value),
    )


def observation_to_arrow(
    observation: VisualObservation,
) -> tuple[pa.Array, dict[str, str]]:
    metadata = {
        "observation_id": str(observation.observation_id),
        "mime_type": "image/jpeg",
    }
    if observation.completed_command is not None:
        metadata["completed_command"] = observation.completed_command
    return pa.array([observation.jpeg], type=pa.binary()), metadata


def observation_from_arrow(
    value: pa.Array, metadata: dict[str, Any]
) -> VisualObservation:
    mime_type = metadata.get("mime_type")
    if mime_type != "image/jpeg":
        raise ValueError(f"Unsupported observation MIME type: {mime_type!r}")
    return VisualObservation(
        observation_id=_observation_id(metadata),
        completed_command=(
            str(metadata["completed_command"])
            if "completed_command" in metadata
            else None
        ),
        jpeg=_binary_from_arrow(value),
    )


def pipeline_error_to_arrow(error: PipelineError) -> pa.Array:
    return _json_to_arrow(asdict(error))


def pipeline_error_from_arrow(value: pa.Array) -> PipelineError:
    data = _json_from_arrow(value)
    return PipelineError(
        source=str(data["source"]),
        observation_id=int(data["observation_id"]),
        detail=str(data["detail"]),
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


def _binary_from_arrow(value: pa.Array) -> bytes:
    values = value.to_pylist()
    if len(values) != 1 or not isinstance(values[0], bytes):
        raise ValueError("Expected one binary value")
    return values[0]


def _observation_id(metadata: dict[str, Any]) -> int:
    observation_id = int(metadata["observation_id"])
    if observation_id < 0:
        raise ValueError("Observation ID must be non-negative")
    return observation_id
