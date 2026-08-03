from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import IntEnum

Vector3 = tuple[float, float, float]


class PlannerMode(IntEnum):
    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    SQUAT = 4
    KNEEL_TWO_LEG = 5
    KNEEL_ONE_LEG = 6
    LYING_FACEDOWN = 7
    HAND_CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_JAB = 11
    RIGHT_JAB = 12
    RANDOM_PUNCHES = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    HAPPY = 17
    STEALTH = 18
    INJURED = 19
    CAREFUL = 20
    OBJECT_CARRYING = 21
    CROUCH = 22
    HAPPY_DANCE = 23
    ZOMBIE = 24
    POINT = 25
    SCARED = 26


@dataclass(frozen=True)
class PlannerSonicInput:
    mode: PlannerMode
    movement_direction: Vector3
    facing_direction: Vector3
    target_vel: float = -1.0
    height: float = -1.0
    random_seed: int = 1234


@dataclass(frozen=True)
class PlannerCommand:
    """Validated VLM command before conversion to planner_sonic inputs."""

    motion: str
    direction: str


_DIAGONAL = 1.0 / math.sqrt(2.0)

_MOTION_MODES: dict[str, PlannerMode] = {
    mode.name.lower().replace("_", "-"): mode for mode in PlannerMode
}
_MOTION_DIRECTIONS: dict[str, Vector3] = {
    "forward": (1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "forward-left": (_DIAGONAL, _DIAGONAL, 0.0),
    "forward-right": (_DIAGONAL, -_DIAGONAL, 0.0),
    "backward-left": (-_DIAGONAL, _DIAGONAL, 0.0),
    "backward-right": (-_DIAGONAL, -_DIAGONAL, 0.0),
}
_MODE_ALIASES = {
    "stand": PlannerMode.IDLE,
    "slowwalk": PlannerMode.SLOW_WALK,
    "kneel": PlannerMode.KNEEL_ONE_LEG,
    "crawl": PlannerMode.HAND_CRAWLING,
}
_STATIONARY_MODES = {
    PlannerMode.IDLE,
    PlannerMode.SQUAT,
    PlannerMode.KNEEL_TWO_LEG,
    PlannerMode.KNEEL_ONE_LEG,
    PlannerMode.LYING_FACEDOWN,
    PlannerMode.IDLE_BOXING,
}
def parse_motion_command(text: str) -> PlannerSonicInput:
    """Parse the VLM's JSON command into planner_sonic's ONNX inputs."""
    command = _parse_vlm_command(text)
    requested_mode = command.motion
    mode = _MODE_ALIASES.get(requested_mode)
    if mode is None:
        try:
            mode = _MOTION_MODES[requested_mode]
        except KeyError as exc:
            choices = ", ".join(_MOTION_MODES)
            raise ValueError(
                f"Unknown planner_sonic mode {requested_mode!r}; "
                f"expected one of: {choices}"
            ) from exc

    try:
        direction = _MOTION_DIRECTIONS[command.direction]
    except KeyError as exc:
        choices = ", ".join(_MOTION_DIRECTIONS)
        raise ValueError(
            f"Unknown planner_sonic direction {command.direction!r}; "
            f"expected one of: {choices}"
        ) from exc

    movement = (0.0, 0.0, 0.0)
    if mode not in _STATIONARY_MODES:
        movement = direction
    return PlannerSonicInput(
        mode=mode,
        movement_direction=movement,
        facing_direction=_MOTION_DIRECTIONS["forward"],
    )


def _parse_vlm_command(text: str) -> PlannerCommand:
    if not text.strip():
        raise ValueError("Command is empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Command must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("Command must be a JSON object")

    expected_fields = {"motion", "direction"}
    if set(payload) != expected_fields:
        missing = expected_fields.difference(payload)
        unexpected = set(payload).difference(expected_fields)
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected fields: {', '.join(sorted(unexpected))}")
        raise ValueError(f"Command must contain only motion and direction ({'; '.join(details)})")

    motion = payload["motion"]
    direction = payload["direction"]
    if not isinstance(motion, str) or not isinstance(direction, str):
        raise ValueError("Command motion and direction must be strings")
    motion = motion.strip().lower().replace("_", "-")
    direction = direction.strip().lower().replace("_", "-")
    if not motion or not direction:
        raise ValueError("Command motion and direction must not be empty")
    return PlannerCommand(motion=motion, direction=direction)
