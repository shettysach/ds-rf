from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

Vector2 = tuple[float, float]

WALK_SPEED_M_S = 0.5

_DIRECTIONS: dict[str, Vector2] = {
    # ARDY uses x/z as its horizontal plane, with +z forward.
    "forward": (0.0, 1.0),
    "backward": (0.0, -1.0),
    "left": (1.0, 0.0),
    "right": (-1.0, 0.0),
}
_MOTIONS = {"stand", "walk"}


@dataclass(frozen=True)
class ArdyCommand:
    motion: str
    direction: Vector2

    @property
    def target_velocity(self) -> Vector2:
        if self.motion == "stand":
            return (0.0, 0.0)
        return (
            self.direction[0] * WALK_SPEED_M_S,
            self.direction[1] * WALK_SPEED_M_S,
        )


def parse_motion_command(text: str) -> ArdyCommand:
    """Parse the VLM JSON command into ARDY text and velocity conditioning."""
    if not text.strip():
        raise ValueError("Command is empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Command must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("Command must be a JSON object")
    payload = cast(dict[str, object], payload)

    expected_fields = {"motion", "direction"}
    if set(payload) != expected_fields:
        missing = expected_fields.difference(payload)
        unexpected = set(payload).difference(expected_fields)
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected fields: {', '.join(sorted(unexpected))}")
        raise ValueError(
            "Command must contain only motion and direction "
            f"({'; '.join(details)})"
        )

    motion = payload["motion"]
    direction_name = payload["direction"]
    if not isinstance(motion, str) or not isinstance(direction_name, str):
        raise ValueError("Command motion and direction must be strings")
    motion = motion.strip().lower()
    direction_name = direction_name.strip().lower()
    if motion not in _MOTIONS:
        raise ValueError(
            f"Unknown ARDY motion {motion!r}; expected one of: {', '.join(sorted(_MOTIONS))}"
        )
    try:
        direction = _DIRECTIONS[direction_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown ARDY direction {direction_name!r}; "
            f"expected one of: {', '.join(_DIRECTIONS)}"
        ) from exc
    return ArdyCommand(motion=motion, direction=direction)
