from __future__ import annotations

import math
from dataclasses import dataclass

PLANNER_SONIC_MODES = {
    "idle": 0,
    "slow-walk": 1,
    "walk": 2,
    "run": 3,
    "squat": 4,
    "kneel-two-leg": 5,
    "kneel-one-leg": 6,
    "lying-facedown": 7,
    "hand-crawling": 8,
    "idle-boxing": 9,
    "walk-boxing": 10,
    "left-jab": 11,
    "right-jab": 12,
    "random-punches": 13,
    "elbow-crawling": 14,
    "left-hook": 15,
    "right-hook": 16,
    "happy": 17,
    "stealth": 18,
    "injured": 19,
    "careful": 20,
    "object-carrying": 21,
    "crouch": 22,
    "happy-dance": 23,
    "zombie": 24,
    "point": 25,
    "scared": 26,
}

_MODE_ALIASES = {
    "stand": "idle",
    "slowwalk": "slow-walk",
    "kneel": "kneel-one-leg",
    "crawl": "hand-crawling",
}

_DIAGONAL = 1.0 / math.sqrt(2.0)
PLANNER_SONIC_DIRECTIONS = {
    "forward": (1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "forward-left": (_DIAGONAL, _DIAGONAL, 0.0),
    "forward-right": (_DIAGONAL, -_DIAGONAL, 0.0),
    "backward-left": (-_DIAGONAL, _DIAGONAL, 0.0),
    "backward-right": (-_DIAGONAL, -_DIAGONAL, 0.0),
}

PLANNER_SONIC_COMMAND_HELP = (
    "Usage: <mode> [direction] [speed] [facing=<direction>] [height=<meters>]"
)


@dataclass(frozen=True)
class PlannerSonicCommand:
    command_id: str
    mode: int
    movement_direction: tuple[float, float, float]
    facing_direction: tuple[float, float, float]
    target_vel: float = -1.0
    height: float = -1.0
    random_seed: int = 1234

    @classmethod
    def parse(cls, text: str, *, command_id: str) -> "PlannerSonicCommand":
        fields = text.strip().lower().replace("_", "-").split()
        if not fields:
            raise ValueError("Command is empty")

        requested_mode = fields.pop(0)
        mode_name = _MODE_ALIASES.get(requested_mode, requested_mode)
        if mode_name not in PLANNER_SONIC_MODES:
            choices = ", ".join(PLANNER_SONIC_MODES)
            raise ValueError(
                f"Unknown planner_sonic mode {requested_mode!r}; expected one of: "
                f"{choices}"
            )

        direction_name: str | None = None
        facing_name = "forward"
        target_vel = -1.0
        height = -1.0
        speed_was_set = False
        height_was_set = False

        for field in fields:
            if field in PLANNER_SONIC_DIRECTIONS and direction_name is None:
                direction_name = field
                continue
            if field.startswith("facing="):
                facing_name = field.partition("=")[2]
                if facing_name not in PLANNER_SONIC_DIRECTIONS:
                    raise ValueError(f"Unknown facing direction: {facing_name!r}")
                continue
            if field.startswith("speed="):
                if speed_was_set:
                    raise ValueError("Speed was provided more than once")
                target_vel = _positive_float(field.partition("=")[2], "Speed")
                speed_was_set = True
                continue
            if field.startswith("height="):
                if height_was_set:
                    raise ValueError("Height was provided more than once")
                height = _nonnegative_float(field.partition("=")[2], "Height")
                height_was_set = True
                continue
            if not speed_was_set:
                target_vel = _positive_float(field, "Speed")
                speed_was_set = True
                continue
            raise ValueError(f"Unexpected command field: {field}")

        if direction_name is None:
            movement = (
                (0.0, 0.0, 0.0)
                if mode_name
                in {
                    "idle",
                    "squat",
                    "kneel-two-leg",
                    "kneel-one-leg",
                    "lying-facedown",
                    "idle-boxing",
                }
                else PLANNER_SONIC_DIRECTIONS["forward"]
            )
        else:
            movement = PLANNER_SONIC_DIRECTIONS[direction_name]

        return cls(
            command_id=command_id,
            mode=PLANNER_SONIC_MODES[mode_name],
            movement_direction=movement,
            facing_direction=PLANNER_SONIC_DIRECTIONS[facing_name],
            target_vel=target_vel,
            height=height,
        )


def _positive_float(value: str, label: str) -> float:
    parsed = _finite_float(value, label)
    if parsed <= 0.0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _nonnegative_float(value: str, label: str) -> float:
    parsed = _finite_float(value, label)
    if parsed < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return parsed


def _finite_float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed
