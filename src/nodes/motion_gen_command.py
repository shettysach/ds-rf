from __future__ import annotations

import math

from motion_gen.planner_sonic import PlannerMode, PlannerSonicInput, Vector3

_DIAGONAL = 1.0 / math.sqrt(2.0)

MOTION_MODES: dict[str, PlannerMode] = {
    mode.name.lower().replace("_", "-"): mode for mode in PlannerMode
}
MOTION_DIRECTIONS: dict[str, Vector3] = {
    "forward": (1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "forward-left": (_DIAGONAL, _DIAGONAL, 0.0),
    "forward-right": (_DIAGONAL, -_DIAGONAL, 0.0),
    "backward-left": (-_DIAGONAL, _DIAGONAL, 0.0),
    "backward-right": (-_DIAGONAL, -_DIAGONAL, 0.0),
}
MOTION_COMMAND_USAGE = (
    "Usage: <mode> [direction] [speed] [facing=<direction>] [height=<meters>]"
)
MOTION_COMMAND_EXAMPLES = (
    "Examples: walk left 0.4 | run forward-right speed=1.2 | squat height=0.6"
)

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
_OPTIONS = {"facing", "speed", "height"}


def parse_motion_command(text: str) -> PlannerSonicInput:
    fields = text.strip().lower().replace("_", "-").split()
    if not fields:
        raise ValueError("Command is empty")

    requested_mode, *arguments = fields
    mode = _MODE_ALIASES.get(requested_mode)
    if mode is None:
        try:
            mode = MOTION_MODES[requested_mode]
        except KeyError as exc:
            choices = ", ".join(MOTION_MODES)
            raise ValueError(
                f"Unknown planner_sonic mode {requested_mode!r}; "
                f"expected one of: {choices}"
            ) from exc

    options: dict[str, str] = {}
    positionals: list[str] = []
    for argument in arguments:
        if "=" not in argument:
            positionals.append(argument)
            continue
        name, value = argument.split("=", 1)
        if name not in _OPTIONS:
            raise ValueError(f"Unknown command option: {name!r}")
        if name in options:
            raise ValueError(f"{name.capitalize()} was provided more than once")
        options[name] = value

    direction: Vector3 | None = None
    speed: str | None = options.get("speed")
    for value in positionals:
        if direction is None and value in MOTION_DIRECTIONS:
            direction = MOTION_DIRECTIONS[value]
        elif speed is None:
            speed = value
        else:
            raise ValueError(f"Unexpected command field: {value}")

    facing_name = options.get("facing", "forward")
    try:
        facing = MOTION_DIRECTIONS[facing_name]
    except KeyError as exc:
        raise ValueError(f"Unknown facing direction: {facing_name!r}") from exc

    movement = (0.0, 0.0, 0.0)
    if mode not in _STATIONARY_MODES:
        movement = MOTION_DIRECTIONS["forward"]
    if direction is not None:
        movement = direction
    return PlannerSonicInput(
        mode=mode,
        movement_direction=movement,
        facing_direction=facing,
        target_vel=_positive_float(speed, "Speed") if speed is not None else -1.0,
        height=(
            _nonnegative_float(options["height"], "Height")
            if "height" in options
            else -1.0
        ),
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
