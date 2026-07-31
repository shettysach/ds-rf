from __future__ import annotations

import math
import time

from dora import Node

from motion_gen.planner_sonic import (
    PlannerMode,
    PlannerSonic,
    PlannerSonicInput,
    Vector3,
)
from motion_gen.resample import resample_motion
from shared.config import MotionGenConfig
from shared.messages import (
    SONIC_FPS,
    PipelineError,
    agent_command_from_arrow,
    motion_to_arrow,
    pipeline_error_to_arrow,
)

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
_OPTIONS = {"facing", "speed", "height"}


def parse_motion_command(text: str) -> PlannerSonicInput:
    fields = text.strip().lower().replace("_", "-").split()
    if not fields:
        raise ValueError("Command is empty")

    requested_mode, *arguments = fields
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
        if direction is None and value in _MOTION_DIRECTIONS:
            direction = _MOTION_DIRECTIONS[value]
        elif speed is None:
            speed = value
        else:
            raise ValueError(f"Unexpected command field: {value}")

    facing_name = options.get("facing", "forward")
    try:
        facing = _MOTION_DIRECTIONS[facing_name]
    except KeyError as exc:
        raise ValueError(f"Unknown facing direction: {facing_name!r}") from exc

    movement = (0.0, 0.0, 0.0)
    if mode not in _STATIONARY_MODES:
        movement = _MOTION_DIRECTIONS["forward"]
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


def main() -> None:
    cfg = MotionGenConfig.from_env()
    node = Node()
    generator = PlannerSonic(cfg.planner_onnx, device=cfg.device)

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] != "command":
            continue

        metadata = dict(event.get("metadata") or {})
        request = agent_command_from_arrow(event["value"], metadata)
        try:
            command = parse_motion_command(request.text)
        except ValueError as exc:
            node.log(
                "warn",
                f"[OBS {request.observation_id}] invalid command: "
                f"{request.text!r} error={str(exc)!r}",
                target="dsrf.motion_gen",
                fields={
                    "event": "invalid_command",
                    "observation_id": str(request.observation_id),
                    "command": request.text,
                    "detail": str(exc),
                },
            )
            error = PipelineError("motion-gen", request.observation_id, str(exc))
            node.send_output("error", pipeline_error_to_arrow(error))
            continue

        started_at = time.perf_counter()
        try:
            planner_qpos = generator.generate(command)
            chunk = resample_motion(
                planner_qpos,
                observation_id=request.observation_id,
                command=request.text,
            )
        except Exception as exc:
            plan_ms = (time.perf_counter() - started_at) * 1000.0
            detail = f"{type(exc).__name__}: {exc}"
            node.log(
                "error",
                f"[OBS {request.observation_id}] motion generation failed: {detail}",
                target="dsrf.motion_gen",
                fields={
                    "event": "motion_generation_error",
                    "observation_id": str(request.observation_id),
                    "command": request.text,
                    "plan_ms": f"{plan_ms:.1f}",
                    "detail": detail,
                },
            )
            raise

        plan_ms = (time.perf_counter() - started_at) * 1000.0
        output_frames = len(chunk.qpos)
        duration_s = output_frames / SONIC_FPS
        node.log(
            "info",
            f"[OBS {request.observation_id}] motion generated: "
            f"command={request.text!r} frames={output_frames} "
            f"duration_s={duration_s:.2f} plan_ms={plan_ms:.1f}",
            target="dsrf.motion_gen",
            fields={
                "event": "motion_generated",
                "observation_id": str(request.observation_id),
                "command": request.text,
                "plan_ms": f"{plan_ms:.1f}",
                "planner_frames": str(len(planner_qpos)),
                "output_frames": str(output_frames),
                "duration_s": f"{duration_s:.2f}",
            },
        )
        data, motion_metadata = motion_to_arrow(chunk)
        node.send_output("motion", data, metadata=motion_metadata)


if __name__ == "__main__":
    main()
