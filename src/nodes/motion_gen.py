from __future__ import annotations

import time

from dora import Node

from motion_gen.planner_sonic import PlannerSonic
from motion_gen.resample import resample_motion
from nodes.motion_gen_command import parse_motion_command
from shared.config import MotionGenConfig
from shared.messages import (
    SONIC_FPS,
    PipelineError,
    agent_command_from_arrow,
    motion_to_arrow,
    pipeline_error_to_arrow,
)


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
