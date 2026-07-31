from __future__ import annotations

from dora import Node

from motion_gen.planner_sonic import PlannerSonic
from motion_gen.resample import resample_motion
from nodes.motion_gen_command import parse_motion_command
from shared.config import MotionGenConfig
from shared.messages import (
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
            error = PipelineError("motion-gen", request.observation_id, str(exc))
            node.send_output("error", pipeline_error_to_arrow(error))
            continue

        planner_qpos = generator.generate(command)
        chunk = resample_motion(
            planner_qpos,
            observation_id=request.observation_id,
            command=request.text,
        )
        data, motion_metadata = motion_to_arrow(chunk)
        node.send_output("motion", data, metadata=motion_metadata)


if __name__ == "__main__":
    main()
