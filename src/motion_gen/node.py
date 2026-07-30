from __future__ import annotations

from dora import Node

from motion_gen.planner_sonic import PlannerSonic, PlannerSonicOutputError
from motion_gen.planner_sonic_command import PlannerSonicCommand
from motion_gen.resample import resample_motion
from shared.config import RuntimeConfig
from shared.messages import (
    MotionCommandRequest,
    RuntimeStatus,
    StatusState,
    command_from_arrow,
    motion_to_arrow,
    status_from_arrow,
    status_to_arrow,
)
from shared.onnx import validate_onnx_device


def main() -> None:
    cfg = RuntimeConfig.from_env()
    cfg.validate_motion_gen()
    validate_onnx_device(cfg.device)
    node = Node()
    generator = PlannerSonic(cfg.planner_onnx, device=cfg.device)
    pending: MotionCommandRequest | None = None
    active_command_id: str | None = None

    def report(
        state: StatusState,
        command_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        status = RuntimeStatus("motion-gen", state, command_id, detail)
        node.send_output("status", status_to_arrow(status))

    report(StatusState.READY, detail=f"device={cfg.device}")

    def generate(request: MotionCommandRequest) -> str | None:
        try:
            command = PlannerSonicCommand.parse(request.text)
        except ValueError as exc:
            report(StatusState.ERROR, request.command_id, str(exc))
            return None

        report(StatusState.GENERATING, request.command_id)
        try:
            planner_qpos = generator.generate(command)
        except PlannerSonicOutputError as exc:
            report(StatusState.ERROR, request.command_id, str(exc))
            return None

        chunk = resample_motion(planner_qpos, command_id=request.command_id)
        data, metadata = motion_to_arrow(chunk)
        node.send_output("motion", data, metadata=metadata)
        return request.command_id

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "command":
            request = command_from_arrow(
                event["value"], dict(event.get("metadata") or {})
            )
            if active_command_id is not None:
                pending = request
            else:
                active_command_id = generate(request)
        elif event["id"] == "sonic_status":
            status = status_from_arrow(event["value"])
            if (
                status.source == "sonic"
                and status.state in {StatusState.DONE, StatusState.ERROR}
                and status.command_id == active_command_id
            ):
                active_command_id = None
                if pending is not None:
                    request, pending = pending, None
                    active_command_id = generate(request)


if __name__ == "__main__":
    main()
