from __future__ import annotations

from dora import Node

from motion_gen.planner_sonic import PlannerSonic
from motion_gen.resample import resample_motion
from nodes.motion_gen_command import parse_motion_command
from shared.config import MotionGenConfig
from shared.messages import (
    MotionCommandRequest,
    RuntimeStatus,
    StatusState,
    command_from_arrow,
    motion_to_arrow,
    status_from_arrow,
    status_to_arrow,
)


def main() -> None:
    cfg = MotionGenConfig.from_env()
    node = Node()
    generator = PlannerSonic(cfg.planner_onnx, device=cfg.device)
    pending: MotionCommandRequest | None = None
    active: MotionCommandRequest | None = None

    def report(
        state: StatusState,
        command_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        status = RuntimeStatus("motion-gen", state, command_id, detail)
        node.send_output("status", status_to_arrow(status))

    report(StatusState.READY, detail=f"device={cfg.device}")

    def generate(request: MotionCommandRequest) -> bool:
        try:
            command = parse_motion_command(request.text)
        except ValueError as exc:
            report(StatusState.ERROR, request.command_id, str(exc))
            return False

        report(StatusState.GENERATING, request.command_id)
        planner_qpos = generator.generate(command)
        chunk = resample_motion(planner_qpos, command_id=request.command_id)
        data, metadata = motion_to_arrow(chunk)
        node.send_output("motion", data, metadata=metadata)
        return True

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "command":
            request = command_from_arrow(
                event["value"], dict(event.get("metadata") or {})
            )
            if active is not None:
                pending = request
            elif generate(request):
                active = request
        elif event["id"] == "sonic_status":
            status = status_from_arrow(event["value"])
            if (
                status.source == "sonic"
                and status.state in {StatusState.DONE, StatusState.ERROR}
                and active is not None
                and status.command_id == active.command_id
            ):
                completed, active = active, None
                replacement, pending = pending, None
                if status.state == StatusState.ERROR:
                    if replacement is not None and generate(replacement):
                        active = replacement
                    continue

                request = replacement if replacement is not None else completed
                if generate(request):
                    active = request
                elif replacement is not None and generate(completed):
                    # A malformed replacement must not stop the last valid
                    # indefinite command.
                    active = completed


if __name__ == "__main__":
    main()
