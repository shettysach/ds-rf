from __future__ import annotations

from dora import Node

from motion_gen.planner_sonic import PlannerSonic
from motion_gen.resample import resample_motion
from shared.config import RuntimeConfig
from shared.messages import (
    PlannerCommand,
    RuntimeStatus,
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
    pending: PlannerCommand | None = None
    busy = False
    node.send_output(
        "status",
        status_to_arrow(
            RuntimeStatus("motion-gen", "ready", detail=f"device={cfg.device}")
        ),
    )

    def generate(command: PlannerCommand) -> None:
        nonlocal busy
        busy = True
        node.send_output(
            "status",
            status_to_arrow(
                RuntimeStatus("motion-gen", "generating", command.command_id)
            ),
        )
        try:
            native = generator.generate(command)
            chunk = resample_motion(native, command_id=command.command_id)
            data, metadata = motion_to_arrow(chunk)
            node.send_output("motion", data, metadata=metadata)
        except Exception as exc:
            busy = False
            node.send_output(
                "status",
                status_to_arrow(
                    RuntimeStatus("motion-gen", "error", command.command_id, str(exc))
                ),
            )

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "command":
            command = command_from_arrow(event["value"])
            if busy:
                pending = command
            else:
                generate(command)
        elif event["id"] == "sonic_status":
            status = status_from_arrow(event["value"])
            if status.state == "done":
                busy = False
                if pending is not None:
                    command, pending = pending, None
                    generate(command)


if __name__ == "__main__":
    main()
