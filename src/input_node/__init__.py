from __future__ import annotations

import queue
import threading
import time
from typing import Any, cast

from dora import Node

from shared.messages import PlannerCommand, command_to_arrow, status_from_arrow


def _read_commands(commands: queue.Queue[str | None]) -> None:
    try:
        while True:
            commands.put(input("motion> "))
    except (EOFError, KeyboardInterrupt):
        commands.put(None)


def main() -> None:
    node = Node()
    commands: queue.Queue[str | None] = queue.Queue()
    threading.Thread(target=_read_commands, args=(commands,), daemon=True).start()
    print("Commands: stand | slow-walk | walk | run [direction] [speed]")

    while True:
        event = cast(Any, node).try_recv()
        if event is not None:
            if event["type"] == "STOP":
                break
            if event["type"] == "INPUT":
                status = status_from_arrow(event["value"])
                suffix = f" ({status.detail})" if status.detail else ""
                print(f"[{status.source}] {status.state}{suffix}")

        try:
            text = commands.get_nowait()
        except queue.Empty:
            time.sleep(0.01)
            continue
        if text is None:
            # Dora may run without an attached terminal. Keep forwarding statuses
            # until the runtime sends STOP even when stdin is closed.
            continue
        if text.strip().lower() in {"quit", "exit"}:
            break
        try:
            command = PlannerCommand.parse(text)
        except ValueError as exc:
            print(f"Invalid command: {exc}")
            continue
        node.send_output("command", command_to_arrow(command))


if __name__ == "__main__":
    main()
