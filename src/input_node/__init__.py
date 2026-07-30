from __future__ import annotations

import time
from typing import Any, cast

from dora import Node

from input_node.socket_server import CommandServer, command_socket_path
from shared.messages import (
    MotionCommandRequest,
    RuntimeStatus,
    StatusState,
    command_to_arrow,
    status_from_arrow,
)


def _format_status(status: RuntimeStatus) -> str | None:
    if status.state != StatusState.ERROR:
        return None
    suffix = f" ({status.detail})" if status.detail else ""
    return f"[{status.source}] {status.state}{suffix}"


def main() -> None:
    node = Node()
    server = CommandServer(command_socket_path())
    print(f"Command socket ready: {server.path}")

    try:
        while True:
            event = cast(Any, node).try_recv()
            if event is not None:
                if event["type"] == "STOP":
                    break
                if event["type"] == "INPUT":
                    status = status_from_arrow(event["value"])
                    message = _format_status(status)
                    if message is not None:
                        print(message)
                        server.broadcast(message)

            commands = server.poll()
            for text in commands:
                try:
                    command = MotionCommandRequest.from_text(text)
                except ValueError as exc:
                    message = f"Invalid command: {exc}"
                    print(message)
                    server.broadcast(message)
                    continue
                data, metadata = command_to_arrow(command)
                node.send_output("command", data, metadata=metadata)

            if event is None and not commands:
                time.sleep(0.01)
    finally:
        server.close()


if __name__ == "__main__":
    main()
