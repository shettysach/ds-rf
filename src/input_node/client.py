from __future__ import annotations

import socket
import threading

from input_node.socket_server import command_socket_path
from motion_gen.planner_sonic_command import (
    PLANNER_SONIC_COMMAND_HELP,
    PLANNER_SONIC_DIRECTIONS,
    PLANNER_SONIC_MODES,
)


def _print_help() -> None:
    print(PLANNER_SONIC_COMMAND_HELP)
    print("Modes: " + " | ".join(PLANNER_SONIC_MODES))
    print("Directions: " + " | ".join(PLANNER_SONIC_DIRECTIONS))
    print("Examples: walk left 0.4 | run forward-right speed=1.2 | squat height=0.6")


def _print_responses(connection: socket.socket) -> None:
    try:
        while response := connection.recv(4096):
            print(f"\n{response.decode('utf-8')}", flush=True)
    except (ConnectionError, OSError):
        pass


def main() -> None:
    path = command_socket_path()
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        connection.connect(str(path))
    except OSError as exc:
        connection.close()
        raise SystemExit(
            f"Could not connect to {path}. Start `dora run dataflow.yml` first: {exc}"
        ) from exc

    threading.Thread(
        target=_print_responses,
        args=(connection,),
        daemon=True,
    ).start()
    print(f"Connected to {path}")
    _print_help()
    try:
        while True:
            command = input("motion> ").strip()
            normalized = command.lower()
            if normalized in {"quit", "exit"}:
                break
            if normalized in {"help", "?"}:
                _print_help()
                continue
            if command:
                connection.send(command.encode("utf-8"))
    except (EOFError, KeyboardInterrupt, BrokenPipeError, ConnectionError):
        pass
    finally:
        connection.close()


if __name__ == "__main__":
    main()
