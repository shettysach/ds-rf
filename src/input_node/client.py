from __future__ import annotations

import socket
import threading

from input_node.socket_server import command_socket_path
from nodes.motion_gen_command import (
    MOTION_COMMAND_EXAMPLES,
    MOTION_COMMAND_USAGE,
    MOTION_DIRECTIONS,
    MOTION_MODES,
)


def _print_help() -> None:
    print(MOTION_COMMAND_USAGE)
    print("Modes: " + " | ".join(MOTION_MODES))
    print("Directions: " + " | ".join(MOTION_DIRECTIONS))
    print(MOTION_COMMAND_EXAMPLES)


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
