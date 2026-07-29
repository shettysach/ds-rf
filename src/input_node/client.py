from __future__ import annotations

import socket
import threading

from input_node.socket_server import command_socket_path


def _print_responses(connection: socket.socket) -> None:
    try:
        with connection.makefile("r", encoding="utf-8") as responses:
            for response in responses:
                print(f"\n{response.rstrip()}", flush=True)
    except (ConnectionError, OSError):
        pass


def main() -> None:
    path = command_socket_path()
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
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
    print("Commands: stand | slow-walk | walk | run [direction] [speed]")
    try:
        while True:
            command = input("motion> ")
            if command.strip().lower() in {"quit", "exit"}:
                break
            connection.sendall(command.encode("utf-8") + b"\n")
    except (EOFError, KeyboardInterrupt, BrokenPipeError, ConnectionError):
        pass
    finally:
        connection.close()


if __name__ == "__main__":
    main()

