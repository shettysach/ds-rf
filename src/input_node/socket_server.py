from __future__ import annotations

import os
import socket
import stat
from pathlib import Path

MAX_COMMAND_BYTES = 4096


def command_socket_path() -> Path:
    configured = os.environ.get("DS_RF_COMMAND_SOCKET")
    if configured:
        return Path(configured)
    return Path(f"/tmp/ds-rf-command-{os.getuid()}.sock")


class CommandServer:
    """Non-blocking, line-oriented command server for the Dora input node."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._listener: socket.socket | None = None
        self._clients: dict[socket.socket, bytearray] = {}

    def start(self) -> None:
        if not self.path.parent.is_dir():
            raise FileNotFoundError(
                f"Command socket directory does not exist: {self.path.parent}"
            )
        try:
            mode = self.path.lstat().st_mode
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISSOCK(mode):
                raise RuntimeError(
                    f"Refusing to replace non-socket command path: {self.path}"
                )
            self.path.unlink()

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.path))
            self.path.chmod(0o600)
            listener.listen()
            listener.setblocking(False)
        except Exception:
            listener.close()
            raise
        self._listener = listener

    def poll(self) -> list[str]:
        if self._listener is None:
            raise RuntimeError("Command server has not been started")

        while True:
            try:
                client, _ = self._listener.accept()
            except BlockingIOError:
                break
            client.setblocking(False)
            self._clients[client] = bytearray()

        commands: list[str] = []
        for client, buffer in list(self._clients.items()):
            while True:
                try:
                    chunk = client.recv(MAX_COMMAND_BYTES)
                except BlockingIOError:
                    break
                except ConnectionError:
                    self._drop(client)
                    break
                if not chunk:
                    self._drop(client)
                    break
                buffer.extend(chunk)
                if len(buffer) > MAX_COMMAND_BYTES:
                    self._send(client, "Invalid command: command is too long")
                    self._drop(client)
                    break

                while b"\n" in buffer:
                    line, _, remainder = buffer.partition(b"\n")
                    buffer[:] = remainder
                    try:
                        commands.append(line.decode("utf-8"))
                    except UnicodeDecodeError:
                        self._send(client, "Invalid command: expected UTF-8 text")
        return commands

    def broadcast(self, message: str) -> None:
        for client in list(self._clients):
            self._send(client, message)

    def close(self) -> None:
        for client in list(self._clients):
            self._drop(client)
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        try:
            is_socket = stat.S_ISSOCK(self.path.lstat().st_mode)
        except FileNotFoundError:
            is_socket = False
        if is_socket:
            self.path.unlink()

    def _send(self, client: socket.socket, message: str) -> None:
        try:
            client.sendall(message.encode("utf-8") + b"\n")
        except (BlockingIOError, ConnectionError):
            self._drop(client)

    def _drop(self, client: socket.socket) -> None:
        self._clients.pop(client, None)
        client.close()
