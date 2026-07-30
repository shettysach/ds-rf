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
    """Non-blocking packet server for local interactive commands."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._client: socket.socket | None = None
        self._listener = self._listen()

    def poll(self) -> list[str]:
        while True:
            try:
                client, _ = self._listener.accept()
            except BlockingIOError:
                break
            client.setblocking(False)
            self._drop_client()
            self._client = client

        commands: list[str] = []
        if self._client is not None:
            while True:
                try:
                    payload = self._client.recv(MAX_COMMAND_BYTES + 1)
                except BlockingIOError:
                    break
                except ConnectionError:
                    self._drop_client()
                    break
                if not payload:
                    self._drop_client()
                    break
                if len(payload) > MAX_COMMAND_BYTES:
                    self._send("Invalid command: command is too long")
                    continue
                try:
                    commands.append(payload.decode("utf-8"))
                except UnicodeDecodeError:
                    self._send("Invalid command: expected UTF-8 text")
        return commands

    def broadcast(self, message: str) -> None:
        self._send(message)

    def close(self) -> None:
        self._drop_client()
        self._listener.close()
        if _is_socket(self.path):
            self.path.unlink()

    def _listen(self) -> socket.socket:
        if self.path.exists() or self.path.is_symlink():
            if not _is_socket(self.path):
                raise RuntimeError(
                    f"Refusing to replace non-socket command path: {self.path}"
                )
            self.path.unlink()

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        listener.bind(str(self.path))
        self.path.chmod(0o600)
        listener.listen()
        listener.setblocking(False)
        return listener

    def _send(self, message: str) -> None:
        if self._client is None:
            return
        try:
            self._client.send(message.encode("utf-8"))
        except (BlockingIOError, ConnectionError):
            self._drop_client()

    def _drop_client(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def _is_socket(path: Path) -> bool:
    try:
        return stat.S_ISSOCK(path.lstat().st_mode)
    except FileNotFoundError:
        return False
