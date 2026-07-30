from __future__ import annotations

import socket
from pathlib import Path

import pytest

from input_node.socket_server import CommandServer


def _poll_until_commands(server: CommandServer) -> list[str]:
    for _ in range(100):
        commands = server.poll()
        if commands:
            return commands
    raise AssertionError("Command server did not receive a command")


def test_command_server_receives_lines_and_broadcasts(tmp_path: Path) -> None:
    path = tmp_path / "command.sock"
    server = CommandServer(path)
    client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        client.connect(str(path))
        server.poll()
        client.send(b"walk forward")
        client.send(b"run left 1.2")

        assert _poll_until_commands(server) == ["walk forward", "run left 1.2"]

        server.broadcast("[motion-gen] generating")
        assert client.recv(1024) == b"[motion-gen] generating"
    finally:
        client.close()
        server.close()

    assert not path.exists()


def test_command_server_refuses_to_replace_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "command.sock"
    path.write_text("keep me")
    with pytest.raises(RuntimeError, match="Refusing to replace non-socket"):
        CommandServer(path)

    assert path.read_text() == "keep me"
