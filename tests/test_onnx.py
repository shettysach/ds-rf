from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from shared import onnx as onnx_utils


class _FakeSession:
    def __init__(self, providers) -> None:
        self.providers = providers

    def get_providers(self) -> list[str]:
        first = self.providers[0]
        return [first[0] if isinstance(first, tuple) else first]


@pytest.mark.parametrize(
    ("device", "current_device", "device_id"),
    [
        ("cuda", 1, "1"),
        ("cuda:1", 0, "1"),
    ],
)
def test_cuda_session_receives_device_and_user_stream(
    monkeypatch,
    device: str,
    current_device: int,
    device_id: str,
) -> None:
    captured: dict[str, object] = {}

    def fake_session(path, *, sess_options, providers):
        captured["path"] = path
        captured["session_options"] = sess_options
        captured["providers"] = providers
        return _FakeSession(providers)

    monkeypatch.setattr(onnx_utils.ort, "preload_dlls", lambda: None)
    monkeypatch.setattr(onnx_utils.ort, "InferenceSession", fake_session)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_device)

    session = onnx_utils.create_onnx_session(
        Path("model.onnx"),
        device=torch.device(device),
        cuda_stream=cast(
            torch.cuda.Stream,
            SimpleNamespace(
                cuda_stream=12345,
                device=SimpleNamespace(index=int(device_id)),
            ),
        ),
    )

    assert session.get_providers() == ["CUDAExecutionProvider"]
    assert captured["providers"] == [
        (
            "CUDAExecutionProvider",
            {"device_id": device_id, "user_compute_stream": "12345"},
        ),
        "CPUExecutionProvider",
    ]
