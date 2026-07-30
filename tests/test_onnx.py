from pathlib import Path
from types import SimpleNamespace

from shared import onnx as onnx_utils


class _FakeSession:
    def __init__(self, providers) -> None:
        self.providers = providers

    def get_providers(self) -> list[str]:
        first = self.providers[0]
        return [first[0] if isinstance(first, tuple) else first]


def test_cuda_session_receives_device_and_user_stream(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_session(path, *, sess_options, providers):
        captured["path"] = path
        captured["session_options"] = sess_options
        captured["providers"] = providers
        return _FakeSession(providers)

    monkeypatch.setattr(onnx_utils, "_load_cuda_libraries", lambda: None)
    monkeypatch.setattr(onnx_utils.ort, "InferenceSession", fake_session)

    session = onnx_utils.create_onnx_session(
        Path("model.onnx"),
        device="cuda:2",
        cuda_stream=SimpleNamespace(cuda_stream=12345),
    )

    assert session.get_providers() == ["CUDAExecutionProvider"]
    assert captured["providers"] == [
        (
            "CUDAExecutionProvider",
            {"device_id": "2", "user_compute_stream": "12345"},
        )
    ]
