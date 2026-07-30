from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import onnxruntime as ort

if TYPE_CHECKING:
    import torch


def create_onnx_session(
    model_path: Path,
    *,
    device: str,
    cuda_stream: torch.cuda.Stream | None = None,
) -> ort.InferenceSession:

    if device == "cpu":
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    _load_cuda_libraries()

    options = {"device_id": "0"}
    if cuda_stream is not None:
        options["user_compute_stream"] = str(cuda_stream.cuda_stream)

    sess_options = ort.SessionOptions()
    sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        model_path,
        sess_options,
        providers=[("CUDAExecutionProvider", options)],
    )
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            f"ONNX Runtime did not activate CUDA: {session.get_providers()}"
        )
    return session


# Importing Torch first makes its CUDA and cuDNN libraries available to ORT.
def _load_cuda_libraries() -> None:
    import torch  # noqa: F401

    ort.preload_dlls()
