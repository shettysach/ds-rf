from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import onnxruntime as ort

from shared.config import normalize_device, parse_cuda_device_index

if TYPE_CHECKING:
    import torch


def create_onnx_session(
    model_path: Path,
    *,
    device: str,
    cuda_stream: torch.cuda.Stream | None = None,
) -> ort.InferenceSession:
    """Create an ONNX session on the requested device without silent fallback."""

    device = normalize_device(device)
    if device == "cpu":
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    device_id = parse_cuda_device_index(device)
    _load_cuda_libraries()
    options: dict[str, str] = {"device_id": str(device_id)}

    if cuda_stream is not None:
        stream_ptr = int(cuda_stream.cuda_stream)
        if stream_ptr == 0:
            raise RuntimeError("Cannot give ONNX Runtime a null CUDA stream")
        options["user_compute_stream"] = str(stream_ptr)

    session_options = ort.SessionOptions()
    session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=[("CUDAExecutionProvider", options)],
    )
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            f"ONNX Runtime did not activate CUDA: {session.get_providers()}"
        )
    return session


def _load_cuda_libraries() -> None:
    # Importing Torch first makes its CUDA and cuDNN libraries available to ORT.
    import torch  # noqa: F401

    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        preload()
