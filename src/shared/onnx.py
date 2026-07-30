from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import onnxruntime as ort

from shared.config import normalize_device, parse_cuda_device_index

if TYPE_CHECKING:
    import torch


def validate_onnx_device(device: str) -> None:
    device = normalize_device(device)
    if device == "cpu":
        return

    import torch

    device_id = parse_cuda_device_index(device)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but Torch cannot access an NVIDIA driver/device"
        )
    if device_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {device_id} does not exist; found {torch.cuda.device_count()}"
        )
    _load_cuda_libraries()
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable. Install the cu128 extra with "
            "`uv sync --extra cu128` and do not install the cpu extra alongside it. "
            f"Available providers: {available}"
        )


def create_onnx_session(
    model_path: Path,
    *,
    device: str,
    cuda_stream: torch.cuda.Stream | None = None,
    require_full_device: bool = False,
) -> ort.InferenceSession:
    """Create an ONNX session on the requested device without silent fallback."""

    device = normalize_device(device)
    if device == "cpu":
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    device_id = parse_cuda_device_index(device)
    validate_onnx_device(device)
    options: dict[str, str] = {"device_id": str(device_id)}
    if cuda_stream is not None:
        stream_ptr = int(cuda_stream.cuda_stream)
        if stream_ptr == 0:
            raise RuntimeError("Cannot give ONNX Runtime a null CUDA stream")
        options["user_compute_stream"] = str(stream_ptr)

    session_options = ort.SessionOptions()
    if require_full_device:
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
