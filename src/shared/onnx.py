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
    import torch

    torch_device = torch.device(device)

    if torch_device.type == "cpu":
        return ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

    ort.preload_dlls()

    device_id = (
        torch.cuda.current_device()
        if torch_device.index is None
        else torch_device.index
    )
    provider_options = {
        "device_id": str(device_id),
    }

    if cuda_stream is not None:
        stream_device_id = cuda_stream.device.index
        assert stream_device_id == device_id
        provider_options["user_compute_stream"] = str(cuda_stream.cuda_stream)

    return ort.InferenceSession(
        model_path,
        sess_options=ort.SessionOptions(),
        providers=[
            ("CUDAExecutionProvider", provider_options),
            # Some operations in the SONIC graphs currently fall back to CPU.
            "CPUExecutionProvider",
        ],
    )
