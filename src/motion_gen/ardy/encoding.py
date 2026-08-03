from __future__ import annotations

from pathlib import Path

import torch


def load_conditioning(
    path: Path,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoding = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(encoding, torch.Tensor):
        raise TypeError(f"ENCODING must contain a tensor, got {type(encoding).__name__}")
    if encoding.shape != (1, 4096):
        raise ValueError(f"ENCODING must have shape [1, 4096], got {encoding.shape}")
    if not encoding.is_floating_point():
        raise ValueError(f"ENCODING must be floating-point, got {encoding.dtype}")
    if not bool(torch.isfinite(encoding).all()):
        raise ValueError("ENCODING contains NaN or infinite values")

    text_feat = encoding.to(device=device, dtype=torch.float32).unsqueeze(1)
    text_pad_mask = torch.ones((1, 1), device=device, dtype=torch.bool)
    return text_feat, text_pad_mask
