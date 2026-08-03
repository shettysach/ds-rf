from __future__ import annotations

import numpy as np
import torch

from shared.messages import ARDY_EMBEDDING_SIZE


def prepare_conditioning(
    embedding: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    array = np.asarray(embedding, dtype=np.float32)
    if array.shape != (ARDY_EMBEDDING_SIZE,):
        raise ValueError(
            f"ARDY embedding must have shape [{ARDY_EMBEDDING_SIZE}], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("ARDY embedding contains NaN or infinite values")

    text_feat = torch.as_tensor(array, device=device).reshape(1, 1, ARDY_EMBEDDING_SIZE)
    text_pad_mask = torch.ones((1, 1), device=device, dtype=torch.bool)
    return text_feat, text_pad_mask
