from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ardy.exports.mujoco import MujocoQposConverter
from ardy.model import load_model
from ardy.motion_rep.tools import length_to_mask

from motion_gen.ardy.encoding import load_conditioning


class ArdyGenerator:
    """Fixed-conditioning ARDY smoke-test backend for Unitree G1."""

    fps = 25

    def __init__(
        self,
        checkpoints_dir: Path,
        encoding_path: Path,
        *,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = load_model(
            "g1",
            device=str(self.device),
            checkpoints_dir=str(checkpoints_dir),
        )
        model_fps = float(self.model.motion_rep.fps)
        if model_fps != self.fps:
            raise ValueError(f"Expected ARDY G1 at {self.fps} FPS, got {model_fps}")

        self.text_feat, self.text_pad_mask = load_conditioning(
            encoding_path,
            device=self.device,
        )
        self.converter = MujocoQposConverter(self.model.skeleton)

    def generate(self, text: str) -> np.ndarray:
        del text  # The smoke test always uses the fixed ENCODING tensor.
        num_frames = int(self.model.gen_horizon_len)
        lengths = torch.tensor([num_frames], device=self.device)

        with torch.inference_mode():
            motion = self.model(
                num_frames,
                num_denoising_steps=int(self.model.diffusion.num_base_steps),
                pad_mask=length_to_mask(lengths),
                first_heading_angle=torch.zeros(1, device=self.device),
                motion_mask=None,
                observed_motion=None,
                text_feat=self.text_feat,
                text_pad_mask=self.text_pad_mask,
                cfg_weight=(2.0, 2.0),
                progress_bar=lambda iterable: iterable,
            )
            decoded = self.model.motion_rep.inverse(motion, is_normalized=True)
            batched_qpos = self.converter.dict_to_qpos(
                decoded,
                str(self.device),
            )

        qpos = np.ascontiguousarray(batched_qpos[0], dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != 36:
            raise ValueError(f"ARDY qpos must have shape [T, 36], got {qpos.shape}")
        if qpos.shape[0] == 0:
            raise ValueError("ARDY generated no motion frames")
        if not np.isfinite(qpos).all():
            raise ValueError("ARDY qpos contains NaN or infinite values")
        return qpos
