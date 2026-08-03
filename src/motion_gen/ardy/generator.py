from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ardy.exports.mujoco import MujocoQposConverter
from ardy.model import load_model
from ardy.motion_rep.tools import length_to_mask

from motion_gen.ardy.encoder import load_conditioning
from motion_gen.ardy.history import build_initial_history
from shared.g1 import standing_qpos


class ArdyGenerator:
    """Fixed-conditioning ARDY smoke-test backend for Unitree G1."""

    fps = 25
    smoke_duration_s = 5

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
        self.history_frames = int(self.model.num_frames_per_token)
        standing_history = np.repeat(
            standing_qpos()[None], self.history_frames, axis=0
        )
        self.initial_history = build_initial_history(
            standing_history,
            self.converter,
            self.model.motion_rep,
            device=self.device,
        )

    def generate(self, text: str) -> np.ndarray:
        del text  # The smoke test always uses the fixed ENCODING tensor.
        generated_frames = self.fps * self.smoke_duration_s
        num_frames = generated_frames + self.history_frames
        lengths = torch.tensor([num_frames], device=self.device)

        with torch.inference_mode():
            motion = self.model(
                num_frames,
                num_denoising_steps=int(self.model.diffusion.num_base_steps),
                pad_mask=length_to_mask(lengths),
                first_heading_angle=None,
                motion_mask=None,
                observed_motion=None,
                text_feat=self.text_feat,
                text_pad_mask=self.text_pad_mask,
                cfg_weight=(2.0, 2.0),
                progress_bar=lambda iterable: iterable,
                init_history_sequence=self.initial_history,
            )
            generated_motion = motion[:, self.history_frames :]
            if generated_motion.shape[1] != generated_frames:
                raise ValueError(
                    f"ARDY generated {generated_motion.shape[1]} frames; "
                    f"expected {generated_frames}"
                )
            decoded = self.model.motion_rep.inverse(
                generated_motion,
                is_normalized=True,
            )
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
