from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NativeMotion:
    """A generator's native G1 MuJoCo qpos sequence."""

    qpos: np.ndarray
    fps: int
