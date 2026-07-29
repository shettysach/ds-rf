from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from shared.messages import PlannerCommand


@dataclass(frozen=True)
class NativeMotion:
    """A generator's native G1 MuJoCo qpos sequence."""

    qpos: np.ndarray
    fps: int


class MotionGenerator(Protocol):
    def generate(self, command: PlannerCommand) -> NativeMotion:
        """Generate a motion and retain any continuity state for the next call."""
