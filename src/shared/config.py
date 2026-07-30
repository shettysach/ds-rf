from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MotionGenConfig:
    planner_onnx: Path
    device: str

    @classmethod
    def from_env(cls) -> "MotionGenConfig":
        return cls(
            planner_onnx=Path(os.environ["DS_RF_PLANNER_ONNX"]),
            device=os.environ["DS_RF_DEVICE"],
        )


@dataclass(frozen=True)
class SonicConfig:
    sonic_dir: Path
    device: str

    @classmethod
    def from_env(cls) -> "SonicConfig":
        return cls(
            sonic_dir=Path(os.environ["DS_RF_SONIC_DIR"]),
            device=os.environ["DS_RF_DEVICE"],
        )
