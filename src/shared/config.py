from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    sonic_dir: Path
    planner_onnx: Path
    device: str
    viewer: str

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        sonic_dir = Path(os.environ.get("DS_RF_SONIC_DIR", "/tmp/GEAR-SONIC"))
        planner_onnx = Path(
            os.environ.get(
                "DS_RF_PLANNER_ONNX",
                "/tmp/GEAR-SONIC/planner_sonic.onnx",
            )
        )
        return cls(
            sonic_dir=sonic_dir,
            planner_onnx=planner_onnx,
            device=os.environ.get("DS_RF_DEVICE", "cpu"),
            viewer=os.environ.get("DS_RF_VIEWER", "native"),
        )
