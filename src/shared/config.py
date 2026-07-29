from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

OnnxProvider = Literal["cpu", "cuda"]
ViewerMode = Literal["native", "headless"]


@dataclass(frozen=True)
class RuntimeConfig:
    sonic_dir: Path
    planner_onnx: Path
    device: str
    onnx_provider: OnnxProvider
    viewer: ViewerMode

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        sonic_dir = Path(os.environ.get("DS_RF_SONIC_DIR", "/tmp/GEAR-SONIC"))
        planner_onnx = Path(
            os.environ.get(
                "DS_RF_PLANNER_ONNX",
                "/tmp/GEAR-SONIC/planner_sonic.onnx",
            )
        )
        provider = os.environ.get("DS_RF_ONNX_PROVIDER", "cpu").lower()
        viewer = os.environ.get("DS_RF_VIEWER", "native").lower()
        if provider not in ("cpu", "cuda"):
            raise ValueError(f"Unsupported ONNX provider: {provider}")
        if viewer not in ("native", "headless"):
            raise ValueError(f"Unsupported viewer mode: {viewer}")
        return cls(
            sonic_dir=sonic_dir,
            planner_onnx=planner_onnx,
            device=os.environ.get("DS_RF_DEVICE", "cpu"),
            onnx_provider=provider,
            viewer=viewer,
        )

    def validate_motion_gen(self) -> None:
        if not self.planner_onnx.is_file():
            raise FileNotFoundError(
                f"Planner checkpoint not found: {self.planner_onnx}"
            )

    def validate_sonic(self) -> None:
        required = (
            self.sonic_dir / "model_encoder.onnx",
            self.sonic_dir / "model_decoder.onnx",
            self.sonic_dir / "observation_config.yaml",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing SONIC bundle files: " + ", ".join(missing))
