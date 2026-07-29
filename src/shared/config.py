from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ViewerMode = Literal["native", "headless"]
_DEVICE_PATTERN = re.compile(r"cuda:(\d+)$")


def normalize_device(device: str) -> str:
    device = device.lower()
    if device == "cuda":
        device = "cuda:0"
    if device != "cpu" and _DEVICE_PATTERN.fullmatch(device) is None:
        raise ValueError(
            f"Unsupported device {device!r}; expected 'cpu' or 'cuda:<index>'"
        )
    return device


def parse_cuda_device_index(device: str) -> int:
    match = _DEVICE_PATTERN.fullmatch(normalize_device(device))
    if match is None:
        raise ValueError(f"Device {device!r} is not a CUDA device")
    return int(match.group(1))


@dataclass(frozen=True)
class RuntimeConfig:
    sonic_dir: Path
    planner_onnx: Path
    device: str
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
        viewer = os.environ.get("DS_RF_VIEWER", "native").lower()
        if viewer not in ("native", "headless"):
            raise ValueError(f"Unsupported viewer mode: {viewer}")
        device = normalize_device(os.environ.get("DS_RF_DEVICE", "cpu"))
        return cls(
            sonic_dir=sonic_dir,
            planner_onnx=planner_onnx,
            device=device,
            viewer=viewer,
        )

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda:")

    @property
    def cuda_device_index(self) -> int:
        return parse_cuda_device_index(self.device)

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
