from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ScoutConfig:
    device: str = "cuda:0"
    image_width: int = 640
    image_height: int = 480
    preview_seconds: float = 3.0

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        if self.preview_seconds < 0:
            raise ValueError("Preview duration must be non-negative")
