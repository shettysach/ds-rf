from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from shared.messages import SONIC_FPS


@dataclass(frozen=True)
class DemoVlmState:
    observation_id: int = -1
    reasoning: str = ""
    command: str = ""


class DemoVideoRecorder:
    """Writes simulation-timed RGB frames with the active VLM decision burned in."""

    def __init__(self, path: Path, *, fps: int = SONIC_FPS) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._writer = iio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            macro_block_size=1,
        )
        self._font = ImageFont.load_default()
        self.frames = 0

    def write_frame(self, rgb: np.ndarray, state: DemoVlmState) -> None:
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").convert(
            "RGBA"
        )
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        reasoning = state.reasoning.strip() or "No reasoning returned."
        lines = ["VLM", "", "Reasoning", ""]
        lines.extend(textwrap.wrap(reasoning, width=58) or [""])
        lines.extend(["", "Decision", "", _decision_label(state.command)])
        lines.extend(["", f"Observation #{state.observation_id}"])

        line_height = 16
        padding = 18
        panel_width = min(image.width - 24, max(300, int(image.width * 0.45)))
        panel_height = padding * 2 + line_height * len(lines)
        draw.rounded_rectangle(
            (12, 12, 12 + panel_width, 12 + panel_height),
            radius=10,
            fill=(0, 0, 0, 200),
        )
        draw.multiline_text(
            (12 + padding, 12 + padding),
            "\n".join(lines),
            fill=(255, 255, 255, 255),
            font=self._font,
            spacing=2,
        )

        image = Image.alpha_composite(image, overlay).convert("RGB")
        self._writer.append_data(np.asarray(image))
        self.frames += 1

    def close(self) -> None:
        self._writer.close()


def _decision_label(command: str) -> str:
    try:
        payload = json.loads(command)
    except (json.JSONDecodeError, TypeError):
        return command or "WAIT"
    if not isinstance(payload, dict):
        return command or "WAIT"
    motion = payload.get("motion")
    direction = payload.get("direction")
    if isinstance(motion, str) and isinstance(direction, str):
        return f"{motion} {direction}".upper()
    if isinstance(motion, str):
        return motion.upper()
    return command or "WAIT"
