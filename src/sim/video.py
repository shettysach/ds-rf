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
        self._font = ImageFont.load_default(size=20)
        try:
            self._bold_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
            )
        except OSError:
            self._bold_font = self._font
        self.frames = 0
        self._closed = False

    def write_frame(self, rgb: np.ndarray, state: DemoVlmState) -> None:
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").convert(
            "RGBA"
        )
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        reasoning = state.reasoning.strip() or "No reasoning returned."
        reasoning = " ".join(reasoning.split())
        if len(reasoning) > 800:
            reasoning = reasoning[:797].rstrip() + "..."
        reasoning_lines = textwrap.wrap(reasoning, width=36) or [""]
        entries = [
            (f"Observation #{state.observation_id}", self._font),
            ("VLM output", self._bold_font),
            ("Reasoning", self._bold_font),
            *[(line, self._font) for line in reasoning_lines],
            ("Command", self._bold_font),
            (_decision_label(state.command), self._font),
        ]

        line_height = 24
        padding = 16
        panel_width = min(image.width - 24, max(220, int(image.width * 0.32)))
        panel_height = padding * 2 + line_height * len(entries) + 4
        panel_x = image.width - panel_width - 12
        panel_y = image.height - panel_height - 12
        draw.rounded_rectangle(
            (panel_x, panel_y, panel_x + panel_width, panel_y + panel_height),
            radius=7,
            fill=(0, 0, 0, 200),
        )
        y = panel_y + padding
        for line, font in entries:
            draw.text(
                (panel_x + padding, y),
                line,
                fill=(255, 255, 255, 255),
                font=font,
            )
            y += line_height

        image = Image.alpha_composite(image, overlay).convert("RGB")
        self._writer.append_data(np.asarray(image))
        self.frames += 1

    def close(self) -> None:
        if self._closed:
            return
        self._writer.close()
        self._closed = True


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
