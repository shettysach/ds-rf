from __future__ import annotations

from io import BytesIO

import imageio.v3 as iio

from sonic.mjlab_env import SonicMjlabEnv


class SonicRenderer:
    def __init__(self, simulation: SonicMjlabEnv, *, jpeg_quality: int) -> None:
        self.simulation = simulation
        self.jpeg_quality = jpeg_quality

    def capture_jpeg(self) -> bytes:
        image = self.simulation.render()
        buffer = BytesIO()
        iio.imwrite(
            buffer,
            image,
            extension=".jpg",
            quality=self.jpeg_quality,
        )
        return buffer.getvalue()
