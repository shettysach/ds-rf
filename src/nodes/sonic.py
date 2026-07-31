from __future__ import annotations

from typing import Optional

from dora import Node

from shared.config import SonicConfig
from sonic.mjlab_env import SonicMjlabEnv
from sonic.policy import SonicPolicy
from sonic.renderer import SonicRenderer
from sonic.runtime import SonicRuntime
from sonic.viewer import NativeSonicViewer, SonicViewer


def main() -> None:
    cfg = SonicConfig.from_env()

    node = Node()
    simulation = SonicMjlabEnv(
        device=cfg.device,
        task=cfg.task,
        image_width=cfg.image_width,
        image_height=cfg.image_height,
    )
    viewer: Optional[SonicViewer] = (
        NativeSonicViewer(simulation) if cfg.viewer == "native" else None
    )

    try:
        with simulation.compute_context():
            policy = SonicPolicy(
                cfg.sonic_dir,
                device=cfg.device,
                cuda_stream=simulation.cuda_stream,
            )
        renderer = SonicRenderer(simulation, jpeg_quality=cfg.jpeg_quality)
        _log_init(node, cfg)
        SonicRuntime(node, simulation, policy, renderer, viewer).run()
    finally:
        if viewer is not None:
            viewer.close()
        simulation.close()


def _log_init(node: Node, cfg: SonicConfig) -> None:
    node.log(
        "info",
        "SONIC initialized",
        target="dsrf.sonic",
        fields={
            "event": "sonic_initialized",
            "task": cfg.task or "none",
            "device": cfg.device,
            "viewer": cfg.viewer,
        },
    )


if __name__ == "__main__":
    main()
