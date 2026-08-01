from __future__ import annotations

from typing import Optional

from dora import Node

from shared.config import SonicConfig
from sonic.mjlab_env import SonicMjlabEnv
from sonic.policy import SonicPolicy
from sonic.reference_ghost import SonicReferenceGhost
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
    viewer: Optional[SonicViewer] = None

    try:
        with simulation.compute_context():
            policy = SonicPolicy(
                cfg.sonic_dir,
                device=cfg.device,
                cuda_stream=simulation.cuda_stream,
            )
        if cfg.viewer == "native":
            if cfg.reference_ghost:
                ghost = SonicReferenceGhost(simulation.unwrapped, policy.reference)
                simulation.add_debug_visualizer(ghost.draw)
            viewer = NativeSonicViewer(simulation)
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
            "reference_ghost": str(cfg.reference_ghost).lower(),
        },
    )


if __name__ == "__main__":
    main()
