from __future__ import annotations

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
    viewer: SonicViewer | None = (
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
        task_name = cfg.task or "none"
        node.log(
            "info",
            f"SONIC initialized: task={task_name!r} device={cfg.device!r} "
            f"viewer={cfg.viewer!r}",
            target="dsrf.sonic",
            fields={
                "event": "sonic_initialized",
                "task": task_name,
                "device": cfg.device,
                "viewer": cfg.viewer,
            },
        )
        SonicRuntime(node, simulation, policy, renderer, viewer).run()
    finally:
        if viewer is not None:
            viewer.close()
        simulation.close()


if __name__ == "__main__":
    main()
