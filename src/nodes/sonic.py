from __future__ import annotations

from dora import Node

from shared.config import SonicConfig
from sonic.mjlab_env import SonicMjlabEnv
from sonic.policy import SonicPolicy
from sonic.renderer import SonicRenderer
from sonic.runtime import SonicRuntime


def main() -> None:
    cfg = SonicConfig.from_env()

    node = Node()
    simulation = SonicMjlabEnv(
        device=cfg.device,
        task=cfg.task,
        image_width=cfg.image_width,
        image_height=cfg.image_height,
        show_viewer=True,
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
            f"SONIC initialized: task={task_name!r} device={cfg.device!r}",
            target="dsrf.sonic",
            fields={
                "event": "sonic_initialized",
                "task": task_name,
                "device": cfg.device,
            },
        )
        SonicRuntime(node, simulation, policy, renderer).run()
    finally:
        simulation.close()


if __name__ == "__main__":
    main()
