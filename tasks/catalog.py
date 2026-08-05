from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from tasks.portrait_corridors import make_portrait_corridors_spec_fn

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]

type TaskName = Literal["portrait-corridors"]
type SpecFn = Callable[["MjSpec"], None]


@dataclass(frozen=True)
class TaskDefinition:
    objective: str
    make_spec_fn: Callable[..., SpecFn]
    camera_distance: float | None = None
    camera_elevation: float | None = None


TASKS: dict[TaskName, TaskDefinition] = {
    "portrait-corridors": TaskDefinition(
        objective="Stand in front of the image of the creator of Linux.",
        make_spec_fn=make_portrait_corridors_spec_fn,
        camera_distance=5.0,
        camera_elevation=-20.0,
    ),
}


def get_task(name: str) -> TaskDefinition:
    if name not in TASKS:
        available = ", ".join(TASKS)
        raise ValueError(f"Unknown task {name!r}. Available: {available}") from None
    return TASKS[name]
