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
    make_spec_fn: Callable[[], SpecFn]
    camera_distance: float | None = None


TASKS: dict[TaskName, TaskDefinition] = {
    "portrait-corridors": TaskDefinition(
        objective="Stand in front of the image of the cartoon.",
        make_spec_fn=make_portrait_corridors_spec_fn,
        camera_distance=2.0,  # Keep the tracking camera inside the back wall at x=-2.
    ),
}


def get_task(name: str) -> tuple[TaskName, TaskDefinition]:
    if name not in TASKS:
        available = ", ".join(TASKS)
        raise ValueError(f"Unknown task {name!r}. Available: {available}") from None
    task_name = name
    return task_name, TASKS[task_name]
