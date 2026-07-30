import math

import pytest

from motion_gen.planner_sonic import PlannerMode
from nodes.motion_gen_command import parse_motion_command


def test_locomotion_direction_and_speed() -> None:
    command = parse_motion_command("run forward-right 1.2")

    assert command.mode is PlannerMode.RUN
    assert command.movement_direction == pytest.approx(
        (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0)
    )
    assert command.facing_direction == (1.0, 0.0, 0.0)
    assert command.target_vel == 1.2


def test_facing_and_height_options() -> None:
    command = parse_motion_command("squat facing=left height=0.6")

    assert command.mode is PlannerMode.SQUAT
    assert command.movement_direction == (0.0, 0.0, 0.0)
    assert command.facing_direction == (0.0, 1.0, 0.0)
    assert command.height == 0.6


@pytest.mark.parametrize(
    ("text", "mode"),
    [("stand", 0), ("slowwalk", 1), ("crawl", 8), ("happy-dance", 23)],
)
def test_modes_and_aliases(text: str, mode: int) -> None:
    assert parse_motion_command(text).mode == mode


@pytest.mark.parametrize(
    "text",
    ["sit", "walk 0", "walk nan", "squat height=-0.1", "walk facing=up"],
)
def test_invalid_commands(text: str) -> None:
    with pytest.raises(ValueError):
        parse_motion_command(text)
