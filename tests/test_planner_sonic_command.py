import math

import pytest

from motion_gen.planner_sonic_command import PlannerSonicCommand


def test_locomotion_direction_and_speed() -> None:
    command = PlannerSonicCommand.parse("run forward-right 1.2", command_id="command")

    assert command.command_id == "command"
    assert command.mode == 3
    assert command.movement_direction == pytest.approx(
        (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0)
    )
    assert command.facing_direction == (1.0, 0.0, 0.0)
    assert command.target_vel == 1.2


def test_facing_and_height_options() -> None:
    command = PlannerSonicCommand.parse(
        "squat facing=left height=0.6", command_id="command"
    )

    assert command.mode == 4
    assert command.movement_direction == (0.0, 0.0, 0.0)
    assert command.facing_direction == (0.0, 1.0, 0.0)
    assert command.height == 0.6


@pytest.mark.parametrize(
    ("text", "mode"),
    [("stand", 0), ("slowwalk", 1), ("crawl", 8), ("happy-dance", 23)],
)
def test_modes_and_aliases(text: str, mode: int) -> None:
    assert PlannerSonicCommand.parse(text, command_id="command").mode == mode


@pytest.mark.parametrize(
    "text",
    ["sit", "walk 0", "walk nan", "squat height=-0.1", "walk facing=up"],
)
def test_invalid_commands(text: str) -> None:
    with pytest.raises(ValueError):
        PlannerSonicCommand.parse(text, command_id="command")
