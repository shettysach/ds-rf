import math

import pytest

from motion_gen.planner_sonic import PlannerMode, parse_motion_command


def test_locomotion_direction() -> None:
    command = parse_motion_command('{"motion":"walk","direction":"forward-right"}')

    assert command.mode is PlannerMode.WALK
    assert command.movement_direction == pytest.approx(
        (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0)
    )
    assert command.facing_direction == (1.0, 0.0, 0.0)
    assert command.target_vel == -1.0


def test_stationary_motion_ignores_direction() -> None:
    command = parse_motion_command('{"motion":"stand","direction":"left"}')

    assert command.mode is PlannerMode.IDLE
    assert command.movement_direction == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("text", "mode"),
    [
        ('{"motion":"stand","direction":"forward"}', 0),
        ('{"motion":"slowwalk","direction":"forward"}', 1),
        ('{"motion":"crawl","direction":"forward"}', 8),
        ('{"motion":"happy-dance","direction":"forward"}', 23),
    ],
)
def test_modes_and_aliases(text: str, mode: int) -> None:
    assert parse_motion_command(text).mode == mode


@pytest.mark.parametrize(
    "text",
    [
        "walk left",
        "[]",
        '{"motion":"walk"}',
        '{"motion":"walk","direction":"left","speed":1}',
        '{"motion":1,"direction":"left"}',
        '{"motion":"sit","direction":"left"}',
        '{"motion":"walk","direction":"up"}',
    ],
)
def test_invalid_commands(text: str) -> None:
    with pytest.raises(ValueError):
        parse_motion_command(text)
