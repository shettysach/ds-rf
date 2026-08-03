from types import SimpleNamespace

import pytest
import torch

from motion_gen.ardy.constraints import build_velocity_constraints
from motion_gen.ardy.parser import WALK_SPEED_M_S, parse_motion_command


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("forward", (0.0, WALK_SPEED_M_S)),
        ("backward", (0.0, -WALK_SPEED_M_S)),
        ("left", (WALK_SPEED_M_S, 0.0)),
        ("right", (-WALK_SPEED_M_S, 0.0)),
    ],
)
def test_walk_direction_becomes_target_velocity(direction, expected) -> None:
    command = parse_motion_command(
        f'{{"motion":"walk","direction":"{direction}"}}'
    )

    assert command.motion == "walk"
    assert command.target_velocity == expected


def test_stand_has_zero_target_velocity() -> None:
    command = parse_motion_command('{"motion":"stand","direction":"left"}')

    assert command.motion == "stand"
    assert command.target_velocity == (0.0, 0.0)


@pytest.mark.parametrize(
    "text",
    [
        "walk left",
        "[]",
        '{"motion":"walk"}',
        '{"motion":"run","direction":"forward"}',
        '{"motion":"walk","direction":"up"}',
    ],
)
def test_invalid_ardy_commands(text: str) -> None:
    with pytest.raises(ValueError):
        parse_motion_command(text)


def test_target_velocity_is_integrated_into_sparse_root_constraints() -> None:
    received: dict[str, object] = {}

    def create_conditions(index, data, length, normalize, device):
        received.update(
            index=index,
            data=data,
            length=length,
            normalize=normalize,
            device=device,
        )
        return torch.zeros((129, 8)), torch.zeros((129, 8))

    motion_rep = SimpleNamespace(create_conditions=create_conditions)
    root_history = torch.zeros((2, 3))

    motion_mask, observed_motion = build_velocity_constraints(
        motion_rep,
        root_history,
        (0.0, 0.5),
        generated_frames=125,
        history_frames=4,
        fps=25,
        device=torch.device("cpu"),
    )

    assert motion_mask.shape == (1, 129, 8)
    assert observed_motion.shape == (1, 129, 8)
    assert received["length"] == 129
    assert received["normalize"] is True
    indices = received["index"]["root_2d"][0]
    root_2d = received["data"]["root_2d"][0]
    assert indices.tolist() == list(range(13, 124, 10))
    assert bool((root_2d[:, 1] > 0.0).all())
    assert bool((root_2d[1:, 1] > root_2d[:-1, 1]).all())
