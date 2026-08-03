from types import SimpleNamespace

import torch

from motion_gen.ardy.constraints import build_waypoint_constraints


def _conditions():
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

    return SimpleNamespace(create_conditions=create_conditions), received


def test_local_waypoint_becomes_ardy_root_endpoint() -> None:
    motion_rep, received = _conditions()
    root_history = torch.tensor([[1.0, 0.0, 2.0], [1.0, 0.0, 2.0]])

    motion_mask, observed_motion = build_waypoint_constraints(
        motion_rep,
        root_history,
        (0.8, 0.3),
        generated_frames=125,
        history_frames=4,
        device=torch.device("cpu"),
    )

    assert motion_mask.shape == (1, 129, 8)
    assert observed_motion.shape == (1, 129, 8)
    assert received["index"]["root_2d"][0].tolist() == [128]
    torch.testing.assert_close(
        received["data"]["root_2d"][0],
        torch.tensor([[1.3, 2.8]]),
    )


def test_stand_holds_root_position() -> None:
    motion_rep, received = _conditions()
    root_history = torch.tensor([[1.0, 0.0, 2.0], [1.0, 0.0, 2.0]])

    build_waypoint_constraints(
        motion_rep,
        root_history,
        None,
        generated_frames=125,
        history_frames=4,
        device=torch.device("cpu"),
    )

    assert received["index"]["root_2d"][0].tolist() == [
        *range(13, 124, 10),
        128,
    ]
    assert received["data"]["root_2d"][0].tolist() == [[1.0, 2.0]] * 13
