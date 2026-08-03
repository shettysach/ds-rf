from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from motion_gen.ardy.encoding import load_conditioning


def test_load_ardy_conditioning(tmp_path: Path) -> None:
    path = tmp_path / "walk_forward.pt"
    torch.save(torch.ones((1, 4096)), path)

    text_feat, text_pad_mask = load_conditioning(
        path,
        device=torch.device("cpu"),
    )

    assert text_feat.shape == (1, 1, 4096)
    assert text_feat.dtype is torch.float32
    assert text_pad_mask.shape == (1, 1)
    assert text_pad_mask.dtype is torch.bool
    assert bool(text_pad_mask.all())


def test_ardy_model_loader_receives_a_device_string(monkeypatch, tmp_path: Path) -> None:
    import motion_gen.ardy.generator as ardy_generator

    received: dict[str, object] = {}
    model = SimpleNamespace(
        motion_rep=SimpleNamespace(fps=25),
        skeleton=object(),
    )
    monkeypatch.setattr(
        ardy_generator,
        "load_model",
        lambda *args, **kwargs: received.update(kwargs) or model,
    )
    monkeypatch.setattr(ardy_generator, "MujocoQposConverter", lambda _: object())
    monkeypatch.setattr(
        ardy_generator,
        "load_conditioning",
        lambda *args, **kwargs: (torch.zeros((1, 1, 4096)), torch.ones((1, 1))),
    )

    ardy_generator.ArdyGenerator(tmp_path, tmp_path / "encoding.pt", device="cuda:0")

    assert received["device"] == "cuda:0"


@pytest.mark.parametrize("shape", [(4096,), (1, 1, 4096), (2, 4096)])
def test_load_ardy_conditioning_rejects_wrong_shape(
    tmp_path: Path,
    shape: tuple[int, ...],
) -> None:
    path = tmp_path / "encoding.pt"
    torch.save(torch.ones(shape), path)

    with pytest.raises(ValueError, match=r"\[1, 4096\]"):
        load_conditioning(path, device=torch.device("cpu"))
