import pytest

from shared.config import RuntimeConfig, normalize_device, parse_cuda_device_index


def test_cuda_device_is_the_single_runtime_switch(monkeypatch) -> None:
    monkeypatch.setenv("DS_RF_DEVICE", "cuda")
    monkeypatch.delenv("DS_RF_ONNX_PROVIDER", raising=False)

    config = RuntimeConfig.from_env()

    assert config.device == "cuda:0"
    assert parse_cuda_device_index(config.device) == 0


def test_device_validation() -> None:
    assert normalize_device("CPU") == "cpu"
    assert parse_cuda_device_index("cuda:3") == 3
    with pytest.raises(ValueError, match="expected 'cpu' or 'cuda:<index>'"):
        normalize_device("gpu")
