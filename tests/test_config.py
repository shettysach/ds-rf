from shared.config import RuntimeConfig


def test_cuda_device_is_the_single_runtime_switch(monkeypatch) -> None:
    monkeypatch.setenv("DS_RF_DEVICE", "cuda:0")
    monkeypatch.delenv("DS_RF_ONNX_PROVIDER", raising=False)

    config = RuntimeConfig.from_env()

    assert config.device == "cuda:0"
