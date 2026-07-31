from pathlib import Path

import yaml


def test_runtime_environment_is_scoped_to_consuming_nodes() -> None:
    descriptor = yaml.safe_load(Path("dataflow.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert "env" not in nodes["input"]
    assert nodes["motion-gen"]["env"] == {
        "DSRF_DEVICE": "${DSRF_DEVICE:-cpu}",
        "DSRF_PLANNER_ONNX": (
            "${DSRF_PLANNER_ONNX:-/tmp/GEAR-SONIC/planner_sonic.onnx}"
        ),
    }
    assert nodes["sonic"]["env"] == {
        "DSRF_DEVICE": "${DSRF_DEVICE:-cpu}",
        "DSRF_SONIC_DIR": "${DSRF_SONIC_DIR:-/tmp/GEAR-SONIC}",
    }
