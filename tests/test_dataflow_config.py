from pathlib import Path

import yaml


def test_runtime_environment_is_scoped_to_consuming_nodes() -> None:
    descriptor = yaml.safe_load(Path("dataflow.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert nodes["agent"]["env"] == {
        "DSRF_VLM_URL": "${DSRF_VLM_URL:-http://127.0.0.1:8080}",
        "DSRF_VLM_TIMEOUT": "${DSRF_VLM_TIMEOUT:-120}",
        "DSRF_VLM_SYSTEM_PROMPT": ("${DSRF_VLM_SYSTEM_PROMPT:-prompt/SYSTEM.md}"),
        "DSRF_VLM_USER_PROMPT": "${DSRF_VLM_USER_PROMPT:-prompt/USER.md}",
    }
    assert nodes["motion-gen"]["env"] == {
        "DSRF_DEVICE": "${DSRF_DEVICE:-cpu}",
        "DSRF_PLANNER_ONNX": (
            "${DSRF_PLANNER_ONNX:-/tmp/GEAR-SONIC/planner_sonic.onnx}"
        ),
    }
    assert nodes["sonic"]["env"] == {
        "DSRF_DEVICE": "${DSRF_DEVICE:-cpu}",
        "DSRF_SONIC_DIR": "${DSRF_SONIC_DIR:-/tmp/GEAR-SONIC}",
        "DSRF_TASK": "${DSRF_TASK:-portrait-corridors}",
        "DSRF_IMAGE_WIDTH": "${DSRF_IMAGE_WIDTH:-640}",
        "DSRF_IMAGE_HEIGHT": "${DSRF_IMAGE_HEIGHT:-480}",
        "DSRF_JPEG_QUALITY": "${DSRF_JPEG_QUALITY:-85}",
        "DSRF_VIEWER": "${DSRF_VIEWER:-native}",
    }
