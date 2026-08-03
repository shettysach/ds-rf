from pathlib import Path

import yaml


def test_runtime_environment_is_scoped_to_consuming_nodes() -> None:
    descriptor = yaml.safe_load(Path("demo.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert nodes["agent"]["env"] == {
        "DSRF_VLM_URL": "http://127.0.0.1:8080",
        "DSRF_VLM_TIMEOUT": "120",
        "DSRF_VLM_SYSTEM_PROMPT": "TASK.md",
        "DSRF_VLM_USER_PROMPT": "prompt/USER.md",
    }
    assert nodes["motion-gen"]["env"] == {
        "DSRF_DEVICE": "cuda",
        "DSRF_MOTION_GENERATOR": ("${DSRF_MOTION_GENERATOR:-planner_sonic}"),
        "DSRF_PLANNER_ONNX": ("/tmp/GEAR-SONIC/planner_sonic.onnx"),
    }
    assert nodes["sonic"]["env"] == {
        "DSRF_DEVICE": "cuda",
        "DSRF_SONIC_DIR": "/tmp/GEAR-SONIC",
        "DSRF_TASK": "portrait-corridors",
        "DSRF_IMAGE_WIDTH": "640",
        "DSRF_IMAGE_HEIGHT": "480",
        "DSRF_JPEG_QUALITY": "85",
        "DSRF_VIEWER": "native",
        "DSRF_REFERENCE_GHOST": "${DSRF_REFERENCE_GHOST:-false}",
    }


def test_ardy_dataflow_wires_encoder_between_agent_and_motion_gen() -> None:
    descriptor = yaml.safe_load(Path("ardy.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert set(nodes) == {"agent", "text-encoder", "motion-gen", "sonic"}
    assert nodes["text-encoder"]["inputs"] == {"command": "agent/command"}
    assert nodes["text-encoder"]["outputs"] == ["encoded_command", "error"]
    assert nodes["text-encoder"]["env"]["DSRF_TEXT_ENCODER_MODEL"] == (
        "${DSRF_TEXT_ENCODER_MODEL:-/tmp/model}"
    )
    assert nodes["agent"]["env"]["DSRF_VLM_USER_PROMPT"] == "prompt/USER.md"
    assert nodes["motion-gen"]["inputs"] == {
        "encoded_command": "text-encoder/encoded_command"
    }
    assert nodes["motion-gen"]["env"]["DSRF_MOTION_GENERATOR"] == "ardy"
    assert nodes["sonic"]["inputs"] == {"motion": "motion-gen/motion"}
    assert nodes["agent"]["inputs"]["observation"] == "sonic/observation"
    assert nodes["agent"]["inputs"]["encoding_error"] == "text-encoder/error"
