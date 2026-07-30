# ds-rf

Runs a three-node Dora dataflow for Unitree G1:

1. `input` accepts manual commands over a local Unix socket.
2. `motion-gen` generates a G1 MuJoCo reference motion.
3. `sonic` runs MJLab together with the SONIC encoder and decoder.

The initial motion generator is `planner_sonic.onnx`. The motion boundary is
independent of that backend so Kimodo or ARDY can replace it later.

## DORA

```bash
  cargo install \
    --git https://github.com/dora-rs/dora.git \
    --tag v1.0.0-rc.4 \
    --locked \
    --force \
    dora-cli
```

## CPU setup

```bash
uv sync --extra cpu
```

The defaults use the checkpoints under `/tmp/GEAR-SONIC`. Override them with:

```bash
export DS_RF_SONIC_DIR=/tmp/GEAR-SONIC
export DS_RF_PLANNER_ONNX=/tmp/GEAR-SONIC/planner_sonic.onnx
export DS_RF_DEVICE=cpu
```

Use the low-latency SONIC bundle without changing code:

```bash
export DS_RF_SONIC_DIR=/tmp/GEAR-SONIC/low_latency
```

## Run

```bash
dora run dataflow.yml
```

Then open a second terminal in the repository and start the interactive client:

```bash
uv run ds-rf-command
```

The command vocabulary follows the V2 `planner_sonic.onnx` interface. Run
`help` in the client to list all 27 modes and the available directions. Append
a direction and optionally a speed in meters per second:

```text
walk forward
walk left 0.4
run forward-right speed=1.2
stand
```

Movement and facing directions are independent. Height-aware modes accept
named options:

```text
walk backward 0.5 facing=forward
squat height=0.6
```

The socket defaults to `/tmp/ds-rf-command-<uid>.sock`. Set
`DS_RF_COMMAND_SOCKET` in both terminals to override it.

Set `DS_RF_VIEWER=headless` to run without a native MuJoCo window.

## CUDA 12.8

Install the mutually exclusive GPU extra and select a CUDA device:

```bash
uv sync --extra cu128
export DS_RF_DEVICE=cuda:0
dora run dataflow.yml
```

`DS_RF_DEVICE` selects MJLab, Torch, and ONNX Runtime together. In the SONIC
process, ONNX Runtime uses CUDA I/O binding on MJLab's Warp stream, so robot
state, policy observations, actions, and simulation stay on the device. The
motion generator runs in a separate Dora process and uses its own CUDA stream.

Before starting the full graph, check the shared-stream integration directly:

```bash
WARP_CACHE_PATH=/tmp/ds-rf-warp-cache \
  uv run pytest -q \
  tests/test_integration.py::test_mjlab_and_sonic_share_one_cuda_stream -s
```

The test verifies that MJLab, the SONIC encoder, and the SONIC decoder use the
same non-null stream pointer and complete one control step without moving the
action back to the CPU.

The `cpu` and `cu128` extras must not be installed together. To switch back:

```bash
uv sync --extra cpu
export DS_RF_DEVICE=cpu
```
