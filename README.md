# ds-rf

Runs a three-node Dora dataflow for Unitree G1:

1. `input` parses manual terminal commands.
2. `motion-gen` generates a G1 MuJoCo reference motion.
3. `sonic` runs MJLab together with the SONIC encoder and decoder.

The initial motion generator is `planner_sonic.onnx`. The motion boundary is
independent of that backend so Kimodo or ARDY can replace it later.

## CPU setup

```bash
uv sync --extra cpu
```

The defaults use the checkpoints under `/tmp/GEAR-SONIC`. Override them with:

```bash
export DS_RF_SONIC_DIR=/tmp/GEAR-SONIC
export DS_RF_PLANNER_ONNX=/tmp/GEAR-SONIC/planner_sonic.onnx
export DS_RF_DEVICE=cpu
export DS_RF_ONNX_PROVIDER=cpu
```

Use the low-latency SONIC bundle without changing code:

```bash
export DS_RF_SONIC_DIR=/tmp/GEAR-SONIC/low_latency
```

## Run

```bash
dora run dataflow.yml
```

Commands in the first phase are `stand`, `slow-walk`, `walk`, and `run`.
Append `forward`, `backward`, `left`, or `right`, and optionally a speed:

```text
walk forward
walk left 0.4
stand
```

Set `DS_RF_VIEWER=headless` to run without a native MuJoCo window.

## CUDA later

The mutually exclusive `cu128` extra reserves the GPU dependency boundary:

```bash
uv sync --extra cu128
```

CUDA ONNX execution, shared streams, and zero-copy Dora transport are deferred.
