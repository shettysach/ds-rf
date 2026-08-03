## Setup

dora-rs

```bash
cargo install \
  --git https://github.com/dora-rs/dora.git \
  --tag v1.0.0-rc.4 \
  --locked \
  dora-cli

uv sync --extra cu128
```

SONIC

```bash
uvx --from huggingface_hub hf download nvidia/GEAR-SONIC \
  model_encoder.onnx \
  model_decoder.onnx \
  observation_config.yaml \
  planner_sonic.onnx \
  low_latency/model_encoder.onnx \
  low_latency/model_decoder.onnx \
  low_latency/observation_config.yaml \
  --local-dir /tmp/GEAR-SONIC
```

```bash
cp .env.example .env
set -a
source .env
set +a
```

Also an OpenAI compatible VLM inference server.

## Run

Run OpenAI compatible VLM inference server.

```bash
dora run dataflow.yml
```

- Set `DSRF_VIEWER=none` to disable the window for headless runs.
- Set `DSRF_REFERENCE_GHOST=true` to show the active motion reference in the
  native viewer.

## ARDY smoke test

The ARDY smoke graph skips the agent and sends one fixed-encoding motion to
SONIC after the simulator publishes its initial observation:

```bash
export DSRF_MOTION_GENERATOR=ardy
export CHECKPOINTS_DIR=/path/to/ardy/checkpoints
export ENCODING=~/Videos/walk_forward.pt
dora run ardy_smoketest.yml
```

`CHECKPOINTS_DIR` may be the G1 checkpoint directory itself (containing
`config.yaml`) or its parent directory containing
`ARDY-G1-RP-25FPS-Horizon52`. The smoke graph uses the `cu128` and
experimental `ardy` extras and defaults `DSRF_DEVICE` to `cuda:0`.
