# dsrf

A three-node Dora closed loop for visual control of a Unitree G1 in MJLab:

1. `sonic` executes one motion segment, freezes MJLab, and captures a JPEG.
2. `agent` sends the observation and conversation history to llama-server.
3. `motion-gen` converts the raw VLM command into one reference segment.

Only one segment is in flight. SONIC runs at 50 Hz and does not advance physics
while waiting for the next motion.

The default scene is the `portrait-corridors` task: three corridors terminate
at portraits of Linus Torvalds, Andrej Karpathy, and Bugs Bunny. Set
`DSRF_TASK=none` to use the empty plane instead.

## Setup

Install Dora 1.0.0-rc.4 and the CPU environment:

```bash
cargo install \
  --git https://github.com/dora-rs/dora.git \
  --tag v1.0.0-rc.4 \
  --locked \
  dora-cli

uv sync --extra cpu
```

Start a multimodal llama.cpp server. Its default OpenAI-compatible URL is used
automatically:

```bash
llama-server -m /path/to/model.gguf --mmproj /path/to/mmproj.gguf
```

The SONIC bundle defaults to `/tmp/GEAR-SONIC`. Then run:

```bash
dora run dataflow.yml
```

SONIC publishes an initial observation immediately after startup. The agent
begins querying llama-server as soon as that image arrives.

The SONIC node also opens a passive MuJoCo viewer. The window only mirrors
SONIC's state: it does not step physics, and the simulation remains frozen
while the loop waits for a new motion.

## Configuration

Shell values override the defaults in `dataflow.yml`.

| Variable | Default | Purpose |
|---|---:|---|
| `DSRF_VLM_URL` | `http://127.0.0.1:8080` | llama-server base URL |
| `DSRF_VLM_TIMEOUT` | `120` | VLM request timeout in seconds |
| `DSRF_VLM_SYSTEM_PROMPT` | `prompt/SYSTEM.md` | System prompt file |
| `DSRF_VLM_USER_PROMPT` | `prompt/USER.md` | Per-observation user prompt file |
| `DSRF_IMAGE_WIDTH` | `640` | Observation width |
| `DSRF_IMAGE_HEIGHT` | `480` | Observation height |
| `DSRF_JPEG_QUALITY` | `85` | JPEG quality from 1 to 100 |
| `DSRF_DEVICE` | `cpu` | Torch, MJLab, and ONNX device |
| `DSRF_PLANNER_ONNX` | `/tmp/GEAR-SONIC/planner_sonic.onnx` | Planner model |
| `DSRF_SONIC_DIR` | `/tmp/GEAR-SONIC` | SONIC model bundle |
| `DSRF_TASK` | `portrait-corridors` | Task scene, or `none` for an empty plane |

For example:

```bash
DSRF_VLM_URL=http://127.0.0.1:9379 \
DSRF_IMAGE_WIDTH=320 \
DSRF_IMAGE_HEIGHT=240 \
dora run dataflow.yml
```

## CUDA 12.8

The CPU and CUDA extras are mutually exclusive:

```bash
uv sync --extra cu128
DSRF_DEVICE=cuda:0 dora run dataflow.yml
```

SONIC's ONNX models use CUDA I/O binding on MJLab's Warp stream, so policy
observations, actions, and simulation remain on the selected device. JPEG
capture intentionally copies the terminal frame to the host.

To switch back:

```bash
uv sync --extra cpu
```
