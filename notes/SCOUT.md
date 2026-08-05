# Scout: launch and test guide

Scout is a local MCP server that inspects a static DSRF task from several
camera presets. A scouting agent uses those views and the bundled skill to
write `TASK.md` for a separate live VLM run.

## Install

Choose one MJLab device extra and include Scout:

```bash
uv sync --extra cu128 --extra scout
```

For CPU-only use:

```bash
uv sync --extra cpu --extra scout
```

The `cpu` and `cu128` extras conflict, so do not enable both.

## MCP configuration

The root `.mcp.json` launches:

```bash
.venv/bin/mjlab-scout --device cuda:0
```

The server communicates over stdio and normally waits silently for an MCP
client. To use CPU instead, change the configured device to `cpu`.

Loading a task also opens the rendered `overview` frame in a small window for
three seconds. This is the same frame returned to the scouting VLM. Pass
`--preview-seconds 0` to disable the window, or choose another duration in
seconds.

The Phase 1 skill is located at:

```text
.agents/skills/mjlab-scout/SKILL.md
```

A typical request is:

```text
Use the mjlab-scout skill to inspect portrait-corridors and write TASK.md in the current working directory.
```

## Generate TASK.md with Pi

Install and enable `pi-mcp-adapter`, then start Pi from the repository root so
it discovers `.mcp.json` and `.agents/skills/mjlab-scout/SKILL.md`. The bundled
Phase 1 request can be run non-interactively with:

```bash
pi -a -p @prompt.md
```

Pi will load the task through Scout, inspect every advertised view, write
`TASK.md` in the repository root, and close the Scout task. The `-a` flag lets
the non-interactive run use the project-local MCP configuration and skill. In
interactive Pi, start `pi`, approve the project when prompted, and enter the
same request from `prompt.md`.

`load_task("portrait-corridors")` advertises the `agent`, `overview`,
`overhead`, `corridor_left`, `corridor_center`, and `corridor_right` views.
The skill requires inspecting every advertised view before writing `TASK.md`.

## Verify

Run the focused tests with:

```bash
uv run --extra cpu --extra scout --with pytest python -m pytest tests/test_scout.py
```

Check the command-line entry point with:

```bash
uv run --extra cpu --extra scout mjlab-scout --help
```
