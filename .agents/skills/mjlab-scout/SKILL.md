---
name: mjlab-scout
description: Inspect an MJLab task with the mjlab-scout MCP tools and write a visually grounded, task-specific TASK.md prompt for a separate VLM execution context. Use when preparing or regenerating a task prompt for any MJLab environment.
---

# MJLab Scout

Inspect one task and create a concise, visually grounded `TASK.md` for a later
live navigation run. Phase 1's only deliverable is `TASK.md`.

## The two phases

- **Phase 1: scouting.** Use the `mjlab-scout` MCP tools to inspect the static
  task through every returned camera preset. Use these views to understand the
  initial surroundings, layout, openings, obstacles, landmarks, target,
  distractors, and observable success condition. Read `prompt/PLANNER_USER.md`
  to understand the live planner's action interface. Do not navigate, send
  motion commands, advance the simulation, or inspect source code.
- **Phase 2: live navigation.** A clean-context VLM controls an MJLab
  simulation. It receives `TASK.md`, a live image from the robot's normal
  forward-facing camera, and the allowed motion commands. It may also receive
  the optional invariant controller prompt. Phase 2—not Phase 1—must use the
  objective's title or role and its own visual reasoning to determine which
  candidate matches.
- **Phase 2 does not have** the Phase 1 conversation or reasoning, Scout tools,
  camera presets, overview or overhead images, view names, or any way to request
  those views. `TASK.md` must therefore carry only the visual knowledge Phase 2
  needs while leaving it to discover the target's location and route live.

## Procedure

1. Read all of `prompt/PLANNER_USER.md` before loading the task. Treat its
   allowed motions, direction semantics, response schema, and other action
   constraints as authoritative. This prompt is interface context, not visual
   evidence about the environment.

2. Call `list_tasks` only if the task name is unknown, then call `load_task`
   once.

3. Inspect **ALL camera preset views** returned by `load_task` with
   `capture_view`. Only then begin interpreting the scene or drafting `TASK.md`.
   Treat the images as separate perspectives of one environment, not as the
   robot's current live view.

4. Inspect the candidates and understand how they can be compared, but do not
   resolve or disclose which candidate matches the objective. A centered,
   nearby, or prominent candidate must not be presented as the answer.

5. Build a compact description using only supported visual facts. Describe the
   initial surroundings and useful scene structure, and derive concise
   navigation principles from the visible geometry. Express actions consistently
   with `prompt/PLANNER_USER.md`, including its direction convention. Do not add
   target-specific appearance cues that reveal which candidate matches the
   objective.

6. Write or replace `TASK.md` in the workspace root using the mandatory format
   below.

7. Call `close_task` after writing. If scouting or writing fails, still call
   `close_task` before reporting the failure.

## Hard identity boundary

- Never write, quote, spell, abbreviate, alias, label, or parenthetically add a
  candidate's personal name anywhere in `TASK.md`.
- Refer to the target only by the exact title or role supplied by the objective.
- Never reveal which visible candidate matches that title or role through a
  name, physical description, clothing, corridor, ordering, location, or other
  distinguishing cue.
- Leave the identity match entirely to the Phase 2 VLM. Phase 2 must derive the
  match itself from the objective and its live visual observations.
- Before finalizing, audit every line of `TASK.md`. If any personal identity or
  candidate-to-title mapping appears, remove it and audit the file again.

## Mandatory TASK.md format

Write raw Markdown using exactly this heading structure and order. Keep the
result close in detail and tone to the template below. The text beneath each
heading is an instruction, not output to copy: replace it with content inferred
by the VLM from the objective and inspected images. Replace `<task-name>` with
the name returned by `load_task`. Do not add an outer title or code fence.

```markdown
# Task: <task-name>

## Objective

Write the task objective and a short, visually grounded description of the
choice the agent must make.

## Identify

Write a numbered procedure for inspecting and comparing every visible candidate
before committing. Explain how to avoid choosing by proximity, prominence, or
the first plausible match and what to do while uncertain.

## Enter safely

Write a numbered procedure for approaching and entering the selected route.
Describe alignment, visible clearance, boundary checks, and corrections that
must happen before entry.

## Navigate

Write a numbered procedure derived from the visible route geometry. Prefer
stable forward progress after alignment, discourage unnecessary corrections,
and explain how to recover from visible drift without contacting boundaries.

## Wrong corridor recovery

Write a safe numbered recovery procedure for leaving an incorrect route,
returning to the decision area, and inspecting the candidates again. Ground the
procedure in traversable space visible in the inspected views.

## Finish

Describe the visual evidence that the objective has actually been reached,
warn against stopping prematurely, and give the final actions and pose without
inventing hidden thresholds.

## Critical rule

End with one short, emphatic rule capturing the most important navigation
lesson derived from the inspected task.
```

## TASK.md rules

- Ground every detail in the inspected images. Omit uncertain details instead of
  guessing; use exact counts only when unambiguous and useful.
- Preserve the objective without broadening, narrowing, reinterpreting, or
  solving its identity match for Phase 2.
- Describe navigation-relevant environment structure, but never bind the target
  to a side, direction, branch, corridor, ordering, adjacency, landmark, turn
  sequence, or other privileged location clue.
- Apply the hard identity boundary without exceptions.
- Keep every procedure actionable from the live forward-facing image and
  supported by visible scene geometry. Use only motions and global direction
  meanings supported by `prompt/PLANNER_USER.md`; never silently reinterpret
  them as robot-relative directions.
- Do not present a privileged camera composition as the robot's current view.
- Make success stricter than merely seeing the target unless the objective says
  otherwise. Do not invent distances, tolerances, or hidden reward conditions.
- Do not mention scouting, phases, MCP, tools, cameras, presets, view names,
  prompt filenames, unavailable context, or prompt-generation instructions.
- Do not add personas, simulator or robot make/model details, coordinates,
  dimensions, source-code names, implementation details, uncertainty
  commentary, or extra sections.
