# Codex guidance for snd-ml

## Start

For substantial work:

1. Read `docs/project/OVERVIEW.md` and `docs/project/CURRENT_STATUS.md`.
2. Read the named file under `docs/tasks/active/`.
3. Run `git status --short` and preserve unrelated work.
4. Inspect task-relevant files only; do not rescan the whole repository.

Maintained areas are `convertData/`, `snakemake/`,
`evaluation_region_partitions/`, `evaluation_data_driven_3d_cnn/`,
`train_3d/`, and `docs/`. Other areas are legacy, specialized, or
exploratory unless the task names them.

## Invariants

- Feature and Hit3D products join on `(runId, eventId, eventIndex)`.
- Preserve or explicitly validate aligned feature/Hit3D entry identity.
- Snakemake is the primary orchestration layer; prefer named aggregate targets.
- Do not edit generated ROOT files, plots, checkpoints, logs, or metadata
  products unless the task requires it.
- Submit long CERN work through Snakemake/HTCondor and record the command, job
  ID, logs, expected outputs, and status command in the task file.

## Continuity

The chat transcript is not authoritative. Task files, Git state, code, tests,
scheduler state, and outputs are authoritative.

For work that may cross sessions:

1. Copy `docs/tasks/TASK_TEMPLATE.md` to
   `docs/tasks/active/<task-name>.md`.
2. Update it after each milestone, before long jobs, before changing direction,
   and before stopping or rotating the chat.
3. Record evidence: files, commit SHAs, commands, results, job IDs, decisions,
   uncertainties, and one exact next action.
4. On resume, verify the record against Git and actual outputs before acting.
5. Start a fresh chat at milestones rather than growing one indefinitely.
6. When complete, move the task file to `docs/tasks/completed/` and move
   durable knowledge into the appropriate project documentation.

## Parallel work

- Use one branch, worktree, VS Code window, and chat per independent task.
- Never let two chats modify the same checkout.
- Use the main checkout for integration and review.
- A new worktree does not contain another checkout's uncommitted files.
- Make task-only checkpoint commits; never include unrelated user changes.
