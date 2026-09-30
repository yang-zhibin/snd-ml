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

## Runtime environments

Read `docs/project/ENVIRONMENT_AND_SNAKEMAKE.md` before running Python,
ROOT, Snakemake, tests, or batch jobs.

- Inspect the relevant rule or wrapper and use its existing environment. Do
  not choose an environment from memory.
- Environment activation and the dependent command must run in the same shell
  block or batch job. A previous terminal, tool call, or login does not carry
  into a new process.
- Before sourcing CVMFS, SNDSW, or Conda setup under strict Bash, run `set +u`;
  enable `set -euo pipefail` only after activation.
- Run a small preflight that records `hostname`, `pwd`, `command -v python3`,
  the Python version, and imports required by the command.
- Treat ROOT/SNDSW and PyTorch as separate runtime stacks unless a documented,
  validated container or overlay explicitly provides both. Split mixed work
  into separate processes with files as the interface.
- Set `PYTHONNOUSERSITE=1` and normally `PYTHONDONTWRITEBYTECODE=1` to prevent
  packages under the user site from changing the selected environment.
- Do not install packages, alter shared environments, or edit activation
  scripts to bypass an import failure unless the user explicitly requests it.
  Diagnose the selected interpreter and required imports first.

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
