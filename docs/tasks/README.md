# Task handoffs

Use one short file per active feature. The user supplies the goal and first
action; the chat discovers scope and constraints from the repository. Record
status, findings, results, evidence, changed files, validation, blockers, and
next actions. Move durable knowledge into a reference document when work
completes.

Keep one local request per task at `requests/<task-name>.json`, set its
action to `auto`, and run `./docs/tasks/manage_task.py --task <task-name>`.
The first run creates the task; later runs resume its branch and worktree. The
launcher also creates the task record and ready-to-copy prompts. The legacy
single-file `TASK_REQUEST.json` and `--request <path>` interfaces remain
available.

Copy `TASK_TEMPLATE.md` to `active/<task-name>.md` when work begins and move
it to `completed/` after validation. See `WORKFLOW.md` for the VS Code Remote
SSH start, checkpoint, resume, and parallel-worktree procedure.
