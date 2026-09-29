# Task handoffs

Use one short file per active feature. Include the goal, acceptance criteria,
status, decisions, changed files, validation results, blockers, and next
actions. Move durable knowledge into a reference document when work completes.

Use the normal one-command path by editing `TASK_REQUEST.txt` and running
`./docs/tasks/manage_task.py`. The launcher creates or resumes the worktree,
task record, and ready-to-copy prompts.

Copy `TASK_TEMPLATE.md` to `active/<task-name>.md` when work begins and move
it to `completed/` after validation. See `WORKFLOW.md` for the VS Code Remote
SSH start, checkpoint, resume, and parallel-worktree procedure.
