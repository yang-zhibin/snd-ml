# Local task requests

Keep one reusable JSON file per task in this directory:

```text
docs/tasks/requests/<task-name>.json
```

These JSON files are ignored by Git. Use `"action": "auto"`, then run:

```bash
./docs/tasks/manage_task.py --task <task-name>
```

The first run starts the task. Later runs resume its existing branch and
worktree. The JSON `name` must match the filename without `.json`.
