# VS Code Remote SSH task workflow

The repository is the durable record; a long chat transcript is not.

## Start new work

1. Choose a committed starting point. If the task depends on uncommitted work,
   checkpoint that work deliberately first.
2. Create an isolated worktree:

   ```bash
   mkdir -p /afs/cern.ch/user/z/zhibin/worktrees
   cd /afs/cern.ch/user/z/zhibin/work/snd-ml
   git worktree add -b task/<name> \
     /afs/cern.ch/user/z/zhibin/worktrees/snd-ml-<name> \
     <starting-commit-or-branch>
   ```

3. Open the worktree in a separate VS Code Remote SSH window.
4. Create its task record:

   ```bash
   cd <task-worktree>
   mkdir -p docs/tasks/active docs/tasks/completed
   cp docs/tasks/TASK_TEMPLATE.md docs/tasks/active/<name>.md
   ```

5. Fill in its goal, acceptance criteria, location, scope, and first action.
6. Start a new Codex chat with:

   ```text
   Start docs/tasks/active/<name>.md. Read AGENTS.md and the task file first.
   Verify the worktree, branch, starting commit, git status, and relevant files.
   Correct stale task state before implementation. Execute the exact next
   action and keep the task record current with evidence and validation.
   ```

Use `/goal` only when the acceptance criteria are precise.

## Checkpoint and rotate

At a meaningful milestone, send:

```text
Checkpoint this task before stopping. Update docs/tasks/active/<name>.md with
verified findings, decisions, modified files, commands and results, running job
IDs, current Git state, and one exact next action. Make a task-only checkpoint
commit if appropriate. Do not start the next action.
```

Review the record and commit. Start a fresh chat when the existing one becomes
long or after a major stage completes.

## Resume

1. Reconnect and open the exact worktree.
2. Inspect:

   ```bash
   pwd
   git status --short
   git branch --show-current
   git log -5 --oneline --decorate
   ```

3. Reopen a healthy, concise chat from Recent chats. Otherwise start a new one.
4. Send:

   ```text
   Resume docs/tasks/active/<name>.md. Read AGENTS.md and the task file first.
   Treat the task record, Git state, current code, outputs, tests, and scheduler
   state as authoritative. Verify completed checkpoints and running jobs, then
   execute the exact next action. Keep the task record current.
   ```

CLI session recovery is optional:

```bash
codex agents
codex resume --all
```

## Long jobs

Use Snakemake/HTCondor so work does not depend on VS Code or SSH remaining
connected. Record the exact target, command, job ID, log, expected outputs, and
status command before disconnecting.

## Complete

Run validation, mark every acceptance criterion, record the final commit and
results, move durable knowledge into the appropriate documentation, then:

```bash
mkdir -p docs/tasks/completed
git mv docs/tasks/active/<name>.md docs/tasks/completed/<name>.md
```
