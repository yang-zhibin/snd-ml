#!/usr/bin/env python3
"""Create or resume an snd-ml Codex task from one editable request file."""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

TASK_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
PLACEHOLDERS = ("replace-with-", "Describe what")


def run(command, cwd=None, check=True, capture=False):
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def git_output(repo, *args):
    result = run(["git", *args], cwd=repo, capture=True)
    return result.stdout.strip()


def text_value(task, key, default=""):
    value = task.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a JSON string")
    return value.strip()


def validate_request(path):
    if not path.is_file():
        raise ValueError(
            f"Request file does not exist: {path}\n"
            "Copy TASK_REQUEST.example.json to TASK_REQUEST.json and edit it."
        )
    try:
        task = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error
    if not isinstance(task, dict):
        raise ValueError("The request must be one JSON object")

    action = text_value(task, "action").lower()
    if action not in {"start", "resume"}:
        raise ValueError("action must be start or resume")
    name = text_value(task, "name")
    if not TASK_RE.fullmatch(name):
        raise ValueError(
            "name must start with a lowercase letter or number and contain only "
            "lowercase letters, numbers, periods, underscores, or hyphens"
        )

    required = {
        "goal": text_value(task, "goal"),
        "next_action": text_value(task, "next_action"),
    }
    if action == "start":
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError("Missing values: " + ", ".join(missing))
        combined = "\n".join(required.values())
        if any(marker in combined for marker in PLACEHOLDERS):
            raise ValueError("Replace all instructional placeholder text first")

    open_vscode = task.get("open_vscode", True)
    if not isinstance(open_vscode, bool):
        raise ValueError("open_vscode must be true or false")
    return task, action, name, required, open_vscode


def render_task(name, branch, worktree, start_commit, values):
    timestamp = datetime.now(ZoneInfo("Europe/Zurich")).strftime(
        "%Y-%m-%d %H:%M Europe/Zurich"
    )
    return f"""# Task: {name}

Status: active
Last updated: {timestamp}

## Goal

{values["goal"]}

## Work location

- Worktree: `{worktree}`
- Branch: `{branch}`
- Starting commit: `{start_commit}`
- Chat/session ID: `not recorded`

## Discovered scope

- To be determined by inspecting the current code and workflow.

## Task-specific constraints

- None provided initially. Record constraints discovered during the work.

## Completed checkpoints

- None yet.

## Findings and decisions

- None yet.

## Results and evidence

- No results yet. Continue until concrete results are ready for user review.

## Modified files

- None yet.

## Validation

- Not run yet.

## Running jobs

- None.

## Blockers and open questions

- None.

## Exact next action

{values["next_action"]}
"""


def prompts(name):
    start = f"""Start docs/tasks/active/{name}.md. Read AGENTS.md and the task file
first. Inspect the current code and workflow to discover the relevant scope and
any task-specific constraints, and record them in the task file. Verify the
worktree, branch, starting commit, git status, and relevant files. Correct stale
task state before implementation. Execute the exact next action and work
iteratively until concrete results are ready for user review. Keep the task
record current with evidence and validation.
"""
    resume = f"""Resume docs/tasks/active/{name}.md. Read AGENTS.md and the task
file first. Treat the task record, Git state, current code, outputs, tests, and
scheduler state as authoritative. Verify completed checkpoints, discovered
scope, constraints, results, and running jobs, then execute the exact next
action. Continue until concrete results are ready for user review and keep the
task record current.
"""
    checkpoint = f"""Checkpoint docs/tasks/active/{name}.md before stopping.
Record verified findings, decisions, modified files, commands and results,
running job IDs, current Git state, and one exact next action. Make a task-only
checkpoint commit if appropriate. Do not start the next action.
"""
    return start, resume, checkpoint


def write_prompts(task_dir, name):
    start, resume, checkpoint = prompts(name)
    paths = {
        "start": task_dir / f"{name}.start-prompt.txt",
        "resume": task_dir / f"{name}.resume-prompt.txt",
        "checkpoint": task_dir / f"{name}.checkpoint-prompt.txt",
    }
    for label, path in paths.items():
        path.write_text(
            {"start": start, "resume": resume, "checkpoint": checkpoint}[label],
            encoding="utf-8",
        )
    return paths


def open_in_vscode(worktree, files):
    code = shutil.which("code")
    if not code:
        print("VS Code CLI not found; open this folder manually:")
        print(worktree)
        return
    run([code, "--new-window", str(worktree), *[str(path) for path in files]])


def main():
    script_path = Path(__file__).resolve()
    repo = script_path.parents[2]
    default_request = script_path.with_name("TASK_REQUEST.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, default=default_request)
    parser.add_argument(
        "--check", action="store_true", help="validate without creating or opening anything"
    )
    args = parser.parse_args()

    try:
        task, action, name, values, open_vscode = validate_request(
            args.request.expanduser().resolve()
        )
        repo_root = Path(git_output(repo, "rev-parse", "--show-toplevel")).resolve()
        if repo_root != repo.resolve():
            raise ValueError(f"Script is not inside the expected Git root: {repo_root}")

        worktree_root = Path(
            text_value(
                task, "worktree_root", "/afs/cern.ch/user/z/zhibin/worktrees"
            )
        ).expanduser()
        destination = (worktree_root / f"snd-ml-{name}").resolve()
        branch = f"task/{name}"
        task_file = destination / "docs" / "tasks" / "active" / f"{name}.md"

        if args.check:
            print(f"Request valid: action={action} task={name}")
            print(f"Worktree: {destination}")
            print(f"Branch: {branch}")
            return 0

        if action == "start":
            base_ref = text_value(task, "base_ref", "develop")
            start_commit = git_output(repo, "rev-parse", "--verify", f"{base_ref}^{{commit}}")
            branch_exists = run(
                ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
                cwd=repo,
                check=False,
            ).returncode == 0
            if branch_exists:
                raise ValueError(
                    f"Branch already exists: {branch}. Use action = resume if appropriate."
                )
            if destination.exists():
                raise ValueError(
                    f"Worktree path already exists: {destination}. "
                    "Use action = resume if appropriate."
                )
            worktree_root.mkdir(parents=True, exist_ok=True)
            run(
                [
                    "git", "worktree", "add", "-b", branch,
                    str(destination), start_commit,
                ],
                cwd=repo,
            )
            task_file.parent.mkdir(parents=True, exist_ok=True)
            (destination / "docs" / "tasks" / "completed").mkdir(
                parents=True, exist_ok=True
            )
            task_file.write_text(
                render_task(name, branch, destination, start_commit, values),
                encoding="utf-8",
            )
            prompt_kind = "start"
        else:
            if not destination.is_dir():
                raise ValueError(f"Existing worktree not found: {destination}")
            actual_root = Path(
                git_output(destination, "rev-parse", "--show-toplevel")
            ).resolve()
            if actual_root != destination:
                raise ValueError(f"Path is not the expected worktree root: {destination}")
            actual_branch = git_output(destination, "branch", "--show-current")
            if actual_branch != branch:
                raise ValueError(
                    f"Expected branch {branch}, found {actual_branch or 'detached HEAD'}"
                )
            if not task_file.is_file():
                raise ValueError(f"Task record not found: {task_file}")
            prompt_kind = "resume"

        prompt_paths = write_prompts(task_file.parent, name)
        print()
        print(f"Task ready: {name}")
        print(f"Worktree: {destination}")
        print(f"Task record: {task_file}")
        print(f"Prompt to copy: {prompt_paths[prompt_kind]}")
        print()
        print(prompt_paths[prompt_kind].read_text(encoding="utf-8"))

        if open_vscode:
            open_in_vscode(
                destination,
                [
                    task_file,
                    prompt_paths[prompt_kind],
                    prompt_paths["checkpoint"],
                ],
            )
        return 0
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
