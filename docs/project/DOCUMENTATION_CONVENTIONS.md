# Documentation conventions

## Where information belongs

- Stable project knowledge: `docs/`.
- Code-specific API or schema notes: beside the owning source code.
- Generated validation and performance reports: beside their outputs.
- Short-lived implementation state: `docs/tasks/<feature>.md`.
- Rejected or superseded designs: `docs/archive/superseded/`.

## Recommended header

```text
Title
Status: authoritative | implemented | active | proposed | paused | superseded
Last updated: YYYY-MM-DD
Scope: one sentence
```

Keep scientific version identifiers such as `V7` and `V8` in filenames. Use
`PLAN` only for genuinely incomplete work. When a plan becomes a maintained
reference, rename it or add a clear `implemented` status.
