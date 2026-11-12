# Task: unified-shower-features

## Current state

- Lifecycle: active
- Worktree: `/afs/cern.ch/work/z/zhibin/worktrees/snd-ml-unified-shower-features`
- Branch: `task/unified-shower-features`
- Last verified: 2026-11-12T16:44:35+01:00
- Implementation checkpoint: `246eb59f62c90c430f48884e10a7d4ecc73fdb8d`
- Implemented: Implementation 559efa5 and plotting update f2d0db9 saved. Output paths now use the AFS work defaults; scientific algorithms unchanged.
- Validation: Previously recorded: 67 unit tests, 2 metadata tests and synthetic ABCD checks; not rerun here. Storage update: 57 task-tool tests, 28 workflow parses, 4 DAG dry-runs and migrated-file checksum verification passed; focused scientific regressions are detailed in the central migration report.
- Remaining limitations: Migration/canary validation remains; production has not been authorized.
- Running jobs: Not queried; verify before running or retiring work.

- Outputs: /afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1 (existing campaign retained); EOS production roots unchanged.

## Goal

Unify shower feature computation and diagnostics.

## Next action

Review the migration/canary plan and validate the selected small sample before production.

## Checkpoint history

Original record preserved verbatim: [earlier checkpoints](../history/unified-shower-features-before-workflow-v2.md). Historical next actions are context, not current authorization.

### 2026-11-12T16:44:35+01:00 — AFS output consolidation

/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1 (existing campaign retained); EOS production roots unchanged.

Compatibility links preserve migrated old paths. No production job was submitted.
The central audit and test plan are in the develop checkout at
`docs/tasks/output-migration-validation-20261112.json` and `docs/tasks/OUTPUT_STORAGE.md`.
Scientific next actions above remain unchanged.
