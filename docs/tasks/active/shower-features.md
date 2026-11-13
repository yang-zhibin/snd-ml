# Task: shower-features

## Current state

- Lifecycle: active
- Worktree: `/afs/cern.ch/work/z/zhibin/worktrees/snd-ml-shower-features`
- Branch: `task/shower-features`
- Last verified: 2026-11-13 (feature-development interface acceptance)
- Implementation checkpoint: `246eb59f62c90c430f48884e10a7d4ecc73fdb8d`
- Implemented: Implementation 559efa5 and plotting update f2d0db9 saved. Output paths now use the AFS work defaults; scientific algorithms unchanged.
- Validation: Previously recorded: 67 unit tests, 2 metadata tests and synthetic ABCD checks; not rerun here. Storage update: 57 task-tool tests, 28 workflow parses, 4 DAG dry-runs and migrated-file checksum verification passed; focused scientific regressions are detailed in the central migration report.
- Remaining limitations: Migration/canary validation remains; production has not been authorized.
- Running jobs: Not queried; verify before running or retiring work.

- Outputs: /afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1 (existing campaign retained); EOS production roots unchanged.

## Goal

Unify shower feature computation and diagnostics.

## Next action

Export a small real sample from each source with the existing SNDSW converter invocation plus `--feature-dev-export`, and compare the development baseline against the original ROOT branches before physics use. See `feature_dev/README.md`.

## Checkpoint history

Original record preserved verbatim: [earlier checkpoints](../history/unified-shower-features-before-workflow-v2.md). Historical next actions are context, not current authorization.

### 2026-11-12T16:44:35+01:00 — AFS output consolidation

/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1 (existing campaign retained); EOS production roots unchanged.

Compatibility links preserve migrated old paths. No production job was submitted.
The central audit and test plan are in the develop checkout at
`docs/tasks/output-migration-validation-20261112.json` and `docs/tasks/OUTPUT_STORAGE.md`.
Scientific next actions above remain unchanged.

### 2026-11-13T10:10:35+01:00 — Task renamed

Renamed task and branch from `unified-shower-features` to `shower-features`. Worktree: `/afs/cern.ch/work/z/zhibin/worktrees/snd-ml-shower-features`.
Old worktree paths remain compatibility symlinks. Scientific output campaign
names and historical checkpoint records are unchanged.

### 2026-11-13T10:16:10+01:00 — Visible shortcuts renamed

Removed the old-name worktree compatibility symlinks at the user request.
The AFS home shortcut is now `~/worktrees/snd-ml-shower-features` and points
to the AFS work checkout. Reopen `shower-features.code-workspace` if an old
VS Code window still displays the previous name.

### 2026-11-13 — Feature-development interface implemented

Implemented `run_feature_dev.sh`, two-function algorithm plugins, schemas, a
versioned synthetic sample, selected-hit export/merge, and the Snakemake DAG
(calculation, validation, histograms, report). Default output is AFS work under
the existing campaign's `feature_dev/`. Content hashes select reusable stage
outputs; conflicting run names and modified cached products are rejected.

Colleague tutorial and real-sample instructions: `feature_dev/README.md`.
Validation: 41 unit/regression tests, 10 end-to-end acceptance checks, and 100
randomized pre-change baseline comparisons passed. Both `tutorial-gap1` and
`tutorial-gap0` reports were generated. Evidence: `feature_dev/TEST_RESULTS.md`.

No production algorithm was switched and no detector conversion was launched.
Real selected-hit exports and event-by-event production parity are the remaining
physics acceptance step. Example gap clustering is an unqualified hypothesis.
