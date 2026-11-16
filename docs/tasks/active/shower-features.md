# Task: shower-features

## Current state

- Lifecycle: active
- Worktree: `/afs/cern.ch/work/z/zhibin/worktrees/snd-ml-shower-features`
- Branch: `task/shower-features`
- Last verified: 2026-11-16T17:13:29+01:00
- Implementation checkpoint: d4f663d458d002f66f3426567548385e9c21a018
- Implemented: Plugin-based feature-development DAG, selected-hit export/merge and baseline comparison interface (4862fc4); independent dataset routing, deterministic sample preparation and registry (431f4e8); managed private Snakemake 9.17.2 setup/check (d4f663d). Production algorithms remain unchanged.
- Validation: Existing `feature_dev/TEST_RESULTS.md` records 41 initial unit/regression tests, 100 randomized baseline comparisons, 10 integration checks; dataset update records 30 unit tests plus 10 dataset and 10 existing integration checks; setup update records 35 unit tests and fresh/repeat/check/minimal-shell synthetic runs. These scientific/runtime tests were not rerun for this documentation checkpoint. Git state, implementation history, report existence and scheduler state verified below.
- Remaining limitations: Real SNDSW selected-hit export and event-by-event parity against original ROOT branches remain unvalidated. Default physics sample directory is absent. Synthetic checks do not qualify physics algorithms; gap clustering remains experimental. Sharded large-input execution and exposure normalization are unimplemented. Temporary environment-audit evidence is unavailable on this host; collaborator-account access remains untested. Production has not been authorized.
- Running jobs: `condor_q -submitter zhibin -af ClusterId ProcId JobStatus Iwd Cmd` returned exit 0 with no rows at verification time; no jobs submitted in this checkpoint. This is the queried scheduler's current view, not a historical or all-pool audit.
- Outputs: Existing campaign `/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1`; three feature-development HTML reports verified present (paths below). EOS production roots unchanged; no output products modified.

## Goal

Unify shower feature computation and diagnostics.

## Next action

Prepare and qualify small real selected-hit exports for `mc-neutrino`, `testbeam`, and `real-data` using the documented SNDSW converter `--feature-dev-export` flow, preserving required source groups and `(runId, eventId, eventIndex)` identity; compare the development baseline event by event with original ROOT branches before physics use. Follow `feature_dev/README.md` and `feature_dev/datasets.yaml`. This action was not started during checkpointing.

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

### 2026-11-13 — Collaborator README

Added `README_SHOWER_FEATURES.md` covering branch logic, dataset locations and
counts, current feature-file sizes, hit/cluster definitions, normalization, and
a complete plugin tutorial. Linked it from both existing READMEs. Counts are
explicitly separated into catalog, selected-event and saved-plot snapshots.
Filesystem inspection found one missing H8 feature file, source
`unified_tb23_src_e596bffdd9b7f0e203ef`, despite the older coverage JSON; this is
recorded in the guide. No report regeneration or production job was launched.

### 2026-11-13 — Independent dataset reruns

Implemented `--dataset mc-neutrino|testbeam|real-data|synthetic|mc-muon|all`,
per-dataset cache/output isolation, fixed-sample preparation with deterministic
per-group/geometry caps, and a registry for small/full exports. `all` preflights
all three physics samples and produces a report index. Provenance checks reject
synthetic/wrong-family inputs for physics registration. Full input is guarded at
256 MiB until a sharded runner exists. Normalization remains unit-area shapes.

Validation: 30 unit tests, 10 new dataset integration checks and 10 existing
integration checks passed. The synthetic wrapper demonstration completed under
`feature_dev/synthetic/dataset-demo-v1/`. Updated both collaborator guides and
`feature_dev/TEST_RESULTS.md`. Next action remains preparing and qualifying
correctly grouped real selected-hit exports; no production jobs were launched.

### 2026-11-15 — Collaborator environment audit

Confirmed LCG 107 has NumPy 1.26.4, uproot 5.3.11, PyYAML 6.0.1 and
matplotlib 3.8.3, but no Snakemake executable/distribution. A personal executable
on PATH was using LCG Python, explaining the apparent shared installation.

Tested a separate LCG-Python virtual environment at
`/tmp/zhibin/shower-snakemake-setup-20261115`. Installing with inherited LCG
PYTHONPATH reused external packages and reproduced missing `humanfriendly` when
running isolated. Installing with PYTHONPATH/PYTHONHOME unset resolved this;
`pip check` passed. Snakemake 9.17.2 completed the synthetic DAG from an `env -i`
shell with only HOME/USER/LOGNAME/PATH/LANG and FEATURE_DEV_SNAKEMAKE supplied.
Evidence: `/tmp/zhibin/shower-clean-login.log`; report:
`/tmp/zhibin/shower-env-audit-20261115/synthetic/clean-login/plot/report.html`.
These are temporary audit artifacts, not production products.

Updated collaborator setup instructions to clear Python path variables during
installation. No AFS permissions changed. Coworker's account, storage quota,
CVMFS access and physics-data permissions have not been tested. Recommend own
Snakemake venv + centrally provided LCG analysis stack, with fixed selected-hit
samples shared separately. No SNDSW or GPU needed for exported-hit development.

### 2026-11-15 — One-command collaborator setup

Implemented `./run_feature_dev.sh setup [--prefix PATH]` and `setup --check`.
The installer uses LCG Python, clears inherited Python/pip target settings,
installs pinned Snakemake 9.17.2 in a managed private venv, verifies dependencies,
and atomically saves a plain executable path in ignored `.feature-dev-snakemake`.
Future wrapper runs read the configuration automatically; explicit environment
variable overrides still take precedence. Existing unrelated environments are
refused, interrupted managed installs can be retried, and healthy repeats do
not reinstall. Nothing changes shell startup files or AFS permissions.

35 development unit tests passed. Fresh install, repeat/check, and a complete
synthetic DAG in a minimal shell without FEATURE_DEV overrides passed. The
collaborator README now gives update/setup/run commands and separate guidance
for collaborators working on their own branches. Setup evidence is recorded in
`feature_dev/TEST_RESULTS.md`. Real source export qualification is unchanged.

### 2026-11-16T17:13:29+01:00 — Verified handoff checkpoint

Verified `pwd` and `git branch --show-current` match the worktree/branch above.
Initial `git status --short` was empty. `git log -8 --format='%h %H %s'`
and `git show --stat d4f663d` verified the implementation commits and later
README update. The implementation checkpoint deliberately names an existing
implementation commit, not this documentation commit.

Implementation files since the previous recorded checkpoint were inspected via
`git diff --name-only 246eb59..HEAD`: `run_feature_dev.sh`, `.gitignore`,
`README.md`, `README_SHOWER_FEATURES.md`, the three converter/clustering files
`convertData/digi_2_features.py`, `convertData/digi_2_features_mc_muon.py`,
`convertData/scifi_plane_clustering.py`, and `feature_dev/` CLI, engine,
Snakefile, algorithms, schemas, selected-hit writer, datasets/registry, setup,
synthetic fixtures, tests and guides. Task record rename/history is preserved.
This checkpoint changes only `docs/tasks/active/shower-features.md`.

Read `docs/project/ENVIRONMENT_AND_SNAKEMAKE.md` and the LCG wrapper before
Python use. Activated `snakemake/envs/env_lcg.sh` with nounset disabled;
set `PYTHONNOUSERSITE=1` and `PYTHONDONTWRITEBYTECODE=1`. Host, directory,
interpreter/version and standard-library import preflight were recorded in the
command output. No scientific tests, detector exports or production were run.

`ls -l` verified these existing reports under the campaign output root:

- `feature_dev/tutorial-gap0/plot/report.html` (612262 bytes)
- `feature_dev/tutorial-gap1/plot/report.html` (614651 bytes)
- `feature_dev/synthetic/dataset-demo-v1/plot/report.html` (614851 bytes)

`ls -ld` found no default `feature_dev/samples` directory under that output
root. The historical `/tmp/zhibin/shower-clean-login.log` and
`/tmp/zhibin/shower-env-audit-20261115/synthetic/clean-login/plot/report.html`
were absent on this host; historical links and recorded results are retained,
not represented as revalidated. Report existence does not revalidate content.

Scheduler command: `condor_q -submitter zhibin -af ClusterId ProcId JobStatus Iwd Cmd`
returned exit 0 with no rows. No new job IDs, job logs or expected job outputs.
Use that command to refresh scheduler status before any future execution.

Handoff validation command (main checkout script, read-only, invoked from this
worktree in the activated environment):
`python3 /afs/cern.ch/work/z/zhibin/snd-ml/docs/tasks/manage_task.py --check-handoff shower-features`.

The first handoff check rejected annotations in the machine-readable timestamp
and SHA fields. Those fields were reduced to the required plain values.
Reviewed HEAD was `edc4068c1f36bd655521c4896e8517aea1f2533b` (collaborator
instructions); the implementation SHA remains the earlier setup commit.

Final handoff check: PASS (exit 0). `git diff --check`: PASS (exit 0).
Only this task record was staged for a documentation checkpoint commit; the
single Next action remains unstarted.
