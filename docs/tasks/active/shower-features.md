# Task: shower-features

## Current state

- Lifecycle: active
- Worktree: `/afs/cern.ch/work/z/zhibin/worktrees/snd-ml-shower-features`
- Branch: `task/shower-features`
- Last verified: 2026-11-16T19:27:43+01:00
- Implementation checkpoint: daf2aff43f19ba5c7595cc7c3d11e95d4956ce50
- Implemented: Automatic bounded source-catalogue preparation for mc-neutrino, testbeam (2024 MC/H4 and 2023 H8), real-data (saved no-veto/veto-10% indices) and optional mc-muon. Separate SNDSW export and LCG parity/assembly processes, Snakemake preparation DAG, deterministic source selection, event-level neutrino truth labels, validated content-addressed caches, explicit prepared-sample override, batch-profile forwarding and updated collaborator instructions. Existing plugin analysis and production clustering remain unchanged.
- Validation: 51 development unit tests, 11 clustering tests, 13 unified-feature regressions and both 10-check integration suites passed. Actual default-registry all-dataset preparation dry-run passed (8 neutrino/45 testbeam/4 real-data files), creating no result directory. Six two-event detector-source canaries passed exact entry identity and 182 baseline scalar comparisons, generated three reports plus index, and reused all caches on identical rerun. See `feature_dev/TEST_RESULTS.md` and dated evidence below.
- Remaining limitations: Full default samples and four-class neutrino coverage were not qualified; the focused two-event neutrino canary deliberately disabled four-class coverage. Optional muon detector canary blocked because none of 15 catalogue rows had both digi and geometry files present from lxplus939. Coworker access and HTCondor execution untested. Full automatic campaigns remain blocked pending sharding; shape normalization only; gap clustering is not physics-qualified. SNDSW shared builds are not immutable images, and digi fingerprints use size/mtime. Temporary canary artifacts are host-local.
- Running jobs: All local validation processes finished. `condor_q -submitter zhibin -af ClusterId ProcId JobStatus Iwd Cmd` returned exit 0 with no rows during final verification. No batch jobs submitted; no job IDs. No full default export or production campaign started.
- Outputs: Temporary canaries under `/tmp/zhibin/feature-auto-canary-20261116/all-output` (three dataset reports, all index, `.samples` plans/exports/parity receipts/logs); initial single-source canary under the sibling `output` directory. Existing campaign `/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1` and EOS products were read only. Default user runs cache samples under their own output root's `.samples/<dataset>/<signature>`.

## Goal

Unify shower feature computation and diagnostics.

## Next action

After the coworker's checkout includes implementation `daf2aff`, verify access from their own account with `./run_feature_dev.sh --dataset all --sample small --algorithm gap_clusters --run access-check --dry-run`. This coworker-account check has not been started.

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

### 2026-11-16 — Automatic catalogue preparation implementation in progress

User authorized implementation for MC neutrinos, testbeam, real data and muon
MC, plus collaborator instructions. Verified worktree/branch and clean Git
before edits. Source inspection found existing unified CSVs with EOS digi,
geometry and cutflow paths, including real-data comparison indexed selections
and H8 testbeam indices. Mixed neutrino truth must be classified per event.
Added preparation planner/workers/Snakemake rules and a separate SNDSW wrapper;
CLI integration, tests and instructions are in progress. No conversion or jobs
started. Existing historical next actions are superseded by this implementation
request; the single current Next action will be refreshed after validation.

### 2026-11-16T19:21:04+01:00 — Preparation validation milestone

35 existing unit tests and 16 preparation tests passed; the 10-check existing
dataset integration passed. Shared-catalogue `--dataset all --dry-run` passed
(8 neutrino, 45 testbeam, 4 real-data source files); no output directory was
created. Log: `/tmp/zhibin/feature-source-preview-20261116.log`.
A two-event single-neutrino-source canary passed export, parity and assembly
through the actual SNDSW/LCG preparation DAG after fixing Snakemake's precreated
output-parent handling. Log: `/tmp/zhibin/feature-auto-canary-20261116/run-retry.log`.

Now validating two selected events per representative source across the three
testbeam families, two real-data groups and upward muon MC. These are bounded
local canaries, not production or HTCondor jobs. Commands use
`./run_feature_dev.sh --dataset all` (then `--dataset mc-muon`),
`--dataset-config /tmp/zhibin/feature-auto-canary-20261116/all-datasets.json`,
`--sample small --run canary-report --algorithm gap_clusters`, and
`--output-root /tmp/zhibin/feature-auto-canary-20261116/all-output`.
Expected reports: `<output-root>/<dataset>/canary-report/plot/report.html`;
converter logs/receipts: `<output-root>/.samples/<dataset>/<signature>/`.
Controller logs: `/tmp/zhibin/feature-auto-canary-20261116/all-run.log` and
`muon-run.log`; inspect with `tail -n 40` while local processes run.
No full default export or production campaign was started.

### 2026-11-16T19:27:43+01:00 — Automatic preparation implementation and handoff

Verified on `lxplus939.cern.ch`: `pwd` and `git branch --show-current` match
this record. Implementation commit `daf2aff43f19ba5c7595cc7c3d11e95d4956ce50`
exists before this documentation checkpoint. No other checkout was edited.

Changed implementation files: `feature_dev/sources.py`, `prepare.py`,
`PrepareSnakefile`, `run_export.sh`, `cli.py`, `datasets.yaml`, `selected_hits.py`,
`tests/test_sources.py`, `run_feature_dev.sh`, and
`convertData/digi_2_features.py`. Updated `README_SHOWER_FEATURES.md`,
`feature_dev/README.md` and `feature_dev/TEST_RESULTS.md`. This separate
checkpoint updates only `docs/tasks/active/shower-features.md`; historical
records and links above are preserved.

Commands/results (LCG activation and preflight in the same shell, user-site and
bytecode disabled):

- `python3 -m unittest discover -s feature_dev/tests -p 'test_*.py'`: 51 passed.
- `python3 -m unittest discover -s convertData -p 'test_scifi_plane_clustering.py'`: 11 passed.
- `python3 -m unittest discover -s convertData -p 'test_unified_shower_features.py'`: 13 passed.
- `python3 feature_dev/tests/dataset_integration.py`: all 10 checks passed.
- `python3 feature_dev/tests/integration.py`: all 10 checks passed.
- `bash -n run_feature_dev.sh feature_dev/run_export.sh`: passed.
- `git diff --check`: passed.
- `./run_feature_dev.sh --dataset all --sample small --algorithm gap_clusters --run source-preview --dry-run --output-root /tmp/zhibin/feature-source-preview-20261116`: all three preparation DAGs passed; `test ! -e /tmp/zhibin/feature-source-preview-20261116` passed.
- Actual detector command and exact temporary registry are recorded in the preceding milestone and `feature_dev/TEST_RESULTS.md`. All three reports and index completed; identical rerun returned three `Reusing validated` messages and three `Nothing to be done` analysis DAGs. Six source parity receipts each recorded 2 events and 182 scalars. Report sizes: neutrino 575912 bytes, testbeam 733676 bytes, real-data 601766 bytes.
- Optional muon default preview and focused canary rejected missing inputs before exporting. Read-only inspection found 0/15 rows with both digi/geometry files present. No alternative physics input was substituted and no shared catalogue was modified.

Evidence logs: `/tmp/zhibin/feature-unit-20261116.log`,
`feature-clustering-20261116.log`, `feature-unified-20261116.log`,
`feature-dataset-integration-20261116.log`, `feature-integration-20261116.log`,
`feature-source-preview-20261116.log`, `feature-muon-preview-20261116.log`
(all in `/tmp/zhibin/`); canary controller logs are
`/tmp/zhibin/feature-auto-canary-20261116/all-run.log` and `reuse.log`.
Reports are under `/tmp/zhibin/feature-auto-canary-20261116/all-output/` at
`<dataset>/canary-report/plot/report.html`; combined index is
`all/canary-report/index.html`. Per-source logs/ROOT/JSONL/validation receipts
remain in `.samples/<dataset>/<signature>/`. These temporary artifacts may
not survive host changes; the tracked test report records the verified results.

No scheduler submissions, job IDs or scheduler logs. Status command remains
`condor_q -submitter zhibin -af ClusterId ProcId JobStatus Iwd Cmd`; it returned
exit 0/no rows at final verification. The single Next action is not started.

Handoff command, invoked read-only from this worktree in LCG:
`python3 /afs/cern.ch/work/z/zhibin/snd-ml/docs/tasks/manage_task.py --check-handoff shower-features`.

Final handoff validation: PASS (exit 0). Final whitespace check: PASS (exit 0).
