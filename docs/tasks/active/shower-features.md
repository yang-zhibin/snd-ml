# Task: shower-features

## Current state

- Lifecycle: active
- Worktree: `/afs/cern.ch/work/z/zhibin/worktrees/snd-ml-shower-features`
- Branch: `task/shower-features`
- Last verified: 2026-11-16T20:43:54+01:00
- Implementation checkpoint: 87b3ff61f7eb44c534aef6f96e4dcbf4998971e1
- Implemented: Automatic deterministic small samples from existing EOS catalogues for mc-neutrino, testbeam and real-data; optional mc-muon adapter. Separate SNDSW export and LCG parity/assembly, Snakemake DAG, content-addressed caches and explicit prepared-input override. Fixed standard testbeam assembly: retain capped candidates one source at a time, preserve 256 MiB guards, retain 25 events per testbeam group/geometry (100 elsewhere). Updated collaborator instructions. Example uses NumPy-style Parameters/Returns/Examples docstrings, executable expected-output examples and compact hit/cluster reference tables.
- Validation: Latest documentation revision: 7 doctest statements passed under LCG107/Python3.11.9; executable AST unchanged excluding docstrings; git diff --check passed. 55 development unit tests and 10 dataset integration checks passed after the fix. Standard EOS-backed HTCondor tests passed: 8 neutrino exports / 633 retained events with all four classes; 45 testbeam exports / 625 events across 25 groups; 4 collision-data exports / 200 events across both selections. All 57 exports passed exact identity and 182 baseline scalar comparisons. Corrected assembly reused audited unchanged exports; all three reports and identical-command cache reuse passed. Combined all-dataset dry-run and actual rerun passed with no analysis work required. Prior implementation also passed 11 clustering, 13 unified regressions and 10 workflow integration checks. Evidence: `feature_dev/TEST_RESULTS.md`. Latest read-only audit located local/SNDSW SciFi density code and identified strict versus inclusive 1 cm boundary; numerical equivalence was not tested.
- Remaining limitations: Coworker-account access, optional --prepare-profile integration, full campaigns and candidate physics acceptance are unqualified. Optional muon detector validation is blocked: none of 15 catalogue rows had both digi and geometry present on lxplus939. Full automatic preparation remains blocked pending sharding. Shape normalization only. SNDSW shared builds are not immutable images; digi fingerprints use size/mtime. Test artifacts are ignored worktree-local files. Comment-only plugin edits change its source hash, so a new run name is required; existing report evidence predates this documentation edit.
- Running jobs: None submitted for the density code-location audit. Previous standard-test jobs exited 0; coworker next action not started.
- Outputs: `feature_dev/output/standard-20261116/results/all/standard-test-fixed-20261116/index.html`; individual reports at `results/<dataset>/standard-test-fixed-20261116/plot/report.html`, samples/receipts at `results/.samples/`, summaries at `results/<dataset>-test-result.json` and `fixed-summary.json`, relative to `feature_dev/output/standard-20261116/`. Logs, harness, job IDs and reuse audits are in that validation directory. Existing EOS inputs and production products were not modified. Latest unit log: `feature_dev/output/standard-20261116/annotated-example-unit.log`.

## Goal

Unify shower feature computation and diagnostics.

## Next action

After the coworker's checkout includes implementation `87b3ff61f7eb44c534aef6f96e4dcbf4998971e1`, verify access from their own account with `./run_feature_dev.sh --dataset all --sample small --algorithm gap_clusters --run access-check --dry-run`. This coworker-account check has not been started.



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

### 2026-11-16 — Standard small-sample validation authorized

User requested testing the implementation here. Verified worktree/branch and
clean Git; setup/check and LCG analysis imports passed. The installed personal
Snakemake 9.17.2 has no cluster-generic executor plugin. The existing HTCondor
profile writes logs into the integration checkout, so it is not used here.
Instead, submit three bounded HTCondor controllers that each run the unchanged
Snakemake-backed CLI for mc-neutrino, testbeam or real-data, then repeat it and
assert export ROOT mtimes are unchanged. No environment installation/change.

Implementation: `daf2aff43f19ba5c7595cc7c3d11e95d4956ce50` (HEAD checkpoint
`20858c3`). Validation helpers/artifacts are ignored under this worktree's
`feature_dev/output/standard-20261116/`. Launch command:
`env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 feature_dev/output/standard-20261116/submit.py`.
Each worker runs `./run_feature_dev.sh --dataset DATASET --sample small
--algorithm gap_clusters --run standard-test-20261116 --output-root
/afs/cern.ch/work/z/zhibin/worktrees/snd-ml-shower-features/feature_dev/output/standard-20261116/results` twice.
Limits: 1 CPU/3 GB RAM/2 hours per controller; 8/45/4 bounded source exports.
Expected output: `results/DATASET/standard-test-20261116/plot/report.html` and
`results/DATASET-test-result.json`, with all standard required groups enabled.
Scheduler IDs will be recorded immediately after submission. Logs:
`feature_dev/output/standard-20261116/DATASET.{out,err,condor.log}`; status:
`condor_q -submitter zhibin -af ClusterId ProcId JobStatus HoldReason`.

Submitted at 2026-11-16T19:41:42+01:00: mc-neutrino `16585787.0`,
testbeam `16585788.0`, real-data `16585789.0`. Submission metadata saved in
`feature_dev/output/standard-20261116/jobs.json`. Exact status command:
`condor_q 16585787 16585788 16585789 -af ClusterId ProcId JobStatus HoldReason`.

First batch attempt exited before source preparation: LCG activation overwrote
the test harness's generic Bash `dataset` variable with `G4CHANNELING`.
The product CLI correctly rejected that invalid argument. Fixed the ignored
validation harness to use `validation_dataset`; product code unchanged. Saved
first-attempt logs and IDs as `*.attempt1.{out,err}` and `jobs-attempt1.json`.
Retrying the same three standard workflows with the corrected harness.

Retry job IDs: mc-neutrino `16585791.0`, testbeam `16585792.0`, real-data
`16585793.0`. Current status command:
`condor_q 16585791 16585792 16585793 -af ClusterId ProcId JobStatus HoldReason`.

### 2026-11-16T19:54:00+01:00 — Two standard dataset tests passed

Neutrino job `16585791.0` completed: 8 source exports passed parity; assembled
633 events (CC_nue 180, CC_numu 298, NC_nue 53, NC_numu 102), generated a
4,818,590-byte report, and repeated the command without changing export ROOT
mtimes. Collision-data job `16585793.0` completed: 4 source exports passed;
100 no-veto + 100 veto-10% events, 673,407-byte report, unchanged export mtimes
on rerun. Machine-readable evidence is in
`feature_dev/output/standard-20261116/results/{mc-neutrino,real-data}-test-result.json`.
Testbeam `16585792.0` continues; 15/45 source receipts validated at this check.

### 2026-11-16T20:11:00+01:00 — Default testbeam size failure found

Testbeam controller `16585792.0` exited 1 after all 45 source exports passed
parity. Assembly rejected the combined ~542 MiB of JSONL at the 256 MiB guard.
No sample/report was published. Fix direction: automatic capped preparation
processes one guarded source at a time, retaining only deterministic per-group
candidates; preserve the guard on the final sample and on manual/full inputs.
Reduce the default testbeam retained cap to 25 per group/geometry. Preserve the
45 validated exports for testing this assembly-only fix after verifying source
plans and converter code hashes are unchanged. No detector physics changes.

Bounded fix implemented in `feature_dev/datasets.py` and automatic assembly:
trim deterministic candidates after each source, enforce per-source/retained/
final JSONL byte guards, preserve aggregate guard for manual/full preparation.
Testbeam default retained cap is now 25 per group/geometry. All 55 development
unit tests passed, including four new aggregate-size, deterministic-equivalence,
full-mode rejection and incomplete-output regression checks.

For validation only, reused previously validated source-export directories under
new cache signatures after checking exact job and catalogue fingerprints and
verifying all export receipts. The only changed preparation code hashes are
`datasets.py` and `prepare.py`; converter code and invocation are unchanged.
Audit: `feature_dev/output/standard-20261116/assembly-reuse-audits.json`.
This is a recorded test-artifact reuse operation, not a general cache migration
feature. Each new cache has its own `reuse-audit.json`. Detector ROOT/JSONL
products were not edited. Retry uses run name `standard-test-fixed-20261116`;
old reports/logs preserved as described above. Standard batch controllers will
re-run assembly, full analysis and cache reuse on all three datasets.

Corrected standard test jobs: neutrino `16585806.0`, testbeam `16585807.0`,
collision data `16585808.0`. Status:
`condor_q 16585806 16585807 16585808 -af ClusterId ProcId JobStatus HoldReason`.

### 2026-11-16T20:23:47+01:00 — Standard EOS tests complete; bounded assembly fixed

Verified `pwd` and `git branch --show-current`: the named worktree and
`task/shower-features`. Implementation commit: `b2bbd1b018383808f74c5b4b92adb8f040757582` (separate from this
checkpoint). Changed implementation/docs: `feature_dev/datasets.py`,
`feature_dev/prepare.py`, `feature_dev/datasets.yaml`,
`feature_dev/tests/test_datasets.py`, `README_SHOWER_FEATURES.md`,
`feature_dev/README.md`, `feature_dev/TEST_RESULTS.md`. This checkpoint changes
only `docs/tasks/active/shower-features.md`; linked historical records preserved.

All corrected HTCondor controllers (16585806.0, 16585807.0, 16585808.0) exited 0;
`condor_q 16585806 16585807 16585808 -af ClusterId JobStatus` returned no jobs.
The 57 exports were parity-qualified in the original test; the assembly-only
fix reused them after verifying identical source jobs/catalogue fingerprints,
unchanged converter hashes and all receipts. Audit:
`feature_dev/output/standard-20261116/assembly-reuse-audits.json`.

Retained samples: neutrino 633 events / 92,262,386 bytes (CC_nue180,
CC_numu298, NC_nue53, NC_numu102); testbeam 625 / 78,175,626 bytes (25 events
in each of 25 groups); collision data 200 / 2,917,300 bytes (100/group).
Every report exists and each identical rerun preserved export ROOT mtimes.
Commands/results: 55 development unit tests PASS (`fix-unit.log`); dataset
integration 10 checks PASS (`fix-dataset-integration.log`); shell syntax PASS;
`git diff --check` PASS. Both logs are under the validation directory.

Combined command (also passed with `--dry-run`):
`./run_feature_dev.sh --dataset all --sample small --algorithm gap_clusters --run standard-test-fixed-20261116 --output-root "$PWD/feature_dev/output/standard-20261116/results"`.
All three samples reused and all three analysis DAGs reported nothing to do.
Combined index and report paths are in Current state; command logs are
`fixed-all-dry-run.log` and `fixed-all-reuse.log`. Summary:
`fixed-summary.json`; per-dataset assertions `results/<dataset>-test-result.json`.
Runtime: LCG107 Python3.11.9 analysis, configured Snakemake9.17.2,
existing SNDSW converter stack. Controllers run Snakemake locally within
HTCondor; optional profile forwarding itself remains untested. No production
jobs remain and no next action was started. Coworker permissions, optional
muon sources, full campaigns and physics acceptance remain unqualified.

Main launcher check:
`python3 /afs/cern.ch/work/z/zhibin/snd-ml/docs/tasks/manage_task.py --check-handoff shower-features`
returned `Handoff shower-features: PASS`; evidence:
`feature_dev/output/standard-20261116/handoff-check.log`. The main checkout was
read only. Final task-only checkpoint follows implementation `b2bbd1b`.

### 2026-11-16T20:33:14+01:00 — Annotated plugin interface example

Implementation commit `71cab7be2b6597a50b24e348cfa977092625c60e` adds inline hit/cluster field descriptions,
units, optional-field handling, member lookup and feature/schema guidance to
`feature_dev/algorithms/gap_clusters.py`. Updated `feature_dev/README.md` with
a data-reference table and linked it from `README_SHOWER_FEATURES.md`.
Verified definitions against `SelectedHitWriter`, converter geometry export and
`scifi_plane_clustering._make_cluster`/`cluster_scifi_hits`. Coordinates are
geometry endpoint A in cm, not reconstructed space points. Timing conventions
remain converter-dependent. Cluster member identity requires station, view and
channel, not channel alone. No new feature or physics definition was introduced.

Validation under LCG107/Python3.11.9 with isolated user site: runtime preflight
passed; `python3 -m unittest discover -s feature_dev/tests` passed 55 tests
(log `feature_dev/output/standard-20261116/annotated-example-unit.log`). AST
comparison against the previous committed plugin passed exactly after removing
the module docstring. `git diff --check` passed. Existing EOS reports were not
regenerated for this documentation-only change; their numerical logic remains
unchanged. Source hashing means future runs need a new run name. No jobs were
submitted and the single coworker-access next action was not started.

Main launcher read-only command
`python3 /afs/cern.ch/work/z/zhibin/snd-ml/docs/tasks/manage_task.py --check-handoff shower-features`
returned `Handoff shower-features: PASS`. Documentation implementation is
`71cab7b`; this separate task-only checkpoint records it.

### 2026-11-16T20:39:50+01:00 — Concise API reference and runnable examples

Implementation `87b3ff61f7eb44c534aef6f96e4dcbf4998971e1` changes only `feature_dev/algorithms/gap_clusters.py`
and `feature_dev/README.md`: replace long inline commentary with NumPy-style
Parameters/Returns/Examples sections; show concrete clusters and expected 0.5
fragmentation; move detailed fields to compact tables and retain coordinate
member lookup. No algorithm or schema change.

Verified pwd and branch: named shower-features worktree, task/shower-features.
LCG107 Python3.11.9 preflight passed. `doctest.testmod(gap_clusters)` passed
all 7 statements. AST comparison against prior committed plugin, stripping
module/function docstrings, passed; `git diff --check` passed. Validation ran
in-process with stdout results; no new detector outputs or jobs. Existing
output paths and detector validation limits remain in Current state. No full
rerun needed for documentation-only changes. Single next action not started.

Main launcher `--check-handoff shower-features` returned
`Handoff shower-features: PASS` using the read-only main-checkout launcher.

### 2026-11-16T20:43:54+01:00 — Locate SciFi density implementations (read-only audit)

Verified pwd/branch and clean initial Git status in the authorized worktree.
Implementation checkpoint remains `87b3ff6` (no algorithm edits). Searches:
`rg -n -i 'density|scifi.*dens|dens.*scifi' convertData feature_dev snakemake/envs`,
targeted `sed` reads, and `readlink -f` of the configured SNDSW release.
`env_sndsw_nue.sh` selects latest-event_level_info_output-release, resolving to
`/afs/cern.ch/work/z/zhibin/public/SndBuild/sw/rhel9_x86-64/sndsw/master-local14`.

Local: `convertData/digi_2_features.py:1043` hitWeightDensity (inclusive +/-1 cm,
self excluded); `:1007` fill_mycode_density (rank by x+y, return min(x,y)).
Older duplicate: `convertData/digi_2_nueAnalysis.py:247`.
SNDSW Python: resolved release `analysis/analyses/snd_analysis_2024_0mu/sciFiTools.py:72`
getSumDensity, called at local converter line 1355. It uses strict distance <1 cm
and endpoint midpoint coordinates, whereas local hit records use endpoint A.
SNDSW C++ source: `/afs/cern.ch/work/z/zhibin/public/SndBuild/sndsw/analysis/tools/sndSciFiTools.cxx:635`
hitWeightComputation and :706 hitDensityPerStation use the inclusive sliding
window corresponding to the local helper. Source inspection only; no compiled
binary provenance or event-level density parity established. No algorithms,
external checkouts, outputs or jobs changed. Only this task checkpoint updated;
linked historical records and the single next action preserved.

Main launcher read-only `--check-handoff shower-features`: PASS.
`git diff --check`: PASS. No next action started.
