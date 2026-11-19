# Task: shower-features

## Current state

- Lifecycle: active
- Worktree: `/afs/cern.ch/work/z/zhibin/worktrees/snd-ml-shower-features`
- Branch: `task/shower-features`
- Last verified: 2026-11-19T13:26:43+01:00
- Implementation checkpoint: d9b330d81a04ca74eb9d359578b4eeb013bdf455
- Implemented: Existing EOS small-sample preparation, feature plugin examples, SciFi density guide and isolated SNDSW runtime retained. Audited prepared-full-sample normalization remains available (MC target-data/parent exposure, data weight1 at effective exposure, small/testbeam shapes). New Snakemake export stage publishes <run>/features.root with unchanged baseline/candidate trees plus source_exposure and normalization TTrees, complete canonical audit JSON/checksum and metadata version/units/sample/calculation hashes. Empty sources and external reference sources are retained and labelled. Unknown shape exposure/weight uses NaN and false validity flags. Export revalidates audited accounting and compares every metadata field and every copied feature branch/identity. Exposure changes regenerate an immutable published copy while reusing calculated features. Updated both guides, normalization reference, tests and results. Local implementation commit created; not pushed.
- Validation: Remote source review found Silvia branch origin/task/silvia-shower-algorithm at d8cd5d08b4da94a0b453749573ba10150da68f72; no coworker code executed or merged. Verified pwd/task branch and clean starting checkout. On lxplus9120, LCG107/Python3.11.9, uproot5.3.11 and NumPy1.26.4: 75 development unit tests passed. Extended four-event normalization integration passed: complete shape/yield DAGs, ROOT metadata round trips, checksum, explicit unknown shape values, empty/reference sources, data effective exposure, same MC weights for both algorithms, changed-exposure refresh with unchanged calculation timestamp, previous export preserved, and deliberately corrupted exported ROOT rejected by cache checks. All 10 existing launcher integration checks passed. git diff --check passed. Prior 11 production histogram tests and earlier EOS/runtime checks remain recorded in feature_dev/TEST_RESULTS.md; production histogram code unchanged this turn.
- Remaining limitations: User prefers production-converter plugins in a separate coworker folder; this replacement design is not implemented. Current production paths are hard-coded to Zhibin and production runtime isolation still needs alignment. Silvia has catalogue read ACLs, but EOS membership/account execution and private testbeam cutflow access remain unverified.  Coworker integration is pending review: energy joins omit source_uid/eventId, plots combine geometries and show candidate-only shapes, and missing coordinates yield zero density. Her engine changes overlap local exposure implementation; remote task/shower-features remains at 4f37831 and lacks local normalization/ROOT metadata commits. Full automatic plugin campaigns and automatic exposure-audit generation remain unimplemented; prepared inputs retain 256 MiB guard. Audit producer must establish complete parent and external reference coverage; embedding the audit does not independently verify EOS coverage. Current real-data zero-active-exposure catalogues are insufficient for yields. Internal calculate/features.root intentionally has no exposure objects; consumers must use published <run>/features.root (copy with links dereferenced). Old outputs are not patched. Shape weights/exposures are unknown rather than fabricated; shape sumw2 omits covariance and yield errors omit luminosity systematics. No new EOS, coworker-account, HTCondor or physical exposure qualification; optional muon availability remains unqualified.
- Running jobs: No batch jobs or EOS conversions launched. All bounded local unit/integration test processes completed; no task test process remains running.
- Outputs: Ignored local evidence: feature_dev/output/root-exposure-20261119/{unit,integration,legacy-integration}.log. Valid four-event ROOT example: feature_dev/output/root-exposure-20261119/integration/results/mc-neutrino/yield/features.root; export/validation.json contains copied-feature and metadata checks, plot/report.html is the report. Sibling shape/ has invalid/unknown exposure metadata. Sibling changed-exposure/ was deliberately corrupted after successful refresh to test tamper detection; do not use it as a valid product. Earlier normalization evidence remains under feature_dev/output/normalization-20261119/. Durable reference: feature_dev/NORMALIZATION.md; results: feature_dev/TEST_RESULTS.md. No EOS inputs/products modified.

## Goal

Unify shower feature computation and diagnostics.

## Next action

Finalize the production-plugin implementation plan: load coworker algorithms from a separate folder in digi_2_features.py, preserve existing features, validate additional schema/branches, parameterize user-owned output/metadata paths, reuse the isolated runtime wrapper and add account-level input checks before full Snakemake/HTCondor production. The new production-plugin implementation has not been started.

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

### 2026-11-16T21:15:00+01:00 — Coworker SciFi density report

Report commit `5ea1dc67c385eb1bbc5f22b71905b2f5fe7ce100` adds `feature_dev/SCIFI_DENSITY.md` and links it from
`README_SHOWER_FEATURES.md` and `feature_dev/README.md`. Algorithm implementation
checkpoint remains `87b3ff6`; this report does not change calculation code.
The report covers source locations, station/view grouping, +/-1 cm neighbour
counts, directed-pair counting, station ranking, ROOT branch meanings, boundary
and coordinate differences, selected-input caveats and a runnable reference.

Verified pwd and branch in the authorized checkout. Targeted source reads
confirmed the C++ station wrapper uses midpoint coordinates and skips invalid
hits. Under LCG107/Python3.11.9 after runtime preflight, extracted/executed the
report's Python block: all five assertions passed (ordinary example, inclusive
boundary, strict boundary, empty input, singleton). Relative file links exist;
`git diff --check` passed. No detector jobs, new data outputs or event-level
parity tests were run. Document path above is the deliverable; runtime checks
printed to stdout. Historical links and the single next action preserved.

Main launcher's read-only `--check-handoff shower-features`: PASS.
No next action started; task-only checkpoint follows the report commit.

### 2026-11-16T21:23:05+01:00 — Commit/push request; destination approval blocked

Verified pwd, task/shower-features branch, clean Git status and existing commits
through `4a17a51` (report `5ea1dc6`). Upstream is origin/task/shower-features;
origin is `git@github.com:yang-zhibin/snd-ml.git`. Separate cern-gitlab remote
exists; no alternate destination was used.

Requested `git push origin HEAD:refs/heads/task/shower-features` followed by
remote verification. Tool execution was rejected before launch: automatic
approval review says the private worktree's commits would go to external GitHub,
the user did not specify that destination, and remote ownership was not
established. No push or remote verification executed. Explicit destination
approval is now the single next action. No code, data outputs or jobs changed.
Coworker message drafted in the response, not sent through Mattermost.

Main launcher read-only `--check-handoff shower-features`: PASS.
`git diff --check`: PASS. Push approval next action not started.

### 2026-11-16T21:25:41+01:00 — Explicit GitHub approval and successful push

User explicitly requested GitHub origin on task/shower-features. Verified pwd,
branch, clean Git status and origin URL `git@github.com:yang-zhibin/snd-ml.git`.
`git push origin HEAD:refs/heads/task/shower-features` succeeded:
`edc4068..fe87462 HEAD -> task/shower-features`.
`git ls-remote --heads origin refs/heads/task/shower-features` verified the
remote commit. Prior automatic-review blocker is resolved. This task-only
checkpoint records the result and will be pushed to the same approved branch.
Implementation remains `87b3ff6`, report `5ea1dc6`; no code or data changes,
new tests, outputs or jobs. Single next action restored to coworker access
verification and not started. Historical records preserved.

Main launcher read-only `--check-handoff shower-features`: PASS.
`git diff --check`: PASS.

### 2026-11-17T13:52:42+01:00 — svicenti source-path and access audit

Verified pwd/task/shower-features/clean initial Git state. Implementation remains
`87b3ff6`; no code changed. Traced datasets.yaml, sources.build_plan and
converter_arguments, run_export.sh, env_sndsw_nue.sh, cli.py output defaults.
LCG107/Python3.11.9 preflight on lxplus973 passed. Parsed all configured CSVs:
1866 neutrino rows, 475 testbeam, 22301 real-data and 15 optional muon rows.
Only digi_path/geo_path/cutflow_path are used as source-file inputs; existing
feature ROOT outputs in catalogue rows are not required.

AFS: fs listacl inspected all nine catalogue-path directories. All grant
system:anyuser l, so no additional parent lookup grants are needed. Three leaf
directories need svicenti read access, relative to
`/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/metadata/`:
- `manifests` (neutrino, 2024 testbeam, optional muon catalogues)
- `testbeam_2023_h8_v2` (H8 catalogue)
- `real_data_distributions/unified_real_data_distributions_v1/source_manifests`
All have only administrator/owner full rights plus anyuser lookup; no svicenti
entry. getent passwd confirms svicenti UID182858. pts membership svicenti
returned permission denied; no effective-account access claim is made.

EOS: eos attr ls sampled first/last source parent per catalogue and each input
kind (full raw results saved). Experiment paths inspected grant
`eos-experiment-sndlhc-readers:rx` and writers:rwx, including:
- MonteCarlo/Neutrinos/Genie/2024 digi+geometry
- users/zhibin/MC_neutrino cutflows
- MonteCarlo/testbeam2024, convertedData/commissioning/testbeam_24 and
  testbeam_June2023_H8 digi+geometry
- users/zhibin/snd-ml/unified_shower_features_v1/selection_indices/testbeam_data_2023_h8
- convertedData/physics/2024 and users/zhibin/snd-ml/unified_shower_features_v1/selection_indices/real_data_2024
All those paths are under `/eos/experiment/sndlhc/`. Her EOS e-group membership
was not established. ACL sampling is not an exhaustive file-level audit.

Additional likely blocker: both 2024 testbeam catalogues reference
`/eos/user/z/zhibin/TestBeam/binary_full_v1/cutflows`. EOS-user attr ls shows
sys.acl `u:187895:rx,u:1066349:rx`, empty user.acl and sys.mask700; neither ID
is svicenti. This share needs targeted read access for testbeam. Do not replace
existing ACLs or open the entire personal EOS area. No EOS grants were applied.
Optional muon path samples include nonexistent directories, consistent with
previous missing-source limitation, not merely an access issue.

Runtime: inspected public/SndBuild ancestors, module directory/BASE module,
master-local14 root/lib/python and analysis package ancestors. All sampled AFS
ACLs grant anyuser rl. No new runtime grant identified; transitive imports and
symlink targets were not exhaustively tested as svicenti. LCG/SNDSW CVMFS paths
remain shared runtime requirements. cli.py chooses the invoking user's own
work output root; she needs no write access to zhibin's outputs or worktree.

Evidence: `feature_dev/output/access-audit-20261117/catalogue-paths.json`,
`afs-acls.json`, `eos-acl-samples.json`. Commands: fs listacl, pts membership,
getent passwd, eos attr ls/ls -ld and standard-library CSV inventory using
PyYAML config. No ROOT processing, batch jobs, source-data changes, ACL changes
or coworker login attempted. Only this tracked checkpoint changes.

Main launcher read-only `--check-handoff shower-features`: PASS.
`git diff --check`: PASS. Permission changes remain unstarted.

### 2026-11-17T14:06:05+01:00 — Authorized AFS catalogue grants applied

Verified pwd/branch and clean initial Git status in the authorized worktree.
User explicitly authorized the previously shown fs setacl command. Before ACLs
had only administrators/owner rlidwka and anyuser l. Ran successfully (exit 0):

```bash
metadata=/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/metadata
fs setacl -dir "$metadata/manifests" "$metadata/testbeam_2023_h8_v2" "$metadata/real_data_distributions/unified_real_data_distributions_v1/source_manifests" -acl svicenti rl
```

Post-change fs listacl and assertions verified svicenti rl on all three targets
and preserved all existing entries. Grants permit reading files directly inside
those directories; no write/delete/admin rights added. No parent, child or EOS
ACLs changed. Evidence: `feature_dev/output/access-audit-20261117/granted-afs-acls.json`.
No code changes (implementation remains `87b3ff6`), processing jobs or data
outputs. Only tracked change is this task record. Verification ran with
LCG107/Python3.11.9 and runtime preflight. User-account end-to-end access remains
untested; no impersonation or next-action execution attempted. Historical
records preserved. Private EOS testbeam cutflows still require separate access.

Main launcher read-only `--check-handoff shower-features`: PASS.
`git diff --check`: PASS. Next action not started.

### 2026-11-17T14:22:24+01:00 — Recheck svicenti versus svicentini

User queried whether the username should be svicentini. Verified pwd/branch
and clean Git status. `getent passwd svicenti` returned Silvia Vicentini,
UID182858; `pts examine svicenti` confirmed AFS identity182858.
`getent passwd svicentini` returned no account and `pts examine svicentini`
reported user/group does not exist. All three target `fs listacl` results
still show svicenti rl and preserved previous entries. No ACL change made:
there is no verified replacement AFS identity. Coworker can confirm her login
with whoami on lxplus; effective access remains untested from her account.
No code changes, jobs or new outputs; implementation remains `87b3ff6`.
Only this checkpoint changed; historical records and single next action retained.

Main launcher read-only `--check-handoff shower-features`: PASS.
`git diff --check`: PASS. No next action started.

### 2026-11-17T15:34:16+01:00 — Reproduce and isolate LHCb Python contamination

Verified authorized pwd, task/shower-features branch, and initial dirty state:
only `feature_dev/run_export.sh` had a pre-existing isolation fix. Preserved and
validated that change without replacing it. Implementation commit
`11ce8a4639da6eb9e668e80c79fe4cab32cae417` includes the wrapper and updated
`feature_dev/README.md`, `feature_dev/TEST_RESULTS.md`. No env_sndsw_nue.sh,
shared installation, ACL or external checkout changes.

Read both Oct7 and June25 setUp.sh files: both capture sys.path from the first
python3 on PATH. Old wrapper removes LCG/miniconda3 paths but leaves LHCb.
Controlled LHCb3.12.11 PATH reproduced init_sys_streams/text_encoding failure
in the old wrapper. Clean-environment wrapper passes ROOT/SndlhcGeo imports
with pinned June25 Python3.9.25, system 3.9 stdlib and ROOT6.36.04. Additional
PYTHONHOME/PYTHONPATH plus stale setup marker probe also passed. Exit codes:
old wrapper 1; new wrapper 0; additional contaminated environment 0. Logs and
probe script paths are recorded in Current state. Shell syntax/diff checks PASS.

Tests ran sequentially on lxplus9111 under timeout and process resource limits;
no data conversion, dataset reads, jobs, installations or node-wide settings.
No attempt to execute under the coworker's identity. Fix committed locally;
not pushed this turn. Full detector parity and account-specific access remain
outside these import-only checks. Single next action not started.

Main launcher's read-only `--check-handoff shower-features`: PASS.
`git diff --check`: PASS. Separate task-only checkpoint records implementation.

### 2026-11-17T15:41:43+01:00 — Publish validated SNDSW runtime fix

Verified authorized pwd, branch task/shower-features, clean Git status and
GitHub origin URL. User requested the push. Command
`git push origin HEAD:refs/heads/task/shower-features` succeeded,
`3d305cc..11f3d34`. `git ls-remote --heads origin refs/heads/task/shower-features`
confirmed `11f3d340df41c6241a89a459928747404163a775`, including implementation
`11ce8a4639da6eb9e668e80c79fe4cab32cae417`. This task-only checkpoint will be
pushed to the same approved branch. No implementation changes, new tests,
outputs or jobs; earlier bounded import evidence remains applicable. Single
coworker-account import-check next action preserved and not started.

Main launcher's read-only `--check-handoff shower-features`: PASS.
`git diff --check`: PASS.

### 2026-11-19T10:21:50+01:00 — Audit previous full-data workflow

Verified pwd, task/shower-features and clean Git status. Implementation remains
11ce8a4; no processing code changed. User reports coworker --sample small works;
--sample full remains deliberately blocked in the development runner.

Read `snakemake/rules/unified_shower_features.smk`: production rule at line608
runs one source per job, max_events=0, and named neutrino aggregate at line826
expands all catalogue sources. Current mc_neutrino.csv has 1866 source rows
(LCG Python csv count). `convertData/digi_2_features.py` reads selected entries
one by one with GetEntry, calculates features, and writes TTree Fill.
`snakemake/metadata/configs/unified_shower_features_v1.yaml` only enables
within-file sharding for mc_muon_up and testbeam_data_2024, above 50000 input
events; shard sizes 50000 and 10000 selected entries respectively. Ranges are
[start,stop), with receipt coverage validation and optional merge/keep-shards.

`snakemake/run_remaining_unified_shower_features_htcondor.sh` selects missing
validated targets and uses HTCondor with default 20 concurrent jobs. Profile
and output paths are owner-specific; no launch/dry-run performed.
`evaluation_region_partitions/unified_feature_distributions.py:398` builds
TChain/RDataFrame over feature ROOT files and Histo1D, instead of collecting
all hits in Python. In contrast `feature_dev/engine.py:54` loads entire JSONL
sample into a list, explaining its development memory guard. Production rules
also retain older environment cleanup, so the new isolated export wrapper must
be reused/adapted when connecting full plugin processing. No full-run candidate
plugin adapter exists in the inspected converter.

Commands: targeted rg/sed source reads, catalogue csv count, runtime preflight;
all inspection completed. No new result files, detector runs, jobs or tests.
Only this checkpoint changes; preserve all linked historical records. The next
integration action is recorded but not started. Earlier advice is refined:
MC neutrino already partitions naturally by source file; a new within-file
sharding system is not a prerequisite for every source.

Main launcher read-only `--check-handoff shower-features`: PASS.
`git diff --check`: PASS. Next integration action not started.

### 2026-11-19T10:27:52+01:00 — Verify current distribution normalization

Verified pwd/branch and clean initial Git status. Source reads:
`feature_dev/engine.py:150` histograms and report text, `feature_dev/README.md`
normalization section, production distribution plot scaling. Current development
curves independently divide bin counts by their in-range sum, per algorithm /
source group / geometry. Missing sentinels excluded; underflow/overflow tracked
separately; uppermost edge included by numpy.histogram. Empty in-range curves
remain zero. No bin-width, luminosity, cross-section or veto-10% yield scaling.
Production code separately supports raw counts, unit-area bin fractions or
probability-density mode, and configured exposure scaling. This is source
inspection, not a new histogram run. No code, outputs or jobs changed;
implementation remains 11ce8a4. Single next integration action preserved and
not started. Only this checkpoint changed; historical records preserved.

Main launcher read-only `--check-handoff shower-features`: PASS.
`git diff --check`: PASS.

### 2026-11-19T10:33:44+01:00 — Trace luminosity storage and yield normalization

Verified pwd/task branch and clean initial Git status. No implementation edits;
checkpoint remains 11ce8a4. Read-only `git show`/`git grep` of task/em-shower
at 7ec4dfc594a390f3bcb8b56f8f4d7b6853b260d6; no checkout switch/edits.

Current branch: `snakemake/metadata/update_path_file.py:97` allocates recorded
run lumi to source files proportional to n_event; :317 assigns 2024 neutrino
15 fb^-1/file. Manifests preserve lumi_per_file and exposure_kind. LCG107
Python3.11 csv read confirms 976 ve files / 14640 fb^-1 and 890 vm / 13350 fb^-1
in current full catalogue (not proof all feature outputs validated).
`evaluation_region_partitions/unified_feature_distributions.py:328` deduplicates
source_uid then sums lumi_per_file; group source_subfolders applied first.
:152 scales common_exposure_19fb by target/source exposure; config target19.
Raw histogram ROOT stores sample_exposure TParameter<double> and exposure_kind
TNamed (:464); plotting summary CSV records exposure and scale. Converter itself
does not embed a lumi field in inspected output-writing code.

Real-data profile manifest builder :138 deliberately sets lumi_per_file=0,
exposure_kind=event_count_only, preserving original source_lumi_per_file.
Verified existing no-veto11151/veto11150 manifest rows: all active lumi0;
probabilities1/0.1 and weights1/10 respectively. Histogram code uses unweighted
Histo1D and does not apply those manifest weights. Real-data plot config offers
unit_area only. For yield integration preserve complete parent exposure,
including zero-selected sources; do not infer luminosity from retained counts.

Other branch: build_region_v2_inspection_input.py defines inspection_event_weight
= target/source lumi before event filtering and writes it to sndData. ROOT
TParameter<double> fields inspection_source_luminosity_fb,
inspection_target_luminosity_fb, inspection_normalization_weight are embedded;
audit/manifest CSV retains the same information. Target lumi is derived from
selected inclusive-data metadata and checked against configured expected_fb.

No detector reads, new result artifacts, batch jobs, package changes or numeric
physics validation. Only checkpoint changed. User normalization requirement
recorded; next integration action not started. Historical links preserved.

Main launcher read-only `--check-handoff shower-features`: PASS.
`git diff --check`: PASS. Next integration action not started.

### 2026-11-19T10:37:37+01:00 — ABCD normalization source audit

Read task/abcd-bkg-estimation at 8d381ad46c9bf75cfeb0c3877d2ff18dbee98656
with git show/grep; no checkout switch or external edits. Target normalization
files in its worktree have no uncommitted changes. Current worktree/branch and
clean initial status verified. No processing-code changes; checkpoint11ce8a4.

ABCD make_abcd_density_mc_hist.py source_luminosity (:185) sums complete source
metadata lumi_per_file, validates expected totals; build(:310) uses target data
exposure/source MC luminosity. Config abcd_density_neutrino_mc_v1.yaml sets
ve14640, vm13350; CC and NC of each flavour share that denominator. No-veto
regions target19.780680927948, veto regions1.960319252235 fb^-1. Raw histogram
is cloned and Scale(weight) applied; sumw2 plus weighted closure retained.
ROOT metadata and summary CSV store source_mc_luminosity, target_data_exposure,
normalization_weight, raw_events, sum_weights, sum_weights_squared. Data builder
checks summed catalogue lumi_per_partition against frozen region exposure and
uses observed counts. Safe-tight/rejection code confirms data weight1.

For integration: use parent-source exposure, retain region data exposure,
fill unweighted data and luminosity-scaled MC, preserve raw/weighted/sumw2 and
provenance. No blanket veto x10 under the ABCD effective-exposure convention;
if explicitly scaling both data and MC to a common target, data scale is
Ltarget/Leffective and MC scale Ltarget/LMC. Do not also apply sampling factor
already absorbed in effective exposure. These ABCD exposures are not assumed
to describe shower branch's different saved 10% selection. Testbeam remains
shape-only per user. Code inspection only, no new ROOT/data runs, output
artifacts or jobs; single integration next action preserved, not started.

Main launcher read-only `--check-handoff shower-features`: PASS.
`git diff --check`: PASS. No next action started.

### 2026-11-19T10:54:34+01:00 — Audited prepared-sample yield normalization

Implementation: `5ea499c9c5345467d33e94ba1c00220486704256` (existing code commit,
not this record). Verified pwd and branch; starting checkout was clean. Only this
worktree was edited. No push, EOS conversion, full processing or batch submission.

Changed implementation files: feature_dev/exposure.py, cli.py, engine.py,
datasets.py, datasets.yaml, sources.py; evaluation_region_partitions/normalization.py
and unified_feature_distributions.py. Tests: feature_dev/tests/test_exposure.py
and normalization_integration.py. Docs: feature_dev/NORMALIZATION.md, README.md,
TEST_RESULTS.md and README_SHOWER_FEATURES.md. This checkpoint updates only the
task record after the implementation commit.

Validation environment: source snakemake/envs/env_lcg.sh with nounset disabled,
then strict shell; PYTHONNOUSERSITE=1 and PYTHONDONTWRITEBYTECODE=1. Preflight
recorded lxplus9120, expected worktree, LCG107 python3 path, Python3.11.9,
NumPy1.26.4/uproot5.3.11 imports. Commands/results:

- `python3 -m unittest discover -s feature_dev/tests -p 'test_*.py'`: 70 passed.
- `python3 -m unittest discover -s evaluation_region_partitions -p 'test_unified_feature_distributions*.py'`: 11 passed, including tiny ROOT fixtures.
- `python3 feature_dev/tests/normalization_integration.py feature_dev/output/normalization-20261119/verified-integration`: passed both full prepared shape/yield DAGs and cache/error/report assertions.
- `python3 feature_dev/tests/integration.py`: 10 checks passed; FEATURE_DEV_SNAKEMAKE selected the existing shower-snakemake-9.17.2-lcg107 environment.
- `git diff --check`: passed.

Evidence and output paths are listed in Current state and TEST_RESULTS.md.
The new ABCD fixture initially had an incorrect test denominator (14630); corrected
the fixture to 14635+5=14640 and reran all 70 unit tests successfully. No algorithm
fix was needed. Runtime software checks do not establish EOS parent completeness,
external data coverage, luminosity systematics or campaign physics acceptance.
The next action remains unstarted.

Main-launcher handoff validation:
`python3 /afs/cern.ch/work/z/zhibin/snd-ml/docs/tasks/manage_task.py --check-handoff shower-features`
returned `Handoff shower-features: PASS`; log:
`feature_dev/output/normalization-20261119/handoff.log`. Final diff check passed.

### 2026-11-19T11:00:12+01:00 — Luminosity storage clarification

Verified pwd, task/shower-features branch and clean initial checkout. Implementation
remains 5ea499c9c5345467d33e94ba1c00220486704256. Read-only `rg`, `sed`, and
`head -2` inspections confirmed mc_neutrino.csv carries lumi_per_file=15.0 in its
first row, plus source_metadata_file pointing to MC_neutrino_2024_ve_metadata.csv.
The real-data no-veto catalogue preserves source_lumi_per_file (first row
0.0016445188131982) while lumi_per_file=0 and exposure_kind=event_count_only.
update_path_file.py allocates run lumi proportional to original n_event.
Selected-hit writer has no per-event luminosity/weight field; new exposure audit
has source luminosity_fb and run manifest plan.exposure_audit/plan.normalization;
histograms.json stores per-curve scales/exposures. These are exposure and event
weights, not a luminosity to sum once per selected event. No new output products
or jobs; only this checkpoint changed. Existing logs/outputs and single Next
action remain valid; that action was not started.

Main launcher `--check-handoff shower-features`: PASS. `git diff --check`: PASS.

### 2026-11-19T11:01:36+01:00 — ROOT luminosity storage clarification

Read-only source inspection after verifying pwd, branch and clean Git status:
feature_dev/engine.py writes baseline/candidate feature trees without luminosity
metadata; exposure remains in JSON run/histogram records. The production
unified_feature_distributions.py writer stores sample_exposure as a ROOT
TParameter<double> and exposure_kind as TNamed. No source changes, output
regeneration or jobs. Implementation SHA, validation limits, output paths and
single Next action remain unchanged; no follow-up implementation started.

Main launcher `--check-handoff shower-features`: PASS; `git diff --check`: PASS.

### 2026-11-19T11:22:49+01:00 — Published ROOT exposure metadata

Implementation: d9b330d81a04ca74eb9d359578b4eeb013bdf455 (existing code commit,
not this checkpoint). Only the authorized worktree/branch was edited. New files:
feature_dev/root_metadata.py and tests/test_root_metadata.py. Changed code:
feature_dev/engine.py, cli.py, Snakefile and tests/normalization_integration.py.
Updated docs: feature_dev/NORMALIZATION.md, README.md, TEST_RESULTS.md, and
README_SHOWER_FEATURES.md. Published <run>/features.root links to export/features.root;
internal calculate/features.root remains normalization-independent. ROOT stores
source/normalization trees, explicit validity flags, audit/checksum and provenance.
Source metadata includes empty and external reference sources. No per-event
luminosity duplication. Every copied feature value and identity was compared.

Verification: shell activated snakemake/envs/env_lcg.sh with nounset off before
strict mode, PYTHONNOUSERSITE=1/PYTHONDONTWRITEBYTECODE=1; preflight recorded
lxplus9120, expected pwd, LCG107 python3, Python3.11.9, uproot5.3.11/NumPy1.26.4.
Commands/results:

- python3 -m unittest discover -s feature_dev/tests -p 'test_*.py': 75 passed.
- python3 feature_dev/tests/normalization_integration.py feature_dev/output/root-exposure-20261119/integration: passed complete DAGs, metadata, exposure refresh/cache reuse, old export preservation and tamper rejection.
- FEATURE_DEV_SNAKEMAKE=/afs/cern.ch/work/z/zhibin/venvs/shower-snakemake-9.17.2-lcg107/bin/snakemake python3 feature_dev/tests/integration.py: all 10 checks passed.
- git diff --check: passed.

Logs and ROOT output paths are listed in Current state; the changed-exposure
fixture is intentionally corrupted for the negative test. No batch jobs, EOS
conversions or push. No actual parent exposure qualification; validation used
bounded artificial inputs. All local tests completed. Existing historical links
are preserved and the single Next action remains unstarted.

Main launcher `python3 /afs/cern.ch/work/z/zhibin/snd-ml/docs/tasks/manage_task.py --check-handoff shower-features`: PASS.
Evidence: feature_dev/output/root-exposure-20261119/handoff.log. Final diff check passed.

### 2026-11-19T12:05:32+01:00 — Review coworker GitHub push

Verified pwd, task/shower-features and clean initial Git state. `git fetch origin`
succeeded and discovered origin/task/silvia-shower-algorithm. Latest commit:
d8cd5d08b4da94a0b453749573ba10150da68f72, Silvia Vicentini,
2026-11-19T11:10:15+01:00, Add energy features and improved report plots. Earlier
feature commit 5c01ce4 adds my_features.py/schema; 22a3609 and merge d6408c5
record environment changes and adoption of the SNDSW fix. `git log`, `git diff`
and `git show` inspected these changes without switching branches or editing
other checkouts. Compared with origin/task/shower-features, changed files are
feature_dev/algorithms/my_features.py, schemas/my_features.yaml, engine.py, and
snakemake/envs/env_sndsw_nue.sh (comment only).

Features include leading QDC fraction and top-two vertical/horizontal cluster
density/fraction/station. Density uses ordered hit pairs within inclusive1cm.
Report adds candidate class overlays, neutrino energy and energy-density 2D plots.
Source-review concerns: energy joins use (runId,eventIndex), omitting eventId and
source_uid; geometry strata are merged; baseline curves are hidden; missing
coordinates are dropped, potentially reporting zero density rather than missing.
No new tests in her diff. These are source findings, not reproduced runtime failures.

Remote task/shower-features is still 4f37831; pre-checkpoint HEAD was ten local
commits ahead of it, and diverged 10-local/4-remote versus Silvia's branch. Local
normalization/export changes overlap engine.py and are not in her branch. No merge,
cherry-pick, push, EOS processing or jobs. Implementation checkpoint remains
d9b330d81a04ca74eb9d359578b4eeb013bdf455; previous output paths/tests unchanged.
Only this task record changed. Single Next action preserved and not started.

Main launcher `--check-handoff shower-features`: PASS; `git diff --check`: PASS.

### 2026-11-19T12:10:31+01:00 — Coworker interface explanation

Verified pwd/task branch and clean initial Git status. Read feature_dev/README.md,
engine.py evaluate/validate_values, and algorithms/gap_clusters.py. Interface is
per-event cluster_hits(hits, parameters), calculate_features(hits, clusters,
parameters), optional validate_algorithm; exact schema keys and finite Python
int/float outputs, read-only selected hits, optional cm/ns coordinates. Plugin
receives no labels, truth/energy, identity, geometry metadata or luminosity.
Existing cluster layout exposes planes and orientation-ranked clusters; ranking
is descending QDC. Runner owns dataset selection, baseline comparison, validation,
normalization and outputs. Silvia's plugin/schema are on her remote branch and
local exposure/ROOT publication additions remain unpublished. No source changes,
tests, jobs or new artifacts; implementation SHA, output paths and validation
limits remain unchanged. Next action retained and not started. Only task record
updated for continuity.

Main launcher `--check-handoff shower-features`: PASS; `git diff --check`: PASS.

### 2026-11-19T12:14:35+01:00 — Whole-branch workflow explanation

Verified pwd/task branch and clean initial Git status. Read README_SHOWER_FEATURES.md,
docs/project/OVERVIEW.md, feature_dev/PrepareSnakefile, prepare.py and Snakefile.
Confirmed two distinct flows: existing production conversion/validation/histogram
bundles and bounded plugin development. Development resolves EOS catalogues, exports
selected hits in SNDSW, checks identities/baseline parity in LCG, assembles a fixed
sample, calculates baseline/candidate, validates, publishes exposure-bearing ROOT
and creates histograms/report. Cached exports and feature calculations are reused
when their inputs permit it. Production still uses its baseline converter; arbitrary
plugin promotion and automatic full-plugin campaigns are not implemented. Latest
local normalization/ROOT metadata additions remain unpublished; Silvia branch is
not merged. Read-only source explanation only: no implementation changes, new
validation jobs, tests or outputs; prior SHA/limits/output paths remain authoritative.
Single Next action preserved and not started. Only this checkpoint changed.

Main launcher `--check-handoff shower-features`: PASS; `git diff --check`: PASS.

### 2026-11-19T12:31:06+01:00 — Add setup to collaborator workflow explanation

Verified pwd/task branch and clean initial status. Read setup and plugin instructions
in README_SHOWER_FEATURES.md and feature_dev/README.md. Confirmed Bash/lxplus
setup via unset FEATURE_DEV_PYTHON FEATURE_DEV_SNAKEMAKE then run_feature_dev.sh
setup; setup installs/configures the coworker's own Snakemake environment.
Synthetic smoke command requires no physics data; real dataset commands resolve
existing catalogues and require account access to inputs and SNDSW. Explanation
combines setup, two-file plugin/schema contract, hit units, dataset execution,
validation/caching/outputs, and contribution workflow. Distinguishes local
unpublished exposure/ROOT updates from her current branch. No setup command
executed, no package installs, source edits, tests, jobs or output artifacts.
Implementation SHA, validation limits, existing outputs and single Next action
remain unchanged; that follow-up was not started. Only this task record changed.

Main launcher `--check-handoff shower-features`: PASS; `git diff --check`: PASS.

### 2026-11-19T12:35:15+01:00 — Locate production feature calculation

Verified pwd/task branch and clean initial status. Read-only rg/sed/git-file
inspection traced unified_shower_features.smk rule unified_shower_feature_v1
(line608) to convertData/digi_2_features.py (MC_muon compatibility wrapper delegates
to same converter). main calls process_hits (definition1222), then new_tree.Fill
(line1858) writes sndData. Cluster wrapper fill_scifi_cluster_features (478) calls
scifi_plane_clustering.cluster_feature_payload (227), which calls cluster_scifi_hits
(117) and payload_from_analysis (230). Other production features are computed
inside process_hits via count/QDC, position, topology, time and density helpers;
external getSumDensity called at1355, local hitWeightDensity/fill_mycode_density
are separate functions. feature_dev/algorithms/baseline.py uses the same cluster
core; arbitrary coworker plugins are not invoked by the production converter.
No source edits, tests, processing jobs or new artifacts; implementation SHA,
validation limits, outputs and single Next action unchanged/unstarted. Only
this continuity record was updated.

Main launcher `--check-handoff shower-features`: PASS; `git diff --check`: PASS.

### 2026-11-19T13:26:43+01:00 — Coworker production execution/access/output assessment

Verified pwd/task branch and clean initial status. Read unified_shower_features.smk
and unified_shower_features_v1.yaml: production feature root is hard-coded under
/eos/experiment/sndlhc/users/zhibin/snd-ml/unified_shower_features_v1/features;
config consistency checks enforce it. Metadata/local-smoke/shard paths are also
Zhibin-owned. Rules rebuild manifests from original metadata and use an older
SNDSW shell setup, not the validated clean development export wrapper. Thus a
new plugin CLI is only proposed, not usable by Silvia yet.

Live permission checks: getent confirms svicenti UID182858; fs listacl confirms
svicenti rl on all three shared catalogue directories; public/SndBuild grants
anyuser rl; /afs/cern.ch/work/s/svicenti exists and grants owner rlidwka.
`eos root://eosuser.cern.ch attr ls /eos/user/z/zhibin/TestBeam/binary_full_v1/cutflows`
still shows sys.acl u:187895:rx,u:1066349:rx, user.acl empty, mask700; no explicit
svicenti grant. This is not an effective-account access test. Experiment EOS
reader membership and all input dependencies must be checked from her account.
Proposed outputs: per-user campaign directory, small outputs/logs/reports in her
AFS work; full ROOT products in an explicitly selected writable EOS directory.
No write access to Zhibin's outputs should be required by the new design.

No files outside this checkout edited, ACL changes, source edits, setup, tests,
conversion or batch submission. No output products generated; implementation
remains d9b330d81a04ca74eb9d359578b4eeb013bdf455. Existing validation and output
paths retained. Next action updated to user's production-plugin direction and
not started. Only this task record changed.

Main launcher `--check-handoff shower-features`: PASS; `git diff --check`: PASS.
