# Task: shower-features

## Current state

- Lifecycle: active
- Worktree: `/afs/cern.ch/work/z/zhibin/worktrees/snd-ml-shower-features`
- Branch: `task/shower-features`
- Last verified: 2026-11-26T17:31:48+01:00
- Implementation checkpoint: a454ef3542fbde1c8c4988c9fe6d22f031f67722
- Implemented: Separate data/MC signed SciFi timing flags, defaults[-0.5,1.2] unchanged; process-local SNDSW and calibrated-muon selection; feature/Hit3D timing metadata; production and original Snakemake forwarding; campaign/reuse protection; shard consistency checks; README/data-format instructions.15 implementation files committed. Shared SNDSW and upstream selections unchanged. Local only, not pushed. Added standalone feature_dev/SCIFI_CLOCK_CYCLE.md with definitions, algorithm, exceptions, code map, usage and provenance; linked from both production/coworker READMEs.
- Validation: Documentation review: relative link targets and balanced code fences pass; git diff --check and main launcher handoff pass. No new runtime tests needed for this prose-only change. Prior implementation evidence:71 tests pass (53production,13converter,4shard,1actual-SNDSW synthetic-hit). MC-muon/Hit3D CLI smoke checks pass; original21-rule Snakemake file parses/compiles. Three HTCondor previews each7/7steps,3sources x20events;18jobs exit0. All source/stage checksums and metadata pass. Event identities/retained-hit containment verified; MC arrays unchanged across windows. Default20MCevents x451branches exactly equal pre-change output. Each preview26PDFs/78histograms; all configured PDF paths verified, six pdfinfo checks pass. See feature_dev/TEST_RESULTS.md and comparison.json. Handoff/whitespace checks run before final checkpoint commit.
- Remaining limitations: Small owner-account preview, not full production/physics qualification. H8sample has only1selected hit;2024data299/300/300hits at1.2/1.8/2.0, addedhit zeroQDC. This does not choose the best window. No adaptive method,7pe change, upstream recut or real Hit3D/muon-MC batch conversion. Historical6.23768058ns vs6.25ns periods and path-based legacy25ns exemption remain. Runtime provenance recorded; no shared environment changes.
- Running jobs: None. All18clusters exited0, targeted condor_q empty; cluster IDs/tokens/log paths in feature_dev/output/timing-selection-20261126/jobs.json. Controller exit0. No full campaign or pending test job.
- Outputs: /afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/testbeam/timing{12,18,20}-20261126/ (9source ROOTs,234histograms,78PDFs total). Evidence feature_dev/output/timing-selection-20261126/{production-tests.log,converter-tests.log,shard-tests.log,sndsw-tests.log,dry-run.log,muon-help.log,hit3d-help.log,batch-*.log,default-parity.json,source-comparison.json,comparison.json,verify_comparison.py,jobs.json,final-queue.txt,pdfinfo-*.txt,testbeam.json,catalogue-*.csv}. Current documentation changes: feature_dev/SCIFI_CLOCK_CYCLE.md, feature_production/README.md, README_SHOWER_FEATURES.md, this task record. Prior validation documentation: feature_dev/TEST_RESULTS.md. Historical records preserved.

## Goal

Unify shower feature computation and diagnostics.

## Next action

Await the user's request to publish the tested timing-window implementation and clock-cycle guide; do not change defaults or launch a full campaign from this checkpoint.

## Checkpoint history

### 2026-11-26T17:29:56+01:00 — Timing code reference review

Verified pwd, branch task/shower-features, clean initial git status and HEAD 2087659; implementation remains a454ef3542fbde1c8c4988c9fe6d22f031f67722 (not pushed). Read explicit installed master-local14 sciFiTools.py: selectHits line9, clock conversion line22, peak centre line41, inclusive cuts lines57/59. Read local scifi_timing.configure and select_calibrated_scifi_hits plus converter call sites. Commands: pwd, git branch --show-current, git status --short, git log -2 --oneline, bounded rg and sed; all passed. Only this task record changed. No tests or production jobs launched; prior validation limits, outputs and job evidence above remain unchanged. Main launcher --check-handoff run for this documentation-only review. Next action unchanged; no publication or campaign started.


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

### 2026-11-19T13:29:01+01:00 — Production-plugin implementation plan

Verified pwd/task branch and clean initial Git status. Read converter branch
creation/event loop, validate_unified_shower_feature_file.py, source CLI builder,
unified production rules and the clean feature_dev/run_export.sh wrapper. User
requests a plan only; no plugin implementation, conversion, tests or jobs started.

Planned sequence:
1. Separate feature_plugins/silvia folder containing algorithm/schema/tests.
Preserve cluster_hits(hits,parameters), calculate_features(hits,clusters,parameters)
and optional validator. Loader supplies standardized copied selected hits with
x_cm/y_cm/z_cm/time_ns and tracks helper imports. Resolve YAML in launcher to
JSON for SNDSW, without importing the LCG development engine into the converter.
Explicit export mapping/prefix prevents existing baseline keys returned by her
plugin (including overlapping station fields) from overwriting production.
2. Add converter plugin arguments, load once/create typed branches before event
loop, initialize sentinel buffers every event, run on final selected SciFi hits
inside process_hits, fill extra outputs before sndData.Fill. Preserve no-plugin
behavior and original branch values; extend MC_muon compatibility arguments.
3. Extend validator/shard merger for extra schema, finite values, identity,
plugin source/config/schema hashes, and consistent metadata across shards.
Embed source exposure and plugin provenance in production ROOT using the
existing metadata conventions, including unknown validity for capped samples;
merge exposure per source without double counting. Retain zero-selected sources
in campaign coverage and refuse yields without a complete validated audit.
4. Add one production launcher with dataset/plugin/small-full/run/output/profile
options and named Snakemake targets; parameterize manifest, feature, shard, log
and plot roots instead of hard-coded Zhibin roots. Read frozen shared input
catalogues, create own campaign snapshot, hash plugin/schema/helpers/config,
reuse the clean runtime wrapper, record commands/job IDs/logs. Small=bounded
local production test; full=per-source/sharded HTCondor production. Keep old
development entry point until parity and migration are validated.
5. Extend existing distribution configuration to plugin outputs; use energy
already in sndData instead of unsafe external joins. Preserve source/geometry
strata, use shapes for capped/testbeam samples and audited luminosity yields
for eligible full MC/data.
6. Add per-account preflight for catalogue/digi/geometry/cutflow/runtime/output
and batch credentials. Existing catalogue grants confirmed, private testbeam
cutflow and experiment EOS effective access remain unresolved; no ACL edits.
Validate synthetic fixtures first, then one bounded real source in the matching
environment, and only subsequently qualify batch execution. No full campaign
as part of initial implementation validation.

Acceptance: inherited baseline returns cannot replace production branches;
plugin results match old interface for identical hits/parameters; original
identity/order and baseline values unchanged; missing coordinate behavior is
explicit (sentinel/error, not false zero); ROOT branches/provenance/exposure
round-trip; deterministic reruns and config invalidation; shard metadata/exposure
deduplication; small local dry-run and full batch dry-run; output isolation and
clear missing-input failures. Final account access qualification requires Silvia
to run preflight with her credentials. Proposed CLI/paths are not available yet.

Only task record changed. Implementation SHA remains
d9b330d81a04ca74eb9d359578b4eeb013bdf455; prior validation limits/output paths
retained. Single Next action updated to first planned milestone, not executed.

Main launcher `--check-handoff shower-features`: PASS; `git diff --check`: PASS.

### 2026-11-19T13:51:55+01:00 — Production plugin implementation and bounded validation

Implementation commit: `799e23d0c082f441aea8565ddc228e2462042fe3` (created before
this checkpoint). Verified worktree/branch; only authorized checkout edited.
Changed files: new convertData/feature_plugin.py; converter and muon wrapper;
production validator and shard merger; feature_dev/exposure.py; new
feature_plugins/ (Silvia implementation/schema/interface guide),
feature_production/ (planner, worker, distributions, Snakefile, tests and guide),
run_shower_features.sh; README_SHOWER_FEATURES.md; feature_dev README,
NORMALIZATION and TEST_RESULTS. Full exact list: `git show --stat 799e23d`.

Commands/results (LCG activation in same shell, user site/bytecode disabled):

- `python3 -m unittest discover -s feature_production/tests -p 'test_*.py'`: 20 passed.
- `python3 -m unittest discover -s feature_dev/tests -p 'test_*.py'`: 75 passed.
- `python3 -m unittest discover -s convertData -p 'test_unified_shower_features.py'`: 13 passed.
- `python3 -m unittest discover -s convertData -p 'test_validate_unified_shower_feature_shards.py'`: 3 passed.
- `./run_shower_features.sh --dataset mc-neutrino --plugin silvia --sample small --files-per-group 1 --events-per-file 2 --run smoke --output-root "$PWD/feature_dev/output/production-plugin-20261119/results"`: passed conversion, validation and report. Identical repeat: nothing to do. Final wrapper-only change forwards setup arguments; checked with setup --check, no physics calculation changed afterward.
- `./run_shower_features.sh --dataset mc-neutrino --plugin silvia --sample full --normalization shape --run full-plan --dry-run --output-root "$PWD/feature_dev/output/production-plugin-20261119/results"`: 1866 per-source jobs planned, no execution/products.
- Original coworker calculation on same four exported events: all 13 outputs match. No-plugin source0 conversion: all 438 original branches and two entry identities match (parity.json).
- `./run_shower_features.sh setup --check`: passed, existing private Snakemake healthy; no installation.
- `bash -n run_shower_features.sh`, `git diff --check`: passed.
- `condor_q -submitter zhibin -af ClusterId ProcId JobStatus Iwd Cmd`: exit1, Collector has no record of schedd/submitter. No submitted task job IDs. Use this command to refresh scheduler evidence; global queue state unverified.

Outputs/logs and validation limits are replaced in Current state above, with
full evidence also in feature_dev/TEST_RESULTS.md. No full campaign started.
Main launcher handoff check is run after this record update; result recorded below.

Main launcher `--check-handoff shower-features`: PASS (exit0).
`git diff --check`: PASS. Single Next action remains unstarted.

### 2026-11-19T14:26:27+01:00 — PDF production report

Implementation commit `864b937251f30a86645173b1285f7cc287f76ee8` predates this
record. Changed feature_production/distributions.py (PdfPages renderer), cli.py
(printed PDF path), README.md (output instructions), feature_dev/TEST_RESULTS.md.
Only authorized worktree on task/shower-features edited. Starting Git clean.

Validation: LCG activation and preflight in same shell, user-site/bytecode
excluded. Called distributions.render with the existing smoke histograms.json
and plan.json, output feature_dev/output/production-pdf-20261119/. No old run
modified. `pdfinfo .../report.pdf`: 61 pages, matching expected count 61;
`pdftoppm -f 2 -singlefile -scale-to 1200 -png .../report.pdf .../page2`:
passed, page visually inspected. `python3 -m unittest discover -s
feature_production/tests -p 'test_*.py'`: 20 passed, unit.log in evidence directory.
`git diff --check`: passed. No new physics validation or HTCondor qualification;
previous limitations retained. No job IDs; no submitted jobs. Single current
Next action remains account/batch qualification and was not started.

Main launcher `--check-handoff shower-features`: PASS; `git diff --check`: PASS.

### 2026-11-19T14:35:52+01:00 — Small MC run in the default output directory

Existing implementation SHA remains `864b937251f30a86645173b1285f7cc287f76ee8`;
executed checkout HEAD `d369eac`. No implementation edits. Verified worktree,
branch and clean initial state. User authorized a small MC neutrino test.

Command: `./run_shower_features.sh --dataset mc-neutrino --plugin silvia --sample small --files-per-group 1 --events-per-file 2 --run mc-small-20261119-143325`.
Exit0, all four tasks completed at 14:34:50 local time. Two source jobs ran
sequentially locally; no batch job IDs. Default output (no output-root override):
`/afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/mc-neutrino/mc-small-20261119-143325`.
PDF: `plots/report.pdf`; ROOT files: `sources/{0,1}/features.root`;
commands/runtime/conversion/validation logs: `logs/source-{0,1}.log`.

LCG107 Python3.11.9 preflight on lxplus9102. Post-run uproot inspection and
verify_receipt: two entries and 13 plugin branches per source; identity and
plugin/exposure metadata present; validation_passed true; source and plot
checksums match. Baseline parity receipts record two events/182 scalars each.
`pdfinfo plots/report.pdf`: valid 61-page PDF; no HTML produced.
Controller log, verification JSON and PDF info saved in
`feature_dev/output/production-small-20261119/` (ignored evidence).
No full run, physical yield qualification, other dataset or coworker-account
execution. Existing limits retained. Single Next action remains unchanged and
unstarted. Only this task checkpoint changed in Git.

Main launcher `--check-handoff shower-features`: PASS; `git diff --check`: PASS.

### 2026-11-19T14:38:37+01:00 — Plotting backend mismatch confirmed

User questioned the PDF style. Verified pwd/branch and clean Git; inspected
feature_production/Snakefile, distributions.py, existing unified distribution
rules/script and plot_comparison.py. New renderer directly uses Matplotlib
stairs and PdfPages. It does not call either existing ROOT plotting script.
Previous statements implying full plotting-backend reuse were inaccurate;
converter and exposure helper reuse do not imply renderer reuse. No code,
products or jobs changed. Implementation SHA remains 864b937251f30a86645173b1285f7cc287f76ee8.
Only task record updated; previous outputs/history preserved. Current Next
action replaced with the plotting integration correction, not executed.

Main launcher `--check-handoff shower-features`: PASS; `git diff --check`: PASS.

### 2026-11-19T14:40:36+01:00 — Exact user plotting reference traced

Read-only inspection: user AFS home/work output alias resolves to the work AFS
em_shower_neutrino_v1/plots directory. snakemake/rules/em_shower_neutrino_evaluation.smk
constructs both requested directory hierarchies from partition/region and
histogram/evaluation versions. 1D uses make_group_hist.py and plot_comparison.py;
2D uses make_group_hist2d.py and plot_group_hist2d.py. Read relevant functions,
configuration hist_features_v1.yaml (em_shower_hist_features_energy_v2),
evaluation_options_v1.yaml (em_shower_evaluation_scifi35_energy_v2), regions_v1.yaml.

Verified existing count_scifi/none/none/summary.csv and
2d/.../total_qdc_vs_hits/scifi35/summary.csv: four neutrino classes; count_scifi
1D integrals normalized to one. Config selects count_scifi > 35, fixed class
colors, per-feature bins/cuts/log axes, overflow folding for 1D. Actual 1D
renderer draws E1 error bars (despite YAML HIST), mean/std legend, and automatic
ratio for exactly two curves; configuration flags alone do not describe it.
2D renderer gives a 2x2 unit-integral COLZ/log-z density page and an overlaid
ProfileX page (profiles made from binned TH2). Current 2D is shape-only.
No products edited/generated, jobs submitted or tests run. Existing
implementation SHA 864b937251f30a86645173b1285f7cc287f76ee8 unchanged.
Only task record updated. Exact reference replaces previous generic backend
proposal; Next action remains unstarted. Previous outputs/limits/history retained.

Main launcher `--check-handoff shower-features`: PASS; `git diff --check`: PASS.

### 2026-11-19T14:49:38+01:00 — ROOT plotting integration implementation plan

Scope: user requests an implementation plan for the previously proposed output
structure, own EM shower plotting backend, automatic ranges when unspecified,
and documented editing locations. No code changes yet. Existing implementation
SHA remains 864b937251f30a86645173b1285f7cc287f76ee8. Verified pwd, branch and
clean Git; read backend contracts and current production Snakefile.

1. Configuration: add feature_production/config/histograms.yaml and
plotting.yaml; CLI --hist-config/--plot-config overrides. Snapshot requested and
resolved YAML under each run/config. Defaults control bins, optional ranges,
expressions, missing sentinels, log axes, selection names, class labels/colors,
normalization and explicitly chosen 2D pairs. Default selection is converter
selected events; scifi35 explicitly selects count_scifi > 35. Preserve original
EM shower config as reference without silently imposing its selection. Plugin
schema supplies feature names/types/units/missing values; historical schema
histogram ranges do not silently defeat automatic ranges in the new plot config.
User plot config takes precedence; retain explicit reference ranges where set.

2. Range resolution: bounded-memory scan of validated source ROOTs, applying the
same selection, truth-group filter, expressions, finite-value and sentinel cuts
as filling. One common min/max across comparison classes per feature/selection/
geometry, including both compared algorithms when requested. Default 50 bins;
integer-aware edges, deterministic constant-value padding; empty fallback [0,1]
with no-valid-data annotation. Pad upper edge to include observed max in ROOT
normal bins. For 2D determine each missing axis range from jointly valid pairs.
No implicit quantile clipping. Reject invalid explicit ranges/log-axis bounds;
record effective edges, accepted counts and fallback reason in resolved config.

3. ROOT histogram integration: call/refactor shared make_group_hist.py and
make_group_hist2d.py entry points; introduce explicit input/output adapters rather
than cloning their filling logic. Current sources contain mixed CC/NC truth:
add explicit per-group event filters and preserve family/source_uid/geometry
identity; do not assign an entire source to one class. Preserve empty groups,
parent coverage and audited exposure without summing duplicated source rows.
Make existing CLI/config behavior backward compatible. Produce hist/metadata
and hist2d/profile_x compatible with the existing plotters, with explicit
missing-value exclusions and under/overflow policy. Profiles retain existing
binned-TH2 ProfileX semantics and errors.

4. Rendering: invoke plot_comparison.py shared drawing functions for 1D and
plot_group_hist2d.py for 2D; extend explicit input/output path support so new
campaigns need not imitate legacy directory layouts. Preserve class colors,
markers/error bars, legends, annotations and per-feature log settings. Honor
ratio configuration explicitly (current renderer auto-enables it for two curves).
Generalize 2D panel layout/group styles for testbeam/real-data; preserve 2x2 for
four neutrino classes. Empty classes produce honest empty annotations. No second
Matplotlib implementation. Retire the new standalone PDF report renderer.

5. Normalization: keep audited coverage validation, including zero-selected
parents, upstream of plots. Feed validated scales into shared plotters: shape
in-range normalization with configured folding, MC target/parent luminosity,
data weight1 at effective exposure, no extra sampling factor. Small/testbeam
remain shapes. Extend 2D's current unconditional shape scaling to explicit
shape/yield modes and correct color-axis labels; profiles remain mean Y, not
scaled event yields. Save scales/exposure and sumw2 in summaries/metadata.
No automatic claim of authoritative exposure from catalogues.

6. DAG/output: source validation -> range resolution/coverage -> grouped ROOT
histograms -> per-feature/pair PDF + summary.csv -> aggregate receipt. Named
Snakemake targets; separate configuration/content fingerprints for conversion,
histograms and rendering, so style-only changes do not require reconversion.
Frozen campaign input settings remain immutable; preserve existing run products.
Layout under run: config/, sources/, histograms/1d/<feature>/<selection>/<geometry>/<class>.root,
histograms/2d/<pair>/<selection>/<geometry>/<class>.root,
plots/1d/<feature>/<selection>/<geometry>/{plot.pdf,summary.csv},
plots/2d/<pair>/<selection>/<geometry>/{plot.pdf,summary.csv}, logs/.
Always use geometry level for stable layout. PDFs per feature or configured
pair; 2D PDF contains density panels and profile overlay. Keep campaign coverage,
normalization and complete receipts; no combinatorial auto-generation of pairs.

7. Acceptance: tests for explicit/automatic ranges, negative/constant/empty and
missing values, exact max-edge inclusion, common class ranges and 2D pair masks;
class selection, no accidental cut changes, empty parents, audited shape/yield
scales/errors, geometry separation, cache invalidation and legacy plotting
regressions. Verify backend calls directly and compare histogram bins and PDF
style with EM shower reference at identical settings. Reuse bounded MC ROOTs
for initial plotting test, then a bounded end-to-end small MC run. Inspect PDF
pages and CSV/ROOT agreement. Full batch remains dry-run only. Update production
and plugin guides with exact configuration examples and output tree.

Only task record edited; no implementation or Next action executed. Historical
output paths, job evidence and validation limits preserved above.

Main launcher `--check-handoff shower-features`: PASS; `git diff --check`: PASS.

### 2026-11-19T15:08:45+01:00 — ROOT backend integration implemented and tested

Implementation commit `5f2c0a453898b925b8e0129dbf3635e835485983` exists before
this record. Verified pwd/branch; only authorized checkout edited. Changes:
evaluation_region_partitions/{make_group_hist.py,make_group_hist2d.py,
plot_comparison.py,plot_group_hist2d.py}; feature_production/{cli.py,worker.py,
distributions.py,plot_settings.py,Snakefile,config/histograms.yaml,
config/plotting.yaml,tests/test_campaign.py,tests/test_root_plots.py,README.md};
feature_plugins/README.md; feature_dev/TEST_RESULTS.md. Exact list:
`git show --stat 5f2c0a4`. This checkpoint changes only the task record.

Implementation, output layout, known limits and job state replaced in Current
state above. Historical linked records retained. No next-action work started.

Commands/results (LCG activation and hostname/pwd/interpreter/import preflight
in same shell, PYTHONNOUSERSITE=1 and PYTHONDONTWRITEBYTECODE=1):

- `python3 -m unittest discover -s feature_production/tests -p 'test_*.py'`: 30 passed, unit.log. Earlier fixture used multiple sndData cycles and emitted ROOT warnings; fixture now recreates consistent schemas and final log has no such errors. Missing branches are explicitly rejected before filling.
- `python3 -m unittest discover -s evaluation_region_partitions -p 'test_em_shower*workflow.py'`: 14 passed, legacy.log.
- `./run_shower_features.sh --dataset mc-neutrino --plugin silvia --sample small --files-per-group 1 --events-per-file 2 --run root-backend-small-20261119`: six stages complete at15:01:44 local; four events. Controller exit143 reported after final completion messages; identical command then exit0/nothing to do. small.log/repeat.log retain both observations.
- Same dataset/plugin/caps with `--run root-backend-reuse-20261119 --reuse-sources /afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/mc-neutrino/mc-small-20261119-143325`: exit0/four plotting preparation stages, no source jobs.
- Same dataset/plugin/caps with `--run root-backend-style-20261119 --reuse-sources /afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/mc-neutrino/root-backend-small-20261119 --plot-config "$PWD/feature_dev/output/root-backend-20261119/style.yaml"`: exit0; only plots/all scheduled. Histograms referenced from original run; new requested style snapshot records CC_numu orange.
- Same dataset/plugin/caps, `--run root-backend-final-20261119 --reuse-sources /afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/mc-neutrino/root-backend-small-20261119`: exit0, four stages. Last subsequent paired-feature-cut accounting change is covered by its dedicated unit case, default MC pair selection unchanged.
- `./run_shower_features.sh --dataset mc-neutrino --plugin silvia --sample full --normalization shape --run root-backend-full-preview --dry-run`: exit0, 1870 DAG tasks (1866 sources + ranges/histograms/plots/all), no full campaign execution.
- `python3 feature_dev/output/root-backend-20261119/verify_outputs.py`: passed (verification.json): two source ROOTs/four events/13 plugin branches, successful validators, recursive checksums, 192 ROOT histograms and variances, 48 PDF page counts, CSV normalization, class counts and style-cache reuse.
- `pdftoppm -f 1 -singlefile -scale-to 1100 -png .../plot.pdf .../one-d` and corresponding two-d: rendered and visually inspected; matches ROOT backend colours/markers/annotations and density panels. PDF profile-page structure covered by tests.
- `bash -n run_shower_features.sh`, `git diff --check`: passed. No new scheduler queries/submissions; no batch job IDs. Existing profile/access limitations remain.

User settings are documented at feature_production/config/histograms.yaml
and plotting.yaml with CLI overrides. New runs no longer emit a combined
report.pdf/HTML; each feature/pair has its own PDF and summary.csv. Class and
geometry identifiers, resolved ranges and normalization are retained for review.

Main launcher `--check-handoff shower-features`: PASS (exit0).
`git diff --check`: PASS. Implementation SHA predates this task-only checkpoint;
Next action remains unstarted.

### 2026-11-19T15:17:34+01:00 — Combined geometry plots

Implementation commit: cd13b6c8d1b3ac47a024b8d6076c2bf80de0c5fa.
Removed geometry directory levels, combining raw shapes and independently weighted
yields while preserving geometry provenance. Changed files and validation limits
are listed in Current state; previous historical records remain preserved.

Commands/results: `python3 -m unittest discover -s feature_production/tests -p
 'test_*.py'`: 31 passed. `./run_shower_features.sh --dataset mc-neutrino
--plugin silvia --sample small --files-per-group 1 --events-per-file 2
--run combined-geometry-20261119 --reuse-sources
/afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/mc-neutrino/root-backend-small-20261119`:
exit0, four tasks complete. Independent uproot/receipt checks: 24 PDFs, 96 ROOT
histograms, four conserved events, no geometry directories. Logs and exact output
paths are recorded above and in feature_dev/TEST_RESULTS.md. No new conversions,
full campaign or batch jobs. Local sessions finished; no job IDs. Handoff command:
`python3 /afs/cern.ch/work/z/zhibin/snd-ml/docs/tasks/manage_task.py --check-handoff shower-features`.
Handoff check: PASS (exit0); git diff --check: PASS. Implementation and task
checkpoint committed separately; no push performed. The single Next action
remains unstarted.

### 2026-11-19T15:19:11+01:00 — Larger MC run prepared

User requested more events. Planned command: `./run_shower_features.sh --dataset mc-neutrino --plugin silvia --sample small --files-per-group 2 --events-per-file 500 --run larger-mc-20261119`. Dry-run exit0: four sources plus ranges/histograms/plots/all. Existing implementation cd13b6c8d1b3ac47a024b8d6076c2bf80de0c5fa; launch HEAD49afb76, only task documentation modified. Sequential local Snakemake, at most2000 selected events; no full campaign or coworker qualification. Expected sources/{0,1,2,3}/features.root, logs/source-<index>.log and plots/1d and plots/2d under output base above. Status: tail controller log and verify plots/complete.json; no scheduler IDs. Access preflight log: feature_dev/output/larger-mc-20261119/preflight.log.

### 2026-11-19T15:23:47+01:00 — Larger MC run verified

Completed the prepared command above, exit0/eight tasks complete. 394 selected
events from four files, compared with four events in the previous test. All
source validators and baseline parity checks passed. Independent verifier
`python3 -m feature_dev.output.larger-mc-20261119.verify` passed receipts,13 plugin
branches,96 ROOT histogram counts/variances,24 PDF page counts and shape
normalization. Exact output/evidence paths and class counts are in Current state
and feature_dev/TEST_RESULTS.md. Implementation remains
cd13b6c8d1b3ac47a024b8d6076c2bf80de0c5fa; launch HEAD49afb76. No running task
processes, no batch IDs, no full campaign or coworker qualification started.
Only task record and validation documentation changed.
Main launcher `--check-handoff shower-features`: PASS (exit0).
`git diff --check`: PASS. Task-only validation checkpoint committed; not pushed.

### 2026-11-19T15:36:34+01:00 — Coworker examples checked

Verified pwd, task branch and clean Git; inspected feature_plugins/README.md,
feature_plugins/silvia/features.py, feature_dev/algorithms/gap_clusters.py and
tutorial links in feature_dev/README.md and README_SHOWER_FEATURES.md. Both
production Silvia example and prior annotated standalone gap-clustering example
remain present. No implementation edits, new tests, generated outputs or jobs;
previous394-event verification and implementation SHA remain current. Only this
checkpoint changed; single Next action unchanged and unstarted.
Main launcher handoff check and git diff --check: PASS (exit0).

### 2026-11-19T15:41:19+01:00 — Coworker README rewritten

Replaced primary coworker guide with current production setup, feature/example
locations, dataset access and parameter table including --files-per-group6 and
--events-per-file200 (actual commands include spaces). Explains per-group/geometry
file selection, per-file event caps, combined PDF outputs, own development branch,
configuration and full/yield references. Preserved previous walkthrough verbatim
under README_SHOWER_FEATURES_STANDALONE.md, apart from a historical notice; linked
from the new guide. Root README and production reference updated. Existing
implementation SHA cd13b6c8d1b3ac47a024b8d6076c2bf80de0c5fa remains valid.

Commands/results: pwd and git branch --show-current match required checkout;
initial git status --short clean. Activated LCG107/preflight Python3.11.9; local
file-link checks pass. ./run_shower_features.sh --help exit0, evidence at
/tmp/zhibin/shower-readme-help-20261119.txt. git diff --check passes. Documentation
only; no new scientific tests, outputs or jobs. Historical records preserved;
coworker/full-campaign Next action unchanged and not started.
Main launcher --check-handoff shower-features: PASS; previous walkthrough
preservation check: PASS. Task-only documentation commit; not pushed.

### 2026-11-19T15:47:31+01:00 — Normalization guide clarified

Production-first guide now identifies required source/target exposure, storage
locations, per-file versus event information, shape/MC/data formulas and geometry
combination. Complete JSON format is in NORMALIZATION_AUDIT.md; standalone
reference preserved separately with scope notice. Changed files listed above.
Existing implementation commit cd13b6c8d1b3ac47a024b8d6076c2bf80de0c5fa retained.

Validation in activated LCG107/Python3.11.9: local links pass; parsed documented
audit passed exposure.validate_coverage and asserted both weights2/30. Actual
source0 ROOT metadata and shape normalization records from larger-mc-20261119
match the guide. This validates documentation/software consistency, not physical
luminosities, new yield outputs or full coverage. No new jobs/IDs or products.
Next action remains unchanged and unstarted.
Main launcher --check-handoff shower-features and git diff --check: PASS.
Task-only documentation commit; no push.

### 2026-11-19T15:51:19+01:00 — Data format guide added

Added feature_dev/DATA_FORMAT.md and linked coworker/production/plugin guides.
Covers raw detector collections, MC truth and saved truth subset, feature/hit
units, geometry coordinate meaning, clusters, separate Hit3D schema and event
identity. Existing implementation cd13b6c8d1b3ac47a024b8d6076c2bf80de0c5fa retained.

Validation commands in activated LCG107: uproot.open on raw MC source and saved
source0 feature ROOT (paths above); keys/typenames/MCTrack leaves inspected.
Asserted70 entries,451 branches,13 plugin branches and documented feature names.
Documented arrays(..., entry_stop=5, library=np) example passed. Local link
checks and git diff --check passed. No new conversion, testsuite, products or
jobs. Only documentation changed; other datasets and Hit3D are code-reviewed
references, not newly file-validated. Single Next action remains unstarted.
Main launcher --check-handoff shower-features: PASS (exit0).
Task-only documentation commit; not pushed.

### 2026-11-19T15:53:10+01:00 — Original luminosity provenance documented

Traced metadata generation and copying into feature ROOT: 2024 neutrino15fb-1
constant, older filename conventions, SND run-luminosity tables, passing-muon
pp24 sampling and muon collision exposure. Added source links, allocation formula
and double-sampling warning to NORMALIZATION.md. Upstream MC CSV row327 agrees
with test plan source0 (same digi, lumi15). Commands: targeted rg/sed code review;
LCG-activated Python json/csv read-only comparison and local link check passed.
Implementation remains cd13b6c8d1b3ac47a024b8d6076c2bf80de0c5fa. No external
physics calibration verification, no new jobs/outputs; Next action not started.
Main launcher --check-handoff shower-features and git diff --check: PASS.
Task-only documentation commit; not pushed.

### 2026-11-19T15:54:22+01:00 — GitHub branch published

User authorized commit/push to origin task/shower-features. Verified pwd, branch,
clean working tree and GitHub origin URL. Fetch succeeded; no remote-only
commits,35 local-only commits. Normal push succeeded:4f37831..68d2dec. All
implementation/doc changes already committed; this task-only checkpoint records
the publication. Implementation SHA remains cd13b6c8d1b3ac47a024b8d6076c2bf80de0c5fa,
not this checkpoint commit. No tests/conversions/jobs or outputs generated.
Scientific Next action unchanged and unstarted.
Main launcher --check-handoff shower-features and git diff --check: PASS.

### 2026-11-20T10:37:54+01:00 — Coworker HTCondor profile lookup

Existing relative path snakemake/.config/snakemake/htcondor is tracked, but
config.v8+.yaml uses executor cluster-generic (config.yaml is legacy cluster).
The existing production wrapper passes that profile to the main snakefile;
feature_production requires executor htcondor. Do not claim the old path works
with the new launcher. Verified via ls/cat/rg/git ls-files and LCG Python YAML
parse. No profile edits, installs, tests or batch jobs; no job IDs/new outputs.
Implementation SHA remains cd13b6c8d1b3ac47a024b8d6076c2bf80de0c5fa. Single Next
action replaced with concrete compatibility prerequisite; not started.
Main launcher handoff check and git diff --check: PASS. Local task-only checkpoint.

### 2026-11-20T10:40:21+01:00 — Batch compatibility fix scope clarified

User asked how to fix profile setup. Read-only review found both restrictive
executor guard and owner-specific submit/status/bootstrap paths. Installer only
installs Snakemake; existing hooks use system Python htcondor2. Runtime is passed
directly to MaxRuntime and must be reconciled with new workflow minute semantics.
Proposed coherent fix: support configured cluster-generic backend, explicit batch
package setup, portable per-run logs/interpreter/bootstrap and resource mapping;
then dry-run and bounded canary before full1866-source campaign. No code edits,
package installs, jobs or tests in this explanation turn. Existing implementation
SHA and single Next action retained; not started.
Main launcher handoff and git diff --check: PASS. Local task-only checkpoint.

### 2026-11-20T10:42:17+01:00 — Upstream HTCondor profile identified

User supplied https://github.com/Snakemake-Profiles/htcondor/tree/master.
Read upstream README, config.v8+.yaml and cookiecutter.json using curl -fsSL
after web fetch cache misses. Confirmed profile is deployed with Cookiecutter,
uses cluster-generic for modern Snakemake and requires HTCondor bindings and
executor plugin. Configuration includes profile name, log directory and CERN
flag. This supports retaining the established profile rather than requiring
native executor htcondor. No package install, code change, submission or new
scientific outputs; Next action remains unstarted. Implementation unchanged.
Main launcher handoff check and git diff --check: PASS; task-only checkpoint.

### 2026-11-20T13:35:59+01:00 — Batch implementation pre-submission checkpoint

Implemented managed setup --batch/check with pinned official upstream profile,
cluster-generic/HTCondor packages, private generated CERN adapter, saved selection,
explicit overrides, concurrency and runtime mapping. Isolated test environment
and profile: feature_dev/output/batch-20261120/{venv,profile}; prior environment
unchanged. Setup failed initially on legacy htcondor import; corrected to
htcondor2. Fresh/repeat/check setup passed. Production40 tests and setup5 tests
passed. Full dry-run planned1866 sources plus stages; no full jobs. Canary access
preflight, klist and condor_ping WRITE passed. All logs under same evidence folder.

Authorized canary command: ./run_shower_features.sh --dataset mc-neutrino
--plugin silvia --sample small --files-per-group 1 --events-per-file 2 --batch
--jobs 2 --run batch-canary-20261120. Expected output base
/afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/mc-neutrino/batch-canary-20261120/
with two source ROOTs and24 combined PDFs. Controller log
feature_dev/output/batch-20261120/canary.log; batch logs under
feature_dev/output/batch-20261120/venv/htcondor_logs/<token>/.
Status: condor_q zhibin and controller/log inspection. Job IDs pending actual
submission. No full production authorized by this canary.

Canary submitted from implementation db10611: source0 job16608353.0, token
1f7fb1f6acac4cc28b9d86959d3c394a; source1 job16608354.0, token
79bb3cd3506f4837b32c2cc29ff97cfe. Initial condor_q status idle. Other checkout
job16608352 observed and left untouched. Status command: condor_q16608353
16608354 (with spaces).

Canary milestone: source validation/receipts pass for both two-event ROOTs;
ranges16608356 completed, histogram16608357 submitted. Controller progresses
from remote event-log status automatically. Implementation db10611 committed
before submission; no code edits during canary. Evidence TEST_RESULTS.md and
feature_dev/output/batch-20261120/.

### 2026-11-20T13:51:23+01:00 — Batch setup and end-to-end canary verified

Implementation db1061162f94aa850a17de9cd1056b40f5f81b4a committed before canary. Commands/results,
changed files, output paths and limits replaced in Current state above. Official
profile deploy/repeat/check,40 production tests,5 setup tests and additional
cancellation mock passed. Full dry-run only:1866 sources. Canary command recorded
in pre-submission entry completed exit0: source jobs16608353/16608354, ranges
16608356, histograms16608357, plots16608359. Four worker events,96 histograms,24
PDFs and all receipts independently verified. Identical rerun nothing to do;
condor_q16608353 16608354 16608356 16608357 16608359 (with spaces) returned no
remaining rows. No full campaign. Local tests/docs follow-up commit only; next
action is coworker-account qualification and is not started.
Main launcher --check-handoff shower-features and git diff --check: PASS (exit0).

### 2026-11-20T13:57:27+01:00 — HTCondor update published to GitHub

Verified required pwd and task/shower-features branch, clean initial tree and
GitHub origin. git fetch origin task/shower-features succeeded;
git rev-list --left-right --count origin/task/shower-features...HEAD returned
0 remote-only and 5 local-only commits. git push origin
HEAD:refs/heads/task/shower-features succeeded (ed65b3c..7915088). Existing
implementation SHA db1061162f94aa850a17de9cd1056b40f5f81b4a retained.
Only docs/tasks/active/shower-features.md changed for this publication checkpoint;
implementation files, prior validation limits and output paths remain above.
No tests rerun, outputs generated or jobs submitted; scheduler not rechecked.
LCG107 Python3.11.9 preflight on lxplus9105 passed. Main launcher command
`python3 /afs/cern.ch/work/z/zhibin/snd-ml/docs/tasks/manage_task.py --check-handoff shower-features`
and git diff --check passed before publication. This checkpoint is committed
and pushed separately. The single Next action remains unchanged and unstarted.

### 2026-11-22T15:04:48+01:00 — Coworker submission failure triage

Read-only code review of feature_production/batch_adapter.py, batch.py and the
installed cluster-generic executor found that the generic error includes stdout
only; traceback stderr may appear separately. Adapter stages: create UUID log
directory, parse job properties, write submission.json, register Kerberos with
Credd, submit to Schedd, write job.json. Missing worker logs alone is inconclusive.
Requested exact command/profile and full traceback; root cause unconfirmed.

Verified pwd/branch and clean initial Git. LCG107/Python3.11.9 on lxplus967:
`python3 -m unittest feature_production.tests.test_batch` passed 10 tests;
`./run_shower_features.sh setup --batch --check` passed the existing profile at
feature_dev/output/batch-20261120/profile (private venv beside it). No packages
installed, no submission or scientific outputs, no scheduler query. Tool output
contains validation results; no new persistent log paths or job IDs. Changed only
this task record; implementation remains db1061162f94aa850a17de9cd1056b40f5f81b4a.
Historical evidence and links preserved. Single Next action awaits coworker
evidence and is not started.

### 2026-11-23T15:17:20+01:00 — Coworker worker SNDSW activation diagnosed

Preserved pre-existing November22 task edits/history; only this task record changed.
Implementation remains db1061162f94aa850a17de9cd1056b40f5f81b4a.
Inspected snakemake/envs/env_sndsw_nue.sh, feature_dev/run_export.sh and CVMFS
setup/alienv. Custom module exists only in public/SndBuild/sw/MODULES/rhel9_x86-64.
Existing wrapper import-only probe succeeded on lxplus912. Clean child shell:
`alienv -a slc9_x86-64 load sndsw/latest-event_level_info_output-release --no-refresh`
reproduced exact missing-module error, exit1; rhel9_x86-64 activation followed by
June25/bin/python importing ROOT and SndlhcGeo passed (ROOT6.36.04; custom
master-local14/python/SndlhcGeo.py). Worker architecture still unconfirmed.
`eval $(alienv load ...)` masks the nonzero substitution status; recommended
explicit architecture and checked assignment before eval. No scripts changed.
Coworker supplied output path: /afs/cern.ch/work/s/svicenti/snd-ml-outputs/
shower-features/mc-neutrino/shape-full-v12, worker b9g02p7746; not inspected.
No new output files/logs, event processing, job IDs or submissions. Evidence in
tool output; prior output paths and limits retained above. Single Next action
is not started. Main launcher handoff and git diff --check run for this checkpoint.

### 2026-11-23T15:19:15+01:00 — Custom SNDSW dependency audit

Read-only reference /afs/cern.ch/work/z/zhibin/public/SndBuild/sndsw Git objects:
70498563c adds MuonDIS event-level pipeline;90fff4b6e adds sndCutFlowTree and
nue event-level pipeline. Installed module master-local14 records70498563c.
convertData/digi_2_features.py imports analysis.analyses.snd_analysis_2024_0mu.
sciFiTools selectHits/getSumDensity; calls implement hit selection and four
density outputs. Shared Oct7 source and install lack this helper; upstream
master raw path HTTP404 (GitHub web fetch failed; curl used). Existing cutflow
files are consumed by feature production, not regenerated by this converter.
No claim that newest SNDSW is a drop-in replacement. Proposed pinned shared
runtime plus preserved/versioned helper logic and event parity before migration.
Only task record updated; pre-existing edits/history preserved, implementation
SHA unchanged. No new output paths, jobs, installs or reference checkout edits.
Main launcher handoff and git diff --check run; single Next action unstarted.

### 2026-11-23T15:32:40+01:00 — Interrupted workflow and preview statistics reviewed

User reports activation error gone without fix; no runtime change made and
connection explanation not verified. pwd/branch checked; existing task edits
preserved. Reviewed feature_production/{cli.py,worker.py,Snakefile,README.md}:
identical plan reuses completed receipts; all sources required before ranges;
reuse-sources checks all sources and same campaign/plugin/conversion settings.
Plot-only settings reuse histograms; histogram changes rebuild from source ROOT;
feature/plugin changes require conversion. No wrapper unlock/rerun-incomplete
options. No automatic partial full-run plot mode.

LCG107 Python3.11.9 on lxplus912: coworker plan read at
/afs/cern.ch/work/s/svicenti/snd-ml-outputs/shower-features/mc-neutrino/shape-full-v12/plan.json
failed PermissionError. Directory scan saw221 source directories; completion
metadata not accessible, not evidence of zero completed events. Existing own
/afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/mc-neutrino/larger-mc-20261119/
validation records confirm70+62+127+135=394 events from4 files. No ROOT files
scanned, output generated, scheduler queries or jobs; only task record changed.
Implementation db1061162f94aa850a17de9cd1056b40f5f81b4a retained.
Main launcher handoff and git diff --check run; single Next action unstarted.

### 2026-11-23T15:55:32+01:00 — Testbeam catalogue sizes and settings audited

Read feature_dev/datasets.yaml, sources.py and feature_production/cli.py.
CSV paths are the three configured catalogues in datasets.yaml. Read-only Python
csv sums and sources.stratum grouping gave MC2024:75 files,95766 input,13 groups;
data2024:15 files,5399810 input,430240 selected,8 groups; H8data2023:385 files,
370929095 input,200000 selected,4 groups. MC selected count not stored.
Current production defaults1 file/group and20 events/file yield25 files and
maximum500 events; prior example200/file yields maximum5000. Grouping uses
family/PDG/energy/geometry. All testbeam shape normalized; full removes caps.
Legacy YAML2files/100events/25events-per-group is not production effective cap.
Metadata only, no source accessibility or ROOT validation. No new outputs/logs,
jobs or scheduler query. Only task record updated, pre-existing changes and
historical links preserved; implementation db1061162f94aa850a17de9cd1056b40f5f81b4a
unchanged. Next action retained and unstarted. Handoff and whitespace checks run.

### 2026-11-24T20:18:05+01:00 — Coworker testbeam EOS permissions inspected

Commands: pwd; git branch --show-current; git status --short; catalogue rg;
ls -ld and eos root://eosuser.cern.ch attr ls on home/TestBeam/binary_full_v1/
cutflows; id svicenti; ls -l on catalogue example
tbfull_src_0891a0e853f2e4ed43bb.root (exists,100176bytes,mode644).
Home mode700 and sys.mask700; all four directories grant sys.acl only
u:187895:rx,u:1066349:rx; user.acl empty. svicenti UID182858 absent, although
member sndlhc-cg. Owner-side evidence explains access barrier; no direct
coworker credential validation. No permission modifications, new outputs, jobs
or scheduler queries. Only task checkpoint changed, preserving existing edits
and linked history. Implementation db1061162f94aa850a17de9cd1056b40f5f81b4a
retained. LCG preflight and main launcher handoff/whitespace checks run.
Single Next action not started.

### 2026-11-24T20:18:43+01:00 — EOS access grant instructions checked

Read eos acl --help (usage exit22) and official EOS permission documentation.
User ACL additions use --user u:182858:+x or +rx and preserve other entries.
user.acl evaluation requires sys.eval.useracl=1, absent in previous attr output;
enabling may require EOS administrator assistance. No ACL commands executed,
permissions unchanged. Only task checkpoint changed; prior edits/history and
implementation SHA retained. No outputs or jobs. Next action remains unstarted.
Handoff and git diff --check run.

### 2026-11-24T20:19:52+01:00 — Authorized EOS cutflow access granted

User explicitly requested granting access. Rechecked pwd/branch, existing task
edits and svicenti UID182858. On /eos/user/z/zhibin, its TestBeam child,
TestBeam/binary_full_v1 and TestBeam/binary_full_v1/cutflows, ran
`eos root://eosuser.cern.ch attr set sys.eval.useracl=1 PATH`. Added
`eos root://eosuser.cern.ch acl --user u:182858:+x PATH` on the three
parents and `--user u:182858:+rx` on cutflows. All exit0. Subsequent attr ls
verified enabled user ACLs and unchanged original sys.acl entries.
No recursive change, write grants, file changes or jobs. Owner-side verification
only; coworker credential/worker tests pending. Only local task record changed;
existing task edits and historical links preserved. Implementation remains
db1061162f94aa850a17de9cd1056b40f5f81b4a. No new output paths/logs or job IDs.
LCG preflight, main launcher handoff and git diff --check run. Single Next
action not started.

### 2026-11-25T11:34:26+01:00 — Three missing H8 selection indices restored

User requested regeneration or removal of three missing indices; restored original
selection without editing catalogue. Verified pwd/branch, preserved existing task
edits and historical links. Read existing writer and its LCG Snakemake rule.
Bounded local repair used convertData.build_indexed_cutflow.write_index with
input_event_count1000000 for each source, counts269/269/562 from authoritative
H8v2 CSV. No full scan or scheduler submission needed.

Commands/results: LCG activation/preflight passed on lxplus9116, Python3.11.9,
ROOT6.34.02, uproot5.3.11. Python csv read plus uproot cbmsim.num_entries confirmed
all3 catalogue counts; Path.iterdir confirmed empty target directories. Writer
created ROOT+JSON sidecars; independent uproot arrays compared exactly against
((2*i+1)*N)//(2*K), including source UID and policy metadata. All PASS. Manifest
SHA unchanged. Full commands in tool transcript; durable machine-readable
inputs/results at feature_dev/output/h8-index-repair-20261125/preflight.json and
verification.json. Outputs:

- `/eos/experiment/sndlhc/users/zhibin/snd-ml/unified_shower_features_v1/selection_indices/testbeam_data_2023_h8/unified_tb23_src_18ef455f032606efb0ea/selected_entries.root` (269 entries), plus `.validated.json`; SHA256 `fc5a3d6583d54dd1f8e65f42595c5dc45febcf46f674152b44315e49c3e01ba7`.

- `/eos/experiment/sndlhc/users/zhibin/snd-ml/unified_shower_features_v1/selection_indices/testbeam_data_2023_h8/unified_tb23_src_b84cfe755b5426cc5f7a/selected_entries.root` (269 entries), plus `.validated.json`; SHA256 `157f274d9ca7da41e59231672ce5d54f4ec54a91355b00bf6bc6c47a331844d2`.

- `/eos/experiment/sndlhc/users/zhibin/snd-ml/unified_shower_features_v1/selection_indices/testbeam_data_2023_h8/unified_tb23_src_e6dcbc75b5f52d5fa85f/selected_entries.root` (562 entries), plus `.validated.json`; SHA256 `1f67f7a370c92b893b24b20e517a508d91d4d9e4ca5de9428d89aa7d2ac90d1e`.

No code changes, unrelated checkout edits, new jobs or logs. Existing implementation
SHA retained; only task checkpoint tracked. Coworker access/production untested.
Main launcher handoff and git diff --check run; single Next action unstarted.

### 2026-11-25T11:37:36+01:00 — Testbeam MC validation failures diagnosed

Verified pwd/branch and preserved pre-existing task edits. Reviewed validator
identity_cutflow_entries/inspect_source_selection, source_provenance.event_source_values,
and prepare_unified_shower_feature_manifests._testbeam_family. Current generator
already copies n_events into selected_event_count; deployed MC CSV lacks column.
Deterministic13-source selection reproduced using sources.stratum/digest and
feature-dev-v1 seed. Read cutFlowSummary entry arrays, raw cbmsim headers and
first MCTrack event only, then retained primary values in summary.json. Compared
/eos/user/z/zhibin/TestBeam/binary_full_v1/catalog/master_sources.csv n_events.
All13 cutflows match upstream and are identity; raw counts match deployed CSV.
Eight selected/raw mismatches, three nominal-energy mismatches, two pass: exactly
coworker pattern. Example electron100GeV1200/18000, pion50GeV252/600. Pion100
truth E100.00009739887399,180 E180.00005411050375,200 E200.0000486994548;
PDG211 correct. Validator tolerance max(1e-7,1e-9*energy) rejects these; converter
math.isclose(rel_tol=.02,abs_tol=.5) accepts them. Truth E includes mass.
No evidence requiring replacing cutflows or removing sources.

Commands: same-shell LCG activation/imports; Python csv/uproot/digest audit and
assertions reproduced8+3+2. Evidence feature_dev/output/testbeam-mc-audit-20261125/
small-source-audit.json and summary.json. No code/catalogue mutations, new ROOT
outputs, conversions or batch jobs. Full75 and20-entry validation untested.
Existing implementation SHA retained. Only task record updated; next action
not started. Main launcher handoff and git diff --check run.

### 2026-11-25T11:52:38+01:00 — MC repair implemented, bounded preview prepared

Implementation 89685db3f68a9e1833bec56d1371fd12068c22db committed before preview; prior task edits preserved.
Repair utility audited all75 raw headers and cutflow identity arrays, then added
selected_event_count atomically from frozen upstream n_events. Backup and hashes
recorded in feature_dev/output/testbeam-fix-20261125/catalogue-repair.json.
Independent original-field/row-order comparison passed. Production44 tests and
converter13 tests passed; final focused/access checks pending.
Planned command: ./run_shower_features.sh --dataset testbeam --plugin silvia
--dataset-config feature_dev/output/testbeam-fix-20261125/testbeam-mc.json
--sample small --files-per-group 1 --events-per-file 20 --batch --jobs 4
--run mc-validation-20261125. Config uses existing MC catalogue only; no source
rows changed for this preview. Expected13sources260events plus plots. Controller
log feature_dev/output/testbeam-fix-20261125/canary.log; source logs/validation
under /afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/testbeam/mc-validation-20261125/.
Jobs pending submission; monitor controller log and targeted condor_q job IDs.
No full production. Single follow-up Next action remains unstarted.

Preview submission milestone (2026-11-25 11:55 Europe/Zurich): access preflight
passed, final4 focused tests passed; catalogue contains33784 selected MC events.
Controller session61891 running. First source clusters16621100,16621101,
16621102,16621103 submitted; exact external IDs in canary.log. Status command:
`condor_q 16621100 16621101 16621102 16621103 -af ClusterId ProcId JobStatus RemoteHost HoldReason`.

Preview milestone 2026-11-25 12:05 Europe/Zurich: all13 sources converted and
validated20events each, including all11 previously failing cases. Source external
IDs and range ID: 1f3ab726cfa64e8992b82e39df1c135d_16621100, b9ce5f19219f4e94a06784297b638278_16621101, 5e217bf968c445298bfbb4e1c6eddefc_16621102, d2671599c8f04e37b2e5ebdf63eb93c8_16621103, 8a52a00076784970b0a033c37937c8af_16621104, 596aa252b7084997b4ab8709b5c7cd5c_16621105, 69c1955c6f5c4922bf2a3648dc836509_16621108, 8857ad3d7483470390703503b6321097_16621109, b57b64e5ca3a49acb7b47f82c0019e91_16621110, 33c5248d9fd444e8a5e63cf3baf7bc73_16621111, 00d4a40396d5488996f8cce7f3771d7a_16621114, 41c2abced37147dd8759c23d3bd3a5f8_16621115, 780aabbe472c4a778ebbfb5983960fd0_16621116, bd912949c4e54b76a7b7b4cb817a3a20_16621124.
Range stage16621124 submitted; remaining histogram/plot stages pending.

### 2026-11-25T12:15:01+01:00 — Testbeam MC fix verified end to end

Implementation 89685db3f68a9e1833bec56d1371fd12068c22db remains the existing implementation checkpoint.
All75 sources audited; repaired deployed MC manifest adds selected_event_count
only (33784 total), backup /afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/metadata/manifests/testbeam_mc_2024.csv.before-selected-counts-4845c54ffebe.
Original SHA 4845c54ffebe6a4b26330897fceaa9249018e2a1a5c10eb7461e264286d2d495; corrected SHA 3c9469cff14dd1b46292a7d0fdc578feaaa08bbe04b2ee83ab681cc462562375.
Production44 tests, converter13 tests and final4 focused tests passed.
Preflight and the exact prepared command above completed exit0. Thirteen source
jobs each converted/validated20events; range/histogram/plot jobs all completed.
Clusters 16621100, 16621101, 16621102, 16621103, 16621104, 16621105, 16621108, 16621109, 16621110, 16621111, 16621114, 16621115, 16621116, 16621124, 16621127, 16621130. Targeted condor_q on these IDs returned
no rows; no other campaign queried or changed. Independent source verifier
(prefix of verify_preview.py) checked receipts, baseline parity presence,13 plugin
branches, event indices and exact PDG/strict energy against raw MC for260events.
verify_outputs.py verified config/histogram/plot receipts and every resolved PDF
path:299 ROOT histograms,23 PDFs. Initial hardcoded24 expectation failed; actual
resolved configuration has23 plot groups, all present. No production defect.
Full commands/configuration/logs and source paths are under the evidence folder
listed in Current state. Only bounded preview; no full/coworker run.
No runtime/library upgrades. Existing task history/pre-existing task edits
preserved. Follow-up docs/checkpoint commit only; not pushed. Single Next action
unstarted. Main launcher --check-handoff shower-features and git diff --check run.

### 2026-11-25T12:23:39+01:00 — Coworker branches reviewed and merged

Implementation: 5c655ed39469557b418fb724ca9278fbbf40202d (parents localccbcf53 and coworkeree3a554). Fetch showed no incoming origin/task/shower-features changes; coworker algorithm7747bc2 is an ancestor of yield-auditee3a554. Used a non-rewriting merge. No conflicts; only four trailing-whitespace cleanups. No push or batch submission.

Changed files: feature_dev/algorithms/my_features.py, feature_dev/algorithms/plot_energy.py, feature_dev/schemas/my_features.yaml, feature_dev/tests/test_my_features.py, feature_plugins/silvia/build_audit.py, feature_production/README.md, feature_production/cli.py, feature_production/config/histograms.yaml, feature_production/tests/test_campaign.py, feature_production/tools/build_audit.py, snakemake/envs/env_sndsw_nue.sh. The environment change is only a commented setup line. Existing testbeam fixes retained.

Commands/results: git fetch origin; git log/diff review; git merge --no-ff --no-commit origin/task/silvia-yield-audit (clean automatic merge); git commit; git merge-base --is-ancestor for ee3a554 and89685db (both exit0). Under source snakemake/envs/env_lcg.sh with PYTHONNOUSERSITE=1/PYTHONDONTWRITEBYTECODE=1: python3 -m unittest discover -s feature_production/tests -p 'test_*.py' (45 before/47 after); PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q feature_dev/tests/test_my_features.py (6 passed); python3 -m unittest discover -s convertData -p 'test_unified_shower_features.py' (13 passed); python3 feature_production/tools/build_audit.py --help (exit0). Logs: feature_dev/output/merge-silvia-20261125/. Git whitespace check passed. No real full-campaign audit, coworker-account run, or new production output tested.

Review findings: shape/yield source reuse retains campaign/plugin/converter checks and records original normalization; histogram signature includes new normalization/audit. Personal older helpers retain fixed paths and were not run. The helper within plugin directory affects source hashes, so existing runs may fail reuse compatibility. Production feature calculation remains the adapted plugin, not the older development example. Added energy range is explicitly0–1000GeV; other unspecified ranges remain automatic. Prior tested260-event/23PDF snapshot predates this plot-config merge.

Previous current-state evidence retained for traceability:

- Lifecycle: active
- Worktree: `/afs/cern.ch/work/z/zhibin/worktrees/snd-ml-shower-features`
- Branch: `task/shower-features`
- Last verified: 2026-11-25T12:15:01+01:00
- Implementation checkpoint: 89685db3f68a9e1833bec56d1371fd12068c22db
- Implemented: 89685db: shared testbeam MC nominal-energy tolerance (2% relative/0.5GeV absolute), exact PDG check and separate errors; all existing numerical/identity checks retained. Early stale-MC-manifest guard and aggregated selection ROOT header checks before submission. Audited atomic catalogue repair utility and regression tests. Shared MC CSV repaired with selected_event_count from frozen upstream; no cutflows/selections changed. Documentation updated. Committed locally, not pushed.
- Validation: lxplus9116, LCG107 Python3.11.9 ROOT6.34.02/uproot5.3.11: all75 raw headers and identity cutflows match upstream counts (33784selected). Backup/original field and row-order preservation verified. Production44 tests, converter13 tests, final4 focused tests passed. SNDSW6.36.04 access preflight passed. Actual HTCondor13-source20-event preview completed17/17 DAG steps,16 jobs,260events; all source validation/parity receipts pass. Independent raw MC comparison: all260 PDGs exact, saved energy strict rtol1e-9/atol1e-7. All source/stage receipts and23 configured PDF paths verified,299histogram files. Initial verifier expected24 PDFs from earlier run; corrected to resolved configuration, no missing plot.
- Remaining limitations: Bounded MC preview on owner account only; no full MC event conversion or coworker-account rerun. The75-source audit checks raw headers and all identity indices, not all raw truth events. Existing custom SNDSW/runtime qualification limits remain. New run name required after catalogue/code/HEAD changes. Source truth verification performed independently for260events, not added as a new full raw-truth scan in production validator.
- Running jobs: All16 preview clusters completed: 16621100,16621101,16621102,16621103,16621104,16621105,16621108,16621109,16621110,16621111,16621114,16621115,16621116,16621124,16621127,16621130. Targeted condor_q returned no rows. Controller exit0; no pending task jobs, no full campaign.
- Outputs: Preview /afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/testbeam/mc-validation-20261125/ has13 source ROOTs/validation/parity receipts,299histograms and23PDFs. Evidence feature_dev/output/testbeam-fix-20261125/{catalogue-repair.json,production-tests.log,converter-tests.log,focused-tests.log,access.log,canary.log,source-verification.json,preview-verification.json,jobs.json,verify_preview.py,verify_outputs.py,testbeam-mc.json}. Backup /afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/metadata/manifests/testbeam_mc_2024.csv.before-selected-counts-4845c54ffebe. Profile per-job logs feature_dev/output/batch-20261120/venv/htcondor_logs/<token>/ (tokens in jobs.json). Changed8 implementation/docs/test files in89685db; follow-up feature_production/README.md,feature_dev/TEST_RESULTS.md and this task record. Prior history preserved.

### 2026-11-25T12:28:06+01:00 — Published integrated branch to GitHub

Verified pwd and branch task/shower-features; working tree clean. git push origin task/shower-features succeeded, advancing GitHub from6f02516 to960a095. Existing implementation commit5c655ed contains the coworker merge and retains89685db testbeam fixes. Publication-only follow-up changes docs/tasks/active/shower-features.md. Main launcher --check-handoff shower-features and git diff --check are required before committing this record. No tests repeated for this documentation-only update; prior47 production/6 density/13 converter tests remain the validation evidence. No new jobs or scientific outputs; earlier paths and validation limits unchanged. Coworker instructions use a fresh small-testbeam run; no next production action started here.

### 2026-11-26T16:00:14+01:00 — Event builder terminology lookup

User asked about sndsw event building and clock-cycle selection. Read upstream https://github.com/SND-LHC/sndsw/pull/277 and its files view: MCEventBuilder in sndFairTasks/MCEventBuilder.cxx; manager shipLHC/run_MCEventBuilder.py exposes --saveFirst25nsOnly. PR merged2025-09-08. Raw master source fetches failed; no installed-version or exact SciFi cycle-selection claim verified. pwd/branch correct, starting tree clean. No code changes, tests, jobs, or scientific outputs. Only this checkpoint changed; implementation stays5c655ed. Earlier pending coworker rerun remains the single next action and was not started.

### 2026-11-26T16:19:50+01:00 — Verified SciFi clock-cycle selection before shower features

Verified pwd/branch; preserved preceding checkpoint edit. Implementation remains5c655ed; only docs/tasks/active/shower-features.md changed. Inspected convertData/digi_2_features.py process_hits, hit_time_selection.py, feature_dev/run_export.sh, env_sndsw_nue.sh and explicit installed sciFiTools.py. Import-only command: bash feature_dev/run_export.sh -c with inspect.getfile(sciFiTools), min_clock_cycle/max_clock_cycle and1e9/160.316e6; exit0, lxplus939, Python3.9.25 ROOT6.36.04. Actual module /afs/cern.ch/work/z/zhibin/public/SndBuild/sw/rhel9_x86-64/sndsw/master-local14/analysis/analyses/snd_analysis_2024_0mu/sciFiTools.py. No detector events read or jobs submitted; no generated scientific outputs.

Normal selector: valid hits, independent5stations x2orientations,200bins over0..100ns, highest-count-bin centre, inclusive window peak-0.5T to peak+1.2T, T=6.237680580852816ns. MC GetTime already ns; data multiplies TDC byT and calls GetCorrectedTime(detID,time,0). Selection precedes density/count/QDC/clustering/plugin. Muon MC types mc_muon/mc_muondis instead subtract first valid arrival across SciFi+MuFilter, use local sparse0.5ns plane peaks (no0..100range), offsets-3.125/+7.5ns, then <=25ns. Testbeam MC uses standard selector plus <=25ns; MC neutrino skips extra25ns cut by existing argument-text mc_neutrino check; real/testbeam data no extra MC cut. Invalid-time handling and path-based exception were read, not changed. Noted numerical mismatch: local/data stored time conversion6.25ns vs SNDSW160.316MHz period; no physics impact measurement. Earlier Next action unchanged and not started.

### 2026-11-26T16:26:52+01:00 — Read26th collaboration meeting slides

Source /afs/cern.ch/user/z/zhibin/work/snd-ml/docs/slides/26th collaboration meeting.pdf (29pages, Riddhi Biswas,2026-09-23). Commands pdfinfo, pdftotext -layout, pdftoppm for pages8,15,20–22. Extracted all text and visually reviewed those5pages; image-viewer sandbox failed, read rendered images through escalated base64 instead. Review artifacts feature_dev/output/slides-26cm-review/. No code changes, jobs, or scientific output; existing implementation5c655ed unchanged. Only task checkpoint edited; earlier history preserved.

Findings: slides13–14 show preferred data timing windows depend on energy and MC configuration (updatedGEANT4,7pe vs3.5pe). Slides15–20 compare hit counts, QDC in peak±0.5cycles, and plane_cross_talk_ratio; slide20 favors occupancy ratio, shows per-plane linear fits to band means. Slides21–22 demonstrate adaptive upper-edge clamp[1.8,2.0]cycles, distinct from tabulated fixed-window optima up to2.2; no exact ratio formula/intercepts supplied. Current converter fixed[-0.5,+1.2] selection does not implement this adaptive method. Slide8 displayed chi-square lacks squared residual and mixes absolute/fractional uncertainty notation; not a literal implementation specification. Slides26–29 still call for updated summary/systematic/significance results. No inference that current inputs use7pe or sameMC. Required handoff and whitespace checks run; pending coworker rerun remains the single Next action, not started.

### 2026-11-26T16:29:06+01:00 — How to configure the SciFi timing window

Read-only review of actual master-local14 sciFiTools.py and local feature_production/{cli,worker}.py, feature_dev/sources.py, convertData/{digi_2_features,digi_2_features_mc_muon,digi_2_hits3D,hit_time_selection}.py, snakemake/rules/unified_shower_features.smk. Verified pwd/branch; preserved prior checkpoint edits. No implementation changes, new jobs, tests, or outputs; implementation remains5c655ed.

Findings/recommendation: SNDSW selectHits reads module globals min_clock_cycle/max_clock_cycle; configure imported module inside each converter process without editing shared installation. Expose signed lower/upper CLI settings via launcher->immutable plan->worker/converter_arguments->converter; existing --parameters goes to plugin after selection and cannot recover discarded hits. Muon MC bypass needs matching local offsets (local lower_clock_cycles is positive width, negate signed lower setting); MC-muon wrapper owns separate parser. Pass consistent policy to Hit3D and original Snakemake rule path. Record window, clock period, selector/source identity and sample policy in ROOT/plan; include timing in conversion signature AND explicit --reuse-sources compatibility checks (currently compares campaign/plugin/code, not conversion_signature alone). Changed timing requires raw-input reconversion and new run, not replot/reuse of old selected hits. Preserve/record existing extra25ns cut, MC-neutrino exemption and first-arrival logic; data-only slide study does not justify silently changing allMC. Proposed flags are not executable today. Adaptive algorithm still requires author code; fixed2.0 is only a comparison candidate. Handoff/whitespace checks required; earlier pending Next action unchanged and not started.

### 2026-11-26T16:42:39+01:00 — Timing configuration implemented; bounded validation prepared

Implementation a454ef3542fbde1c8c4988c9fe6d22f031f67722. Verification commands: python3 -m unittest discover -s feature_production/tests -p 'test_*.py' (53pass); corresponding convertData test_unified_shower_features.py (13pass) and test_validate_unified_shower_feature_shards.py (4pass); bash feature_dev/run_export.sh convertData/test_scifi_timing_sndsw.py (1pass). LCG107 Python3.11.9 ROOT6.34.02; SNDSW Python3.9.25 ROOT6.36.04, lxplus939. Parser probe initially used outdated signature, corrected to installedAPI;21rules parse/compile. Dry-run3sources passes. Evidence feature_dev/output/timing-selection-20261126/.

Authorized batch command for each upper in1.2,1.8,2.0: ./run_shower_features.sh --dataset testbeam --plugin silvia --dataset-config feature_dev/output/timing-selection-20261126/testbeam.json --sample small --files-per-group 1 --events-per-file 20 --batch --jobs 2 --scifi-data-window-cycles -0.5 UPPER --scifi-mc-window-cycles -0.5 1.2 --run timingTAG-20261126. Includes oneMC source from preceding validated preview, one2024 data source and one2023H8 source. Logs batch-TAG.log; output root as Current state. Status via controller logs and targeted condor_q cluster IDs. This is current authorized validation, not the pending publication/full-production action.

Prior current state preserved:

## Current state

- Lifecycle: active
- Worktree: `/afs/cern.ch/work/z/zhibin/worktrees/snd-ml-shower-features`
- Branch: `task/shower-features`
- Last verified: 2026-11-26T16:29:06+01:00
- Implementation checkpoint: 5c655ed39469557b418fb724ca9278fbbf40202d
- Implemented: Merged origin/task/silvia-yield-audit (ee3a554), including origin/task/silvia-shower-algorithm (7747bc2), with local testbeam fix89685db and checkpointccbcf53. Merge5c655ed preserves both histories; no conflicts. Incoming density example/schema/tests, energy helper, default energy and two density-versus-energy plots, audit helpers, and shape-to-yield source reuse. Removed four incoming trailing-whitespace errors. Production plugin missing-coordinate handling and testbeam safeguards retained. Published to GitHub origin/task/shower-features throughfdb7ee5 (remote verified after publication); implementation remains5c655ed.
- Validation: Read-only timing configuration trace checked worker, converter_arguments, launcher reuse checks, separate MC-muon parser, Hit3D converter and unified-shower Snakemake rule. Read29-page local 26th collaboration meeting PDF via pdfinfo/pdftotext; visually inspected slides8,15,20,21,22. SciFi source trace and import-only production-wrapper probe passed on lxplus939 (Python3.9.25/ROOT6.36.04); selector resolves to master-local14 with limits(-0.5,1.2) and clock6.237680580852816ns. Publication: git push origin task/shower-features succeeded (6f02516..960a095); pwd/branch and clean tree verified before push. pwd/branch verified and starting tree clean; git fetch origin; branch logs/diffs reviewed. LCG107 on lxplus9116, Python3.11.9, ROOT6.34.02, numpy1.26.4, uproot5.3.11. Before merge45 production tests passed; after merge47 production,6 density pytest,13 converter tests passed. Audit CLI --help passed. git diff --cached --check passed. Both coworker tip and89685db verified ancestors of merge. Main launcher --check-handoff shower-features: PASS; final git diff --check: PASS.
- Remaining limitations: Configurable timing-window flags are proposed only, not implemented; launcher/plugin parameters currently cannot select this window. Slides do not fully specify occupancy-ratio formula, fit coefficients, or timing-window sign convention; adaptive selection not implemented or validated here. SciFi selection source and actual import verified; no event-level timing comparison or data campaign executed. No merged production campaign or full yield-audit generation run. Older personal helper scripts feature_dev/algorithms/plot_energy.py and feature_plugins/silvia/build_audit.py retain hard-coded coworker paths; not executed or general-purpose entry points. Prefer parameterized feature_production/tools/build_audit.py. Added plugin-directory helper changes plugin fingerprint, so some older runs cannot reuse sources; converter/catalogue changes also retain compatibility checks. Existing testbeam full-production/coworker-account and custom SNDSW qualification limits remain. No fresh batch validation after merge.
- Running jobs: No jobs submitted during integration; prior16 testbeam preview clusters were verified complete in preceding checkpoint. No new output campaign.
- Outputs: Slide renderings feature_dev/output/slides-26cm-review/page-{08,15,20,21,22}.png (ignored review artifacts). Integration evidence feature_dev/output/merge-silvia-20261125/{before-tests.log,production-tests.log,density-tests.log,converter-tests.log,audit-help.log}. Prior preview /afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/testbeam/mc-validation-20261125/ and catalogue repair evidence feature_dev/output/testbeam-fix-20261125/ retained. Merge changed11 files listed in history; publication follow-up changes only this task record; no new scientific output.

### 2026-11-26T16:52:12+01:00 — Baseline timing preview source validation

All3sources passed20-event source validation/parity. MC source compared against mc-validation-20261125: all451branches exactly equal (rtol=atol=0); evidence default-parity.json. Jobs16624920,16624921,16624961 sources and16624979 ranges complete;16624991 histograms pending. Job tokens/controller logs recorded in feature_dev/output/timing-selection-20261126/batch-12.log and feature_dev/output/batch-20261120/venv/htcondor_logs/<token>/. MC-muon and Hit3D --help smoke checks pass in actualSNDSW; logs muon-help.log/hit3d-help.log. No code changes after a454ef3.

### 2026-11-26T17:03:27+01:00 — First preview complete; wider-window sources verified

timing12 finished7/7 steps with ROOT/PDF outputs. timing18 all3sources validated20events each; event identities unchanged. Selected-hit totals at1.2/1.8: H8data1/1,2024data299/300,MC22884/22884. H8preview is sparse and unsuitable for choosing a physics window. Current histogram cluster16625148; submitted job IDs/tokens in feature_dev/output/timing-selection-20261126/jobs.json. No code changes after a454ef3.

### 2026-11-26T17:11:36+01:00 — Two timing previews complete; final preview underway

timing12 and timing18 controllers exited0 with source/histogram/PDF outputs. timing20 clusters16625194/16625195 source conversion complete; final source cluster16625231 submitted. Code remains frozen at a454ef3; no full campaign or publication started. Final event/hit/receipt/PDF comparison will follow completion.

### 2026-11-26T17:20:56+01:00 — Fixed timing windows fully tested and checkpointed

Implementation a454ef3542fbde1c8c4988c9fe6d22f031f67722; no implementation edits during batch runs. All3 previews completed7/7steps;18jobs exit0, final targeted condor_q empty. Clusters: 16624920, 16624921, 16624961, 16624979, 16624991, 16625029, 16625055, 16625056, 16625091, 16625104, 16625148, 16625179, 16625194, 16625195, 16625231, 16625252, 16625279, 16625292. Per-job token/log mapping jobs.json. Commands and detailed results in feature_dev/TEST_RESULTS.md; tests53+13+4+1=71pass. Verification command python3 feature_dev/output/timing-selection-20261126/verify_comparison.py passed (source/stage hashes, metadata, event identity, hit containment, exactMC451branch default parity and cross-window equality, all78configuredPDFpaths). Six pdfinfo checks pass. Three output paths timing12/timing18/timing20-20261126 under testbeam root above. Bounded180converted entries represent60distinct source events repeated across3policies. H8sparse/zero-QDC extra datahit limits physics conclusions; no adaptive/default change or full campaign. Final checkpoint includes prior review history and only task documentation; not pushed.

Implementation changed files:

```text
convertData/digi_2_features.py
convertData/digi_2_features_mc_muon.py
convertData/digi_2_hits3D.py
convertData/merge_unified_shower_feature_shards.py
convertData/scifi_timing.py
convertData/test_scifi_timing_sndsw.py
convertData/test_validate_unified_shower_feature_shards.py
convertData/validate_unified_shower_feature_file.py
feature_dev/DATA_FORMAT.md
feature_dev/sources.py
feature_production/README.md
feature_production/cli.py
feature_production/tests/test_scifi_timing.py
feature_production/worker.py
snakemake/rules/unified_shower_features.smk
```

Follow-up docs: feature_production/README.md clarifies upstream event selection is unchanged; feature_dev/TEST_RESULTS.md records acceptance; docs/tasks/active/shower-features.md records handoff. Required main-launcher --check-handoff and git diff --check performed before commit. Pending Next action not started.

### 2026-11-26T17:31:48+01:00 — Standalone SciFi clock-cycle guide

Added feature_dev/SCIFI_CLOCK_CYCLE.md and discovery links in README_SHOWER_FEATURES.md and feature_production/README.md. Documents cycle vs histogram bin, per-plane peak, inclusive signed offsets, data/MC routing, muon first-arrival/rounded-period exceptions, legacy path-based 25 ns cut, CLI/YAML usage, fresh conversion requirements, function locations and ROOT provenance. Verified against implementation a454ef3542fbde1c8c4988c9fe6d22f031f67722 and installed master-local14 selector; no implementation edits. Preserved preceding uncommitted code-reference checkpoint and linked historical records.

Commands/results: pwd and git branch --show-current confirmed authorized worktree/branch; targeted sed/rg inspected code and README; LCG Python link/fence check passed; git diff --check passed; main launcher python3 /afs/cern.ch/work/z/zhibin/snd-ml/docs/tasks/manage_task.py --check-handoff shower-features passed. Changed files: the new guide, two READMEs, this checkpoint. No production tests/jobs or generated output changes; previous output/evidence paths and validation limits remain as recorded above. Documentation-only task commit appropriate; no push or next-action execution.
