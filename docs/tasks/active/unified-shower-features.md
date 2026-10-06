# Task: unified-shower-features

Status: active
Last updated: 2026-10-06 Europe/Zurich

## Goal

Integrate the established SciFi shower features with the existing ABCD and
region-evaluation workflow, then extend the unified feature schema to the
approved datasets without changing established feature semantics.

## Work location

- Worktree: `/afs/cern.ch/user/z/zhibin/worktrees/snd-ml-unified-shower-features`
- Branch: `task/unified-shower-features`
- Starting commit: `5a2821cdda6fccc3d1c440c22a09760f1a2400ed`
- Chat/session ID: `not recorded`

## Discovered scope

- The implementation sequence is recorded in
  `docs/plans/proposed/UNIFIED_SHOWER_FEATURE_OPERATION_PLAN.md`.
- The architecture and dataset scope are recorded in
  `docs/plans/proposed/UNIFIED_SHOWER_FEATURE_PRODUCTION_AND_REGION_EVALUATION_PLAN.md`.
- The integration branch starts from the checkpointed ABCD workflow and
  contains the complete committed `task/em-shower` history.
- Testbeam support remains a selective future port. The whole
  `task/3d-cnn-testbeam` branch was not merged.

## Task-specific constraints

- Do not regenerate feature files until the user explicitly authorizes
  production.
- Preserve the established feature branches and add the shower schema
  backward-compatibly.
- Preserve `(runId, eventId, eventIndex)` identity and validate aligned
  feature/Hit3D products.
- Keep testbeam input adaptation separate from this baseline merge.
- Use Snakemake aggregate targets and the documented runtime environments for
  future canaries and production.

## Completed checkpoints

- `361e94935d605416fb64e77209186b12f8aebc75` - checkpointed the EM-shower
  implementation and both unified-workflow plans on `task/em-shower`.
- `5a2821cdda6fccc3d1c440c22a09760f1a2400ed` - checkpointed the recorded ABCD
  task changes after all six focused LCG regression tests passed.
- `7a9ef4410006b63d9c6fde4ed60431c317b580bd` - merged the complete EM-shower
  history into the integration branch without conflicts.

## Findings and decisions

- The EM and ABCD branches share starting commit
  `5793ce82cfc4a1f0bd11c551e2032266ac53c504` and merged cleanly.
- Testbeam committed reference `481e2426301dcf1ef113e9d170da2ea186680075`
  is read-only source material. Its worktree also contains later uncommitted
  selection fixes, so none of that work was copied or merged.
- The merged baseline exposes both EM aggregate targets and Region V1 ABCD
  targets through the main Snakefile.
- No production target, feature conversion, histogram production, plotting
  production, or HTCondor submission was run for this merge.

## Results and evidence

- Merge strategy: Git `ort`, no conflicts.
- LCG preflight: Python 3.11.9, ROOT 6.34.02, NumPy 1.26.4, pandas 2.2.2,
  PyYAML 6.0.1.
- Post-merge EM tests passed: 11 SciFi clustering, 2 partition-builder,
  9 event-display, 8 neutrino-workflow, and 6 MC-muon-workflow tests.
- Post-merge focused ABCD tests passed: DS--SciFi selection and end-to-end
  rejection, original-binning transfer comparison, signal-ratio comparison,
  Veto edge selection, and Veto central rejection.
- `snakemake --list-target-rules` completed successfully and listed the merged
  EM and ABCD targets.

## Modified files

- `docs/tasks/active/unified-shower-features.md` - integration checkpoint and
  continuity record.
- All other merged paths are carried by the source commits and merge commit
  listed above; no source file was edited during conflict resolution.

## Validation

- Focused LCG Python test sequence recorded above - passed with exit code 0.
- `/afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake
  --snakefile snakemake/snakefile --list-target-rules` - passed with exit code
  0 and ran no jobs.
- `git diff --check` - passed before adding this task record.

## Running jobs

- Target/command: none
- Scheduler and job ID: none
- Submitted: no
- Log: none
- Expected outputs: none
- Status command: `condor_q zhibin -af ClusterId ProcId JobStatus JobBatchName`

## Blockers and open questions

- The exact testbeam behavior must later be selectively ported from a clean,
  reviewed source boundary; the current testbeam worktree is not clean.
- Production schema names, output roots, canonical labels, and eligible
  testbeam source scope still require the Phase 2 freeze in the operation
  plan.

## Exact next action

Wait for explicit user instruction. Then perform Phase 2 of the operation
plan: review and freeze the unified production version, output paths, source
fields, canonical labels, and testbeam source scope. Do not regenerate any
feature file or start a canary or production job before that review.
