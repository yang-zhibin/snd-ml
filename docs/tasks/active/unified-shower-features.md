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

- Phase 2 contract review completed on 2026-10-06. The production version,
  output roots, common source fields, canonical labels, and testbeam physical
  source boundary are frozen below. This documentation milestone is not yet
  committed.

## Phase 2 frozen production contract

### Version and paths

- Production/schema version: `unified_shower_features_v1`.
- Metadata output-path column: `feature_scifi_cluster_v1_path`.
- The existing `feature_path` column keeps its current meaning and must not be
  overwritten or repointed.
- Feature ROOT products use these isolated EOS roots:
  - `/eos/experiment/sndlhc/users/zhibin/snd-ml/unified_shower_features_v1/features/mc_neutrino`
  - `/eos/experiment/sndlhc/users/zhibin/snd-ml/unified_shower_features_v1/features/mc_muon_up`
  - `/eos/experiment/sndlhc/users/zhibin/snd-ml/unified_shower_features_v1/features/real_data_2024_skim`
  - `/eos/experiment/sndlhc/users/zhibin/snd-ml/unified_shower_features_v1/features/testbeam_mc_2024`
  - `/eos/experiment/sndlhc/users/zhibin/snd-ml/unified_shower_features_v1/features/testbeam_data_2024`
- Metadata, summaries, and plots use the AFS work-volume root
  `/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/`,
  with `metadata/`, `summaries/`, and `plots/` subdirectories. No large ROOT
  product belongs on the AFS user volume.

### Common event-level source fields

- Retain `mc_incoming_energy` and all existing branches unchanged.
- Add `source_energy_gev` as ROOT `Double_t` (`/D`). Its unavailable sentinel
  is finite value `-1.0`.
- Add `source_pdg_code` as ROOT `Int_t` (`/I`). Its unavailable sentinel is
  `0`.
- Add `source_domain_id` as ROOT `Int_t` (`/I`) with this stable enumeration:
  `0=unknown/reserved`, `1=mc_neutrino`, `2=mc_muon_up`,
  `3=real_data_2024_skim`, `4=testbeam_mc_2024`, and
  `5=testbeam_data_2024`.
- Production manifests also retain the corresponding string domain name; the
  ROOT branch uses the integer ID above.
- For neutrino MC, MC-muon-up, and testbeam MC, `source_energy_gev` and
  `source_pdg_code` come from the incoming `MCTrack[0]` truth. For neutrinos,
  `source_pdg_code` is the unmodified incoming PDG code; the existing
  `pdgCode` branch keeps its established CC/NC encoding.
- Testbeam-MC catalogue values are validation inputs, not truth overrides.
  Require exact PDG agreement and the established energy comparison
  (`rtol=0.02`, `atol=0.5 GeV`) against MC truth.
- For testbeam data, both values come together from the validated catalogue;
  a missing member of the pair is an error. Catalogue beam energy must not be
  written to `mc_incoming_energy` or described as MC truth.
- For skimmed collision data, use `source_energy_gev=-1.0`,
  `source_pdg_code=0`, and `source_domain_id=3`.

### Canonical sample labels

The only production labels are:

```text
real_data
CC_nue
CC_numu
NC_nue
NC_numu
MC_muon_up
testbeam_mc_electron
testbeam_mc_pion
testbeam_data_electron
testbeam_data_pion
```

`muon` is not an alias for `MC_muon_up`. Testbeam `e-` maps to `electron`;
both `pi+` and `pi-` map to `pion`, while their signed PDG codes remain in
`source_pdg_code`.

### Frozen testbeam v1 source boundary

- V1 covers the validated full-campaign physical-source catalogue, not every
  eligible row in the two legacy metadata files.
- The source boundary is the 90 rows in
  `/eos/user/z/zhibin/TestBeam/binary_full_v1/catalog/master_sources.csv`
  (SHA-256
  `b57ee0e8f8404de2ae665a84360aff79c2b6dc6e12116a6dcf3afe61b52fc735`).
- Its geometry evidence is
  `/eos/user/z/zhibin/TestBeam/binary_full_v1/catalog/geometry_manifest.csv`
  (SHA-256
  `07f6464fda566eacbc95b8956f960651894ae3407ab46fcaaea76514eb58c090`),
  and its validation report is
  `/eos/user/z/zhibin/TestBeam/binary_full_v1/catalog/validation_report.json`
  (SHA-256
  `4086beb307f8db45096e40ab4926dd38955bf32c2bfb94c057932f401f012c27`).
- The implementation reference remains commit
  `481e2426301dcf1ef113e9d170da2ea186680075`; it is read-only source material.
- All 90 selected sources use audited `W_2walls_28plates` geometry. The
  geometry audit contains 106 valid W2 paths and one invalid `no_absorber`
  path corresponding to the recorded `Fe_1wall` exclusion. `Fe_1wall` must
  not enter v1.

Selected physical-source counts:

| Domain | Particle | Energy (GeV) | Source files |
| --- | --- | --- | ---: |
| testbeam MC | electron | 50, 100, 150, 200, 250, 300 | 5, 4, 5, 6, 6, 6 |
| testbeam MC | positive pion | 50, 100, 150, 180, 200, 250, 300 | 6, 5, 6, 9, 6, 5, 6 |
| testbeam data | electron | 50, 100, 150, 200, 250, 300 | 2, 1, 1, 4, 1, 1 |
| testbeam data | positive pion | 180 | 4 |
| testbeam data | negative pion | 180 | 1 |

The source files therefore comprise 75 MC files and 15 data files. Relative
to the legacy catalogues (105 MC and 860 data rows), the exclusion report is:

| Domain | Exclusion reason | Rows |
| --- | --- | ---: |
| testbeam MC | eligible W2 row outside the validated physical catalogue | 30 |
| testbeam data | eligible W2 row outside the validated physical catalogue | 365 |
| testbeam data | geometry is not valid W2 (`Fe_1wall`/`no_absorber`) | 479 |
| testbeam data | missing or invalid beam energy | 1 |

Source identity is the catalogue `source_uid`; event identity remains
`(source_uid, runId, eventId, eventIndex)` because raw event IDs can repeat
across independent testbeam files.

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
- Phase 2 reused the established names `feature_scifi_cluster_v1_path` and
  `MC_muon_up` where the merged workflow already had stable conventions.
- The neutral source fields are additive. They separate raw incoming truth
  from the legacy neutrino classification encoding and separate catalogue beam
  metadata from MC truth.

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
  continuity record, including the Phase 2 frozen contract and scope report.
- All other merged paths are carried by the source commits and merge commit
  listed above; no source file was edited during conflict resolution.

## Validation

- Focused LCG Python test sequence recorded above - passed with exit code 0.
- `/afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake
  --snakefile snakemake/snakefile --list-target-rules` - passed with exit code
  0 and ran no jobs.
- `git diff --check` - passed before adding this task record.
- Phase 2 read-only LCG preflight: host `lxplus924.cern.ch`, Python 3.11.9,
  pandas 2.2.2.
- A pandas audit of the two legacy testbeam metadata files, the geometry
  manifest, and the frozen master-source catalogue reproduced 105 MC rows,
  860 data rows, 90 selected physical sources, every selected
  domain/particle/energy count above, and complete exclusion accounting.
- No Python command wrote a product; no Snakemake target or scheduler job was
  run.

### Phase 3 implementation checkpoint (2026-10-06)

- Added the frozen source-provenance contract to both feature converters,
  including schema metadata, domain IDs, truth/catalogue validation, and
  collision-data sentinels.
- Added deterministic five-family manifest construction. Repeated in-memory
  construction produced identical frames with 1,866 neutrino, 15 muon-up,
  11,368 collision-data, 75 testbeam-MC, and 15 testbeam-data sources. Each
  family has exactly one deterministic canary.
- Added per-file ROOT validation and JSON validation records. A published
  feature is complete only when both the ROOT file and its validation record
  exist; a zero-byte marker is not accepted.
- Added separate canary and full aggregate targets for every dataset family.
  These family aggregates are also the supported Snakemake `--batch` rules,
  so large DAGs need not be built monolithically.
- Focused LCG preflight used Python 3.11.9, ROOT 6.34.02, pandas 2.2.2, and
  PyYAML 6.0.1. All 35 focused new and regression tests passed.
- Target listing passed and exposed all 11 unified manifest/canary/full
  aggregate targets. The combined five-canary dry-run passed with 15 jobs.
- A monolithic five-family full dry-run was intentionally interrupted during
  filesystem inventory after several minutes. No rule ran. It was replaced
  by successful independent batch-1 dry-runs for every family:
  `mc_neutrino=1/20`, `mc_muon_up=1/3`,
  `real_data_2024_skim=1/120`, `testbeam_mc_2024=1/5`, and
  `testbeam_data_2024=1/3`.
- No converter rule, file transfer, feature production, scheduler submission,
  histogram production, plotting production, or Region V1 production ran.

### Stage 1 distribution-code checkpoint (2026-10-06)

- Added a separate `unified_shower_distributions_v1` configuration containing
  exactly `density_sndsw_scifi`, `scifi_cluster_qdc_per_hit`, and
  `scifi_qdc_per_hit_direct`, with the established `count_scifi > 35`
  selection.
- Added raw, unscaled histogram production for `CC_nue`, `CC_numu`, `NC_nue`,
  `NC_numu`, and `MC_muon_up`. Electron- and muon-neutrino groups are bound to
  their own `2024_ve` and `2024_vm` source sets so exposure is not double
  counted across flavors.
- Added explicit plot-only normalization. Unit-area output is a probability
  density; 19 fb^-1 scaling accepts only validated simulated/recorded
  luminosity and rejects MC muon-up and event-count-only samples.
- Added family-completeness receipts as an explicit boundary between batched
  feature production and downstream histograms. The audit reads every JSON
  validation record and requires every published feature to be readable, but
  does not pull thousands of converter jobs into a distribution DAG.
- Added Stage 1 aggregate targets for feature audits, 15 raw histograms, three
  unit-area plots, three neutrino-only 19 fb^-1 plots, and the complete Stage 1
  distribution set.
- All 43 focused tests passed, including a synthetic ROOT test that preserves
  raw underflow/overflow, folds only the plot clone, and verifies unit-area
  density normalization. The full Stage 1 dry-run passed with 26 jobs. A
  `1/15` raw-histogram batch dry-run passed with exactly three jobs: manifest,
  completeness audit, and one histogram.
- An initial histogram batch dry-run that directly depended on every feature
  marker was interrupted because it reproduced the large upstream DAG. No
  rule ran; the completeness-receipt boundary above replaced that design.
- No feature, histogram, plot, audit receipt, transfer, or scheduler job ran.

### Canary-harness and local-test checkpoint (2026-10-06)

- Added bounded `--max-events` smoke-test support with ROOT metadata recording
  the available selected entries and configured limit. Production defaults to
  zero and remains unlimited.
- Added isolated local-smoke targets and an entrywise ROOT comparison report.
- The neutrino local smoke test wrote and validated 48/48 events in 6.680 s.
  Its comparison with the existing EM-shower `2024_ve/1` file passed with zero
  mismatches across both trees and every common branch; the only candidate-only
  branches were `source_energy_gev`, `source_pdg_code`, and
  `source_domain_id`.
- The MC-muon-up local smoke test wrote and validated 100/94,120 events in
  3.768 s. It used the legacy `FairEventHeader` compatibility path as expected.
- The 46-test focused LCG regression suite passed. Both HTCondor-profile canary
  dry-runs passed and each resolved one `unified_shower_feature_v1` conversion
  plus its aggregate target.
- Local smoke outputs and the neutrino legacy-comparison JSON are under
  `/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/local_smoke/`.
- No HTCondor job or canonical unified feature output has been produced yet.

## Running jobs

- Target/command: none
- Scheduler and job ID: none
- Submitted: no
- Log: none
- Expected outputs: none
- Status command: `condor_q zhibin -af ClusterId ProcId JobStatus JobBatchName`

## Blockers and open questions

- The implementation remains uncommitted and must be reviewed before any
  runtime authorization.
- Full production must be invoked one family at a time and, for large
  families, one documented Snakemake batch at a time. Batches are numbered
  from 1 through the chosen total and all batches are required for family
  completion.
- Runtime gates remain closed: no real canary or production invocation is
  authorized yet.

## Exact next action

Create a task-only checkpoint commit so canary provenance names the code
actually executed. Then submit only `unified_features_mc_neutrino_canary_v1`
through the HTCondor profile, monitor it to completion, validate and compare it
with the local and legacy outputs, and checkpoint the result before deciding
whether to submit the MC-muon-up canary. Do not run full-family production.
