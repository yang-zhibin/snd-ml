# Task: unified-shower-features

## Current preservation checkpoint — 2026-11-12

- User authorized checkpointing current task work and moving worktrees to AFS work.
- Branch: `task/unified-shower-features`; pre-checkpoint HEAD: `559efa55cfd1e738e855ab8c1b0fecc6ec9ce0c1`.
- Existing integrity/coverage/partition fixes are committed at 559efa5. Preserve the remaining energy-summary plot frame/range change. Production metadata migration, feature regeneration and scientific qualification remain separate work.
- Selected source files passed Python/YAML/JSON syntax checks as applicable;
  `git diff --check` passed. Prior scientific test results below are historical
  evidence and were not rerun as part of this storage operation.
- No queued HTCondor jobs or local training/Snakemake controllers were found.
- Exact administrative next action: complete migration/launcher verification,
  then await the user's next task instruction. Do not execute historical
  production commands automatically.


Status: active
Last updated: 2026-10-09 Europe/Zurich

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

### HTCondor neutrino-canary checkpoint (2026-10-06)

- Checkpoint commit: `03b3d97f0a256238a20b5ad1c3e0b36ea2363fba`.
- Target: `unified_features_mc_neutrino_canary_v1`.
- HTCondor cluster: `16420968`; final status 4, exit code 0.
- Worker: `b9p13p0054.cern.ch`; converter time 11.816 s; recorded peak
  resident set about 425,000 KiB.
- Output: `/eos/experiment/sndlhc/users/zhibin/snd-ml/unified_shower_features_v1/features/mc_neutrino/unified_src_79ff57f0d342b7b283ba/feature_scifi_cluster_v1_unified_src_79ff57f0d342b7b283ba.root`.
- Validation passed for 48/48 entries with the frozen unified and SciFi-cluster
  schemas. The JSON records the checkpoint commit and SHA-256
  `dd21eb5e42f88147297b19bf6094baf9f9e50cea169843cc23cf317b444c3334`.
- Local-versus-remote and legacy-EM-shower-versus-remote comparisons both
  passed with zero mismatches across `sndData`, `cutFlowSummary`, and every
  common branch. The legacy comparison accepted only the three frozen unified
  source branches as additions.
- Worker log directory:
  `/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/htcondor_log/1_2f13394b-a5c8-4e06-9a00-8db10b6e7c74/`.
- Comparison reports are under the local-smoke `mc_neutrino` directory.

### Count-contract fix checkpoint (2026-10-07)

- The attempted full neutrino production submitted 373 workers before
  stopping: 46 production jobs completed, 325 workers failed because the
  manifest input count was incorrectly treated as the selected output count,
  and two workers failed during transient SNDSW activation. No jobs remain in
  the queue. Together with the original canary, 47 validation records exist.
- Root cause was reproduced with `2024_ve/244`: the digi `cbmsim` tree and
  source cutflow each have 70 rows, but source row/raw entry 40 has
  `SciFiMinHits=0`. The intended converter selection therefore writes 69
  entries, matching the existing EM-shower feature file. The old validator
  incorrectly required 70.
- The manifest now records `input_event_count`. The converter and validator
  share the frozen selection expression. The validator independently checks
  the digi input count, derives selected raw entries from the source cutflow,
  checks converter count metadata and the conversion limit, and validates
  output `eventIndex` plus copied-cutflow `entry` order against the selected
  source entries.
- All 47 focused LCG tests passed: 39 feature/catalogue/partition tests and
  eight Stage 1 distribution tests. New synthetic coverage verifies a
  three-row `[1,0,1]` cutflow produces raw indices `[0,2]` and that a bounded
  conversion keeps only the first selected event.
- The real local `2024_ve/244` regression passed with 70 input, 69 available
  selected, 69 written, and 69 entrywise-validated events. Its SHA-256 is
  `d9b46e3359d05030d645df2cc18c727b9b29bb7e1142486e6dbf8ae2d21283d3`.
  The legacy comparison passed with zero mismatches across 31,050 common
  values. The four expected unified-only branches are
  `scifi_qdc_per_hit_direct`, `source_energy_gev`, `source_pdg_code`, and
  `source_domain_id`.
- Refreshed local smoke targets passed: neutrino 48/48 and bounded muon-up
  100/94,120. The exact HTCondor-profile dry-run for the partition-244
  validation record resolved one `unified_shower_feature_v1` job.
- Local regression output and comparison report:
  `/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/local_smoke/mc_neutrino_filtered_partition244/`.
- Checkpoint commit `1f40e97703759cd3983d781a3a3edf1247ff29f8` was
  submitted for the exact canonical partition-244 validation record only.
  HTCondor cluster `16421376` completed with status 4, exit code 0 on
  `b9p16p0041.cern.ch`; wall time was 52 s and recorded resident set about
  300,000 KiB against a 3,000 MiB request.
- The canonical remote validation passed with 70 input, selection expression
  `SciFiMinHits == 1`, 69 available selected, 69 expected/written/validated,
  and entrywise source identity. Feature SHA-256:
  `8497dffa121219c8d296f0c203cef312bd1ef89617113d5c3e987f4e26dd7a5f`.
- Local-versus-remote comparison passed with zero mismatches across 31,326
  common values. Legacy-versus-remote passed with zero mismatches across
  31,050 common values and only the four expected unified-only branches.
- Canonical output:
  `/eos/experiment/sndlhc/users/zhibin/snd-ml/unified_shower_features_v1/features/mc_neutrino/unified_src_2cd65923b96bca90d6a8/feature_scifi_cluster_v1_unified_src_2cd65923b96bca90d6a8.root`.
- Worker log directory:
  `/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/htcondor_log/0_17578932-ea3c-488f-94f4-430f6619f018/`.
- Comparison reports are `local_remote_comparison.json` and
  `legacy_remote_comparison.json` in the local partition-244 regression
  directory.

### HTCondor muon-up-canary checkpoint (2026-10-07)

- Cluster `16420969` completed with status 4 and exit code 0 after 3,108 s;
  recorded peak resident set was about 1,000,000 KiB against a 3,000 MiB
  request.
- The canonical validation record passed all 94,120 entries. Its first 100
  entries match the refreshed local smoke output with zero mismatches across
  both trees and all common branches.
- Worker log directory:
  `/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/htcondor_log/1_3b875bcc-859a-409d-accd-17b9cac96eff/`.
- Comparison report:
  `/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/local_smoke/mc_muon_up/local_remote_comparison.json`.

### Stage 2 testbeam-canary test (2026-10-07)

- The combined HTCondor-profile dry-run passed for exactly the
  `unified_features_testbeam_mc_2024_canary_v1` and
  `unified_features_testbeam_data_2024_canary_v1` targets.
- The dry-run resolved six jobs: two manifest preparations, two feature
  conversions, and two canary aggregates. It did not run any job.
- The selected MC canary is source
  `tbfull_src_0817c7185e2cbdfa7683`; the selected data canary is source
  `tbfull_src_0891a0e853f2e4ed43bb`.
- Only these two canaries are authorized for the live test; full Stage 2
  production remains out of scope.
- The testbeam-MC canary cluster `16421400` completed with exit code 0 after
  194 s. Validation passed for 960 selected/written events from 1,000 input
  events; feature SHA-256 is
  `0fd46950373c1d32b6aa5a2febbc2ffcfef263339f0a732ce98b4e73ab0977c3`.
- The first testbeam-data attempt, cluster `16421399`, failed before
  conversion because the transient SNDSW setup did not find
  `sndsw/latest-event_level_info_output-release` and could not import
  `SndlhcGeo`. Snakemake submitted retry cluster `16421401`.
- At user request, active waiting stopped while retry `16421401` was still
  running unheld on `b9p16p0041.cern.ch`. The local Snakemake controller then
  exited because its grid-status helper could not initialize the retry's
  HTCondor event log; this controller error does not establish the worker's
  final result.

### Large-file sharding implementation checkpoint (2026-10-07)

- Implemented selected-entry range conversion with inclusive start/exclusive
  stop arguments. Raw `eventIndex` and copied-cutflow `entry` values retain
  their original input identity; no per-shard renumbering is performed.
- Enabled sharding only for `mc_muon_up` and `testbeam_data_2024` inputs above
  50,000 events. Shards contain at most 50,000 selected entries and are stored
  under the isolated EOS `unified_shower_features_v1/shards` root.
- Added independent validation records for every shard and an ordered merger
  that requires validated checksums, identical provenance, contiguous complete
  coverage, matching `sndData`/`cutFlowSummary` counts, and full canonical
  revalidation after merge. Canonical feature and validation-record paths are
  unchanged, so downstream Stage 1 and Stage 2 rules need no path changes.
- Avoided materializing a full `TEntryList` for no-selection identity/direct
  sources. Selected ordinals map directly to tree ordinals, eliminating a
  repeated million-entry Python loop in every large-file shard process.
- Added a global wildcard exclusion so legacy `process_features` does not
  claim internal shard outputs.
- LCG 107 regression validation passed 53 relevant tests across clustering,
  partition/catalogue, comparison, unified feature, EM-shower workflow, and
  Stage 1 distribution modules. The shard merger unit test verifies ordered
  entries and aligned `sndData`/`cutFlowSummary` trees.
- A real local muon-up smoke run passed for 100/100 events. A separate real
  selected-range smoke passed for `[10,20)`, writing and validating 10 events
  with raw `eventIndex` values beginning at 10. The temporary range output is
  `/tmp/zhibin/unified_range_smoke.ABgxGS/`.
- HTCondor-profile dry-runs ran no jobs and selected the intended DAGs:
  source `unified_src_2b2faed04c777347f5f7` (205,446 events) resolved to five
  shard jobs plus one merge; testbeam source
  `tbfull_src_933230a6f85f25fe0031` (100,000 events) resolved to two shard jobs
  plus one merge. A forced neutrino target retained the direct single-file
  conversion rule.
- No production or HTCondor job was submitted by this implementation test.
- Added executable helper
  `snakemake/run_remaining_unified_shower_features_htcondor.sh`. It discovers
  missing canonical validation records across neutrino, muon-up, and 2024
  testbeam data, then submits only those exact targets. It defaults to 20
  concurrent jobs and accepts a positive job-count argument. Shell syntax and
  its non-submitting invalid-argument path passed; the submission path was not
  run.

### Compiled validation optimization checkpoint (2026-10-07)

- Replaced the Python per-entry loop for contiguous direct/identity products
  with one ROOT `RDataFrame` scan. ROOT reads only referenced scalar branches;
  the scan still checks every event's raw `eventIndex`, source domain and
  source values, cluster closures, and finite shower-feature values. A
  contiguous expected `eventIndex` also proves tuple identity uniqueness.
- Identity cutflows are now checked in compiled ROOT code and represented as
  a Python `range`, avoiding million-row Python cutflow and list construction.
- Final merged-file validation no longer repeats the full scan already passed
  by every shard. It checks schema and metadata, tree counts, the merge
  checksum, and the first/last event and cutflow identity at every shard
  boundary. Merge receipts now record the ordered shard ranges.
- A real canonical muon-up file with 94,120 events passed the optimized full
  scan in 20.36 s wall time under LCG 107 and 28.59 s under the production
  SNDSW/ROOT 6.36 environment. Both reproduced its existing SHA-256. The
  receipt records `validation_execution=compiled_branch_pruned_full_scan`.
- A deliberately shifted 100-event expected range was rejected for all
  100 entries. The complete 53-test feature/workflow/distribution regression
  suite passed.
- Earlier local shards `part_0000` and `part_0001` for source
  `unified_src_2b2faed04c777347f5f7` are complete with readable ROOT and JSON
  outputs. A dry-run using `--rerun-triggers input params` reuses both and
  schedules only the three missing shards plus the merge. The remaining-files
  helper now uses this policy so newer implementation mtimes do not rerun
  already validated shards.

### Testbeam shard-size adjustment (2026-10-07)

- The user canceled the first 50,000-event local testbeam-data shard after the
  muon shard completed. No testbeam shard ROOT or validation marker had been
  published, so changing the part boundaries cannot reuse a conflicting
  product.
- Replaced the single shard size with dataset-specific configuration:
  `mc_muon_up` remains 50,000 selected events per shard and
  `testbeam_data_2024` now uses 10,000. The measured earlier testbeam rate
  predicts about 20 minutes per 10,000-event shard instead of roughly
  100 minutes for 50,000 events.
- The focused unified-feature suite now has 12 passing tests, including a
  frozen assertion for both family sizes. The 100,000-event testbeam source
  dry-run resolves ten shard jobs plus one merge and manifest preparation.
- Manifest refreshes would otherwise reconsider completed shards. The
  remaining-files helper now marks the shard rule's manifest input ancient
  while retaining `input` and `params` rerun triggers. A muon dry-run reuses
  completed parts 0000--0002 and schedules only parts 0003--0004 plus merge.
- Converter progress reporting is now adaptive. A 10,000-event shard reports
  every 1,000 events and a 50,000-event shard every 5,000; this is logging
  only and does not change feature content.

## Running jobs

- Target/command: all remaining canonical neutrino-MC, muon-up-MC, and 2024
  testbeam-data feature/validation products, submitted with
  `./snakemake/run_remaining_unified_shower_features_htcondor.sh` from commit
  `b78bbc10061c169e217411bd714059585c68c86d`. The helper uses the repository
  HTCondor profile with 20 concurrent jobs, `--rerun-triggers input params`,
  the shard manifest input marked ancient, `--rerun-incomplete`,
  `--latency-wait 120`, `--keep-going`, and `--drop-metadata`.
- Submitted: 2026-10-07 12:14 Europe/Zurich.
- Scope: nine missing canonical validation targets (one neutrino, five
  muon-up, and three 2024 testbeam-data sources). The resolved DAG contains
  219 jobs: three manifests, 207 shards, one direct neutrino conversion, and
  eight merges.
- Scheduler and job IDs: the first 20 shard workers were clusters
  `16425257`--`16425276`. All 20 moved from idle to running
  (`JobStatus=2`) at 13:11 CEST with no hold reasons. Workers then exposed a
  deterministic testbeam shard-planning failure described below. The
  Snakemake controller was stopped at 13:13 CEST to prevent a retry storm;
  because the profile has no cluster-cancel hook, its 18 remaining orphaned
  retry clusters `16425288`--`16425305` were explicitly removed. A 13:13:59
  queue check showed no remaining jobs with `ClusterId >= 16425257`.
- Snakemake log:
  `.snakemake/log/2026-10-07T121400.564002.snakemake.log`.
- Expected outputs: canonical ROOT feature files and `.validated.json`
  records for source UIDs `unified_src_3086e020d1583cf5f19e`,
  `unified_src_2b2faed04c777347f5f7`,
  `unified_src_2d53f3013ca0df5a8f37`,
  `unified_src_792fd96cf7aedb0fd9d2`,
  `unified_src_7a429b72e0822899defc`,
  `unified_src_a5375fa9e082fc6fa6a4`,
  `tbfull_src_69f577a86697a8eaff05`,
  `tbfull_src_933230a6f85f25fe0031`, and
  `tbfull_src_a8c40f4cde8dfe5890e3`.
- Status commands: `condor_q zhibin -constraint 'ClusterId >= 16425257' -af
  ClusterId ProcId JobStatus JobBatchName RemoteHost HoldReason` and
  `tail -f .snakemake/log/2026-10-07T121400.564002.snakemake.log`.

### Production submission failure checkpoint (2026-10-07 13:13 CEST)

- This is a deterministic planning/count-contract failure, not a worker,
  environment, geometry-warning, memory, disk, or walltime failure.
- The generated testbeam manifest records `input_event_count=1000000` for
  source `tbfull_src_a8c40f4cde8dfe5890e3`, so the checkpointed 10,000-entry
  shard planner generated 100 parts.
- The frozen identity cutflow for that source exposes only 50,000 selected
  entries. The converter therefore rejected every attempted range ending
  above 50,000 before event processing. Representative worker failures were
  `ValueError: selected-entry stop 150000 exceeds available selected entries
  50000` (cluster `16425266`), the same error for stop 450,000 (cluster
  `16425267`), and stop 460,000 (cluster `16425270`).
- Before the stop, the Snakemake log recorded 25 shard-rule errors and 20
  restart attempts. The controller had submitted clusters through
  `16425305`; no job remains queued from this submission.
- Failed jobs terminated normally with exit code 1 after about 40 seconds and
  used under 150 MB, confirming that increasing resources would not help.
- No implementation fix was made during monitoring. The required correction
  is for manifest/shard planning to use the selected-entry cardinality of an
  `identity_cutflow` source (50,000 here), while retaining the raw digi entry
  count separately for provenance/validation. The regenerated DAG should
  contain five 10,000-entry shards for this source, not 100.

### Identity-cutflow shard-count fix checkpoint (2026-10-07 14:43 CEST)

- Added a separate `selected_event_count` manifest field. Direct-input
  sources set it to the raw input count; testbeam identity-cutflow sources
  read it from the frozen catalogue `n_events` field. The existing
  `input_event_count` remains the raw digi count used for provenance.
- Sharding eligibility still uses the configured raw-input threshold, while
  shard boundaries now use `selected_event_count`. The range helper rejects
  negative counts, selected counts above the raw count, and invalid shard
  sizes.
- Extended validation with `--expected-selected-event-count`. Identity
  cutflow cardinality is checked against the selected count while the digi
  tree is independently checked against the raw input count.
- The focused LCG suite passed all 13 unified-feature tests. New regression
  coverage proves that a 1,000,000-entry raw source with 50,000 selected
  entries produces exactly five 10,000-entry ranges and preserves the
  shortened final MC-muon shard behavior.
- Regenerated the testbeam-data manifest. Representative source
  `tbfull_src_a8c40f4cde8dfe5890e3` now records
  `input_event_count=1000000` and `selected_event_count=50000`.
- The exact HTCondor-profile dry-run resolved five shard jobs plus one merge,
  with ranges ending at 50,000. It submitted no job.
- The first local canary conversion completed 10,000 events but exposed the
  validator's remaining raw-versus-selected assumption before publication.
  That validator contract was fixed and the focused suite passed again.
- The repeated local canary then passed end to end for range `[0,10000)`:
  10,000 written and compiled-validated events out of 50,000 available
  selected entries and 1,000,000 raw input entries. Converter time was
  1,001.675 s. The published feature is 72,968,635 bytes with SHA-256
  `430a1cd2cb61026dbe6c585bbaaf33054a7942ebc4b131d8540d40a76bd1fa1b`.
- Canary validation record:
  `/eos/experiment/sndlhc/users/zhibin/snd-ml/unified_shower_features_v1/shards/testbeam_data_2024/tbfull_src_a8c40f4cde8dfe5890e3/part_0000/feature_scifi_cluster_v1_tbfull_src_a8c40f4cde8dfe5890e3_part_0000.root.validated.json`.
- Successful canary Snakemake log:
  `.snakemake/log/2026-10-07T142542.222237.snakemake.log`.
- No remaining shard, merge, full-production, or downstream-distribution job
  was started.

### Selective remaining-production helper checkpoint (2026-10-07 14:50 CEST)

- Extended `snakemake/run_remaining_unified_shower_features_htcondor.sh`
  with repeatable `--exclude-source-uid` arguments, `--jobs N`, and a
  non-submitting `--dry-run` option. The original positional positive job
  count remains supported.
- Exclusions are validated for UID syntax and presence in the three input
  manifests before Snakemake is invoked. Excluded sources are removed while
  discovering missing canonical validation targets, so none of their shard
  or merge dependencies enter the DAG.
- Shell syntax and help output passed. An invalid UID was rejected with exit
  code 2 before Snakemake invocation.
- The exact dry-run excluding testbeam source
  `tbfull_src_a8c40f4cde8dfe5890e3` and MC-muon source
  `unified_src_792fd96cf7aedb0fd9d2` passed. It selected seven remaining
  canonical sources and resolved 37 jobs: two manifest preparations, 28
  shards, six merges, and one direct neutrino conversion.
- No HTCondor job was submitted by this test.

## Blockers and open questions

- The testbeam `identity_cutflow` raw-versus-selected count defect is fixed
  and has passed the required five-shard dry-run and real local shard canary.
- The corrected neutrino workflow has passed the formerly failing
  partition-244 case locally and through HTCondor, including both comparisons.
  The combined production attempt was authorized, but was stopped while its
  first testbeam wave was failing; it did not reach the neutrino and muon-up
  work in this submission.
- Two production workers also saw transient failure to load
  `sndsw/latest-event_level_info_output-release`; this is distinct from the
  corrected count-contract defect and remains a retry/activation-hardening
  concern for mass production.

## Exact next action

Run the tested filtered helper command only when the user is ready:
`./snakemake/run_remaining_unified_shower_features_htcondor.sh --jobs 20
--exclude-source-uid tbfull_src_a8c40f4cde8dfe5890e3
--exclude-source-uid unified_src_792fd96cf7aedb0fd9d2`. Do not start the two
excluded sources or downstream distributions.

### Real-data selection-index milestone (2026-10-07 15:54 CEST)

- The existing production still had 20 idle HTCondor jobs (`16425813` through
  `16425832`), so no converter, validator, unified-feature configuration, or
  active unified-feature rule used by those jobs was changed.
- Added an isolated, standalone selection-index workflow at
  `snakemake/unified_real_data_selection.smk`. It is not included by the main
  Snakefile and cannot change the active production DAG.
- Added the frozen `real_data_2024_selection_v1` policy: base collision-data
  quality plus SciFi selection, valid system-1 MuFilter hits for the veto
  category, and deterministic SHA-256 sampling of
  `(runId,eventId,eventIndex)` into bucket 0 of 10.
- Added deterministic 11,368-source manifest construction, paired no-veto and
  veto-10% ROOT index production, independent index validation, and local
  smoke/canary/full-index aggregate targets.
- Four focused pure-Python policy tests passed. The standalone Snakefile
  parsed and exposed four targets. An in-memory manifest audit reproduced
  11,368 unique sources and one canary.
- Metadata row 0 had zero base-selected events and was rejected as a useful
  canary. Read-only cutflow auditing selected row 10, which has 36,930
  base-selected entries.
- The bounded 1,000-entry local smoke passed end to end with count
  conservation: 998 veto events, 2 no-veto events, and 102 deterministic
  veto-10% events. Independent validation checked category membership,
  sampling membership, ordering, bounds, hashes, and source identity.
- Successful smoke log:
  `.snakemake/log/2026-10-07T155310.132564.snakemake.log`.
- Smoke products are under
  `/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/local_smoke/real_data_selection/`.
- No real-data HTCondor job or full-source selection canary was started.
- Integration of the new arbitrary-entry indexes into unified feature
  production still requires the planned `indexed_cutflow` mode, catalog
  aggregation, two feature families, sharding, and feature-level canaries.

## Exact next action (supersedes the preceding action)

Wait until HTCondor jobs `16425813` through `16425832` have drained before
editing shared converter or unified-feature rule files. Then add and test the
`indexed_cutflow` integration for the two real-data feature families.

### Three-source real-data selection test (2026-10-07 16:08 CEST)

- The first local attempt was interrupted with the chat/tool process before
  publication. A process and output audit found no surviving worker and only
  empty EOS directories, so the same targets were safely restarted with
  `--rerun-incomplete`.
- Three full-source selection-index jobs ran locally with two workers; no
  HTCondor job was submitted.
- All three builders conserved the base selection and all three independent
  validators reported `status=passed`:
  - `unified_src_7a5432a6b551bafb15f4`: 36,930 base-selected, 68 no-veto,
    36,862 veto, and 3,622 veto-10% events.
  - `unified_src_3fb91153d0c208ac2011`: 37,242 base-selected, 55 no-veto,
    37,187 veto, and 3,682 veto-10% events.
  - `unified_src_d4101f3abedc9cecd745`: 37,012 base-selected, 65 no-veto,
    36,947 veto, and 3,680 veto-10% events.
- Every ROOT index, summary, and validation record was published and readable
  under the configured EOS selection-index root.
- A repeat dry-run reported: `Nothing to be done (all requested files are
  present and up to date)`.
- Successful Snakemake log:
  `.snakemake/log/2026-10-07T160430.995494.snakemake.log`.

## Exact next action (supersedes the preceding action)

Wait until the existing unified-feature HTCondor jobs have drained, then
implement and test `indexed_cutflow` integration. Do not submit the full
11,368-source selection-index production yet.

### Real-data bad-source exclusion checkpoint (2026-10-07)

- Added configuration-level exclusions for
  `unified_src_0ee91078a115e62f0a98` and
  `unified_src_17d1dbbd67bc0d9ebda2`; both have missing Run-9258
  `nueAnalysisFilter` inputs.
- The standalone selection workflow validates that every configured excluded
  UID exists in the source manifest and removes exclusions only from aggregate
  targets. Direct source rules and the full provenance manifest remain intact.
- The deterministic source manifest still has 11,368 rows; the aggregate has
  11,366 valid candidates. Installed Snakemake 9.17.2 resolves batch `1/10`
  to exactly 1,137 validation targets.
- The exact HTCondor-profile batch dry-run completed successfully with no
  `MissingInputException`. No selection job was submitted.
- `git diff --check` passed and no patch backup/reject artifacts remain.

## Exact next action (supersedes the preceding action)

When the user explicitly authorizes submission and no other Snakemake
controller is using this worktree, run batch `1/10` without `--dry-run`.
Do not run batches `2/10` through `10/10`.

### Frozen partial-distribution workflow checkpoint (2026-10-07 17:25 CEST)

- Added the isolated standalone workflow
  `snakemake/unified_partial_distributions.smk`; it is not included by the
  main Snakefile and writes only beneath versioned `partial` output roots.
- Added `unified_partial_distributions_20261007_v1`, covering the three frozen
  SciFi shower features with the common `count_scifi > 35` base selection.
  Stage 1 provides unit-area overlays for neutrino/muon-up MC and preliminary
  19 fb^-1 scaling for neutrino MC only. Stage 2 provides unit-area testbeam
  MC/data comparisons for inclusive electrons, 50/100/200 GeV electrons, and
  180 GeV pions.
- Added a snapshot builder that accepts a source only when both the feature
  ROOT file and a matching successful validation record exist. It records
  source-manifest SHA-256 values, availability counts/fractions, status
  counts, and available exposure in JSON and CSV coverage products.
- Froze the snapshot at
  `/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/metadata/partial_distributions/unified_partial_distributions_20261007_v1/`:
  neutrino MC 1865/1866 (27,975 fb^-1 simulated exposure), muon-up MC 10/15,
  testbeam MC 75/75, and testbeam data 12/15. Missing testbeam data are the
  150, 250, and 300 GeV electron sources, so this snapshot intentionally does
  not schedule those three per-energy comparisons.
- Verified representative testbeam MC and data files contain the
  `source_pdg_code` and `source_energy_gev` branches used for group selection.
- Corrected distribution compatibility checking to compare the shared base
  selection rather than the full group-specific selection. This permits the
  intended CC/NC and MC/data overlays while still rejecting histograms made
  with another base selection.
- The focused LCG suite passed all 10 tests, including real ROOT histogram and
  two-group plotting coverage. `git diff --check` passed before the final task
  checkpoint, and no histogram, plot, feature-production, or HTCondor job was
  run.
- Snakemake 9.14.1 parsed the standalone workflow. The complete aggregate
  dry-run passed with 71 jobs: four audits, 45 raw histograms, six Stage-1
  plots, 15 Stage-2 plots, and one aggregate. Snapshot log:
  `.snakemake/log/2026-10-07T172321.562292.snakemake.log`.

## Exact next action (supersedes the preceding action)

Wait for user authorization before running
`unified_partial_distributions_v1`. Do not submit histogram or plot jobs yet.

### Geometry-controlled testbeam-distribution implementation (2026-10-07)

- Added the standalone `snakemake/unified_testbeam_distributions.smk` workflow
  and versioned `unified_testbeam_distributions_v2` configuration. It is not
  included by the main Snakefile and does not enlarge the existing Stage-1 or
  Stage-2 production DAG.
- Added fail-closed legacy H8 geometry classification from the audited active
  placements/checksums. The primary measured-pion family accepts only
  `Fe_3walls_H8_2023` with checksum
  `4194869a465bfbe2f049effe9407577eba99111334f1960c8d26f857c12de21c`;
  unknown geometry filenames are rejected.
- Added deterministic 2023 H8 manifest construction for positive pions at
  100, 140, and 180 GeV. It reproduced 56, 54, and 89 source files and
  allocates exactly 50,000 raw events per energy across every source in the
  cell.
- Added `indexed_cutflow` as an explicit selection mode and source domain 6
  (`testbeam_data_2023_h8`). Sparse indexes use sorted, unique, evenly spaced
  raw-entry values. The converter already follows an index tree's `entry`
  branch; the independent validator now exercises the arbitrary-entry path
  rather than identity-cutflow validation.
- Added local index and feature rules for the 199 measured 2023 sources. Each
  feature is produced with the established SNDSW converter, independently
  validated for schema/provenance/index membership, and published only after
  validation.
- Added five-feature, unit-area plots for 2024 W2 MC pion/electron energy
  scans, 2023 Fe-three-wall measured positive-pion scans, 2024 W2 measured
  electron scans, six exact-energy W2 MC particle comparisons, the MC
  180/200-GeV mismatch control, and the corresponding W2 measured-data
  mismatch plot.
- Histogram production filters source manifests before validation, requires
  one geometry ID/checksum per geometry-declared group, and records source
  count, available/used entries, event cap, sampling policy, geometry,
  particle charge, and energy in plot summaries. MC comparison cells use
  equal 2,520-event caps; MC electron scans use 2,880; measured cells use
  50,000.
- The focused LCG/ROOT regression suite passed 27 tests. A temporary CLI smoke
  reproduced the 199-source/150,000-event manifest and verified a 7-of-100
  sparse ROOT index. Snakemake 9.17.2 parsed all five target rules. A
  temporary checkpoint plus dry-run expanded to 199 index jobs and 199
  feature jobs (399 jobs including the aggregate).
- `git diff --check` passed and no patch backup/reject artifacts remain. No
  production index, feature, histogram, plot, or HTCondor job was started.

## Exact next action (supersedes the preceding action)

Wait for the user's instruction. Before production, run one local 2023 H8
feature canary from the standalone workflow and compare its event identities
against the generated sparse index. Do not submit all 199 feature jobs or run
the distribution aggregate yet.

### Leading-cluster QDC-spread plotting checkpoint (2026-10-07 17:34 CEST)

- Added `scifi_vertical_cluster1_qdc_std` and
  `scifi_horizontal_cluster1_qdc_std` to the frozen partial-distribution
  feature set. Both use `[0,15]` with width `0.3`.
- Each histogram applies its matching cluster-existence cut in addition to the
  shared SciFi and group selections, preventing `-999` missing-slot sentinels
  from entering the distribution.
- Updated the distribution helper to accept the two approved optional
  features while retaining the three core features, and to compose
  feature-specific cuts explicitly.
- The focused LCG suite passed all 10 tests. Representative validated files
  from all four frozen families contain both new QDC-standard-deviation
  branches and both matching `cluster1_exists` branches.
- Refreshed only the frozen snapshot metadata; coverage is unchanged. Log:
  `.snakemake/log/2026-10-07T173240.211786.snakemake.log`.
- The expanded aggregate dry-run passed with 115 jobs: four audits, 75 raw
  histograms, ten Stage-1 plots, 25 Stage-2 plots, and one aggregate. No
  histogram, plot, feature-production, or HTCondor job ran.

## Exact next action (supersedes the preceding action)

Wait for user authorization before running
`unified_partial_distributions_v1`. Do not submit histogram or plot jobs yet.

### Testbeam source-selection plan checkpoint (2026-10-07)

- Added
  `docs/plans/proposed/TESTBEAM_DISTRIBUTION_SOURCE_SELECTION_PLAN.md` and
  linked it from the proposed-plan index.
- The plan freezes geometry-controlled sources, event balancing, and filters
  for MC pion/electron energy scans, measured pion/electron energy scans, and
  separate MC/data particle comparisons.
- It records the 2023 legacy geometry-metadata defect, the three iron setup
  checksums, the 2023 unified-production prerequisites, and the absence of an
  exact-energy measured pion/electron pair.
- No production, histogram, plot, or HTCondor job ran for this documentation
  action.

## Exact next action (supersedes the preceding action)

Wait for user authorization before running
`unified_partial_distributions_v1`. Do not submit histogram or plot jobs yet.

### Testbeam-distribution implementation final handoff (2026-10-07 22:36 CEST)

- The geometry-controlled implementation checkpoint and its validation
  evidence are recorded above. They were completed after this source-selection
  plan checkpoint.
- Final verification remains 27 passing focused LCG/ROOT tests, a passing
  Snakemake parser/preparation dry-run, and a 399-job temporary-path feature
  DAG dry-run. No production job was started.

## Exact next action (supersedes the preceding action)

Wait for the user's instruction. Before production, run one local 2023 H8
feature canary from `snakemake/unified_testbeam_distributions.smk` and compare
its event identities against its sparse index. Do not submit all 199 feature
jobs or run `unified_testbeam_distributions_v2` yet.

### Shard-native downstream implementation (2026-10-07 23:00 CEST)

- Added `validate_unified_shower_feature_shards.py` and the
  `unified_shower_feature_shards_complete_v1` rule. A source becomes ready
  without a merge only after its validation records prove contiguous
  `[0, selected_event_count)` coverage, consistent source/provenance fields,
  exact digi/cutflow binding, per-shard counts, readable files, and matching
  SHA-256 checksums.
- Added shard-aware family aggregates for MC muon-up and measured 2024
  testbeam data. Existing canonical products remain valid terminal products;
  unfinished sharded sources terminate at `shards.complete.json`.
- Extended the remaining-feature launcher with repeatable `--family` and
  `--keep-shards` options. The two explicitly held sources remain excludable.
  The verified no-submit command selected six unfinished sources and produced
  a 37-job DAG: 31 shard conversions plus six completeness validations, with
  zero merge jobs.
- Extended frozen partial snapshots to prefer an existing canonical feature
  and otherwise expand a validated completeness receipt into ordered shard
  rows. Each row has a unique `feature_unit_uid`; physical `source_uid` is
  retained across shards.
- Extended distribution audit/histogram loading to accept repeated physical
  sources only when feature-unit IDs are unique. `TChain` consumes every
  shard while source counts and luminosity are deduplicated by `source_uid`.
- Focused validation passed: 3 SNDSW/ROOT shard-validator tests and 13
  LCG/ROOT distribution/snapshot tests. The ROOT regression proved identical
  histogram bins for a canonical file and the equivalent two-shard chain,
  with one source and one luminosity contribution. `git diff --check` passed.
- No production, merge, histogram, plot, or HTCondor job was started.

## Exact next action (supersedes the preceding action)

Wait for explicit user authorization. To submit only the unfinished MC
muon-up and measured-testbeam shard jobs while retaining shards, rerun the
verified launcher command without `--dry-run`. Keep excluding
`tbfull_src_a8c40f4cde8dfe5890e3` and
`unified_src_792fd96cf7aedb0fd9d2`. After those jobs finish, rebuild the
partial feature snapshot before running downstream distributions.

### Cross-geometry testbeam V3 implementation (2026-10-08)

- Added isolated `unified_testbeam_distributions_v3` configuration while
  retaining the V2 configuration and outputs. V3 defines four explicitly
  opted-in measured-data overlays: 100/100, 140/150, 180/200, and 300/300 GeV
  pion/electron comparisons across 2023 Fe three-wall and 2024 W two-wall
  geometries.
- Generalized the 2023 H8 manifest builder to configuration-driven
  particle/energy cells. The real 865-row catalogue resolves to 385 sources:
  56 pi+ 100 GeV, 54 pi+ 140 GeV, 89 pi+ 180 GeV, and 186 pi- 300 GeV.
  Each cell receives exactly 50,000 deterministic selected events, for
  200,000 total.
- Removed the hardcoded positive-pion PDG from the 2023 feature rule. Both the
  converter and validator now receive manifest-derived PDG and energy values;
  the validator performs a branch-pruned full-file equality check and records
  the expected values in its receipt.
- Added fail-closed plot geometry policies. Existing plots default to
  `same_geometry`; mixed geometry is accepted only under
  `cross_geometry_diagnostic`, requires complete geometry metadata, and is
  restricted to unit-area normalization. Plot summaries record the policy.
- Added a shard-aware V3 feature-snapshot checkpoint, allowing retained 2024
  data shards to feed histograms without merge files. The dedicated target is
  `unified_testbeam_cross_geometry_distributions_v3`.
- The real-catalogue temporary build confirmed 385 sources and the four
  50,000-event cells. The focused LCG/ROOT suite passed 20 tests, the focused
  SNDSW converter/validator suite passed 16 tests, and `git diff --check`
  passed. The HTCondor-profile dry-run resolved an initial 63-job plotting
  DAG: two checkpoints, 40 histograms, 20 plots, and one aggregate. The
  checkpoint-expanded 2023 feature jobs were not executed.
- No production manifest, selection index, feature, histogram, plot, or
  HTCondor job was started.

## Exact next action (supersedes the preceding action)

Wait for explicit user authorization. After the Stage-1/Stage-2 remainder
finishes, generate the V3 2023 manifest and run one local pi+ 100 GeV feature
canary and one local pi- 300 GeV feature canary. Verify their identities, PDG,
energy, and geometry receipts before submitting the remaining 2023 feature
jobs. Do not run `unified_testbeam_cross_geometry_distributions_v3` until the
required 2024 150 and 300 GeV electron shard receipts and all four 2023 pion
cells are complete.

### V3 manifest-checkpoint invocation fix (2026-10-08 00:29 CEST)

- Diagnosed log `.snakemake/log/2026-10-08T002500.380455.snakemake.log`:
  the local manifest checkpoint exited with argparse status 2 because its
  builder invocation omitted the required `--config` argument. It failed
  before DAG expansion and submitted no HTCondor jobs.
- Added `--config {input.config}` to the checkpoint and a static regression
  assertion. All six focused V3 workflow tests pass, `git diff --check`
  passes, and the repaired HTCondor-profile dry-run reaches the checkpoint
  without an invocation error. No production job was restarted.

## Exact next action (supersedes the preceding action)

Wait for user authorization to rerun `unified_testbeam_2023_features_v1`.
The checkpoint will first write the V3 manifest and then expand the 385-source
selection-index/feature DAG. Do not launch distribution targets yet.

### Two-cohort 2024 real-data distribution implementation (2026-10-08)

- Added the standalone `snakemake/unified_real_data_distributions.smk`
  workflow. It keeps feature production separate from downstream snapshot,
  histogram, and plot targets so plotting the currently complete snapshot
  does not pull missing conversions into the DAG.
- Added a fail-closed manifest builder and configuration for
  `real_data_2024_no_veto` and `real_data_2024_veto_10pct`. The builder
  verifies selection receipts, count conservation, source identity, index
  existence, and SHA-256 before exposing a source. Empty selections are
  omitted, and the smallest non-empty selection is the local canary.
- Feature conversion reuses the EM-shower converter with
  `indexed_cutflow` and real-data domain ID 3. The converter and validator
  enforce the automatic collision-data PDG 0 and energy -1 sentinels. The
  downstream workflow makes ten raw histograms and
  five unit-area overlays for the requested density, two QDC-per-hit, and two
  leading-cluster QDC-standard-deviation features with `count_scifi > 35`.
- The actual selection manifest was checked read-only: 11,368 total source
  rows, 1,140 validated selection sources, 1,110 non-empty sources per
  cohort, 77,051 no-veto selected events, and 3,925,280 sampled veto events.
  The other 10,228 rows currently lack selection-validation receipts.
- Two focused LCG tests pass. The Snakemake target parser, checkpoint dry-run,
  and `git diff --check` pass. No production manifest, feature, histogram,
  plot, or HTCondor job was run.

## Exact next action (supersedes the preceding action)

Run `unified_real_data_distribution_feature_canaries_v1` locally. This first
writes the verified source manifests and then converts the smallest non-empty
source once for each cohort (4 no-veto events and 232 sampled veto events).
Inspect both validation receipts before authorizing full feature production.

### Real-data canary provenance fix (2026-10-08)

- The first two feature canaries failed before conversion because the new
  rule explicitly passed PDG 0 and energy -1. Domain 3 collision data must
  receive those sentinel values internally, not as testbeam-style CLI input.
- Removed the paired PDG/energy options from both converter and optional
  validator expectations. The validator still enforces the collision-data
  provenance contract entry by entry.
- Added a static regression test. Three focused LCG tests, the expanded
  two-feature canary dry-run, and `git diff --check` pass. No canary was
  restarted and no feature output was published during validation.

## Exact next action (supersedes the preceding action)

Rerun `unified_real_data_distribution_feature_canaries_v1` locally and inspect
the two validation receipts. Do not submit the 2,220 full feature jobs until
both canaries pass.

### Combined testbeam particle-energy plots (2026-10-08)

- Added `data_combined_particle_energy` and
  `mc_combined_particle_energy` V3 plot sets. Each produces the five frozen
  feature overlays with unit-area normalization.
- The data plot combines four pion/electron pairs at 100/100, 140/150,
  180/200, and 300/300 GeV. It remains explicitly
  `cross_geometry_diagnostic` because pion and electron geometry are confounded.
- The MC plot combines exact pion/electron pairs at 50, 100, 150, 200, 250,
  and 300 GeV under the shared W two-wall geometry.
- Added paired styling: each comparison pair shares a color, pions are solid,
  electrons are dashed, crowded legends use two columns, and style choices are
  recorded in each CSV summary.
- Added the dedicated aggregate target
  `unified_testbeam_combined_particle_energy_plots_v3`.
- Seventeen focused LCG/ROOT tests pass. The expanded dry-run resolves 113
  jobs: two local checkpoints, 100 reusable raw histograms, ten plot jobs, and
  one aggregate. No histogram or plot job ran.

## Exact next action (supersedes the preceding action)

Wait until no other Snakemake controller is using this worktree, then run
`unified_testbeam_combined_particle_energy_plots_v3` locally with four cores.
No feature-production job is required by this target.

### Downstream-only combined testbeam plotting fix (2026-10-08)

- The first combined target was still connected to the 2023 preparation
  checkpoint. After checkpoint expansion, a refreshed manifest pulled 381
  selection-index and 382 feature jobs into the DAG.
- Added `snakemake/unified_testbeam_combined_plots.smk`, which directly
  reads the frozen feature snapshots and cannot schedule preparation, index,
  or feature-production rules. Existing data histograms are external inputs.
- The EOS inventory has all 40 required data histograms and none of the 60
  required MC histograms. The corrected run therefore builds only those 60
  MC histograms, ten plots, and one aggregate.
- Eighteen focused LCG/ROOT tests pass. The corrected expanded dry-run
  contains exactly 71 jobs and no feature-production jobs.

## Exact next action (supersedes the preceding action)

Run `unified_testbeam_combined_particle_energy_plots_v3` locally with
four cores through the downstream-only Snakefile. This produces histograms
and plots only; it cannot schedule selection-index or feature jobs.

### Combined testbeam plot readability refinement (2026-10-08)

- Combined plots now draw points with statistical error bars. Pions use
  filled circles, electrons use open circles, and paired energies share color.
- Compact particle/energy legend labels, a wider canvas, smaller legend text,
  and geometry abbreviations for data remove the two-column label overlap.
- Reduced the density bin width from 500 to 250 and the four QDC-related bin
  widths from 0.3 to 0.2.
- The isolated workflow now regenerates all 100 combined histograms because
  smaller bins cannot be recovered from the old coarse histograms.
- One frozen 180 GeV pion source currently lacks both feature and validation
  files. Explicit partial-snapshot mode omits only missing pairs, remains
  strict for present validation records, and records the omitted count in
  ROOT and CSV metadata. It never schedules feature production.
- Eighteen focused LCG/ROOT tests pass. The force-all dry-run resolves exactly
  111 jobs: 100 histograms, ten plots, and one aggregate.

## Exact next action (supersedes the preceding action)

Run the isolated `unified_testbeam_combined_particle_energy_plots_v3` target
locally with four cores and `--forceall` so every old coarse-bin histogram is
replaced consistently. This workflow cannot schedule selection-index or
feature-production jobs.

### Combined plot header and line-style refinement (2026-10-08)

- Combined particle-energy overviews now use line histograms: pions are solid,
  electrons are dashed, and equal-energy pairs retain the same color.
- Added a fixed-width 900 by 1050 canvas with a separate upper metadata and
  legend pad, leaving the lower plotting pad free of legend overlap.
- Detailed two-curve comparisons retain point markers and statistical errors.
- Data annotations state the distinct beam periods and geometries; MC annotations
  state the shared geometry. Data legend entries retain their full configured labels.
- Added `unified_testbeam_combined_scifi_cluster_qdc_per_hit_plot_v3` for an
  isolated all-cluster SciFi QDC-per-hit plot test.
- Confirmed all 20 required existing histograms use 75 bins over 0--15, matching
  the configured 0.2 bin width. Eighteen focused LCG/ROOT tests pass. The isolated
  force-plot dry-run contains exactly two plot jobs and one aggregate job.

### Combined normalization, range, and label fix (2026-10-08)

- The two combined overviews now use bin fractions whose bin contents sum to
  one, rather than continuous density; the y axis says `Fraction of events / bin`.
- Replaced Unicode em dashes in data legend labels with ROOT-safe ASCII hyphens.
- All-cluster SciFi QDC per hit now spans 0--7 with 0.1-wide bins. Values above
  7 are excluded rather than folded into an artificial last-bin spike.
- Eighteen focused LCG/ROOT tests pass. The isolated dry-run contains exactly
  20 histogram jobs, two plot jobs, and one aggregate job.

## Exact next action (supersedes the preceding action)

Run `unified_testbeam_combined_scifi_cluster_qdc_per_hit_plot_v3` locally with
four cores, forcing `unified_testbeam_combined_raw_hist_v3` and
`unified_testbeam_combined_plot_v3` only. This regenerates only the affected
feature histograms and combined plots; it cannot schedule feature production.

### Full combined testbeam plot regeneration (2026-10-08)

- Ran the isolated `unified_testbeam_combined_particle_energy_plots_v3`
  target locally on `lxplus9101.cern.ch` with four cores, `--forceall`,
  `--rerun-triggers input params`, `--rerun-incomplete`, `--latency-wait 120`,
  `--keep-going`, and `--drop-metadata`.
- The workflow completed successfully: 111 of 111 steps (100%), comprising
  100 raw-histogram jobs, ten combined plot jobs, and one aggregate job. No
  selection-index or feature-production rule was scheduled.
- Complete Snakemake log:
  `.snakemake/log/2026-10-08T143901.298505.snakemake.log`.
- Verified five non-empty PDFs and five CSV summaries in each output folder:
  `data_combined_particle_energy/unit_area` and
  `mc_combined_particle_energy/unit_area` under the V3 plot directory.
- Each data CSV contains eight plotted groups and each MC CSV contains twelve.
  The summaries include the marker/style, statistical-error draw option, and
  omitted-source metadata.
- The frozen 180 GeV pion data group remains the documented partial snapshot:
  one source missing both its feature file and validation marker is omitted
  explicitly and recorded in output metadata.

## Exact next action (supersedes the preceding action)

Visually inspect the ten regenerated PDFs for readable axes, legends, points,
and statistical error bars. Do not start further production or modify the
plotting configuration until that inspection is complete.

### Combined plot-only rerun correction (2026-10-08)

- The first completed full regeneration exposed stale `histogram_lines`
  settings in both combined plot sets. Changed both to `points_errors`, so
  CSV metadata and ROOT drawing use `E1 P` with statistical errors.
- Restored the documented all-cluster QDC range to 0--7 with 0.1-wide bins
  and removed contradictory unused per-plot range entries.
- The focused LCG/ROOT suite passes: 18 tests in 12.72 seconds.
- A subsequent full regeneration was interrupted and no Snakemake controller
  remains active.
- Verified by dry-run that forcing `unified_testbeam_combined_plot_v3` while
  treating the raw-histogram rule's config and script inputs as ancient
  schedules only the ten plot jobs plus the aggregate. It does not schedule
  raw histograms, selections, or feature production.

## Exact next action (supersedes the preceding action)

Run the plot-only combined target with `--forcerun
unified_testbeam_combined_plot_v3` and `--consider-ancient
unified_testbeam_combined_raw_hist_v3=config,script`, then inspect the ten
PDFs. This intentionally reuses the current histogram files.

### Faceted and energy-summary testbeam plots (2026-10-09)

- Added two downstream-only plot modes to
  `evaluation_region_partitions/unified_feature_distributions.py`:
  `faceted-plot` draws one pion/electron pair per panel, while
  `energy-summary-plot` draws the median and 16--84% distribution interval
  against beam energy.
- MC faceted outputs use a 2-by-3 layout for exact 50, 100, 150, 200, 250,
  and 300 GeV comparisons. Data uses a 2-by-2 layout for the configured
  100/100, 140/150, 180/200, and 300/300 GeV comparisons and labels every
  panel as a cross-geometry diagnostic.
- Pions are blue filled circles and electrons are red open circles. QDC and
  leading-cluster-spread faceted plots use compact visible x ranges without
  changing or regenerating the input histograms.
- Added the isolated rules
  `unified_testbeam_combined_faceted_plot_v3` and
  `unified_testbeam_combined_energy_summary_plot_v3`, plus aggregate target
  `unified_testbeam_combined_readable_plots_v3`.
- Focused LCG/ROOT regression suite: 18 tests passed in 9.67 seconds.
- The isolated dry-run resolved exactly 21 jobs: ten faceted plots, ten
  energy-summary plots, and one aggregate. It scheduled no raw-histogram,
  selection, or feature-production jobs.
- The local four-core run completed 21 of 21 steps. Complete log:
  `.snakemake/log/2026-10-09T134716.334786.snakemake.log`.
- Verified five PDFs and five CSVs in each `faceted` and `energy_summary`
  subdirectory for both `data_combined_particle_energy` and
  `mc_combined_particle_energy`.

## Exact next action (supersedes the preceding action)

Visually review the generated faceted and energy-summary PDFs. If their
layout is accepted, keep them as the primary comparison products and retain
the original all-energy overlays only as overview diagnostics.

### Reusable V4 testbeam histogram bundles and expanded plots (2026-10-09)

- Added separate histogram and plotting configurations:
  `unified_testbeam_histograms_v4.yaml` owns 30 TH1 definitions, seven TH2
  definitions, selections, and bundle locations; `unified_testbeam_plots_v4.yaml`
  owns cohort layout and presentation only.
- Added `unified_testbeam_histogram_bundles.py`. It audits required branches,
  builds one ROOT bundle per physics group, books all TH1/TH2 actions before
  executing the RDataFrame graph, applies the deterministic event cap before
  feature-specific cuts, embeds provenance/config metadata, and validates the
  object inventory, binning, configuration hash, group identity, and finite bins.
- Added `plot_unified_testbeam_histogram_bundles.py`. It reads only validated
  bundles and produces PDF-only paired 1D facets, energy-summary plots, and
  normalized 2D facets. Data and MC remain separate. No per-plot CSV is written;
  the workflow produces one consolidated JSON plot manifest.
- Added downstream-only `snakemake/unified_testbeam_distributions_v4.smk` with
  named bundle and plot aggregate targets. Plot rules depend on ROOT bundles,
  validation receipts, the renderer, and the plot config; style-only changes do
  not rescan feature trees.
- Complete dry-run for `unified_testbeam_all_plots_v4` resolves 196 jobs:
  20 audits, 20 bundles, 20 validators, 60 faceted TH1 PDFs, 60 energy-summary
  PDFs, 14 faceted TH2 PDFs, one manifest, and one aggregate. It contains no
  selection-index or feature-production rule.
- Real 1,000-event canaries passed for data pion/electron and MC pion/electron
  at 100 GeV. Each bundle contains and validates 37 histograms (30 TH1 + 7 TH2).
  The post-optimization MC canary confirms availability counting and all
  histogram actions execute through one booked RDataFrame graph.
- Focused LCG/ROOT tests pass: three tests cover the configuration boundary,
  workflow invariants, and all three PDF renderers, including the no-CSV rule.
- Moved the accepted V3 plot tree to the reversible sibling backup
  `unified_testbeam_distributions_v3_backup_20261009`. No production V4 bundle
  or plot job has been submitted.

## Exact next action (supersedes the preceding action)

After user approval, run `unified_testbeam_histogram_bundles_v4` through the
HTCondor profile to produce and validate the 20 production ROOT bundles. Do not
start the V4 PDF target until that bundle target completes successfully.

### Stop checkpoint (2026-10-09 Europe/Zurich)

#### Verified repository and scheduler state

- Worktree: `/afs/cern.ch/user/z/zhibin/worktrees/snd-ml-unified-shower-features`.
- Branch: `task/unified-shower-features`.
- Pre-checkpoint HEAD: `b78bbc10061c169e217411bd714059585c68c86d`.
- `git status --short` contained 46 task-related paths: 13 tracked
  modifications and 33 untracked source/config/test/plan files. No generated
  ROOT files, plots, logs, scheduler products, or metadata outputs were staged.
- `git diff --check` passed.
- No Snakemake controller or testbeam worker command was running in the
  worktree. `condor_q -af ClusterId ProcId JobStatus JobBatchName Cmd Args`
  returned no jobs, so there are no running job IDs to preserve.

#### Modified files included in this checkpoint

- Feature conversion, provenance, validation, sharding, and real-data
  selection: `convertData/sample_selection.py`,
  `convertData/source_provenance.py`,
  `convertData/validate_unified_shower_feature_file.py`,
  `convertData/build_indexed_cutflow.py`,
  `convertData/build_real_data_2024_selection_indices.py`,
  `convertData/real_data_2024_selection.py`,
  `convertData/validate_real_data_2024_selection_indices.py`,
  `convertData/validate_unified_shower_feature_shards.py`, and their focused
  `test_*.py` files.
- Distribution code and tests:
  `evaluation_region_partitions/unified_feature_distributions.py`,
  `evaluation_region_partitions/unified_testbeam_histogram_bundles.py`,
  `evaluation_region_partitions/plot_unified_testbeam_histogram_bundles.py`,
  `evaluation_region_partitions/test_unified_feature_distributions.py`,
  `evaluation_region_partitions/test_unified_feature_distributions_root.py`,
  `evaluation_region_partitions/test_unified_partial_feature_snapshots.py`,
  `evaluation_region_partitions/test_unified_testbeam_distribution_plan.py`,
  and `evaluation_region_partitions/test_unified_testbeam_histogram_bundles.py`.
- Snakemake workflows and batch support:
  `snakemake/rules/unified_shower_features.smk`,
  `snakemake/unified_partial_distributions.smk`,
  `snakemake/unified_real_data_selection.smk`,
  `snakemake/unified_real_data_distributions.smk`,
  `snakemake/unified_testbeam_distributions.smk`,
  `snakemake/unified_testbeam_combined_plots.smk`,
  `snakemake/unified_testbeam_distributions_v4.smk`,
  `snakemake/run_remaining_unified_shower_features_htcondor.sh`, and
  `snakemake/.config/snakemake/htcondor/grid-status.py`.
- Metadata preparation and tests:
  `snakemake/metadata/prepare_unified_shower_feature_manifests.py`,
  `snakemake/metadata/build_unified_partial_feature_snapshots.py`,
  `snakemake/metadata/prepare_unified_real_data_feature_manifests.py`,
  `snakemake/metadata/prepare_unified_real_data_selection_manifest.py`,
  `snakemake/metadata/prepare_unified_testbeam_2023_manifest.py`, and the two
  new metadata tests.
- Configuration: unified partial distributions, real-data selection and
  distributions, testbeam V2/V3 distributions, and separate V4 histogram and
  plot configuration files under `snakemake/metadata/configs/`.
- Documentation: this task file, `docs/plans/proposed/README.md`,
  `REAL_DATA_2024_PARTITION_V1_DISTRIBUTION_PLAN.md`, and
  `TESTBEAM_DISTRIBUTION_SOURCE_SELECTION_PLAN.md`.

#### Commands and verified results

- V4 bundle dry-run:
  `snakemake --snakefile snakemake/unified_testbeam_distributions_v4.smk
  --cores 1 -n unified_testbeam_histogram_bundles_v4` resolved 61 jobs:
  20 branch audits, 20 bundles, 20 validators, and one aggregate.
- Complete V4 dry-run:
  `snakemake --snakefile snakemake/unified_testbeam_distributions_v4.smk
  --cores 1 -n unified_testbeam_all_plots_v4` resolved 196 jobs and no feature
  production or selection-index jobs.
- Real V4 canaries used `unified_testbeam_histogram_bundles.py build` with
  `--max-events 1000`, followed by `validate`, for 100 GeV data pion, data
  electron, MC pion, and MC electron groups. All four produced bundles with
  30 TH1 plus seven TH2 objects and validation receipts. A repeated optimized
  MC-pion canary also passed all 37 checks.
- LCG command
  `python3 -m unittest -v
  evaluation_region_partitions.test_unified_testbeam_histogram_bundles`
  passed all three tests, including all three PDF render modes and the
  no-per-plot-CSV invariant.
- The accepted V3 plot tree was moved, without deletion, to
  `/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/plots/testbeam_distributions/unified_testbeam_distributions_v3_backup_20261009`.
- No production V4 histogram bundle or PDF job was submitted.

#### Decisions frozen at stop

- Keep data and MC comparisons in separate plot cohorts.
- Build and validate one ROOT bundle per physics group before plotting.
- Plot rules read only validated bundles; styling changes must not rescan
  feature trees.
- Produce 30 paired/faceted 1D feature distributions, seven 2D distributions,
  and energy summaries per cohort, with PDF products and one consolidated JSON
  manifest but no individual CSV files.
- Production remains a two-stage operation: finish the 20 HTCondor bundles
  first, then run the lightweight PDF target locally.

## Exact next action (supersedes the preceding action)

Run only the dry-run of `unified_testbeam_histogram_bundles_v4` from this
worktree using `snakemake/unified_testbeam_distributions_v4.smk`; inspect that
it still resolves only audits, bundles, validators, and the aggregate. Do not
submit production or start `unified_testbeam_all_plots_v4` in the same action.

### Full 2024 real-data input exclusion audit (2026-10-09)

- The first unbatched real-data selection dry-run exposed another unavailable
  upstream `nueAnalysisFilter` file. An EOS namespace comparison then checked
  every manifest-listed digi and cutflow path.
- All 11,368 digi inputs exist. Nine cutflow files are absent: the two sources
  already excluded plus seven newly identified sources.
- Added the seven verified UIDs to `excluded_source_uids` in
  `unified_real_data_selection_v1.yaml`. The provenance manifest remains
  unchanged; exclusions affect only aggregate production targets.
- Re-ran the full unbatched `unified_real_data_selection_indexes_v1` dry-run
  with `--rerun-triggers input params --rerun-incomplete`. It built the DAG and
  exited successfully with no `MissingInputException`.
- No selection-index or feature job was submitted.

## Exact next action (supersedes the preceding action)

Run the unbatched HTCondor command for
`unified_real_data_selection_indexes_v1` with `--jobs 200`,
`--rerun-triggers input params`, `--rerun-incomplete`, `--latency-wait 120`,
`--keep-going`, and `--drop-metadata`. Do not refresh the feature manifests or
start feature production until this selection-index target finishes.

### Real-data V2 histogram-bundle workflow implementation (2026-10-12)

#### Verified starting state

- Branch `task/unified-shower-features`, starting HEAD
  `42e0f09582533437b5b2a2532ca0b860d93b027d`.
- Preserved pre-existing user edits in this task file and
  `snakemake/metadata/configs/unified_real_data_selection_v1.yaml`.
- Selection-index production is complete for the 11,359 non-excluded sources,
  but the frozen V1 feature snapshots still contain only 1,109 sources per
  cohort and report one missing feature in each old 1,110-source manifest.
- No Snakemake controller, histogram-bundle build, or HTCondor job for this
  task is running. One unrelated event-display process belongs to another
  worktree and was not touched.

#### Implemented files and decisions

- Added the standalone downstream workflow
  `snakemake/unified_real_data_distributions_v2.smk`. It does not include or
  schedule selection-index or feature-file production.
- Added `unified_real_data_histograms_v2.yaml` and
  `unified_real_data_plots_v2.yaml`. They reuse the exact testbeam V4 set of
  30 TH1 and seven TH2 definitions, compare no-Veto with the deterministic
  10% Veto sample, use unit-area normalization, and write PDF products plus
  one JSON manifest but no per-plot CSV files.
- Added dataset-neutral wrapper entry points for bundle construction and
  plotting, plus a strict feature-snapshot coverage validator. The coverage
  gate rejects incomplete or non-available source records before audits or
  bundle jobs can run.
- Generalized the established testbeam bundle engine to inherit histogram
  definitions from another config and to record group labels, sampling
  probability, event weight, the common selection, the group selection, and
  their effective combined selection.
- Generalized the renderer to support config-defined group labels/styles,
  configurable facet sizes and TH2 columns, shared log-Z floors, and dataset
  annotations. The testbeam energy-summary renderer now derives its frame
  from both particle series so one series cannot clip the other.
- Fixed capped canary builds: `--max-events` no longer books an uncapped entry
  count that forces a full event scan. Uncapped production still evaluates
  and records the complete selected-entry count.
- Added `test_unified_real_data_histogram_bundles.py`, covering configuration,
  downstream-only workflow structure, strict coverage rejection, and PDF-only
  TH1/TH2 rendering.

#### Commands and results

- Focused LCG regression command for the testbeam and real-data bundle test
  modules passed all seven tests. The final repeated run completed in 3.649 s.
- `git diff --check` and `python3 -m py_compile` passed for the new and modified
  Python sources.
- The V2 aggregate dry-run resolved exactly 46 jobs: one coverage gate, two
  branch audits, two bundle builds, two bundle validators, 30 TH1 plots,
  seven TH2 plots, one plot manifest, and one aggregate. It resolved no
  feature or selection-index jobs.
- Strict audits of both current snapshots passed and found all 37 required
  feature branches. The remote validation-record scan took about four minutes
  per 1,109-file cohort.
- Built real-file 100-event canary bundles for both cohorts under
  `/tmp/zhibin/real_data_v2_canary/`; each contains 30 TH1 plus seven TH2
  objects. Both validation commands passed all 37 objects. Representative
  `count_scifi.pdf` and `total_qdc_vs_hits.pdf` comparisons were produced.
- Running the new coverage validator against the current frozen production
  metadata failed as intended: both cohorts report 1,109 available of 1,110,
  with one `missing_feature`. No V2 production output was created or submitted.
- Attempted visual raster inspection of the temporary PDFs, but the local
  image-view helper could not run because the host filesystem sandbox lacks
  its bubblewrap isolation dependency. ROOT PDF creation and bundle validation
  succeeded.

#### Current Git state

- HEAD remains `42e0f09582533437b5b2a2532ca0b860d93b027d`; no commit was made.
- Task changes are the two modified reusable testbeam modules plus seven new
  real-data/generic source, test, config, and workflow files. The pre-existing
  modifications to this task file and the real-data selection config remain
  uncommitted and preserved.

## Exact next action (supersedes the preceding action)

Run only a Snakemake dry-run of
`unified_real_data_distribution_features_v1` using
`snakemake/unified_real_data_distributions.smk`, with
`--rerun-triggers input params --rerun-incomplete`, and verify that the
refreshed checkpoint expands the 11,359 non-excluded sources for both cohorts.
Do not submit feature production or run the V2 histogram target in the same
action.

### Explicit partial real-data plotting mode (2026-10-12)

- The refreshed source manifests contain 11,151 non-empty No-Veto sources and
  11,150 non-empty Veto-10% sources. Existing feature storage contains 1,109
  validated canonical files per cohort, so the available plot input is a
  preliminary processed-source subset rather than complete production.
- Added an explicit `coverage.policy: partial` configuration with a positive
  minimum source count and `missing_feature` as the only allowed unavailable
  status. Invalid receipts, failed validation, mismatches, empty snapshots,
  inconsistent counts/fractions, changed source-manifest checksums, and stale
  snapshot row counts remain fatal.
- Coverage receipts now include source and snapshot checksums, exact available
  and total source counts, fractions, status counts, and the selected policy.
  The final plot manifest embeds this validated receipt.
- V2 plot annotations identify the products as a preliminary partial
  processed-source snapshot. Legend labels dynamically include the source
  count recorded in each histogram bundle.
- Added cached manifest frames and source-UID dictionaries to
  `unified_real_data_distributions.smk`. Dynamic DAG expansion now parses each
  17 MB manifest once instead of reparsing and scanning it roughly nine times
  per one of 22,301 jobs.
- The combined testbeam/real-data focused suite passed all seven tests.
  `git diff --check` passed. A snapshot-refresh dry-run resolved exactly one
  local snapshot job, and the partial V2 coverage dry-run resolved exactly one
  local rule under the HTCondor profile. No production or snapshot job was
  executed.

## Exact next action (supersedes the preceding action)

Run only `unified_real_data_distribution_snapshot_v1` locally with 16 cores
and `--rerun-triggers input params --rerun-incomplete`. Verify the resulting
coverage records approximately 1,109 validated sources per cohort against the
new 11,151/11,150-source manifests. Do not start V2 histogram production in
the same action.

### Real-data V2 HTCondor bookkeeping-rule fix (2026-10-12)

- Diagnosed log `2026-10-12T132524.312758.snakemake.log`: no Condor job was
  submitted. The profile submit helper requires `resources["runtime"]`, but
  the lightweight coverage rule had no batch resources and submission failed
  twice before obtaining a cluster ID.
- Declared `unified_real_data_feature_snapshot_coverage_v2` and
  `unified_real_data_plot_manifest_v2` as local rules. Both only read/write
  small AFS metadata products and do not belong on HTCondor.
- Added regression assertions for both local-rule declarations. The focused
  real-data test module passed all four tests.
- An HTCondor-profile dry-run of
  `unified_real_data_feature_snapshot_coverage_v2` passed and resolved the
  rule locally with `tmpdir=/tmp/zhibin`; it no longer attempted submission
  with `runtime=<TBD>`.
- No jobs were submitted. Current feature storage remains at 1,109 ROOT files
  plus 1,109 receipts for each cohort, so the coverage gate will still reject
  an actual V2 run until feature production and snapshot refresh are complete.

## Exact next action (supersedes the preceding action)

Run only a Snakemake dry-run of
`unified_real_data_distribution_features_v1` using
`snakemake/unified_real_data_distributions.smk`, with
`--rerun-triggers input params --rerun-incomplete`, and verify that the
refreshed checkpoint expands the 11,359 non-excluded sources for both cohorts.
Do not submit feature production or run the V2 histogram target in the same
action.

### Review-fix implementation checkpoint (2026-11-04)

- User explicitly authorized implementing the six fixes in `CODE_REVIEW_FINDINGS.md` section L. This supersedes the previous inspection-only next action for this implementation session. Production feature regeneration remains unauthorized.
- Starting HEAD: `42e0f09582533437b5b2a2532ca0b860d93b027d`. Preserved all five pre-existing tracked modifications and seven pre-existing implementation/config/test untracked files plus the review report. No commits or production jobs.
- Implemented ROOT-free feature integrity and snapshot membership helpers; frozen snapshot file/receipt digests; exact source-status membership and shard accounting; selection-policy-bound upstream ledger with per-profile zero counts and explicit exclusions; normalized sample classification; serial partition snapshots with input/output identity checks and split-feature metadata digest verification; numerical ABCD test tolerance and model-consistent positive control.
- Added explicit helper/source/policy/ledger workflow dependencies. Existing snapshots and old coverage records require validated refresh before downstream use. Existing split partition metadata without a feature digest must also be refreshed through validated partition production; do not populate hashes blindly.
- LCG 107 on lxplus915, Python 3.11.9, ROOT 6.34.02, pyhf 0.7.6: existing 54 focused regressions passed; initial nine adversarial tests and two metadata pytest cases passed; combined/split ROOT partition integration then passed (10-test adversarial suite). Full `test_abcd_blinded_expected.py` passed with existing optimizer/covariance warnings; these remain a separate statistical robustness limitation.
- Final additional shard/timing tests, exclusion reconciliation, and Snakemake parse/dry-run checks are in progress. Temporary test products are under automatically cleaned temporary directories. No HTCondor submission or production output refresh occurred.
- Exact next action: finish the running verification commands, address any failures, review the implementation diff, and record final evidence before stopping. Do not start production migration or feature regeneration.

### Review-fix verification completed (2026-11-04)

- Final LCG combined regression: 66 tests passed in 13.517 s; two metadata pytest cases passed in 0.75 s. Complete synthetic ABCD test passed with the previously documented optimizer/covariance warnings. Changed Python syntax and `git diff --check` passed.
- New integration test runs actual ROOT selection and snapshots in both combined and split feature/Hit3D modes with a two-thread request, checks exact ordered output identities, and rejects replaced feature products through metadata digest verification.
- Read-only inventory of the original 11,368 selection receipt paths found nine absent; the missing UID set exactly equals the configured exclusion UID set. No receipt contents were revalidated in this inventory.
- Snakemake 9.17.2/Python 3.13.12 environment: real-data V2 coverage dry-run resolved one job; testbeam V4 bundle dry-run resolved 61 audit/bundle/validator/aggregate jobs; main workflow target listing passed. No jobs executed or submitted.
- Implementation and migration details are recorded in `CODE_REVIEW_FINDINGS.md` section M. Temporary final test log: `/tmp/zhibin/review_fix_final_tests.log`; main workflow target listing: `/tmp/zhibin/review_fix_targets.txt`.
- All six review findings have implementation and regression evidence; production deployment/canary and the broader scientific review remain pending. Old snapshots/coverage records and split partition metadata require validated migration. No production artifacts were refreshed and no commit was made.

## Exact next action (supersedes the preceding action)

Wait for the user's instruction on validated metadata migration or a detector
conversion canary. Do not automatically refresh production metadata,
regenerate features, or start histogram/ML production.

### Commit acceptance verification (2026-11-04)

- User authorized testing the fixes against the implementation plan, then committing on success.
- Reinspection closed canonical snapshot range/status consistency and empty-source-list split-Hit3D reference-check gaps, with regression coverage. Explicit LCG activation added to the real-data coverage rule.
- Exported staged tree `0da3057fc409a0b071d65369d2ff4a12ccca03d5` to `/tmp/zhibin/review_fix_commit_k1c9aeaw` and tested that standalone export: 67 unittest cases passed in 17.324 s, two metadata pytest cases passed in 0.69 s, and the complete synthetic ABCD regression passed. Existing statistical optimizer/covariance warnings remain documented. Staged Python syntax and whitespace checks passed.
- Staged workflow checks passed: real-data coverage dry-run (one job), testbeam bundle dry-run (61 jobs), main target listing. Existing production coverage was rejected as obsolete without writing a receipt. No production jobs or artifact migration.
- Related V2 support is included so the fix commit is runnable independently. The independent pre-existing testbeam energy-summary plotting adjustment remains uncommitted in `plot_unified_testbeam_histogram_bundles.py`.
- Final evidence and log locations: `CODE_REVIEW_FINDINGS.md` section N. Commit subject: `Fix unified feature integrity, coverage and partition alignment`. Resolve its SHA with `git log -1 --format=%H --grep='Fix unified feature integrity, coverage and partition alignment'`.

## Exact next action (supersedes the preceding action)

After creating the authorized verified fix commit, wait for the user's next
instruction. Do not automatically migrate production metadata, regenerate
features, or start histogram/ML production.
