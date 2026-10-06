# Unified Shower Feature Production and Region Evaluation Plan

Status: proposed

Last updated: 2026-10-06

Scope: Integrate the EM-shower SciFi cluster features with the established
event-feature schema, produce versioned features for five 2024 dataset
families, and compare their distributions through collision-region and
testbeam-specific partitions.

## Objective

Produce one backward-compatible event-feature schema containing both the
established `digi_2_features.py` branches and the new SciFi clustering/profile
branches for:

1. 2024 electron- and muon-neutrino MC;
2. 2024 MC muon-up;
3. 2024 testbeam MC;
4. 2024 testbeam data; and
5. the 2024 skimmed collision-data catalogue.

Then materialize validated partitions, histograms, and comparisons without
overwriting the existing canonical feature products.

## Guiding decisions

- Keep `convertData/digi_2_features.py` as the single owner of feature
  calculations. It already preserves the established features and appends the
  SciFi cluster/profile features.
- Do not merge old and new ROOT trees after conversion. Regenerate each source
  once with the unified converter.
- Preserve thin compatibility adapters only where the input contract differs.
  The existing MC-muon adapter handles its legacy event header. Testbeam may
  use a similarly thin adapter or common converter options, but must not copy
  the feature algorithms.
- Do not overwrite the existing metadata `feature_path` products. Introduce a
  versioned feature-path column and isolated output roots.
- Preserve event identity. Every downstream product must retain or validate
  `(runId, eventId, eventIndex)` together with source-file identity where MC
  files can reuse those values.
- Use Snakemake named aggregate targets as the production interface and
  HTCondor for long campaigns.
- Use the collision Region V1 definitions only for collision-like samples.
  Testbeam comparisons require testbeam-specific particle/energy partitions.
- Begin with shape comparisons using unit-area normalization. Absolute-yield
  comparisons require an explicit exposure or luminosity policy.

## Branch integration strategy

The EM-shower and ABCD branches share starting commit `5793ce8`. Their
committed changes do not overlap at the file level, but the ABCD worktree has
substantial uncommitted task work.

The safe integration sequence is:

1. Checkpoint and commit all current task-only changes on
   `task/abcd-bkg-estimation`.
2. Create a dedicated integration branch and worktree from that checkpoint,
   provisionally named `task/unified-shower-features`.
3. Merge `task/em-shower` into the integration branch, preserving its three
   commits and provenance.
4. Resolve and test any integration issues before adding new dataset support.
5. Port only the required testbeam selection, identity-cutflow, geometry, and
   beam-truth conventions from `task/3d-cnn-testbeam`. Do not merge that whole
   branch because it also contains unrelated Hit3D, training, and evaluation
   development.
6. Keep each independent task worktree unchanged while the integration branch
   is being validated.

No branch integration should start from a dirty worktree.

## Unified feature contract

The output `sndData` tree must continue to contain the established feature
families:

- event identity and MC/data markers;
- detector hit counts and QDC sums;
- unweighted and QDC-weighted positions;
- SciFi density variables;
- detector timing variables;
- applicable MC truth quantities; and
- existing optional reconstructed-track quantities.

It must additionally contain the validated SciFi cluster/profile schema,
including:

- vertical and horizontal cluster counts;
- leading- and other-cluster hit and QDC totals;
- leading-cluster fractions and QDC per hit;
- QDC variation and roughness;
- effective cluster counts;
- per-orientation station profiles, slopes, and curvatures;
- orientation asymmetry; and
- `scifi_qdc_per_hit_direct`.

The output must retain `metadata_scifi_cluster_schema` and the ordinary
selection/input provenance markers.

### Common source-energy fields

Retain `mc_incoming_energy` as MC truth. Add dataset-neutral provenance fields
for comparisons that include real testbeam data, provisionally:

```text
source_energy_gev
source_pdg_code
source_domain
```

For MC, populate these from authoritative MC truth or validated source
metadata. For real testbeam data, populate them from the testbeam catalogue.
For collision real data, use an explicit unavailable value. Do not represent
catalogue beam energy as MC truth.

## Versioned metadata and output layout

Add a new metadata column such as:

```text
feature_scifi_cluster_v1_path
```

Do not change the meaning of the existing `feature_path` column. Proposed
isolated roots are:

```text
features_v2/mc_neutrino/
features_v2/mc_muon_up/
features_v2/real_data_2024_skim/
features_v2/testbeam_mc_2024/
features_v2/testbeam_data_2024/
```

Every prepared production manifest should record at least:

- source metadata file and row;
- digi and geometry paths;
- input tree and selection mode;
- external or identity cutflow path when used;
- output feature path;
- source domain, particle information, and beam energy when applicable;
- expected source entries where known; and
- schema/validation marker path.

## Dataset adapters

### MC neutrino

Reuse the implemented isolated workflow and external neutrino cutflow. Keep
the existing CC/NC classification:

- `pdgCode = +/-12`: CC nue;
- `pdgCode = +/-112`: NC nue;
- `pdgCode = +/-14`: CC numu; and
- `pdgCode = +/-114`: NC numu.

The current catalogue contains 976 electron-flavour and 890 muon-flavour
sources.

### MC muon-up

Reuse `digi_2_features_mc_muon.py` as a thin adapter around the common
converter. Keep the legacy `FairEventHeader` compatibility isolated there.
The current catalogue contains 15 sources.

Use one canonical downstream particle-group name, preferably `MC_muon_up`,
instead of mixing `MC_muon_up` and `muon` for the same production.

### 2024 skimmed collision data

Reuse the generic metadata-driven production contract and the external
collision-data cutflow. The `real_data_2024_skim_run` production profile must
continue to preserve events with both Veto and US activity so all eight
Region V1 populations remain available.

The catalogue contains 11,368 sources, so this is the largest campaign and
must run only after a representative canary passes.

### 2024 testbeam MC

Use the audited testbeam geometry catalogue and identity cutflows. Preserve
the input event-tree identity and obtain particle/energy truth from the
validated MC source contract.

The legacy metadata contains 105 MC source rows. The full testbeam campaign
has a stricter valid-W2 physical-source catalogue; the intended production
scope (all eligible metadata rows or the validated W2 subset) must be frozen
before production.

### 2024 testbeam data

Use identity cutflows rather than the collision neutrino filter. Bypass
collision-only `StableBeams`, `IP1`, and event-time requirements without
changing ordinary collision-data behavior.

Supply particle PDG and beam energy together from the validated testbeam
catalogue. Missing or inconsistent beam metadata must fail validation rather
than silently use a sentinel in a labelled comparison.

The legacy metadata contains 860 real-data source rows. As for MC, freeze
whether production covers every eligible W2 row or only the full-campaign
physical-source catalogue.

## Partition strategy

### Collision Region V1

For MC neutrino, MC muon-up, and 2024 skim data, reuse the eight orthogonal
Region V1 populations after `count_scifi > 35`:

```text
Veto/no Veto x US/no US x SciFi fiducial inside/outside
```

The canonical particle groups should be:

```text
real_data
CC_nue
CC_numu
NC_nue
NC_numu
MC_muon_up
```

Use the existing `build_region_partitions.py` identity and count-closure
checks. Produce a new versioned partition catalogue that points only to the
new feature products.

### Testbeam partitions

Do not label testbeam samples as Signal/SB1--SB7. Define an isolated testbeam
partition configuration with axes such as:

```text
domain: MC or data
particle: electron or pion
beam energy
geometry setup
optional beam-entry position
```

Required first comparisons are:

- MC electron versus MC pion at fixed energy;
- data electron versus data pion;
- MC versus data for the same particle and energy; and
- energy dependence within one particle type.

## Histogram and plotting design

Keep feature production independent from plotting. Use a shared, versioned
histogram-feature configuration for the legacy and cluster branches, with
dataset-specific evaluation options.

### Shared one-dimensional observables

At minimum include:

- `count_scifi` and `qdc_scifi`;
- `density_sndsw_scifi`;
- `scifi_qdc_per_hit_direct` and `scifi_cluster_qdc_per_hit`;
- total and per-orientation cluster counts;
- leading-cluster hits, QDC per hit, and QDC fraction;
- other-cluster QDC fraction;
- cluster QDC variation and roughness;
- effective cluster count;
- station-profile mean, slope, and curvature; and
- energy variables where available.

### Shared two-dimensional observables

At minimum include:

- total SciFi QDC versus hit count;
- leading-cluster QDC versus hit count;
- roughness versus cluster QDC per hit;
- fragmentation versus leading-cluster fraction; and
- energy versus the principal shower observables.

### Normalization policy

- Use unit-area normalization for initial shape comparisons.
- Treat real-data Region V1 exposures explicitly in rate plots.
- Apply authoritative neutrino-MC luminosity weights for expected-yield plots.
- Do not interpret pooled MC muon-up events as an absolute yield without a
  documented normalization.
- Treat testbeam event counts as beam-sample statistics, not collision
  luminosity.
- Label the already inspected collision signal region as unblinded
  development data; do not use it for tuning and later claim an independent
  blind evaluation.

## Validation gates

### Static and unit validation

- Compile every changed Python module in its documented runtime.
- Run existing clustering, partition, histogram, plotting, MC-muon, and
  testbeam-focused tests.
- Add tests for common source-energy fields and testbeam real-data selection.
- Confirm old feature names and types remain unchanged.
- Confirm the generic legacy workflow does not select the new versioned paths
  unless explicitly configured.

### One-source canaries

Before full production, run and audit one representative job for each input
contract:

1. MC neutrino;
2. legacy MC muon-up;
3. 2024 skimmed collision data;
4. testbeam MC; and
5. testbeam real data.

Each canary must validate:

- the output ROOT file opens and contains `sndData`;
- the established and cluster branches are present;
- the schema marker is correct;
- cluster hit and QDC closure holds;
- direct QDC-per-hit closure holds;
- all required scalar values are finite;
- selected/written counts agree with the selection contract;
- `(runId, eventId, eventIndex)` is preserved; and
- sample-specific truth/provenance fields are correct.

If aligned Hit3D exists, validate entry count and event identity against it.

### Production completeness

For every dataset family, require:

- one source-manifest row per intended source;
- one nonempty feature output and validation marker per row;
- no duplicate output path or source identity;
- recorded failed/missing sources rather than silent omission; and
- a no-op final Snakemake dry run.

### Partition and histogram validation

- Validate source-to-partition event-count closure.
- Validate particle and region selection expressions.
- Validate partition identities and source provenance.
- Fail clearly when a requested histogram branch is unavailable.
- Require histogram selected-entry and flow accounting to close.
- Inspect representative 1D and 2D plots for each comparison family.

## Staged execution order

Do not begin full production until all five canaries pass. Then process one
family at a time in this order:

1. MC muon-up, because it has only 15 sources and exercises the compatibility
   adapter;
2. MC neutrino, using the already validated canary contract;
3. testbeam MC, after its adapter and geometry/truth contract pass;
4. testbeam data, after metadata-provided particle/energy validation passes;
5. 2024 skimmed collision data, last because it has 11,368 sources.

For each family, complete and record this sequence before starting the next:

```text
feature generation
-> schema and identity validation
-> source-completeness audit
-> partitions
-> partition audit
-> histograms
-> representative plots
```

## Proposed named targets

Exact names should be finalized during implementation, but the interface
should expose one aggregate per milestone rather than one monolithic target:

```text
unified_features_mc_neutrino_v1
unified_features_mc_muon_up_v1
unified_features_real_data_2024_skim_v1
unified_features_testbeam_mc_2024_v1
unified_features_testbeam_data_2024_v1

unified_collision_region_partitions_v1
unified_testbeam_partitions_v1

unified_collision_feature_hists_v1
unified_testbeam_feature_hists_v1

unified_collision_feature_plots_v1
unified_testbeam_feature_plots_v1
```

Canary targets should be separate and must not implicitly trigger full
production.

## Completion criteria

The plan is complete only when:

- the integration branch contains one unified, backward-compatible converter;
- all five dataset manifests and versioned output paths are frozen;
- all five canaries pass;
- intended full-source feature production is complete and validated;
- collision Region V1 and testbeam-specific partition catalogues close;
- the requested legacy, cluster, profile, energy, and 2D histograms are
  produced;
- representative plots pass visual and machine-readable checks;
- normalization and blinding labels are explicit; and
- commands, logs, job IDs, output locations, failures, and final Git state are
  recorded in the active task file.

## Work not authorized by this plan record

Saving this plan does not authorize a branch merge, code implementation,
feature regeneration, partition production, histogram production, plotting,
or scheduler submission. Each next action must be explicitly requested and
recorded.
