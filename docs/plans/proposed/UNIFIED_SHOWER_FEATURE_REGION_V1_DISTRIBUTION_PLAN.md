# Unified Shower Feature Region V1 Distribution Plan

Status: proposed

Last updated: 2026-10-06

## Objective

Produce one backward-compatible feature schema, derived from the EM-shower
implementation, for 2024 neutrino MC, MC muon-up, skimmed 2024 collision data,
testbeam MC, and testbeam data.

For collision-like samples only, use the established eight Region V1
populations to produce validated feature partitions, raw feature histograms,
and comparison plots. Keep these responsibilities separate:

```text
feature production
    -> Region V1 partitioning and generic distributions
        -> optional, separately versioned ABCD statistical inference
```

No layer should silently regenerate products owned by another layer.

## Frozen design decisions

### Feature contract

- `convertData/digi_2_features.py` remains the only owner of feature
  calculations.
- `digi_2_features_mc_muon.py` remains a thin legacy-header adapter.
- All established branches and ROOT types remain unchanged.
- SciFi shower cluster/profile branches remain additive.
- Preserve `mc_incoming_energy` with its established meaning.
- Add `source_energy_gev` as `Double_t` (`/D`), `source_pdg_code` as
  `Int_t` (`/I`), and `source_domain_id` as `Int_t` (`/I`).
- Use the frozen domain enumeration: `0=unknown/reserved`, `1=mc_neutrino`,
  `2=mc_muon_up`, `3=real_data_2024_skim`, `4=testbeam_mc_2024`, and
  `5=testbeam_data_2024`.
- Preserve the frozen metadata column name
  `feature_scifi_cluster_v1_path`. Do not introduce a competing
  `feature_unified_shower_v1_path` name.
- Do not overwrite or repoint the legacy `feature_path` column.

### Event and source identity

The event-level identity remains:

```text
(runId, eventId, eventIndex)
```

The globally meaningful identity also includes `source_uid`:

```text
(source_uid, runId, eventId, eventIndex)
```

Require entrywise feature/Hit3D alignment only when both products were made
from the same source and selection contract. Otherwise validate a keyed join.

### Region scope

Collision Region V1 applies only to:

```text
real_data
CC_nue
CC_numu
NC_nue
NC_numu
MC_muon_up
```

Testbeam samples use the same converter but a separate
particle/energy/geometry evaluation workflow. The expressions in
`snakemake/metadata/configs/regions/region_v1.yaml` remain unchanged.

### Output isolation

Use new versioned roots for unified feature files, projected metadata, Region
V1 partitions, audits, raw histograms, plots, and summaries. Do not overwrite
accepted feature, partition, histogram, or ABCD products.

## Target architecture

```text
frozen source catalogues
        |
        v
 deterministic unified manifests
        |
        +----> separate input-availability audit
        |
        v
 unified feature conversion
        |
        v
 feature validation record
        |
        v
 collision metadata projection
        |
        v
 feature-only Region V1 partitions
        |
        v
 partition identity/exposure audit
        |
        v
 raw 1D and 2D histograms
        |
        +-------------------+
        |                   |
        v                   v
 unit-area plots      common-exposure plots
        |
        v
 optional separately versioned ABCD studies
```

## Why the validation record and metadata projection are separate

### Feature validation record

A ROOT file existing on EOS does not prove that conversion and validation
completed successfully. For every feature output, publish a validation record
only after the converter, schema checks, identity checks, and EOS transfer all
succeed.

Prefer a JSON record such as:

```text
feature_scifi_cluster_v1_<source_uid>.root.validated.json
```

It should record at least:

```text
manifest_version
source_uid
input identities
feature output path
feature schema version
converter commit
written entry count
validation result
output checksum or stable file identity
validation timestamp
```

Downstream workflows consume the validation record rather than trusting file
existence alone. This prevents partial transfers, stale outputs, wrong-schema
files, and failed conversions from entering Region V1.

### Collision metadata projection

The unified production manifest describes how to create feature files. The
Region V1 partition builder expects analysis-oriented metadata with dataset,
particle-group, exposure, and feature-path columns. The projection converts
validated production rows into that interface without modifying the original
source catalogues.

It combines:

```text
unified manifest row
+ successful validation record
+ source exposure metadata
```

and produces a small deterministic CSV/JSON catalogue for the collision-like
families. It does not copy or regenerate ROOT data.

For compatibility with the current partition builder, the projection may set:

```text
output_base_path = parent directory of feature_scifi_cluster_v1_path
feature_path = basename of feature_scifi_cluster_v1_path
```

These compatibility columns exist only in the projected catalogue. This keeps
legacy metadata unchanged, excludes testbeam samples from collision Region
V1, and prevents unvalidated feature files from entering downstream work.

## Phase 0: reconcile the current implementation

The worktree already contains uncommitted unified-feature changes. Do not
restart implementation from the beginning.

Review the current converter, provenance, selection, manifest, configuration,
and Snakemake changes. Classify every change as required, incomplete,
incorrect, or unrelated.

Verify specifically:

- frozen names and paths have not drifted;
- testbeam PDG/energy overrides are paired;
- unified MC domains require `MCTrack[0]`;
- testbeam-MC catalogue values validate rather than overwrite truth;
- collision data uses the frozen sentinels;
- identity cutflows bypass collision-only quality requirements;
- the muon adapter duplicates no feature calculations;
- frozen external cutflows cannot be hijacked by broad legacy rules; and
- canary targets do not imply full production.

Exit gate: current implementation state is understood and reconciled with the
frozen contract.

## Phase 1: complete the converter and validator contract

Implement and validate dataset-neutral source fields, domain validation,
paired testbeam arguments, geometry/source provenance, explicit selection
modes, identity-cutflow behavior, and failure for missing incoming MC truth in
unified MC domains.

Add an explicit unified schema marker while retaining the existing SciFi
cluster schema marker.

The validator must check:

- `sndData` and required metadata objects exist;
- legacy and shower branches exist with the expected ROOT types;
- required scalars are finite or valid sentinels;
- source domain and provenance agree with the manifest;
- event identity is unique within the source;
- selected and written event counts close;
- cluster hit totals equal `count_scifi`;
- cluster QDC totals agree with `qdc_scifi`;
- direct QDC per hit agrees with `qdc_scifi / count_scifi`;
- direct and cluster-derived QDC-per-hit values agree within tolerance; and
- testbeam-MC truth/catalogue agreement passes.

For fixed representative events, prove that no established branch, type, or
legacy feature value changed. Cover empty inputs, no-SciFi-hit events, missing
MC truth, incomplete overrides, and invalid domain/sample combinations.

Exit gate: static and synthetic contract tests pass without producing real
feature files.

## Phase 2: finalize deterministic manifests

Use one manifest builder driven by the frozen unified configuration. Each row
should contain:

```text
manifest_version
family
source_uid
source_metadata_file
source_metadata_row
data_type
subfolder
partition or source key
digi_path
geo_path
cutflow_path
input_tree
selection_mode
source_energy_gev
source_pdg_code
source_domain
source_domain_id
source_geometry_id
source_geometry_checksum
lumi_per_file
exposure_kind
expected_event_count
feature_scifi_cluster_v1_path
validation_record
is_canary
```

Require exact family counts, deterministic sorting, byte-stable output,
unique source and output identities, valid domain/sample combinations, paired
testbeam values, frozen catalogue checksums, valid geometry, and one
deterministic canary per family.

Keep storage availability checks in a separate audit containing source UID,
input availability/readability, audit host/time, readiness, and failure reason.
This avoids making the deterministic manifest depend entirely on transient EOS
availability.

Exit gate: manifest counts and hashes reproduce, and the availability audit
accounts for every row.

## Phase 3: complete isolated feature targets

Keep the implementation in `snakemake/rules/unified_shower_features.smk`.
Provide full and separately named canary targets for all five families.

Each conversion job must:

1. activate the documented SNDSW environment;
2. print the runtime preflight;
3. verify source inputs and the frozen cutflow;
4. convert to node-local temporary storage;
5. run the unified validator;
6. copy the ROOT file to its isolated EOS destination;
7. verify that the destination is readable;
8. publish the validation JSON only after successful transfer; and
9. remove temporary files.

The rules consume unified manifests, not mutable global metadata lists.

Exit gate: target listing and canary/full dry runs show exact expected source
counts without executing jobs.

## Phase 4: run five feature canaries

Run one explicitly authorized source for each input contract: neutrino MC, MC
muon-up, skimmed collision data, testbeam MC, and testbeam data.

For each canary perform:

```text
target dry run
-> local conversion
-> schema validation
-> identity audit
-> provenance audit
-> runtime/memory record
-> task checkpoint
```

For an existing Hit3D product, validate keyed identity. Require entrywise
alignment only when the selection contracts match. The final no-op check
applies to the individual canary target, not the unproduced full target.

Exit gate: all five contracts pass independently.

## Phase 5: create the collision metadata projection

Project only `mc_neutrino`, `mc_muon_up`, and `real_data_2024_skim` rows with
successful validation records.

Retain source UID, source metadata provenance, dataset identifiers, expected
and validated counts, exposure fields, source domain, validation record, and
the absolute unified feature path.

Requirements:

- deterministic ordering and byte-stable output;
- unique source and output identity;
- no legacy feature path;
- no unvalidated feature product;
- no testbeam row; and
- explicit exposure provenance.

Suggested target:

```text
unified_collision_region_metadata_v1
```

Exit gate: the projected catalogue contains exactly the intended validated
collision-like sources.

## Phase 6: create unified Region V1 partitions

Create a new partition version such as `unified_shower_partition_v1`. Reuse
`region_v1.yaml` unchanged. Initially produce feature partitions only and use
the six canonical particle groups.

### Empty selections

An empty but valid assignment must retain a catalogue row with:

```text
n_events = 0
feature_available = false
partition_status = valid_empty
assigned source identities
assigned exposure
selection expression
```

Do not create a dummy ROOT file. Distinguish `valid_nonempty`, `valid_empty`,
`missing_input`, `validation_failed`, and `not_scheduled`.

### Exposure semantics

Exposure belongs to the assigned source set, not to the selected event count.
The same exposure may legitimately appear in multiple regions or particle
groups. Therefore:

- validate exposure separately for each `(region, particle_group)`;
- do not sum exposure across the eight regions;
- do not sum CC and NC exposures as independent datasets when they share
  source files;
- retain the source identities supporting each exposure; and
- record `exposure_kind`, value, source, and validation status.

MC muon-up remains shape-only until its equivalent-exposure convention is
validated.

### Partition validation

Require that every event with `count_scifi > 35` enters exactly one of the
eight regions, events below the threshold enter none, region intersections are
empty, region union closes, particle classification closes, identities are
unique, nonempty counts match ROOT entries, and valid empty assignments remain
represented.

Suggested targets:

```text
unified_collision_region_partitions_v1
unified_collision_region_partition_audit_v1
```

## Phase 7: produce raw feature histograms

Create a versioned histogram configuration combining established Region V1
observables, SciFi cluster/profile observables, meaningful source-energy
observables, and selected diagnostic 2D relationships.

Produce one raw histogram per `region x particle_group x feature`. Do not scale
bins during histogram creation.

Record before any flow folding:

```text
selected_entries
underflow_raw
in_range_raw
overflow_raw
integral_all_raw
```

Require selected-entry and flow closure. If flow is folded, also record the
post-fold integral and policy. Store histogram, partition, region, feature,
selection, source, exposure, flow-policy, and creation-commit provenance.

For 2D histograms record entries, x/y flow accounting, sum of weights, bin-area
policy, and whether the result represents counts, unit sum, or density.

Suggested target:

```text
unified_collision_region_hists_v1
```

Exit gate: branch availability, selected counts, and flow accounting pass for
every configured histogram.

## Phase 8: implement explicit plotting normalization

Raw histogram files remain unchanged. Normalization is selected explicitly;
conflicting CLI and configuration values must fail.

### Unit-area mode

For a nonempty histogram:

```text
scale_factor = 1 / integral_after_flow_policy
```

Empty histograms remain empty and labelled. Unit-area mode requires no
luminosity.

### Common-exposure mode

Use:

```text
scale_factor = reference_exposure / sample_exposure
```

Enable it only when both exposures are validated and compatible and no missing
generator/cross-section weighting is implied. Reference exposure comes from
the source/exposure catalogue, not from opening a signal-region histogram.
The potentially blinded signal region must not be the normalization authority.

Always allow unit-area MC-muon plots. Disable common-exposure MC-muon plots
until its equivalent-luminosity convention is documented and tested.

For 2D products distinguish `unit_sum`, `probability_density`,
`common_exposure_counts`, and `profile_mean`. Profiles remain conditional means
and are never multiplied by luminosity.

Every plot summary records normalization mode, raw/scaled integrals, sample and
reference exposure, exposure kind, scale factor, selected entries, and empty
status.

Suggested targets:

```text
unified_collision_region_plots_unit_area_v1
unified_collision_region_plots_exposure_v1
unified_collision_region_plots_v1
```

## Phase 9: validate normalization synthetically

Test known synthetic inputs for unit integral, empty handling, exposure ratios,
equal-exposure scale one, partial-exposure scaling, error scaling, immutable raw
histograms, summary reproduction, missing/zero/incompatible exposure failures,
flow closure, distinct 2D unit-sum/density behavior, and unchanged profiles.

Exit gate: normalization is reproducible without physics production inputs.

## Phase 10: run an end-to-end collision canary

Use a frozen bounded set containing one electron-neutrino source, one
muon-neutrino source, one MC-muon-up source, and complete collision-data source
files. Label its exposure as canary exposure, not full-campaign luminosity.

Run:

```text
feature validation
-> collision metadata projection
-> eight-region partitioning
-> partition audit
-> raw histograms
-> unit-area plots
-> permitted exposure plots
-> summary audit
```

Require count, identity, particle, exposure, histogram-flow, and normalization
closure. The canary aggregate target must become a no-op after success.

## Phase 11: staged feature production

After explicit authorization, process MC muon-up, neutrino MC, skimmed
collision data, testbeam MC, and testbeam data. Testbeam ordering may be
changed independently because it does not block collision Region V1.

For each family use:

```text
full-target dry run
-> review source/job/resource count
-> HTCondor submission
-> monitor and diagnose
-> validate intended outputs
-> completeness report
-> final no-op dry run
-> task checkpoint
```

Stop on schema, identity, count, truth/catalogue, finite-value, output-count,
storage, or validation failure.

## Phase 12: full collision Region V1 evaluation

Run only after the collision-like feature campaigns are complete:

```text
validated feature completeness
-> projected metadata
-> Region V1 partitions
-> partition audit
-> raw histograms
-> unit-area plots
-> permitted exposure plots
-> completeness/accounting report
-> final no-op dry run
```

The final report covers intended versus validated sources, region/particle
counts, valid empty assignments, identity/exclusivity closure, exposure by
region/group, histogram flow, disabled normalization modes, and output
inventory.

## Phase 13: optional ABCD integration

Do not automatically feed all new shower features into the accepted ABCD fit.
For each candidate observable, freeze binning and flow policy, document its
physics motivation, quantify correlation with the Region V1 axes, compare
control-region data/MC, test closure and stability, define weighting and
blinding, and create a separately versioned ABCD configuration.

Generic distribution production remains independent of the statistical model.

## Dataset-isolated execution and DAG batching

Never request all five full feature families in one Snakemake invocation. Run
one family aggregate at a time so failures, accounting, and recovery remain
isolated. For large families, use the family aggregate itself as the Snakemake
batching rule:

```text
snakemake --snakefile snakemake/snakefile --dry-run --quiet --batch unified_features_mc_neutrino_v1=1/20 unified_features_mc_neutrino_v1

snakemake --snakefile snakemake/snakefile --dry-run --quiet --batch unified_features_real_data_2024_skim_v1=1/120 unified_features_real_data_2024_skim_v1
```

Use the same batch fraction for the eventual authorized execution after its
dry-run passes. Complete batches in numeric order, and run every batch from 1
through the denominator. The final batch is the one that includes the
aggregate rule. Recommended initial batch counts are 20 for neutrino MC, 3
for muon-up, 120 for collision data, 5 for testbeam MC, and 3 for testbeam
data. The denominator is operational rather than part of the data contract;
record any change in the task file before execution.

Canaries remain separate, unbatched targets. A successful batch dry-run is not
authorization to execute that batch.

## Recommended aggregate targets

```text
unified_feature_manifests_v1
unified_feature_input_audit_v1

unified_features_<family>_canary_v1
unified_features_<family>_v1

unified_collision_region_metadata_v1
unified_collision_region_partitions_v1
unified_collision_region_partition_audit_v1
unified_collision_region_hists_v1
unified_collision_region_plots_unit_area_v1
unified_collision_region_plots_exposure_v1
unified_collision_region_plots_v1
unified_collision_region_evaluation_v1
```

## Exact next action

Review the completed Phase 3 converter, provenance, manifest, validator, and
Snakemake diff. After explicit runtime authorization, run the five real
feature canaries one family at a time and validate each before proceeding to
the next. Do not begin full-family batches or the Region V1 adaptation in the
same action.
