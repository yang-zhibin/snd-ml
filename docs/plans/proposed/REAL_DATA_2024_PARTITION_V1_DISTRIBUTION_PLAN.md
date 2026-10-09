# 2024 real-data Partition V1 distribution plan

Status: proposed

## Objective

Produce reproducible shower-feature distributions for the 2024 skimmed
collision-data snapshot in the eight regions defined by `region_v1`, while
preserving the existing no-Veto and deterministic 10% Veto sampling policies.

The implementation must remain separate from the complete-sample workflow and
must label its outputs as preliminary snapshot products.

## Inputs and cohorts

Use the validated selection products from
`snakemake/unified_real_data_selection.smk` and keep two explicit cohorts:

1. `real_data_2024_no_veto`, selected by each source's `no_veto.root` index.
2. `real_data_2024_veto_10pct`, selected by each source's `veto_10pct.root`
   index.

Freeze the exact set of processed source files and the selection receipts in
versioned manifests. Do not rebuild the snapshot implicitly when more source
files become available.

The Veto cohort has already been sampled at probability 0.1. Downstream code
must not apply the `partition_v1` real-data Veto fraction a second time.

## Feature products

Convert each non-empty selection index with the established unified converter
using `indexed_cutflow`, collision-data provenance, and separate output roots
for the two cohorts. Preserve `(runId, eventId, eventIndex)` and validate the
output identities against the selected index entries.

Manifest and validation metadata must record:

- the physical source UID and input paths;
- the selection profile and selection-index checksum;
- input and selected event counts;
- sampling probability (`1.0` for no-Veto and `0.1` for Veto);
- event weight (`1.0` for no-Veto and `10.0` for Veto);
- feature and validation-record paths.

Sources with zero selected entries remain part of coverage accounting but do
not require an empty feature product.

## Region contract

Read the authoritative expressions from
`snakemake/metadata/configs/regions/region_v1.yaml`; do not copy independent
region expressions into the new workflow.

The eight regions are the product of:

- `count_veto == 0` or `count_veto > 0`;
- `count_us == 0` or `count_us > 0`;
- SciFi fiducial inside or outside;

with the common requirement `count_scifi > 35`.

The four no-Veto regions consume only the no-Veto snapshot. The four
Veto-active regions consume only the Veto-10% snapshot.

Before plotting, require an audit proving that the four applicable regions are
exclusive and exhaustive within each cohort's `count_scifi > 35` population.
Also require the cohort-level Veto predicate and unique event identities.

## Distributions and normalization

Produce these five distributions in every region:

- `density_sndsw_scifi`;
- `scifi_cluster_qdc_per_hit`;
- `scifi_qdc_per_hit_direct`;
- `scifi_vertical_cluster1_qdc_std`;
- `scifi_horizontal_cluster1_qdc_std`.

The primary outputs are unit-area shape comparisons. Also retain raw entries,
`sumw`, `sumw2`, underflow, overflow, source count, sampling policy, snapshot
checksum, and the complete region expression in histogram metadata.

For sampling-corrected yields, use event weight 1 for no-Veto and 10 for the
Veto-10% cohort. Do not infer full-2024 or 19 fb-1 yields by multiplying the
source-file snapshot by ten. An absolute-exposure plot requires validated
luminosity for the exact frozen source-file set.

## Plot products

For each feature, produce:

- a 2-by-4 unit-area small-multiple view of all eight regions;
- four paired no-Veto versus Veto overlays at fixed US/fiducial state;
- individual region histograms and CSV summaries;
- a region-population table or heatmap with raw and sampling-corrected counts.

Keep an all-region target separate from a control-region-only target so signal
region inspection can remain fail-closed when analysis blinding requires it.

## Proposed implementation files

- `snakemake/metadata/prepare_unified_real_data_partition_manifests.py`
- `snakemake/metadata/configs/unified_real_data_partition_distributions_v1.yaml`
- `snakemake/unified_real_data_partition_distributions.smk`
- `evaluation_region_partitions/audit_unified_region_partition.py`
- `evaluation_region_partitions/plot_unified_region_grid.py`
- backward-compatible weighted-histogram support in
  `evaluation_region_partitions/unified_feature_distributions.py`
- focused manifest, identity, region-audit, weighting, ROOT-histogram, and plot
  tests

## Aggregate targets

Use separate production and downstream boundaries:

- `unified_real_data_partition_features_v1`
- `unified_real_data_partition_snapshot_v1`
- `unified_real_data_partition_audit_v1`
- `unified_real_data_partition_raw_hists_v1`
- `unified_real_data_partition_control_regions_v1`
- `unified_real_data_partition_all_regions_v1`

The plotting DAG must consume a frozen feature snapshot and must not pull all
feature-conversion jobs into DAG construction.

## Validation sequence

1. Test deterministic two-cohort manifest construction and exclusions.
2. Run one local no-Veto and one local Veto feature canary.
3. Verify exact selected-index identity and provenance.
4. Test region exclusivity and exhaustiveness on synthetic and canary ROOT
   files.
5. Test Veto weighting and prove no second 10% selection is applied.
6. Test the 2-by-4 plot and its CSV metadata.
7. Dry-run each aggregate target.
8. Complete feature production before freezing the production snapshot.
9. Run histograms and plots only after the snapshot and region audit pass.

## Exit criteria

- Every published feature has a passing identity-aware validation receipt.
- Snapshot coverage and exclusions are explicit and versioned.
- Region audit proves exclusive and exhaustive assignment within each cohort.
- Veto sampling is applied exactly once and its inverse-probability weight is
  recorded.
- All five feature distributions and summary metadata are produced for the
  authorized region scope.
- No output is represented as full-2024 exposure without validated luminosity
  evidence for the frozen source set.
