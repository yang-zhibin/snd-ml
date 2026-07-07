# Region Cutflow Efficiency Evaluation Plan

## Goal

Add a metadata-driven cutflow efficiency study to the current region-partition
evaluation workflow.

The new study should answer:

```text
For each configured region and particle group, how many events survive each
configured cut, and what are the cumulative and relative efficiencies?
```

It should also answer the same question after combining regions:

```text
For one particle group, what is the efficiency after merging all selected
regions?

For all selected particle groups and regions together, what is the total
efficiency?
```

This workflow should live in the newer region evaluation system:

```text
evaluation_region_partitions/
snakemake/rules/region_evaluation.smk
snakemake/metadata/configs/region_evaluation/cutflow_efficiency_v1.yaml
```

The existing `evaluation_options_v1.yaml` still provides shared metadata
settings and the configured region/particle allow-list. The cutflow-specific
studies and cut definitions live in the separate `cutflow_efficiency_v1.yaml`
config so they can evolve independently from histogram and fiducial studies.

It should not be implemented in the older `evaluation/` folder. The old folder
is useful as a reference, especially `nueAnalysis_eff.py` and
`plot_eff_table.py`, but the old workflow is filename-driven. The region
workflow should remain metadata-driven.

## Source Of Truth

Use the combined region partition metadata CSV:

```text
snakemake/metadata/region_partitions/partition_v1_region_v1.csv
```

or, later, a split-aware catalog such as:

```text
snakemake/metadata/region_partitions/partition_v1_region_v1_split_v1.csv
```

The current region evaluation config already points to this file:

```yaml
metadata:
  partition_csv: snakemake/metadata/region_partitions/partition_v1_region_v1.csv
  tree_name: sndData
```

Important metadata columns:

```text
region
particle_group
particle_family
particle_id
partition_id
feature_partition_path
n_events
lumi_per_partition
feature_available
selection_expression
particle_filter_expression
```

The efficiency script should open `feature_partition_path` and count events in
the `sndData` tree.

Do not infer categories from ROOT filenames. Do not scan EOS directories for
`hist_*.root` or `eff_*.root`. The old workflow does that, but the region
workflow should not.

## Efficiency Definitions

The user-requested two efficiency types are:

```text
cumulative_efficiency = count(denominator && cut1 && ... && cut_i) / count(denominator)
relative_efficiency   = count(denominator && cut_i) / count(denominator)
```

In this plan, `relative_efficiency` means the single-cut efficiency of each cut
against the same initial denominator. It does not mean the old step-to-previous
cutflow efficiency.

For a cut sequence:

```text
Total
cut1
cut2
cut3
```

the counts are:

```text
initial_count = count(denominator)

cumulative_count_cut1 = count(denominator && cut1)
cumulative_count_cut2 = count(denominator && cut1 && cut2)
cumulative_count_cut3 = count(denominator && cut1 && cut2 && cut3)

relative_count_cut1 = count(denominator && cut1)
relative_count_cut2 = count(denominator && cut2)
relative_count_cut3 = count(denominator && cut3)
```

and the efficiencies are:

```text
cut1:
  cumulative_efficiency = cumulative_count_cut1 / initial_count
  relative_efficiency   = relative_count_cut1 / initial_count

cut2:
  cumulative_efficiency = cumulative_count_cut2 / initial_count
  relative_efficiency   = relative_count_cut2 / initial_count

cut3:
  cumulative_efficiency = cumulative_count_cut3 / initial_count
  relative_efficiency   = relative_count_cut3 / initial_count
```

This differs from the old `nueAnalysis_eff.py` / `plot_eff_table.py` meaning
of relative efficiency, where relative efficiency was computed against the
previous cut step. The new implementation must use the definitions above.

## Denominator Meaning

The first implementation should compute efficiency inside already-produced
region partitions.

That means:

```text
initial_count = events in selected region-partition feature files passing the
configured denominator expression
```

If the denominator expression is `1`, then the initial count is simply all
events already assigned to that region and particle group.

This is the safest first implementation because region partition files contain
only events that already passed the region selection. Events outside the region
are no longer present in these files.

### Important Limitation

This first implementation does not measure efficiency relative to the original
unpartitioned source feature files before region selection.

For example, if the region is:

```text
signal_no_veto_has_us_fiducial_inside
```

then `denominator: 1` means:

```text
all events already inside signal_no_veto_has_us_fiducial_inside
```

not:

```text
all events from the original MC sample before signal-region selection
```

If a future analysis needs absolute pre-region efficiency, add a second mode
that opens `source_feature_paths` from the partition metadata and applies region
selection during the efficiency calculation. That mode is more expensive and
must avoid double-counting source files that appear in multiple partitions.

## Config Design

Add a separate cutflow config:

```text
snakemake/metadata/configs/region_evaluation/cutflow_efficiency_v1.yaml
```

The existing region-evaluation config still provides shared metadata settings
and comparison group selection:

```yaml
metadata:
  partition_csv: snakemake/metadata/region_partitions/partition_v1_region_v1.csv
  tree_name: sndData

comparisons:
  regions:
  particle_groups:
```

The cutflow config owns cutflow-specific cuts and studies. It can duplicate
definitions from `evaluation_options_v1.yaml` when that keeps the cutflow study
self-contained.

Use a structure like:

```yaml
version: cutflow_efficiency_v1

metadata:
  tree_name: sndData

defaults:
  denominator:
    name: region_events
    expression: "1"
  aggregation_modes:
    - per_region_particle
    - overall_particle
    - overall_all

cuts:
  definitions:
    avg_scifi_fiducial:
      expression: cutFlowSummary_AvgSFChan == 1
      applies_to: all

studies:
  signal_cutflow:
    enabled: true
    cuts:
      - avg_scifi_fiducial
      - no_veto_hit
```

The initial study can use cut definitions copied from nueAnalysis:

```yaml
avg_scifi_fiducial:
  expression: cutFlowSummary_AvgSFChan == 1
  applies_to: all

no_veto_hit:
  expression: cutFlowSummary_NoVetoHits == 1
  applies_to: all

consecutive_scifi_hits:
  expression: cutFlowSummary_At_least_two_consecutive_SciFi_planes == 1
  applies_to: all
```

Add new cuts as named definitions in the cutflow config's
`cuts.definitions` section.

Example separated-plane fiducial cut:

```yaml
separate_plane_fiducial:
  expression: >
    (count_scifi1 <= 2 || (avg_scifi1_x > -44 && avg_scifi1_x < -10 && avg_scifi1_y > 18 && avg_scifi1_y < 52)) &&
    (count_scifi2 <= 2 || (avg_scifi2_x > -44 && avg_scifi2_x < -10 && avg_scifi2_y > 18 && avg_scifi2_y < 52)) &&
    (count_scifi3 <= 2 || (avg_scifi3_x > -42 && avg_scifi3_x < -12 && avg_scifi3_y > 20 && avg_scifi3_y < 50)) &&
    (count_scifi4 <= 2 || (avg_scifi4_x > -42 && avg_scifi4_x < -12 && avg_scifi4_y > 20 && avg_scifi4_y < 50)) &&
    (count_scifi5 <= 2 || (avg_scifi5_x > -42 && avg_scifi5_x < -12 && avg_scifi5_y > 20 && avg_scifi5_y < 50))
  applies_to: all
```

This makes the separated-plane fiducial available to any study:

```yaml
cuts:
  - separate_plane_fiducial
```

### Cut Application Rules

Keep supporting the existing `applies_to` logic:

```text
all
real_data_only
mc_only
```

For `real_data_only` cuts, skip the cut for MC groups.

For `mc_only` cuts, skip the cut for real data.

Skipped cuts should not silently appear as zero-efficiency cuts. The output
should mark them clearly.

Recommended behavior:

```text
If a cut does not apply to the current aggregation:
  do not filter the dataframe
  keep cumulative_count equal to the previous cumulative_count
  set relative_count equal to initial_count
  set relative_efficiency = 1
  set cumulative_efficiency unchanged
  set applied = false
```

This matches the old idea that data-quality cuts are pass-all placeholders for
MC, but makes the behavior explicit.

## Aggregation Modes

The script should support three aggregation modes.

### 1. per_region_particle

Group by:

```text
region
particle_group
```

This answers:

```text
Inside this region, for this particle group, what is the cutflow?
```

Example:

```text
region = signal_no_veto_has_us_fiducial_inside
particle_group = CC_nue
```

### 2. overall_particle

Group by:

```text
particle_group
```

and combine all configured regions for each particle group.

This answers:

```text
Across all selected regions, for this particle group, what is the cutflow?
```

Example:

```text
particle_group = CC_nue
region = ALL
```

### 3. overall_all

Combine all configured regions and all configured particle groups.

This answers:

```text
Across all selected events in the region catalog, what is the cutflow?
```

This is mostly useful as a global sanity check.

## Region And Particle Selection

Use the existing region-evaluation selection lists:

```yaml
comparisons:
  regions:
  particle_groups:
```

The cutflow script should only include metadata rows whose `region` and
`particle_group` are allowed by these lists.

If the lists are empty or absent, include all rows in the metadata CSV.

This keeps the efficiency study aligned with histogram comparisons.

## Output Files

Add one output directory under the existing region-evaluation study layout:

```text
evaluation_region_partitions/
  partition_v1_region_v1/
    hist_features_v1__evaluation_options_v1/
      cutflow_efficiency/
```

Recommended files:

```text
summary.csv
summary_by_study.csv
```

The first implementation can write only `summary.csv`. It should contain one
row per:

```text
study
aggregation_mode
region label
particle_group label
cut step
```

Suggested columns:

```text
study
aggregation_mode
region
particle_group
cut_index
cut_name
cut_label
applied
applies_to
initial_count
cumulative_count
relative_count
cumulative_efficiency
relative_efficiency
cumulative_efficiency_error
relative_efficiency_error
denominator_name
denominator_expression
cumulative_expression
relative_expression
cut_expression
n_partitions
n_added_files
n_skipped_files
lumi
skipped_feature_partition_paths
```

Use these placeholder values for aggregate labels:

```text
region = ALL
particle_group = ALL
```

Examples:

```text
per_region_particle:
  region = signal_no_veto_has_us_fiducial_inside
  particle_group = CC_nue

overall_particle:
  region = ALL
  particle_group = CC_nue

overall_all:
  region = ALL
  particle_group = ALL
```

## Error Calculation

Use binomial errors for both cumulative and relative efficiencies.

For:

```text
efficiency = numerator / denominator
```

use:

```text
error = sqrt(efficiency * (1 - efficiency) / denominator)
```

with safe behavior:

```text
if denominator <= 0:
  efficiency = 0
  error = 0
```

For cumulative efficiency:

```text
numerator = cumulative_count
denominator = initial_count
```

For relative efficiency:

```text
numerator = relative_count
denominator = initial_count
```

## Script Design

Create:

```text
evaluation_region_partitions/cutflow_efficiency.py
```

Command-line interface:

```text
python evaluation_region_partitions/cutflow_efficiency.py \
  --metadata-csv snakemake/metadata/region_partitions/partition_v1_region_v1.csv \
  --eval-config snakemake/metadata/configs/region_evaluation/evaluation_options_v1.yaml \
  --cutflow-config snakemake/metadata/configs/region_evaluation/cutflow_efficiency_v1.yaml \
  --output evaluation_region_partitions/.../cutflow_efficiency/summary.csv \
  --repo-root /afs/cern.ch/work/z/zhibin/snd-ml
```

Optional arguments:

```text
--tree sndData
--study signal_cutflow
```

If `--study` is omitted, run all enabled studies.

### Internal Steps

1. Load region-evaluation YAML config.
2. Load cutflow YAML config.
3. Load partition metadata CSV.
4. Filter rows by configured regions and particle groups from
   `evaluation_options_v1.yaml`.
5. Build aggregation groups.
6. For each enabled study and aggregation group:
   - open a `ROOT.TChain`;
   - add all valid `feature_partition_path` files;
   - record skipped files;
   - create `ROOT.RDataFrame`;
   - apply the denominator expression;
   - count `initial_count`;
   - loop through cuts in order;
   - apply only cuts that apply to this aggregation;
   - compute `cumulative_count` from the cumulative selection;
   - compute `relative_count` from the current cut alone under the denominator;
   - compute cumulative efficiency, relative efficiency, and errors;
   - append output rows.
7. Write `summary.csv`.

### Path Resolution

Reuse the pattern from existing region-evaluation scripts:

```text
If path starts with root:// or is absolute:
  use it directly
else:
  resolve relative to --repo-root
```

### Building A TChain

For each metadata row:

```text
path = feature_partition_path
chain.Add(path)
```

If `chain.Add(path) <= 0`, record the path in `skipped_feature_partition_paths`
and continue.

If no files can be added for a group, do not crash the entire study unless the
config asks for strict behavior. Recommended first behavior:

```text
write zero-count rows for that group
```

However, if all groups fail, raise an error. This prevents producing a
misleading all-zero CSV when paths are wrong.

## Aggregation And `applies_to` Edge Case

For `per_region_particle`, deciding whether a group is real data is simple:

```text
particle_group == real_data or particle_family == real_data
```

For `overall_particle`, it is also simple because there is still one
particle group.

For `overall_all`, the group may contain both real data and MC. In that case,
`real_data_only` and `mc_only` cuts are ambiguous.

Recommended first implementation:

```text
overall_all only runs cuts with applies_to: all
```

If a study includes `real_data_only` or `mc_only` cuts and also requests
`overall_all`, write skipped rows for those cuts with:

```text
applied = false
skip_reason = mixed_real_data_and_mc
```

This avoids applying a real-data-only cut to MC or vice versa in a mixed chain.

If mixed overall cutflow becomes important later, implement separate
sub-aggregations:

```text
overall_real_data
overall_mc
```

## Duplicate Counting Risk

When using `feature_partition_path`, duplicate counting is not expected inside
one aggregation because each partition row points to one already-produced
region/particle partition.

Still, protect against duplicate metadata rows by de-duplicating:

```text
partition_id
feature_partition_path
```

within each aggregation group before building the chain.

Do not de-duplicate only by `feature_partition_path` globally because different
studies may intentionally reuse the same paths.

## Snakemake Integration

Modify:

```text
snakemake/rules/region_evaluation.smk
```

Add:

```python
REGION_EVAL_CUTFLOW_EFFICIENCY_CSV = (
    f"{_region_eval_study_output_dir('cutflow_efficiency')}/summary.csv"
)
```

Add a rule:

```python
rule region_eval_cutflow_efficiency:
    input:
        metadata=REGION_EVAL_METADATA_CSV,
        eval_config=REGION_EVAL_OPTIONS_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/cutflow_efficiency.py",
    output:
        summary=REGION_EVAL_CUTFLOW_EFFICIENCY_CSV,
    threads: 1
    resources:
        runtime=45 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        echo "summarizing region evaluation cutflow efficiencies"
        set +u
        source {env_script_lcg}
        set -euo pipefail

        python {input.script} \
            --metadata-csv {input.metadata} \
            --eval-config {input.eval_config} \
            --output {output.summary} \
            --repo-root {PERSONAL_WORK_SPACE}
        """
```

Then add the output to:

```python
rule region_eval_studies:
    input:
        REGION_EVAL_FIDUCIAL_SUMMARY_CSV,
        REGION_EVAL_FIDUCIAL_OVERLAP_CSV,
        REGION_EVAL_EVENTBUILDER_VETO_TIMING_CSV,
        REGION_EVAL_CUTFLOW_EFFICIENCY_CSV,
```

## Relationship To Existing Fiducial Efficiency

Keep the existing script:

```text
evaluation_region_partitions/fiducial_efficiency.py
```

That script is a specialized overlap and migration study:

```text
current average fiducial vs QDC-weighted fiducial vs SNDSW fiducial
```

The new `cutflow_efficiency.py` is a sequential cutflow study:

```text
cut1, then cut2, then cut3, ...
```

They are complementary and should not be merged.

## Validation Plan

### 1. Dry Config Validation

Run the script on the config without opening ROOT files if possible.

Check:

```text
all study cuts exist in cuts.definitions
all cut expressions are non-empty when applied
all aggregation modes are recognized
metadata CSV has required columns
```

If not implementing a separate dry-run mode, these checks should still happen
before the first ROOT event loop.

### 2. Small Single-Group Test

Run one study on one known group first:

```text
region = signal_no_veto_has_us_fiducial_inside
particle_group = real_data
```

Expected behavior:

```text
initial_count equals total entries in the selected feature partition chain
cumulative_count is monotonically non-increasing for applied cuts
cumulative_efficiency is monotonically non-increasing
relative_efficiency is between 0 and 1
```

### 3. MC Data-Quality Cut Behavior

If a study includes data-quality cuts:

```text
stable_beams
ip1_bunch_crossing
pre_evt_clock_cycle_100
```

and those cuts are marked `real_data_only`, check an MC group.

Expected:

```text
applied = false
cumulative_count remains unchanged
relative_count = initial_count
relative_efficiency = 1
cumulative_efficiency remains unchanged
```

### 4. Aggregate Count Consistency

For a particle group:

```text
overall_particle initial_count
```

should equal the sum of the corresponding:

```text
per_region_particle initial_count
```

for the same particle group and configured regions, assuming no skipped files.

### 5. Old Workflow Cross-Check

For an equivalent selection, compare the new output to old `cutflow_data.csv`
or `cutflow_summary.csv`.

Exact equality is not guaranteed if the region-partition denominator is already
post-region-selection, but the cumulative and relative efficiency definitions
should match.

## Implementation Order

1. Add `cutflow_efficiency_v1.yaml` and any missing cut definitions.
2. Create `evaluation_region_partitions/cutflow_efficiency.py`.
3. Add Snakemake output path and rule.
4. Add the new output to `region_eval_studies`.
5. Run a small direct script test.
6. Run the Snakemake target:

```text
snakemake -s snakemake/snakefile region_eval_cutflow_efficiency
```

7. Inspect `summary.csv`.
8. Only after the CSV is trusted, optionally add plotting/table rendering.

## Future Extensions

Possible later additions:

```text
overall_real_data
overall_mc
per_particle_family
per_region_particle_family
split-aware efficiency using partition_v1_region_v1_split_v1.csv
pre-region absolute efficiency using source_feature_paths
PDF/PNG cutflow tables
ROOT histograms matching old cutflow_counts/cutflow_cumeff/cutflow_releff
```

Do not add these until the CSV logic is validated.
