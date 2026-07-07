# Weighted Position and Veto Timing Evaluation Plan

## Goal

Study the new QDC-weighted position features and veto timing features in the
region-partition evaluation workflow, without repeatedly reprocessing expensive
inputs.

The immediate study should answer:

1. How do current average position features compare with the new QDC-weighted
   position features?
2. How do these comparisons look separately for real data, MC neutrino truth
   groups, EventBuilder neutrino groups, and different analysis regions?
3. What do veto hit timing distributions look like in veto-tagged and
   no-veto regions?
4. How do veto hit timing distributions compare between normal MC neutrino
   samples and EventBuilder MC neutrino samples?
5. Later, if weighted positions look better, how should a new SciFi fiducial
   definition be compared against the current average-position fiducial and the
   SND software fiducial?

## Main Principle

Separate expensive production from cheap evaluation.

The desired flow is:

```text
feature.root with all old and new branches
        |
        v
region feature partitions with current region_v1 definition
        |
        v
many evaluation histograms, comparisons, and summaries
        |
        v
only later: new region version using weighted-position fiducial
```

Do not change the region definition while first comparing features. If the
region definition changes too early, the event samples change at the same time
as the feature definition, making the comparison harder to interpret.

## Inputs

The upstream feature files should contain both the existing average-position
branches and the new weighted-position branches.

Existing average-position branches:

```text
avg_scifi_x
avg_scifi_y
avg_scifi1_x ... avg_scifi5_x
avg_scifi1_y ... avg_scifi5_y
avg_us_y, avg_us1_y ... avg_us5_y
avg_ds_x, avg_ds_y, avg_ds1_x ... avg_ds4_x, avg_ds1_y ... avg_ds4_y
avg_veto_x, avg_veto_y, avg_veto1_y, avg_veto2_y, avg_veto3_x
```

New QDC-weighted branches:

```text
qdcAvg_scifi_x
qdcAvg_scifi_y
qdcAvg_scifi1_x ... qdcAvg_scifi5_x
qdcAvg_scifi1_y ... qdcAvg_scifi5_y
qdcAvg_us_y, qdcAvg_us1_y ... qdcAvg_us5_y
qdcAvg_ds_x, qdcAvg_ds_y, qdcAvg_ds1_x ... qdcAvg_ds4_x, qdcAvg_ds1_y ... qdcAvg_ds4_y
qdcAvg_veto_x, qdcAvg_veto_y, qdcAvg_veto1_y, qdcAvg_veto2_y, qdcAvg_veto3_x
```

Veto timing branches:

```text
vetoHitTime_earlist
vetoHitTime_latest
vetoHitTime_earlist_veto1
vetoHitTime_latest_veto1
vetoHitTime_earlist_veto2
vetoHitTime_latest_veto2
vetoHitTime_earlist_veto3
vetoHitTime_latest_veto3
```

Note: `earlist` is the current branch spelling in the feature producer. Keep
that spelling in configs unless the producer branch name is changed.

## Stage 1: Build One Stable Evaluation Sample

### Purpose

Create region feature partitions with the current region definitions, but using
feature files that include the new weighted-position and veto timing branches.

### Logic

Use the current region definition:

```text
snakemake/metadata/configs/regions/region_v1.yaml
```

This region version uses current average SciFi position branches for the SciFi
fiducial classification. That is intentional for the first study.

Use the current partition policy:

```text
snakemake/metadata/configs/partitions/partition_v1.yaml
```

For real data, use the skim metadata, not the full 2024 metadata:

```python
subset_metadata_csv_list = [
    "real_data_2024_skim_runs_metadata.csv",
]
```

Keep the full real-data metadata disabled:

```python
# "real_data_2024_metadata.csv",
```

### Expected Output

The central catalog remains:

```text
snakemake/metadata/region_partitions/partition_v1_region_v1.csv
```

Each region feature partition should now contain both old and new feature
families. After this stage, all feature comparisons should be evaluation-only.

## Stage 2: Compare Average Position and Weighted Position

### Purpose

Compare old average-position features and new QDC-weighted-position features
for each particle group and each analysis region.

This should show whether weighted positions:

- shift the reconstructed event position;
- reduce tails or outliers;
- separate signal-like neutrino events from sideband or real-data events better;
- behave consistently between MC neutrino, EventBuilder, and real data.

### Implementation

Add weighted-position histogram features to:

```text
snakemake/metadata/configs/region_evaluation/hist_features_v1.yaml
```

For direct comparisons, add features like:

```yaml
qdcAvg_scifi_x:
  expression: qdcAvg_scifi_x
  x_min: -60
  x_max: 0
  bin_width: 1
  axis_title: QDC-weighted Average SciFi X Position
  logy: true

qdcAvg_scifi_y:
  expression: qdcAvg_scifi_y
  x_min: 0
  x_max: 60
  bin_width: 1
  axis_title: QDC-weighted Average SciFi Y Position
  logy: true
```

For per-plane features, add:

```text
qdcAvg_scifi1_x ... qdcAvg_scifi5_x
qdcAvg_scifi1_y ... qdcAvg_scifi5_y
```

Use feature cuts for per-plane average positions, matching the existing
average-position logic:

```yaml
feature_cut: count_scifi1 > 2
```

For direct old-vs-new difference features, add expressions like:

```yaml
qdcAvg_scifi_x_minus_avg_scifi_x:
  expression: qdcAvg_scifi_x - avg_scifi_x
  x_min: -30
  x_max: 30
  bin_width: 1
  axis_title: QDC-weighted SciFi X minus Average SciFi X
  logy: false
  feature_cut: count_scifi > 2
```

And per-plane versions:

```text
qdcAvg_scifi1_x - avg_scifi1_x
qdcAvg_scifi2_x - avg_scifi2_x
...
qdcAvg_scifi1_y - avg_scifi1_y
qdcAvg_scifi2_y - avg_scifi2_y
...
```

### Comparisons Produced

The existing region evaluation plotting already makes two useful comparison
modes:

1. Particle groups within one fixed region.
2. Regions for one fixed particle group.

Therefore, after adding the histogram definitions, the same workflow can compare:

```text
real_data
CC_nue
CC_numu
NC_nue
NC_numu
CC_nue_EventBuilder
CC_numu_EventBuilder
NC_nue_EventBuilder
NC_numu_EventBuilder
```

inside each configured region, and can also compare all regions for one
particle group.

## Stage 3: Compare Veto Hit Timing

### Purpose

Study veto timing distributions for data and MC, especially in veto-tagged
sidebands. This can help understand whether veto timing behaves consistently
between samples and whether veto-tagged regions contain different timing
components.

This stage should also compare normal MC neutrino samples against the
EventBuilder versions of the same neutrino classes. This is important because
EventBuilder can change the detector activity around the neutrino interaction,
and veto timing is one of the observables most likely to reveal such a
difference.

### Features

Add whole-veto timing histograms:

```yaml
vetoHitTime_earlist:
  expression: vetoHitTime_earlist
  x_min: -10
  x_max: 200
  bin_width: 2
  axis_title: Earliest Veto Hit Time
  logy: true
  feature_cut: count_veto > 0

vetoHitTime_latest:
  expression: vetoHitTime_latest
  x_min: -10
  x_max: 200
  bin_width: 2
  axis_title: Latest Veto Hit Time
  logy: true
  feature_cut: count_veto > 0
```

Add per-veto-plane timing histograms:

```yaml
vetoHitTime_earlist_veto1:
  expression: vetoHitTime_earlist_veto1
  x_min: -10
  x_max: 200
  bin_width: 2
  axis_title: Earliest Veto 1 Hit Time
  logy: true
  feature_cut: count_veto1 > 0
```

Repeat for:

```text
vetoHitTime_latest_veto1 with count_veto1 > 0
vetoHitTime_earlist_veto2 with count_veto2 > 0
vetoHitTime_latest_veto2 with count_veto2 > 0
vetoHitTime_earlist_veto3 with count_veto3 > 0
vetoHitTime_latest_veto3 with count_veto3 > 0
```

### Important Detail

When plotting a per-plane veto time, require that the corresponding veto plane
has at least one hit. Otherwise sentinel values such as `-1` or `-999` can
dominate the histogram and hide the physical timing distribution.

For example:

```text
veto plane 1 timing -> feature_cut: count_veto1 > 0
veto plane 2 timing -> feature_cut: count_veto2 > 0
veto plane 3 timing -> feature_cut: count_veto3 > 0
whole veto timing   -> feature_cut: count_veto > 0
```

### Expected Behavior

No-veto regions should have empty or nearly empty veto timing distributions
after these feature cuts. Has-veto regions should contain the relevant timing
events.

If no-veto regions still show many entries, that indicates either:

- the region definition and veto count branch are inconsistent;
- the feature cut is wrong;
- sentinel values are slipping through;
- the partition was built from stale feature files.

### EventBuilder vs Non-EventBuilder Comparison

Compare each normal MC neutrino group with its EventBuilder counterpart:

```text
CC_nue  vs CC_nue_EventBuilder
CC_numu vs CC_numu_EventBuilder
NC_nue  vs NC_nue_EventBuilder
NC_numu vs NC_numu_EventBuilder
```

This comparison should be done for each veto timing feature:

```text
vetoHitTime_earlist
vetoHitTime_latest
vetoHitTime_earlist_veto1
vetoHitTime_latest_veto1
vetoHitTime_earlist_veto2
vetoHitTime_latest_veto2
vetoHitTime_earlist_veto3
vetoHitTime_latest_veto3
```

And in each relevant region:

```text
sideband_has_veto_has_us_fiducial_inside
sideband_has_veto_no_us_fiducial_inside
sideband_has_veto_has_us_fiducial_outside
sideband_has_veto_no_us_fiducial_outside
```

The no-veto regions can still be plotted as a validation check, but the main
physics comparison should focus on has-veto regions because the timing feature
cuts require veto hits.

The existing region evaluation can already show these samples on the same page
when using the `particles_in_region` comparison mode, because the configured
particle groups include both normal MC and EventBuilder MC groups. If a cleaner
one-to-one comparison is needed, add an EventBuilder-specific plotting mode or
post-processing summary that creates paired overlays and ratios for:

```text
EventBuilder / normal MC
```

for each matched neutrino class.

Useful summary quantities:

```text
normal_integral
eventbuilder_integral
eventbuilder_over_normal_integral
normal_mean_time
eventbuilder_mean_time
eventbuilder_minus_normal_mean_time
```

These should be reported separately for whole-veto timing and each veto plane.

Implemented Snakemake target:

```text
region_eval_eventbuilder_veto_timing
```

Current implemented output:

```text
evaluation_region_partitions/partition_v1_region_v1/hist_features_v1__evaluation_options_v1/eventbuilder_veto_timing/summary.csv
```

## Stage 4: Compare Fiducial Efficiencies

### Purpose

Before replacing the current SciFi fiducial definition, compare three fiducial
definitions:

1. Current average-position SciFi fiducial.
2. New QDC-weighted-position SciFi fiducial.
3. SND software SciFi fiducial, currently represented by:

```text
cutFlowSummary_AvgSFChan == 1
```

### Why This Should Be Separate

This is not only a histogram-shape comparison. It is a cut-efficiency and event
migration study.

Changing the SciFi fiducial changes which events are inside and outside the
fiducial region. Therefore, it affects:

- signal region membership;
- sideband region membership;
- event yields;
- real-data normalization per region;
- training and evaluation samples downstream.

### Recommended Denominator

Use a denominator before the fiducial choice, for example:

```text
count_scifi > 35
```

Then define numerators:

```text
current_avg_fiducial_pass
qdcAvg_fiducial_pass
sndsw_fiducial_pass
```

Efficiency:

```text
efficiency = numerator / denominator
```

Compute this separately for:

```text
particle_group
region category or source sample
real_data / MC / EventBuilder
```

### Overlap Table

The overlap table is more informative than efficiency alone.

For each sample, count:

```text
current_avg_pass && qdcAvg_pass
current_avg_pass && !qdcAvg_pass
!current_avg_pass && qdcAvg_pass
!current_avg_pass && !qdcAvg_pass
```

And also compare to SND software:

```text
sndsw_pass && current_avg_pass
sndsw_pass && qdcAvg_pass
sndsw_pass && !current_avg_pass
sndsw_pass && !qdcAvg_pass
```

This tells us which events move when replacing the current average-position
fiducial with the weighted-position fiducial.

### Suggested Implementation

Add a small evaluation script rather than encoding all of this as ordinary
histograms.

The script should:

1. Read the partition metadata CSV.
2. Open feature partition ROOT files.
3. For each particle group and optional region/source category, count pass/fail
   for named fiducial expressions.
4. Write a CSV summary with denominator, numerator, efficiency, and binomial
   uncertainty.
5. Write an overlap CSV for event migration between definitions.

Suggested output:

```text
evaluation_region_partitions/fiducial_efficiency/
  partition_v1_region_v1/
    fiducial_efficiency_summary.csv
    fiducial_overlap_summary.csv
```

Implemented Snakemake target:

```text
region_eval_fiducial_efficiency
```

Current implemented outputs:

```text
evaluation_region_partitions/partition_v1_region_v1/hist_features_v1__evaluation_options_v1/fiducial_efficiency/summary.csv
evaluation_region_partitions/partition_v1_region_v1/hist_features_v1__evaluation_options_v1/fiducial_efficiency/overlap.csv
```

## Stage 5: Define a New Region Version

### Purpose

Only after the weighted-position feature and fiducial studies are understood,
define a new region version that actually uses the weighted-position fiducial.

Do not overwrite:

```text
region_v1.yaml
```

Instead create a new file, for example:

```text
snakemake/metadata/configs/regions/region_qdcAvg_v1.yaml
```

### New Fiducial Logic

The current fiducial macro uses branches like:

```text
avg_scifi1_x
avg_scifi1_y
...
avg_scifi5_x
avg_scifi5_y
```

The weighted-position version would use:

```text
qdcAvg_scifi1_x
qdcAvg_scifi1_y
...
qdcAvg_scifi5_x
qdcAvg_scifi5_y
```

Example shape:

```yaml
expression_macros:
  __SCIFI_FIDUCIAL_INSIDE__: >
    (
      (count_scifi1 <= 2 || (qdcAvg_scifi1_x > -44 && qdcAvg_scifi1_x < -10 && qdcAvg_scifi1_y > 18 && qdcAvg_scifi1_y < 52)) &&
      (count_scifi2 <= 2 || (qdcAvg_scifi2_x > -44 && qdcAvg_scifi2_x < -10 && qdcAvg_scifi2_y > 18 && qdcAvg_scifi2_y < 52)) &&
      (count_scifi3 <= 2 || (qdcAvg_scifi3_x > -42 && qdcAvg_scifi3_x < -12 && qdcAvg_scifi3_y > 20 && qdcAvg_scifi3_y < 50)) &&
      (count_scifi4 <= 2 || (qdcAvg_scifi4_x > -42 && qdcAvg_scifi4_x < -12 && qdcAvg_scifi4_y > 20 && qdcAvg_scifi4_y < 50)) &&
      (count_scifi5 <= 2 || (qdcAvg_scifi5_x > -42 && qdcAvg_scifi5_x < -12 && qdcAvg_scifi5_y > 20 && qdcAvg_scifi5_y < 50))
    )
```

### What Changes When This Is Done

Changing the region config changes event membership. Therefore, this requires a
new partition metadata output, for example:

```text
partition_v1_region_qdcAvg_v1.csv
```

At that point, rebuild region partitions for the new region version and compare
the old and new region outputs at the yield and efficiency level.

## Recommended Order of Work

1. Make sure upstream feature files contain both `avg_*` and `qdcAvg_*`
   branches.
2. Use skim real-data metadata instead of full real-data metadata.
3. Rebuild only stale or wrong real-data feature partitions; keep valid MC
   partitions when possible.
4. Rebuild the combined partition metadata CSV.
5. Add weighted-position and difference histogram features to
   `hist_features_v1.yaml`.
6. Add veto timing histogram features with correct `feature_cut` requirements.
7. Run `region_eval_group_hists` or `region_eval_comparison_plots`.
8. Inspect comparison PDFs and summary CSVs.
9. Add a dedicated fiducial efficiency and overlap script.
10. Only after the study is complete, create a new weighted-position region
    config and rebuild partitions under a new region version.

## What Should Not Be Done Yet

Do not immediately replace `avg_scifi*_x/y` with `qdcAvg_scifi*_x/y` inside
`region_v1.yaml`.

Do not overwrite existing region version names when changing event selection.

Do not mix feature files with and without `qdcAvg_*` branches in the same
partition study unless the missing branches are handled explicitly and the
result is clearly labeled as a partial or compatibility study.

## Commands to Remember

Dry-run evaluation target:

```bash
snakemake -s snakefile region_eval_comparison_plots -n
```

Build grouped histograms:

```bash
snakemake -s snakefile region_eval_group_hists \
  --profile ../.config/snakemake/htcondor \
  --rerun-triggers mtime
```

Build comparison plots:

```bash
snakemake -s snakefile region_eval_comparison_plots \
  --profile ../.config/snakemake/htcondor \
  --rerun-triggers mtime
```

Build fiducial and EventBuilder veto timing CSV studies:

```bash
snakemake -s snakefile region_eval_studies \
  --profile ../.config/snakemake/htcondor \
  --rerun-triggers mtime
```

Build region feature partitions with grouped HTCondor jobs:

```bash
snakemake -s snakefile region_feature_partitions \
  --profile ../.config/snakemake/htcondor \
  --rerun-triggers mtime \
  --groups build_region_feature_partition=feature_partition_group \
  --group-components feature_partition_group=10
```
