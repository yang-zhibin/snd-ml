# Region Partition Evaluation Workflow Plan

## Goal

Build a new evaluation workflow for the region/particle partition outputs.

The main purpose is to compare feature distributions across regions and
particles after the region partition stage has produced aligned feature and
hit3D ROOT files.

The workflow should answer questions like:

```text
In the same region, how do different particles compare?
For the same particle, how do different regions compare?
```

This evaluation should be separate from the older `evaluation/` workflow. The
older workflow is useful as a reference for histogram settings, cut handling,
ROOT plotting style, luminosity scaling, and Snakemake structure, but it is
mostly filename-driven and expects old partition names.

The new workflow should instead be metadata-driven and consume the combined
region partition catalog:

```text
snakemake/metadata/region_partitions/partition_v1_region_v1.csv
```

or, later, a split-aware catalog such as:

```text
snakemake/metadata/region_partitions/partition_v1_region_v1_split_v1.csv
```

The important design rule is:

```text
Do not rediscover samples from filenames.
Use the partition metadata columns.
```

## Starting Point

The region partition stage produces one metadata row per produced partition.

Each row describes one pair of aligned ROOT files:

```text
feature_partition_path
hit3d_partition_path
```

with the invariant:

```text
feature_partition.root:sndData entry i == hit3d_partition.root:hit3D entry i
```

For this evaluation workflow, the first implementation should use:

```text
feature_partition_path
```

and plot branches or expressions from the `sndData` tree.

Important metadata columns for evaluation are:

```text
partition_id
region
particle_group
particle_family
particle_id
feature_partition_path
hit3d_partition_path
n_events
lumi_per_partition
rdf_selection
particle_filter
```

The workflow should not redefine region boundaries. Region definitions have
already been applied when the partition files were produced.

## New Folder Layout

Create a new evaluation folder, for example:

```text
evaluation_region_partitions/
```

Suggested files:

```text
evaluation_region_partitions/
  make_partition_hist.py
  plot_comparison.py
  config.py
  README.md
```

Add a new Snakemake rule file:

```text
snakemake/rules/region_evaluation.smk
```

Add two config files:

```text
snakemake/metadata/configs/region_evaluation/hist_features_v1.yaml
snakemake/metadata/configs/region_evaluation/evaluation_options_v1.yaml
```

The two-config split is intentional:

```text
hist_features_v1.yaml
  describes what can be plotted and how each histogram is binned

evaluation_options_v1.yaml
  describes cuts, comparison options, normalization, and selected groups
```

This keeps feature definitions separate from analysis choices.

## Config 1: Histogram Feature Settings

The histogram feature config should define all feature histograms.

Path:

```text
snakemake/metadata/configs/region_evaluation/hist_features_v1.yaml
```

This config should contain:

```text
feature name
ROOT branch or expression
x_min
x_max
bin_width
axis_title
logy
optional feature-specific cut
optional overflow/underflow handling
```

Most settings can be learned from the old evaluation scripts:

```text
evaluation/plot_hist_nueAnalysis.py
evaluation/find_had_scale_factor.py
```

Example structure:

```yaml
version: hist_features_v1

features:
  density_sndsw_scifi:
    expression: density_sndsw_scifi
    x_min: 100
    x_max: 40000
    bin_width: 500
    axis_title: Sum of SciFi Density Weight (SNDSW)
    logy: true
    feature_cut: ""
    fold_underflow: false
    fold_overflow: false

  count_scifi:
    expression: count_scifi
    x_min: 0
    x_max: 800
    bin_width: 10
    axis_title: SciFi Hit Total Count
    logy: true
    feature_cut: ""
    fold_underflow: false
    fold_overflow: false

  avg_scifi1_x:
    expression: avg_scifi1_x
    x_min: -60
    x_max: 0
    bin_width: 1
    axis_title: Plane 1 Average SciFi X Position
    logy: true
    feature_cut: count_scifi1 > 2
    fold_underflow: false
    fold_overflow: false
```

The first version should include the old evaluation features:

```text
density_scifi
density_scifi1
density_scifi2
density_scifi3
density_scifi4
density_scifi5
density_sndsw_scifi

count_scifi
count_scifi1
count_scifi2
count_scifi3
count_scifi4
count_scifi5

avg_scifi_x
avg_scifi1_x
avg_scifi2_x
avg_scifi3_x
avg_scifi4_x
avg_scifi5_x

avg_scifi_y
avg_scifi1_y
avg_scifi2_y
avg_scifi3_y
avg_scifi4_y
avg_scifi5_y

avg_ds_x
avg_ds1_x
avg_ds2_x
avg_ds3_x
avg_ds4_x

avg_ds_y
avg_ds1_y
avg_ds2_y
avg_ds3_y
avg_ds4_y

avg_us_y
avg_us1_y
avg_us2_y
avg_us3_y
avg_us4_y
avg_us5_y

count_us
count_us1
count_us2
count_us3
count_us4
count_us5

count_ds
count_ds1
count_ds2
count_ds3
count_ds4

qdc_scifi
qdc_scifi1
qdc_scifi2
qdc_scifi3
qdc_scifi4
qdc_scifi5

qdc_us
qdc_us1
qdc_us2
qdc_us3
qdc_us4
qdc_us5
```

The config should also be ready to include newer feature branches when they
exist in the partition feature ROOT files. Candidate examples are:

```text
density_sndsw_scifi_second
count_veto
count_veto1
count_veto2
qdc_veto
avg_veto_x
avg_veto_y
```

Before implementation, the actual branch list in the produced feature
partition files should be checked and the config should be updated
accordingly.

## Config 2: Evaluation Options And Cuts

The evaluation options config should define analysis choices.

Path:

```text
snakemake/metadata/configs/region_evaluation/evaluation_options_v1.yaml
```

This config should contain:

```text
input metadata CSV
normalization reference
cut definitions
named base cut options
named extra cut options
enabled comparison modes
selected regions
selected particle groups
plot defaults
```

Example structure:

```yaml
version: evaluation_options_v1

metadata:
  partition_csv: snakemake/metadata/region_partitions/partition_v1_region_v1.csv

normalization:
  default: signal_data_lumi
  reference:
    region: signal_no_veto_has_us_fiducial_inside
    particle_group: real_data

cuts:
  definitions:
    stable_beams:
      expression: cutFlowSummary_StableBeams == 1
      applies_to: real_data_only

    ip1_bunch_crossing:
      expression: cutFlowSummary_IP1 == 1
      applies_to: real_data_only

    pre_evt_clock_cycle_100:
      expression: cutFlowSummary_EventDeltat_1_100 == 1
      applies_to: real_data_only

    avg_scifi_fiducial:
      expression: cutFlowSummary_AvgSFChan == 1
      applies_to: all

    no_veto_hit:
      expression: cutFlowSummary_NoVetoHits == 1
      applies_to: all

    consecutive_scifi_hits:
      expression: cutFlowSummary_At_least_two_consecutive_SciFi_planes == 1
      applies_to: all

    scifi_continuity:
      expression: cutFlowSummary_SciFiContinuity == 1
      applies_to: all

  base_cut_options:
    none: []

    old_default:
      - stable_beams
      - ip1_bunch_crossing
      - pre_evt_clock_cycle_100
      - avg_scifi_fiducial
      - no_veto_hit
      - consecutive_scifi_hits
      - scifi_continuity

    region_basic:
      - no_veto_hit
      - consecutive_scifi_hits
      - scifi_continuity

  extra_cut_options:
    none: ""
    density_sndsw_scifi_gt_1000: density_sndsw_scifi > 1000
    count_scifi_gt_200: count_scifi > 200

comparisons:
  modes:
    - particles_in_region
    - regions_for_particle

  regions:
    - signal_no_veto_has_us_fiducial_inside
    - sideband_no_veto_no_us_fiducial_inside
    - sideband_no_veto_has_us_fiducial_outside
    - sideband_no_veto_no_us_fiducial_outside
    - sideband_has_veto_has_us_fiducial_inside
    - sideband_has_veto_no_us_fiducial_inside
    - sideband_has_veto_has_us_fiducial_outside
    - sideband_has_veto_no_us_fiducial_outside

  particle_groups:
    - real_data
    - CC_nue
    - CC_numu
    - NC_nue
    - NC_numu
    - muonDIS
    - muon
```

This config can also include kaon and neutron groups when those partitions are
enabled:

```text
kaon_5_10GeV
kaon_10_20GeV
neutron_5_10GeV
neutron_10_20GeV
...
```

## Cut Logic

The old evaluation code uses numeric cut IDs such as:

```text
2 3 4 5 8 9 10
```

The new workflow should use named cuts instead.

This is clearer and safer:

```yaml
old_default:
  - stable_beams
  - ip1_bunch_crossing
  - pre_evt_clock_cycle_100
  - avg_scifi_fiducial
  - no_veto_hit
  - consecutive_scifi_hits
  - scifi_continuity
```

Each cut definition should declare where it applies:

```text
applies_to: all
applies_to: real_data_only
applies_to: mc_only
```

This replaces the old hardcoded rule:

```text
For MC, drop cuts 2, 3, and 4.
```

The histogram builder should construct the final selection as:

```text
base cut expressions
&& extra cut expression
&& feature-specific cut expression
```

For a real-data row, `real_data_only` cuts are included.

For an MC row, `real_data_only` cuts are skipped.

For all rows, `all` cuts are included.

This makes the rule explicit in the config instead of hidden inside code.

## Normalization Logic

All plotted histograms should be normalized to the signal-region real-data
luminosity.

The reference luminosity is:

```text
L_ref = sum(lumi_per_partition)
```

over rows satisfying:

```text
region == signal_no_veto_has_us_fiducial_inside
particle_group == real_data
```

For each plotted group:

```text
L_group = sum(lumi_per_partition for rows in that group)
scale = L_ref / L_group
hist_scaled = hist_raw * scale
```

This applies to every group:

```text
real_data in signal region
real_data in sideband regions
MC in signal region
MC in sideband regions
muonDIS
kaon
neutron
muon
neutrino groups
```

This is especially important for sideband data. If only a fraction of the
whole sideband data set was processed, then `L_group` is the processed
sideband luminosity. Scaling by:

```text
L_ref / L_group
```

puts the sideband distribution onto the same luminosity scale as the signal
region data.

The first implementation only needs one normalization mode:

```text
signal_data_lumi
```

Other modes can be added later if needed:

```text
raw
unit_area
per_fb
```

but they are not required for the first version.

## Comparison Modes

Only two comparison modes are needed for the first implementation.

Real data should be treated as one particle type in both modes:

```text
particle_group == real_data
```

Real data is special only for cut handling and for defining the reference
luminosity. In comparison plots, it is just another compared group.

### Mode A: Compare Particles In The Same Region

Mode name:

```text
particles_in_region
```

Fixed:

```text
region
feature
base_cut_key
extra_cut_key
```

Compared groups:

```text
particle_group
```

Example question:

```text
In the signal region, how do real data, neutrinos, muonDIS, kaons, and neutrons
compare for density_sndsw_scifi?
```

Example selection:

```text
region = signal_no_veto_has_us_fiducial_inside
feature = density_sndsw_scifi
base_cut_key = old_default
extra_cut_key = none
```

Curves in the plot might be:

```text
real_data
CC_nue
CC_numu
NC_nue
NC_numu
muonDIS
muon
kaon_5_10GeV
neutron_5_10GeV
```

The plot builder should:

1. select metadata rows with the requested region;
2. group selected rows by `particle_group`;
3. sum partition histograms within each particle group;
4. compute each group's luminosity from metadata;
5. scale each histogram to `L_ref`;
6. draw one curve per particle group;
7. write a PDF and summary CSV.

### Mode B: Compare Regions For The Same Particle

Mode name:

```text
regions_for_particle
```

Fixed:

```text
particle_group
feature
base_cut_key
extra_cut_key
```

Compared groups:

```text
region
```

Example question:

```text
For real data, how does density_sndsw_scifi differ between the signal region
and the sideband regions?
```

Example selection:

```text
particle_group = real_data
feature = density_sndsw_scifi
base_cut_key = old_default
extra_cut_key = none
```

Curves in the plot might be:

```text
signal_no_veto_has_us_fiducial_inside
sideband_no_veto_no_us_fiducial_inside
sideband_no_veto_has_us_fiducial_outside
sideband_no_veto_no_us_fiducial_outside
sideband_has_veto_has_us_fiducial_inside
sideband_has_veto_no_us_fiducial_inside
sideband_has_veto_has_us_fiducial_outside
sideband_has_veto_no_us_fiducial_outside
```

The plot builder should:

1. select metadata rows with the requested particle group;
2. group selected rows by `region`;
3. sum partition histograms within each region;
4. compute each region's luminosity from metadata;
5. scale each histogram to `L_ref`;
6. draw one curve per region;
7. write a PDF and summary CSV.

## Plot Style

The first implementation should use overlay plots, not stacked plots.

Reason:

```text
The goal is comparison, not composition.
```

Each plot should draw:

```text
one line or marker style per compared group
legend with group name and scaled integral
axis labels from hist_features_v1.yaml
log-y according to feature config
```

Real data can use a black line or black markers. Other particle groups or
regions can use colored lines.

A ratio panel is optional for the first version. If added later, it should be
defined carefully because the denominator depends on the comparison mode.

## Stage 1: Grouped Histogram Jobs

The first stage should build reusable histograms grouped by region and
particle group.

One job should correspond to:

```text
one region
one particle_group
one feature
one base_cut_key
one extra_cut_key
```

Input:

```text
partition metadata CSV
hist_features_v1.yaml
evaluation_options_v1.yaml
all feature_partition_path files matching region + particle_group
```

Output:

```text
evaluation_region_partitions/hists/
  partition_v1_region_v1/
    {feature}/
      {base_cut_key}/
        {extra_cut_key}/
          {region}/
            {particle_group}.root
```

The output ROOT file should contain:

```text
hist
```

and metadata objects or sidecar fields for:

```text
partition_id
region
particle_group
particle_family
lumi_per_partition
n_events
feature
base_cut_key
extra_cut_key
selection
raw_integral
source_partition_ids
source_feature_partition_paths
```

The histogram script should:

1. read the partition metadata CSV;
2. select rows matching `region` and `particle_group`;
3. add all matching `feature_partition_path` files to one ROOT `TChain`;
4. load the feature histogram config;
5. load the selected base and extra cuts;
6. build the final ROOT selection expression;
7. build a ROOT `RDataFrame` from the chain;
8. draw or fill one combined histogram;
9. fold underflow/overflow if configured;
10. save the histogram and metadata.

## Stage 2: Comparison Plot Jobs

The second stage should combine grouped histograms into comparison plots.

Input:

```text
partition metadata CSV
hist_features_v1.yaml
evaluation_options_v1.yaml
all relevant grouped histogram ROOT files
```

Output:

```text
evaluation_region_partitions/plots/
  partition_v1_region_v1/
    {comparison_mode}/
      {comparison_key}/
        {feature}/
          {base_cut_key}/
            {extra_cut_key}/
              plot.pdf
              summary.csv
```

For `particles_in_region`:

```text
comparison_key = region
```

For `regions_for_particle`:

```text
comparison_key = particle_group
```

The plot script should:

1. read metadata;
2. calculate `L_ref`;
3. select rows for the requested comparison;
4. group rows by the comparison axis;
5. load and sum per-partition histograms per group;
6. calculate each group's luminosity;
7. scale each group to `L_ref`;
8. draw the overlay plot;
9. write the PDF;
10. write the summary CSV.

## Snakemake Integration

Add a new Snakemake rule file:

```text
snakemake/rules/region_evaluation.smk
```

The top-level `snakefile` can include it later:

```python
include: "rules/region_evaluation.smk"
```

The first implementation should define at least two rules.

### Rule: Build One Grouped Histogram

Conceptual rule:

```text
rule region_eval_group_hist
```

Wildcards:

```text
region
particle_group
feature
base_cut_key
extra_cut_key
```

Output:

```text
evaluation_region_partitions/hists/
  partition_v1_region_v1/{feature}/{base_cut_key}/{extra_cut_key}/{region}/{particle_group}.root
```

This gives Snakemake the desired granularity:

```text
one feature one job, per region + particle_group + cut option
```

### Rule: Build One Comparison Plot

Conceptual rule:

```text
rule region_eval_comparison_plot
```

Wildcards:

```text
comparison_mode
comparison_key
feature
base_cut_key
extra_cut_key
```

Output:

```text
evaluation_region_partitions/plots/
  partition_v1_region_v1/{comparison_mode}/{comparison_key}/{feature}/{base_cut_key}/{extra_cut_key}/plot.pdf
```

Allowed comparison modes:

```text
particles_in_region
regions_for_particle
```

The input function for this rule should use metadata and the comparison mode
to determine which partition histogram files are needed.

For example:

```text
particles_in_region + signal_no_veto_has_us_fiducial_inside
```

needs all partition histograms where:

```text
region == signal_no_veto_has_us_fiducial_inside
```

while:

```text
regions_for_particle + real_data
```

needs all partition histograms where:

```text
particle_group == real_data
```

## Summary CSV

Every plot should save a CSV summary next to the PDF.

Recommended columns:

```text
comparison_mode
comparison_key
feature
base_cut_key
extra_cut_key
group_name
region
particle_group
n_partitions
sum_n_events
lumi
reference_lumi
scale_factor
raw_integral
scaled_integral
selection
```

For mode A:

```text
group_name = particle_group
```

For mode B:

```text
group_name = region
```

This summary is important for debugging luminosity scaling and checking which
partitions actually contributed to each curve.

## Relationship To The Old Evaluation Code

The old evaluation code should be used as a reference for:

```text
histogram feature names
histogram bin ranges
axis labels
log-y defaults
ROOT plotting style
cut expression examples
luminosity scaling pattern
Snakemake job structure
```

But the new code should avoid copying the old partition parsing logic.

Old behavior:

```text
parse sample identity from filenames like hist_MC_muonDIS_Max10-1.root
```

New behavior:

```text
read sample identity from metadata columns:
  region
  particle_group
  particle_family
  particle_id
```

This is the core design change.

## First Implementation Scope

The first version should implement:

1. `hist_features_v1.yaml`;
2. `evaluation_options_v1.yaml`;
3. per-partition histogram builder;
4. comparison plot builder;
5. Snakemake rule for partition histograms;
6. Snakemake rule for comparison plots;
7. comparison mode `particles_in_region`;
8. comparison mode `regions_for_particle`;
9. normalization to signal-region real-data luminosity;
10. PDF and summary CSV outputs.

The first version does not need:

```text
stacked Data/MC plots
Data/MC ratio panels
signal-vs-sideband transfer plots
heatmaps
ML prediction overlays
unit-area normalization
per-fb normalization
```

Those can be added later after the metadata-driven histogram and comparison
workflow is stable.

## End-To-End Logic

The final workflow should look like:

```text
partition_v1_region_v1.csv
        |
        v
hist_features_v1.yaml + evaluation_options_v1.yaml
        |
        v
one histogram per partition + feature + cut option
        |
        v
group histograms by particle_group or region
        |
        v
scale every group to signal-region real-data luminosity
        |
        v
draw overlay comparison plot
        |
        v
write PDF + summary CSV
```

The essential rule is:

```text
Region partitioning defines the samples.
Evaluation only compares those samples.
```
