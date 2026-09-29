# Run-8329 Data-Quality/Passing-Muon Comparison and Partition-v2

> **Current definition:** the denominator is the existing real-data feature
> sample selected by its external data-quality cutflow. PassingMuonReco events
> are intersected with that sample by `(source_file_index, runId, eventId)`.
> Earlier direct-inclusive design notes later in this historical plan are
> superseded by this definition.

## Goal

Build a versioned, exposure-matched evaluation of the upstream passing-muon
reconstruction using only run 8329:

```text
existing data-quality-selected run-8329 real-data features
    versus
PassingMuonReco features whose event keys occur in the real-data sample
```

The numerator is an exact subset of the denominator after event-key
intersection. No cutflow or feature conversion is rerun.

## Implementation status

The v2 code and configuration are implemented as an isolated workflow. The
existing passing-muon feature and partition-v1 targets retain their original
names, configuration, and default converter behavior.

Implemented components:

```text
paired source/chunk metadata construction and range validation
per-source PassingMuonReco intersection with existing real-data DQ features
per-source count, duplicate-key, wrong-run, and rejection audits
partition balancing by observed real-data DQ event counts
source-aligned real-DQ and passing-DQ partition construction
partition schema, count, luminosity, and exact event-containment validation
common-feature yield, shape, and binomial-efficiency histograms
passing-only reconstructed-track inspection
staged and final Snakemake targets
```

Primary commands:

```bash
snakemake -s snakemake/snakefile \
  passing_muon_run8329_comparison_metadata_v2 --cores 1

snakemake -s snakemake/snakefile \
  passing_muon_run8329_dq_intersection_v2 --cores 4

snakemake -s snakemake/snakefile \
  passing_muon_track_partitions_v2 --cores 4

snakemake -s snakemake/snakefile \
  passing_muon_run8329_comparison_v2 --cores 4
```

## Current comparison-plot output

The implemented comparison stage produces one multi-page PDF and one ROOT
histogram file per symmetric selection:

```text
evaluation_region_partitions/passing_muon_run8329_v2/
  comparison/{selection}.pdf

/eos/experiment/sndlhc/users/zhibin/snd-ml/
  evaluation_region_partitions/passing_muon_run8329_v2/
    histograms/{selection}.root
```

The configured selections are `none`, `scifi_gt_35`, `no_veto`, `has_veto`,
`no_us`, and `has_us`. Each PDF has 20 pages in configuration order: 15 1D
features followed by 5 2D Region-V2 veto-inspection features. A 1D page shows
the real-DQ and PassingMuonReco-after-DQ yields in the upper pad and their
exact-subset binomial ratio in the lower pad. A 2D page shows the two raw-yield
histograms side by side with a shared z range and has no ratio panel.

No per-feature efficiency CSV is produced. Counts, luminosity, selection
provenance, feature order, and exact-subset provenance are stored in the ROOT
file. The inclusive veto-time views follow the Region-V2 convention of placing
no-veto events at `-1 ns` and excluding veto-active events with invalid time.

## Verified selection and efficiency

The upstream SNDSW real-data writer applies:

```python
self.ExecuteEvent(self.eventTree)
if not self.MonteCarlo and self.fittedTracks.GetEntries() == 0:
    continue
self.fSink.Fill()
```

Therefore:

```text
PassingMuonReco == 1
    <=> at least one enabled tracking task produced a fitted track
```

The enabled tasks may contribute simple or Hough SciFi and downstream tracks.
For Hough tracks, pattern recognition requires the configured minimum hit-plane
conditions and only a converged Kalman fit is stored. There is no additional
final event-level veto, US, timing, fiducial, density, momentum, or chi2/NDF
cut in `TrackSelector.Execute`.

The authoritative run totals are:

```text
canonical run:                8329
sampling scale:                  1
source files:                  299
source/attempted events: 298,285,666
reconstructed chunks:         2,983
PassingMuonReco events:  82,145,120
shared luminosity:       0.5323492746290897 fb^-1
```

The raw, pre-DQ efficiency is:

```text
82,145,120 / 298,285,666 = 0.275391... ~= 27.54 percent
```

Scale 1 means that every source event was attempted. Only events with at least
one fitted track were written to the reconstructed output.

## Run-8329 event flow and efficiency accounting

The real-data and passing-muon populations below are two branches derived from
the same 298,285,666 raw events in run 8329; they are not two independent runs.
The real-data branch applies the existing combined external DQ requirement,
while the passing branch applies the upstream fitted-track requirement.

| Population | Events | Fraction of raw run | Removed from raw |
|---|---:|---:|---:|
| Raw run-8329 events | 298,285,666 | 100.000000% | 0 |
| `PassingMuonReco` | 82,145,120 | 27.539077% | 216,140,546 |
| Real data after combined external DQ | 11,115,982 | 3.726623% | 287,169,684 |
| `PassingMuonReco` intersect real-data DQ | 11,025,227 | 3.696197% | 287,260,439 |

The exact event-key comparison gives the complete 2-by-2 accounting:

| | DQ yes | DQ no | Total |
|---|---:|---:|---:|
| `PassingMuonReco` yes | 11,025,227 | 71,119,893 | 82,145,120 |
| `PassingMuonReco` no | 90,755 | 216,049,791 | 216,140,546 |
| Total | 11,115,982 | 287,169,684 | 298,285,666 |

Consequently, the efficiencies answer different conditional questions:

```text
inclusive passing efficiency
    = 82,145,120 / 298,285,666
    = 27.539077%

passing efficiency inside the real-data DQ sample
    = 11,025,227 / 11,115,982
    = 99.183563%

fraction of DQ events missing from the passing output
    = 90,755 / 11,115,982
    = 0.816437%

DQ acceptance inside the passing sample
    = 11,025,227 / 82,145,120
    = 13.421646%
```

The missing 90,755 events pass the external DQ requirement but do not occur in
the reconstructed passing-muon output. Under the verified SNDSW writer logic,
this means that none of the enabled tracking tasks supplied a saved fitted
track for those events. Passing the DQ hit requirements does not guarantee a
converged and accepted fitted track.

The current validation records the count after the combined DQ expression. It
does not record cumulative counts after each individual DQ flag, so a
flag-by-flag sequential cutflow cannot be reconstructed from these summary
counts alone.

### Counts after the configured plotting selections

Each row below starts from the same DQ denominator and its matched passing
subset. These selections are alternative symmetric views, not cumulative
steps. `Difference` is `real DQ - passing after DQ`, and efficiency is computed
within that row.

| Selection | Real-data DQ | Passing after DQ | Difference | Efficiency |
|---|---:|---:|---:|---:|
| `none` | 11,115,982 | 11,025,227 | 90,755 | 99.183563% |
| `scifi_gt_35` | 10,711,967 | 10,626,868 | 85,099 | 99.205571% |
| `no_veto` | 20,158 | 8,023 | 12,135 | 39.800575% |
| `has_veto` | 11,095,824 | 11,017,204 | 78,620 | 99.291445% |
| `no_us` | 20,687 | 5,534 | 15,153 | 26.751100% |
| `has_us` | 11,095,295 | 11,019,693 | 75,602 | 99.318612% |

Feature-specific validity expressions can reduce the entries in an individual
histogram further. In particular, `hitTime_veto_min_ns` has its own timing
availability requirement, so its histogram totals need not equal the event
counts in this table.

## Current analysis contract

1. Use only canonical run 8329 on both sides.
2. Reuse the existing real-data external-cutflow feature files as the DQ denominator.
3. Apply no sampling correction and do not apply `scale_factor` again.
4. Do not reconvert real data and do not rerun its cutflow.
5. Reuse the existing passing-muon track-feature chunks.
6. Apply the real-data DQ selection to passing muons only through exact event membership.
7. Preserve that the DQ-intersected passing sample is nested inside the real-DQ sample.
8. Apply optional data-quality or region selections symmetrically at evaluation.
9. Use track variables only for conditional studies inside the passing sample.
10. Never stack inclusive and passing samples as independent data yields.

## Why the existing real-data features are the denominator

Normal real-data feature production reads the external `cutFlowSummary`, which
contains the desired data-quality selection. The passing-muon production has
no equivalent cutflow. For each original source, v2 therefore uses the real
feature `(runId,eventId)` keys as an allow-list and filters all associated
passing chunks. `eventIndex` is not used across the two products because it
can be local to a reconstructed chunk.

## Versioned implementation files

Add the following without changing partition v1:

```text
snakemake/metadata/configs/passing_muon_run8329_comparison_v2.yaml
snakemake/metadata/configs/partitions/passing_muon_track_partition_v2.yaml
snakemake/metadata/configs/regions/passing_muon_track_comparison_region_v2.yaml
snakemake/metadata/configs/region_evaluation/passing_muon_run8329_hist_v2.yaml
snakemake/metadata/configs/region_evaluation/passing_muon_run8329_eval_v2.yaml
snakemake/metadata/make_passing_muon_run8329_comparison_metadata.py
snakemake/metadata/finalize_passing_muon_run8329_dq_metadata.py
convertData/filter_passing_muon_by_real_data_dq.py
convertData/validate_passing_muon_track_partitions_v2.py
evaluation_region_partitions/make_passing_muon_efficiency_hist.py
evaluation_region_partitions/plot_passing_muon_efficiency.py
snakemake/rules/passing_muon_track_partitions_v2.smk
```

Use versioned output roots:

```text
/eos/experiment/sndlhc/users/zhibin/snd-ml/
  passing_muon_run8329_direct_features_v2/
  passing_muon_track_partitions/
    passing_muon_track_partition_v2/
```

## Stage 1: build paired source metadata

Read:

```text
snakemake/metadata/updated/real_data_2024_metadata.csv
snakemake/metadata/updated/passing_muon_track_features_v1_metadata.csv
```

Select `run == 8329` and write:

```text
inclusive_run8329_source_metadata.csv
passing_muon_run8329_source_metadata.csv
passing_muon_run8329_source_pairs.csv
```

One source-pair row represents one of the 299 original source files:

```text
source_file_index
source_path
source_n_event
inclusive_feature_path
passing_chunk_paths
attempt_start/attempt_stop ranges
attempted_event_count
saved_passing_event_count
lumi_per_source
```

Validate:

```text
299 unique source files
2,983 unique reconstructed chunks
chunk ranges are ordered, gap-free, non-overlapping, and cover each source
sum(source_n_event) == sum(n_attempted_source_events) == 298,285,666
sum(n_saved_event) == 82,145,120
sum(lumi_per_source) == 0.5323492746290897 fb^-1
```

Use the dedicated passing-muon luminosity as the shared run value. Do not let
the two sides retain slightly different values from independently constructed
metadata.

## Stage 2: generalize direct feature conversion

`digi_2_features.py` currently couples direct iteration to the passing-muon
data type. Introduce an explicit option while preserving current behavior:

```text
--selection-mode external_cutflow
--selection-mode inclusive_direct
--selection-mode passing_muon_direct
```

Common direct behavior:

```text
iterate every entry in the requested input tree
do not open an external cutflow file
do not apply a physics-region cut
do not remove veto+US events
write runId, eventId, and local eventIndex
write identical detector features and timing summaries
record selection provenance
```

For `inclusive_direct`, read the 299 original run-8329 `rawConv` files without
track extraction. For `passing_muon_direct`, read the 2,983 reconstructed
chunks and enable `--extract-reco-tracks`.

Use honest direct cutflow schemas:

```text
inclusive: entry, InclusiveDirect
passing:   entry, PassingMuonReco
```

These marker branches are constant one in their already-defined input
populations; neither is a newly evaluated downstream cut.

## Stage 3: establish event identity

Current `eventIndex` values cannot join the samples:

```text
inclusive eventIndex -> local entry in the original source file
passing eventIndex   -> local entry in the reconstructed chunk
```

Use the source-aware event key:

```text
source_file_index + eventId
```

Validate its uniqueness within the inclusive and passing populations and
confirm every passing key occurs exactly once in the inclusive sample. If
`eventId` is not unique inside an original source file, recover and persist the
original source entry from reconstructed production and replace it with:

```text
source_file_index + original_source_entry
```

Do not claim a binomial subset efficiency until containment passes. If exact
matching cannot initially be established, report exposure-matched rate ratios
and label them as such.

## Stage 4: build paired partition v2

Use one no-cut region:

```yaml
regions:
  all_events:
    expression: "true"
```

Use two groups:

```text
inclusive_real_data_run8329
passing_muon_run8329
```

Build 20 paired partitions. Choose boundaries once from the ordered 299 source
files, balancing attempted events or luminosity. Assign all passing chunks to
the partition containing their `source_file_index`:

```text
part001 inclusive source files A-B
part001 passing chunks derived from source files A-B
...
part020 inclusive source files Y-Z
part020 passing chunks derived from source files Y-Z
```

Do not balance passing partitions independently by saved-event count. Paired
source boundaries enable closure and source-dependent stability tests.

Both sides must contain the common feature schema. Passing partitions may
additionally retain all `reco_track_*` branches; validation should compare the
common subset rather than requiring identical full schemas.

## Stage 5: validate partition closure

Require:

```text
20 inclusive and 20 passing partitions
each source_file_index occurs in exactly one paired partition
inclusive event total == 298,285,666
passing event total == 82,145,120
both groups have luminosity 0.5323492746290897 fb^-1
no source file or reconstructed chunk is missing or duplicated
all common branches have compatible types
all passing events match an inclusive key when matching is enabled
partition-level passing count never exceeds inclusive count
```

Write one validation row per partition plus an overall row. Include the raw
passing fraction in each row to expose source-dependent anomalies.

## Stage 6: common-feature evaluation

For every common feature `x`, produce three views.

### Exposure-normalized yields

Plot inclusive and passing counts or rates per fb^-1. Since luminosities are
identical, the raw count ratio already represents the exposure-matched ratio.
Do not unit-normalize the primary rate plot.

### Shape comparison

Unit-normalize separately to expose topology bias:

```text
(1 / N_inclusive) dN_inclusive/dx
(1 / N_passing)   dN_passing/dx
```

### Efficiency versus feature

For bin `i`:

```text
epsilon_i = N_passing,i / N_inclusive,i
```

Use binomial intervals after event containment is validated. Otherwise use a
ratio uncertainty and call the result an exposure-matched rate ratio.

Priority variables:

```text
count_veto, veto-plane counts, qdc_veto, earliest veto time
count_scifi and station counts
SciFi average and QDC-weighted positions
SciFi fiducial distance when available
SciFi density and topology variables
count_us, US plane multiplicity, qdc_us
count_ds and qdc_ds
```

Support symmetric optional selections: none, data quality, region-v1, and
later region-v2 working points. Apply the same expression to numerator and
denominator.

## Stage 7: passing-only track evaluation

Retain existing partition-v1 track plots and add:

```text
track convergence fraction
track npoints
track momentum components
track slopes and angles
track-type fractions
unknown track-type rate
events with multiple track types
```

These are conditional distributions
`p(track property | PassingMuonReco)`. They have no inclusive denominator
because non-passing events have no saved reconstructed-track objects.

## Stage 8: Snakemake targets

Provide staged targets:

```text
passing_muon_run8329_comparison_metadata_v2
passing_muon_run8329_inclusive_features_v2
passing_muon_track_partitions_v2
validate_passing_muon_track_partitions_v2
passing_muon_run8329_common_hists_v2
passing_muon_run8329_efficiency_v2
passing_muon_run8329_track_inspection_v2
passing_muon_run8329_comparison_v2
```

The final target requires metadata validation, paired partitions, common
feature plots, efficiency tables/plots, and passing-only track plots.

## Implementation order

1. Build and test the paired metadata/source-range validator.
2. Generalize `digi_2_features.py` selection mode with unit tests.
3. Produce one source file and its reconstructed chunks.
4. Prove event-identity mapping on that small sample.
5. Build one paired partition and compare matched common features.
6. Produce all direct inclusive features for run 8329.
7. Build and validate all 20 paired partitions.
8. Implement yield, shape, and efficiency histograms.
9. Add symmetric selection configurations.
10. Run the full comparison and freeze its configuration and report.

## Completion criteria

The comparison is complete only when:

```text
source exposure closes exactly
passing production ranges close exactly
both samples use identical common feature definitions
no asymmetric production selection remains
paired partition membership closes
event containment is measured and reported
raw efficiency reproduces approximately 27.54 percent
binned efficiencies have documented statistical treatment
all outputs carry version, selection, exposure, and source provenance
```
