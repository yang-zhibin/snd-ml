# Region V2 Veto Inspection Plotter

## Purpose

This workflow inspects the veto axis before any `region_v2` boundary is
defined. It reads the six validated consolidated samples produced by
`region_v2_inspection_inputs` and studies:

- veto hit count;
- earliest veto-hit time;
- earliest veto-hit time versus veto hit count.
- earliest veto-hit time versus total SciFi hit count;
- earliest veto-hit time versus SNDSW SciFi density;
- veto hit count versus total SciFi hit count;
- veto hit count versus SNDSW SciFi density.

It does not apply a Region V1 or Region V2 partition.

Two configurations are available:

- `veto_inspection_v1.yaml` includes every event and displays no-veto events
  at the plotting-only value `-1 ns`;
- `veto_inspection_v2.yaml` applies the common base selection
  `count_veto > 0`, so no-veto events are absent from every histogram.

V1 and V2 write to separate output directories. The V2 summary retains both
the source-event count and selected-event count, making the exclusion
auditable without putting the rejected events into a histogram.

## Input

The authoritative input list is:

```text
snakemake/metadata/region_v2_inspection/
  region_v2_inspection_v1_manifest.csv
```

The configured samples are inclusive real data, passing-muon data, and four
EventBuilder neutrino categories: CC nue, NC nue, CC numu, and NC numu.

## No-veto and invalid-time semantics

The physical timing branch is:

```text
hitTime_veto_min_ns
```

For plotting only, an event with `count_veto == 0` is assigned:

```text
veto_earliest_time_plot_ns = -1 ns
```

This is a categorical display value labelled `no veto hit`; it is not a
measured time. A veto-active event with invalid timing is not assigned to this
bin. The workflow records three exclusive categories:

```text
no veto hit
veto hit with valid timing
veto hit with invalid timing
```

The builder fails if a valid physical time falls into the configured `-1 ns`
display bin.

## Architecture

The first stage runs once per dataset:

```text
make_veto_inspection_hist.py
```

It books all raw and weighted TH1/TH2 objects and counters together and calls
`ROOT.RDF.RunGraphs`, so each large consolidated file is scanned once.
Histogram statistical errors use `Sumw2`. Overflow folding is configured in
YAML and preserves variances.

The second stage runs:

```text
plot_veto_inspection.py
```

It reads only the six compact histogram files. Changing labels or styles does
not require rescanning the consolidated samples.

## Plot layouts

Every 1D plot overlays all six samples on one canvas. Each bin is drawn as a
marker with its statistical error bar. Both independent unit-area shapes and
luminosity-weighted yields are produced.

Every 2D plot uses a `1 x 3` canvas:

```text
real data | passing-muon data | combined neutrino MC
```

The neutrino panel is the luminosity-weighted sum of CC nue, NC nue, CC numu,
and NC numu. For the unit-area page, the four components are combined with
their physical weights before the combined histogram is normalized.

The SciFi observables are the total hit count `count_scifi` and the official
SNDSW density proxy `density_sndsw_scifi`. Correlations containing veto time
include the no-veto category at the plotting-only value `-1 ns` and exclude
veto-active events with invalid timing. Correlations containing veto count but
not veto time include every event.

Passing-muon data is always an independent enriched reference. It is never
added to inclusive real data.

## Configuration and code

```text
snakemake/metadata/configs/region_v2_inspection/veto_inspection_v1.yaml
evaluation_region_partitions/make_veto_inspection_hist.py
evaluation_region_partitions/plot_veto_inspection.py
snakemake/rules/region_v2_inspection_plots.smk
evaluation_region_partitions/test_veto_inspection.py
```

## Run

Dry run:

```bash
snakemake \
  -s snakemake/snakefile \
  region_v2_veto_inspection \
  --cores 4 \
  -n -p
```

Production:

```bash
snakemake \
  -s snakemake/snakefile \
  region_v2_veto_inspection \
  --cores 4 \
  --rerun-incomplete \
  -p
```

Veto-active-only V2:

```bash
snakemake \
  -s snakemake/snakefile \
  region_v2_veto_inspection \
  --cores 4 \
  --rerun-incomplete \
  --config \
    region_v2_veto_config=snakemake/metadata/configs/region_v2_inspection/veto_inspection_v2.yaml \
  -p
```

Synthetic integration test:

```bash
source snakemake/envs/env_lcg.sh
python evaluation_region_partitions/test_veto_inspection.py
```

## Outputs

Outputs are written below:

```text
evaluation_region_partitions/region_v2_inspection/veto_inspection_v1/
```

The workflow produces:

- four 1D studies in unit-area and weighted modes;
- one 2D study in unit-area and weighted modes;
- one single-page PDF per plot (PNG output is disabled);
- six compact per-dataset ROOT files;
- six per-dataset summary rows;
- one combined summary CSV;
- one combined ROOT file containing the displayed 2D histograms;
- a completion marker.

The summary reports total, no-veto, valid-timed-veto, and invalid-timed-veto
counts, weights, fractions, valid-time extrema, and status for every dataset.
