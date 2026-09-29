# Run 8329 Muon-DIS transform comparison V1

## Purpose

This study compares detector-feature shapes for three Run-8329 populations:

1. original Muon-DIS candidates that have a successful transformed partner;
2. those same events after detector-ID transformation;
3. the complete original no-Veto signal region.

All population assignments are frozen from the original RealDataDQ feature row:

```text
passing-muon candidate = count_veto > 0 and loose selection passes
Muon-DIS candidate     = count_veto > 0 and loose selection fails
signal region          = count_veto == 0
```

The three populations are exclusive and close to the complete RealDataDQ input.

## Methods and matching

Two PDFs are produced independently:

- `muon_dis_reco_scifi_transform.pdf` uses the reconstructed SciFi-track transform;
- `muon_dis_simple_transform.pdf` uses the five-station activity-line transform.

Within a PDF, original and transformed Muon-DIS histograms use the identical
set of successful `(runId,eventId)` matches. Feature validity must pass in both
rows. The signal-region histogram is independent and uses all original no-Veto
events for which that feature is valid.

Each distribution is normalized to unit visible integral. One-dimensional
pages show transformed/original and signal-region/original ratios. Two-dimensional
pages use three panels with common axes and z scale.

## Reused production

The V1 histogram layer reuses the already validated manifests and transformed
feature shards below the historical V8 production directory. It does not rerun
the detector-ID transformations or feature extraction. New histogram products
are written below:

```text
/eos/experiment/sndlhc/users/zhibin/snd-ml/evaluation_region_partitions/
passing_muon_run8329_transform_comparison_v1/histograms/
```

Plots are written below:

```text
evaluation_region_partitions/passing_muon_run8329_transform_comparison_v1/
```

## Run

From `snakemake/`:

```bash
snakemake --profile .config/snakemake/htcondor --jobs 100 \
  passing_muon_run8329_transform_comparison_v1
```

The histogram-only and plot-only targets are:

```text
passing_muon_run8329_transform_comparison_v1_hist
passing_muon_run8329_transform_comparison_v1_plots
```
