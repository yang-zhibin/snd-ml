# Run-8329 passing-muon detector study V8

V8 starts from the V7 `coverage_loose` detector requirements but publishes a
single working point named `passing_muon`. It does not modify any V7 product.

## Exclusive populations

The full RealDataDQ and MC baseline samples are divided into three populations:

1. `passing_muon_candidate`: `count_veto > 0` and the V7 loose detector
   requirements pass;
2. `muon_dis_candidate`: `count_veto > 0` and those requirements fail;
3. `signal_region`: `count_veto == 0`.

The three populations are mutually exclusive and must close to the full input
sample. The no-Veto signal region is therefore never included in the
Muon-DIS-candidate curve.

## Outputs

- `passing_muon_selection.pdf` compares data and pooled upward-muon MC for the
  passing-muon-candidate and Muon-DIS-candidate populations.
- `signal_region.pdf` compares data and pooled upward-muon MC for the no-Veto
  signal region only.
- `histograms/passing_muon_detector_study_v8.root` stores the three populations,
  their closure counts, and the catalog metadata used by the first PDF page.

Each PDF begins with the 2024 Run-8329 data event count and integrated
luminosity. It also lists every configured upward-muon MC catalog, source-file
count, generated event count, detector-activity baseline count, and equivalent
luminosity. The combined MC curves remain the unweighted event sum used by V7.

## Run

From `snakemake/`:

```bash
snakemake --profile .config/snakemake/htcondor --jobs 100 \
  passing_muon_run8329_detector_study_v8
```

The histogram and plot-only targets are:

```text
passing_muon_run8329_detector_study_v8_hist
passing_muon_run8329_detector_study_v8_plots
```
