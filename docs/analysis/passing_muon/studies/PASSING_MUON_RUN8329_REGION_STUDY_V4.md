# Run-8329 Passing-Muon Region Study v4

## Purpose

This workflow studies detector-feature distributions in exclusive regions of
veto hit count, earliest veto-hit time, and US hit count. Every expression is
applied identically to the validated `PassingMuonReco` population and its exact
`NoPassingMuonReco = RealDataDQ - PassingMuonReco` complement.

The workflow is independent of the v3 track-type working points. It reuses the
validated input partitions but does not change or overwrite v3 outputs.

## Region definitions

The veto-count categories are:

```text
veto_n0:    count_veto == 0
veto_n1_2:  1 <= count_veto < 3
veto_n3:    count_veto == 3
veto_nge4:  count_veto >= 4
```

The main veto-time categories are:

```text
veto_time_none:       count_veto == 0
veto_time_0_2:        positive veto count and 0 <= earliest time <= 2 ns
veto_time_gt2_lt7:    positive veto count and 2 < earliest time < 7 ns
veto_time_ge7:        positive veto count and earliest time >= 7 ns
```

Exactly zero is included in the first valid-time interval. Otherwise a valid
zero-time hit would be silently unassigned. `veto_time_none` is defined from
`count_veto == 0`, never from the `-999` storage sentinel.

Two positive-hit exceptional categories are counted and stored separately:

```text
veto_time_invalid:   non-finite or <= -900 ns
veto_time_negative:  finite, > -900 ns, and < 0 ns
```

These audit events do not enter the main joint regions. This prevents missing
or unusual timing information from being merged with genuine no-veto events.

The US-count categories are:

```text
us_n0:    count_us == 0
us_n1_4:  1 <= count_us <= 4
us_n5_7:  5 <= count_us <= 7
us_nge8:  count_us >= 8
```

## Marginal and joint comparisons

The three marginal PDFs compare the four RealDataDQ category shapes for each
axis:

```text
veto_count_regions.pdf
veto_time_regions.pdf
us_count_regions.pdf
```

RealDataDQ histograms are formed exactly as the sum of passing and missing
histograms. Each marginal PDF has one count/closure summary and 27 common
feature pages. One-dimensional pages contain four unit-normalized overlays and
a shape-ratio pad. Two-dimensional pages use a 2-by-2 layout and a shared
normalized z scale.

The common 2D features include two SciFi spatial maps with identical axes and
binning:

```text
scifi_x_vs_y               = avg_scifi_x versus avg_scifi_y
qdc_weighted_scifi_x_vs_y  = qdcAvg_scifi_x versus qdcAvg_scifi_y
```

The first map gives every accepted SciFi hit equal weight. The second weights
each hit position by its non-negative QDC. An event enters a map only when both
of that map's coordinates are finite and are not the `-999` invalid-value
sentinel. Thus the weighted map also requires positive QDC weight in both the
vertical (x) and horizontal (y) views. Detector-coordinate underflow and
overflow bins are deliberately not folded into the visible edge bins.

The naive Cartesian product contains 64 cells, but only 40 are physically
compatible:

```text
(1 no-veto count/time state + 3 positive count x 3 valid time states)
  x 4 US states
= 40 regions
```

All 40 region expressions and passing/missing counts are stored in ROOT. The
five-page `joint_region_summary.pdf` contains an introduction and one set of
count/fraction heatmaps per US category. Incompatible count/time cells are
left empty.

Full feature histograms are booked only for the configurable
`selected_joint_regions` allowlist. Each selected-region PDF compares
`PassingMuonReco` with `NoPassingMuonReco`: 1D normalized overlays plus a
missing/passing shape ratio, and side-by-side normalized 2D maps.

## Required closure

For every marginal and joint region `R`:

```text
N_RealDataDQ(R) = N_PassingMuonReco(R) + N_NoPassingMuonReco(R)
```

The veto-count and US axes must each assign every event exactly once. The four
main veto-time categories plus the invalid and negative audit categories must
also assign every event exactly once. The 40 joint regions must close to the
main veto-time-eligible population for both passing and missing samples.

Any non-exclusive, unassigned, total-count, or joint-region closure failure is
fatal. Expressions, counts, feature-selected counts, category order, and
closure flags are stored in the ROOT metadata. No efficiency CSV is produced.

## Files and outputs

Implementation files:

```text
snakemake/metadata/configs/region_evaluation/passing_muon_region_study_v4.yaml
evaluation_region_partitions/make_passing_muon_region_hist_v4.py
evaluation_region_partitions/plot_passing_muon_region_study_v4.py
evaluation_region_partitions/test_passing_muon_region_study_v4.py
snakemake/rules/passing_muon_region_study_v4.smk
```

The production writes one histogram ROOT file and nine PDFs with the initial
five-region allowlist:

```text
/eos/experiment/sndlhc/users/zhibin/snd-ml/
  evaluation_region_partitions/passing_muon_run8329_region_study_v4/
    histograms/regions.root

evaluation_region_partitions/passing_muon_run8329_region_study_v4/
  veto_count_regions.pdf
  veto_time_regions.pdf
  us_count_regions.pdf
  joint_region_summary.pdf
  region__<selected-region>.pdf
```

## Commands

Run the synthetic boundary and closure test:

```bash
cd /afs/cern.ch/user/z/zhibin/work/snd-ml
source snakemake/envs/env_lcg.sh
python evaluation_region_partitions/test_passing_muon_region_study_v4.py
```

Inspect the production DAG:

```bash
snakemake -s snakemake/snakefile \
  passing_muon_run8329_region_study_v4 \
  --cores 4 --rerun-triggers mtime -n
```

Run the production:

```bash
snakemake -s snakemake/snakefile \
  passing_muon_run8329_region_study_v4 \
  --cores 4 --rerun-triggers mtime
```
