# Region V1 DS--SciFi passing-muon rejection plan

Status: frozen inclusive working point; implementation and production complete  
Recorded: 2026-10-02  
Analysis: Region V1 ABCD background estimation

## Objective

Study a modified passing-muon-like rejection that uses only SciFi and
Downstream MuFilter (DS) information. Veto and Upstream MuFilter (US)
quantities remain reserved for the Region V1 ABCD axes.

This is a new selection. It must not replace or silently modify the canonical
Run-8329 V7 `coverage_loose` working point. All new code, configuration, and
outputs must use a distinct versioned name such as
`ds_scifi_muon_rejection_v1`.

## Current data format

The existing `feature.root:sndData` trees already contain every branch needed
for this study:

- `density_sndsw_scifi`;
- `scifi_n_active_stations_xy`;
- `count_scifi1` through `count_scifi5` and `count_scifi`;
- `count_ds1` through `count_ds4` and `count_ds`.

`count_ds1` through `count_ds4` are total valid-hit counts for the four DS
stations. The current scalar feature format does not provide separate DS X-
and Y-view count branches. Therefore, "all DS planes fired, regardless of X
or Y" is implemented as activity in every DS station, not as proof that both
views fired in every station.

`scifi_n_active_stations_xy` is the number of SciFi stations with both X-view
and Y-view activity. Since SciFi has five stations, requiring at least five is
equivalent to requiring exactly five.

## Frozen Region V1 boundary

Retain the existing common preselection

```cpp
count_scifi > 35
```

and leave all Veto and US expressions unchanged. The new tag is evaluated as
an additional common selection and must not be embedded differently in
individual ABCD regions.

## Proposed SciFi component

Define

```cpp
S =
    scifi_n_active_stations_xy == 5
    && count_scifi1 < 20
    && count_scifi2 < 20
    && count_scifi3 < 20
    && count_scifi4 < 20
    && count_scifi5 < 20;
```

This retains the SciFi coverage and per-station multiplicity requirements from
the V7 loose selection. There is no additional total-SciFi multiplicity bound.

## Proposed DS component

The recommended definition, analogous to the V7 US condition, is

```cpp
D =
    count_ds1 > 0
    && count_ds2 > 0
    && count_ds3 > 0
    && count_ds4 > 0
    && count_ds1 <= 2
    && count_ds2 <= 2
    && count_ds3 <= 2
    && count_ds4 <= 2
    && count_ds <= 5;
```

The total threshold is five because DS has four stations and the selection
allows one hit beyond the minimum one-per-station topology. It accepts, for
example, `(1,1,1,1)` and `(2,1,1,1)`.

### Frozen station threshold

The phrase "each plane has less than two hits" literally gives
`count_dsN < 2`. Together with `count_dsN > 0`, this forces exactly one hit in
every station, so `count_ds <= 5` becomes redundant. To match the format and
intent of the V7 US loose cut, this plan recommends `count_dsN <= 2`.

The implementation uses the recommended inclusive definition: every
`count_dsN <= 2`, with `count_ds <= 5`. The user's instruction to implement
the saved plan is recorded as acceptance of its recommended working point.
The strict alternative (`count_dsN < 2`, implying `count_ds == 4`) is not
scanned or used.

## Tag and retained analysis population

Define the modified passing-muon-like tag

```cpp
M = S && D;
```

and reject tagged events with

```cpp
analysis_keep = !(S && D);
```

The complete common retained selection is therefore

```cpp
count_scifi > 35 && !(S && D)
```

Neither `S`, `D`, nor `M` may contain Veto, US, reconstructed-track, truth, or
signal-region-observation conditions.

## Efficiency definitions

For each Region V1 region `R`, calculate raw nested event counts

```text
N_before[R] = events in the existing region;
N_tag[R]    = events in the existing region that satisfy M;
N_keep[R]   = events in the existing region that satisfy !M.
```

Require exact closure

```text
N_before[R] = N_tag[R] + N_keep[R].
```

Report

```text
removed_fraction[R] = N_tag[R]  / N_before[R];
keep_efficiency[R]  = N_keep[R] / N_before[R]
                    = 1 - removed_fraction[R].
```

Because tag and retained counts are nested subsets of the same population,
use a binomial interval, preferably a 68.27% Clopper--Pearson or Wilson
interval. Do not treat the numerator and denominator as independent Poisson
counts.

The primary table must contain the signal region, SB1 through SB7, their
effective exposure, all three raw counts, both fractions, and interval bounds.

## Definition of total-data efficiency

The current stored Region V1 products mix exposures: no-Veto regions use
approximately `19.78 fb^-1`, while Veto-active regions use approximately
`1.96 fb^-1`. A raw sum over the eight stored region products is therefore a
sample-catalog efficiency, not a physical full-data efficiency.

Produce both quantities with explicit labels:

1. `sampled_catalog_efficiency`, calculated from the stored events; and
2. `exposure_corrected_efficiency`, calculated from regional rates,

```text
sum_R(N_tag[R] / L[R]) / sum_R(N_before[R] / L[R]).
```

The preferred full-2024 result is a separate evaluation on the full common-
exposure source data before the Veto-region ten-percent partitioning. Do not
overwrite or rebuild the frozen Region V1 partition products to obtain it.

## MC acceptance requirements

In addition to data efficiencies, calculate the retained acceptance for:

- target `CC_nue` signal MC;
- `CC_numu` background MC;
- `NC_nue` background MC;
- `NC_numu` background MC.

Record raw selected entries and luminosity-normalized yields. For weighted
quantities use `sum(w)` for the yield and `sum(w^2)` for the MC statistical
variance. The target-signal retention is mandatory because the modified tag
can reject genuine neutrino interactions with a long outgoing muon.

## Density histogram products

For each region, materialize three `density_sndsw_scifi` histograms:

- before rejection;
- tagged by `M`;
- retained by `!M`.

Use the original comparison binning for the first diagnostic:

```text
80 bins over [100, 40000].
```

Preserve and record the underflow and overflow policy. The requested report
should contain:

1. an all-region retained-density overlay;
2. one before/after comparison for each region;
3. a lower-panel density-dependent retention ratio
   `N_after[i] / N_before[i]`;
4. the inclusive efficiency table and exact selection definition.

For an all-region rate overlay, scale each data histogram to the common
reference exposure

```text
N_scaled[R,i] = N_keep[R,i] * 19.78 / L[R].
```

This scaling is for visualization only. Efficiencies and ABCD calculations
must use raw event counts or explicitly modelled rates, not plot-normalized
contents.

## Implementation structure

Keep the current baseline immutable. Add separate task-specific files, for
example:

```text
snakemake/metadata/configs/region_evaluation/
  abcd_ds_scifi_muon_rejection_v1.yaml

evaluation_region_partitions/
  evaluate_abcd_ds_scifi_muon_rejection.py
  test_abcd_ds_scifi_muon_rejection.py
```

Use a dedicated output directory:

```text
evaluation_region_partitions/abcd/
  partition_v1_region_v1/ds_scifi_muon_rejection_v1/
```

and a named aggregate Snakemake target such as

```text
region_v1_abcd_ds_scifi_muon_rejection_v1
```

The worker should use one `ROOT.RDataFrame` graph per input population, book
before/tag/keep counts and density histograms together, and execute them with
`ROOT.RDF.RunGraphs`. Persist the fully expanded C++ expression, version,
input catalogue, exposures, binning, flow policy, and count-closure flags in
the ROOT and tabular metadata. Write outputs atomically.

The existing grouped-histogram code already demonstrates named extra
expressions, `RDataFrame.Filter`, `selected_entries`, luminosity metadata, and
ROOT output handling. Reuse those conventions, but do not add this study to a
broad default target that would rebuild unrelated plots.

## Validation gates

Before interpreting a post-rejection ABCD result, require all of the
following:

1. every required feature branch exists in every input;
2. no negative detector counts enter the selection;
3. `N_before = N_tag + N_keep` in every data and MC population;
4. histogram integrals close to the corresponding selected counts under the
   stated flow policy;
5. full-source and partition-catalog provenance is recorded;
6. per-region and total efficiencies are reported with correct denominators;
7. target-signal and background-MC retention are quantified;
8. sparse-bin sufficiency is re-evaluated;
9. sideband validation and density-dependent ABCD closure are rerun;
10. no cut threshold is retuned after inspecting nominal signal-region data.

The absence of Veto and US variables from the selection prevents their direct
reuse in the cut, but it does not prove statistical independence. DS and SciFi
topology can still correlate with Veto, US, fiducial containment, and density.
The modified selection is accepted only if the new closure and signal-
acceptance results support it.

## Authoritative references

- `snakemake/metadata/configs/region_evaluation/passing_muon_binary_region_study_v7.yaml`
  contains the canonical V7 loose working point.
- `convertData/passing_muon_selection.py` constructs its frozen expression.
- `convertData/digi_2_features.py` defines the detector-count branches.
- `convertData/FEATURE_DESCRIPTION.md` documents their meanings.
- `snakemake/metadata/configs/regions/region_v1.yaml` contains the current
  ABCD regions and the `count_scifi > 35` common preselection.
- `evaluation_region_partitions/make_group_hist.py` demonstrates the current
  region-level selection and histogram conventions.

## Exact next action

Review the completed efficiency and density report with the user and decide
whether this extremely low-rejection working point should be accepted or
redefined. Do not start post-selection ABCD closure or change the frozen cut
without that decision.
