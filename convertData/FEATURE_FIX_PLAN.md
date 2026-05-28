# digi_2_features Fix Plan

## Goal

Make `convertData/digi_2_features.py` produce feature ROOT files that are
consistent with the SND analysis pipeline calibration, SciFi hit filtering, and
output schema.

This plan is intentionally ordered so fixes can be made and checked one by one.

## Current Context

The Snakemake rule `process_features` runs:

```bash
/cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python \
  convertData/digi_2_features.py \
  -p <nueAnalysisFilter_path> \
  -d <digi_path> \
  -g <geo_path> \
  -o <tmp_output> \
  -t <data_type>
```

The script reads:

- input digi tree: `cbmsim`
- preselection/filter tree: `cutFlowSummary`
- geometry file through `SndlhcGeo.GeoInterface`

It writes:

- `sndData`: event-level feature tree
- `cutFlowSummary`: selected copy of the input cut-flow tree

## Findings To Fix

### 1. Missing per-event detector initialization

After `raw_tree.GetEntry(raw_entry)`, the script uses SND geometry/time helpers
without calling:

```python
snd_geo.modules["Scifi"].InitEvent(raw_tree.EventHeader)
snd_geo.modules["MuFilter"].InitEvent(raw_tree.EventHeader)
```

SND's own `sndEventFilter` initializes these modules each event before running
cuts. This is needed for run-dependent SciFi time calibration and position
alignment.

Risk if unfixed:

- real-data corrected hit times can be wrong
- real-data SciFi positions can use the wrong alignment tag
- `selectHits`, `GetCorrectedTime`, and `GetSiPMPosition` can be inconsistent
  with the SND pipeline

Acceptance check:

- `InitEvent` is called exactly once per processed event after loading the raw
  entry and before any feature extraction.

### 2. Mixed SciFi hit-filter conventions

The script currently mixes two SciFi hit selections:

- `density_sndsw_*` uses SND `selectHits` and `getSumDensity`
- `count_scifi*`, `qdc_scifi*`, `avg_scifi*`, and `density_mycode*` use a local
  Python `filter_SciFiHits`

The local filter is not equivalent to SND's filter. SND's nue pipeline uses
`sciFiHitFilter(0.5*6.25, 1.2*6.25, 52, 0, 26)`, which applies corrected times
for real data and a fixed time histogram range. The local filter uses
`hit.GetTime()/6.25`, dynamic histogram ranges, and no real-data time
correction.

Risk if unfixed:

- different branches in the same output event describe different SciFi hit sets
- downstream comparisons between SND density and custom density are biased by
  selection differences
- real-data features are especially vulnerable because corrected time is skipped

Planned fix:

- For normal nue-style samples, build `SciFi_hits` from SND `selectHits`.
- For muonDIS samples, keep the current special behavior of using all valid
  SciFi hits, because the muonDIS pipeline intentionally skips `sciFiHitFilter`.
- Refactor SciFi hit dictionary creation into a helper so selected SND hit
  objects and raw valid hit objects go through the same feature extraction.

Acceptance check:

- `process_hits` has one authoritative `selected_scifi_hits` collection.
- All SciFi feature branches are computed from `selected_scifi_hits`.
- The local `filter_SciFiHits` is either removed or clearly unused/debug-only.

### 3. Empty selected outputs have the wrong schema

When `n_match == 0`, the script writes `sndData` before branches are created.
This produces an empty tree with no feature branches.

Risk if unfixed:

- downstream ROOT/uproot code can fail on missing branches
- zero-selected files have a different schema from non-empty files

Planned fix:

- Move branch declaration and optional vector branch setup before the
  `n_match == 0` early return.
- For zero selected events, write the empty `sndData` tree with the full branch
  schema and an empty selected `cutFlowSummary`.

Acceptance check:

- Empty and non-empty output files expose the same `sndData` branch names.

### 4. `start_z` is declared but never filled

The output branch `start_z` is created but never assigned. It remains `-999` for
all events.

Risk if unfixed:

- the branch looks meaningful but contains only sentinel values
- downstream users may unknowingly use invalid data

Decision:

- `start_z` was obsolete, so it was removed from the branch list.

Acceptance check:

- `start_z` is removed from the output schema.

### 5. `vetoHits` branch is created but not filled

The script creates a `vetoHits` branch, but the code that would fill it is
commented out.

Risk if unfixed:

- the output schema advertises veto-hit detail objects that are empty
- downstream code may assume the branch contains veto hit information

Decision:

- `vetoHits` was empty and not used by feature calculation, so the branch
  creation was removed. Scalar veto features are still produced.

Acceptance check:

- The branch is removed.

## Fix Order

1. Add per-event `InitEvent`.
2. Refactor SciFi selection so feature branches use SND-selected hits.
3. Preserve full output schema for zero-selected files.
4. Fill or remove `start_z`.
5. Remove or implement `vetoHits`.

## Implementation Notes

### Suggested helper for SciFi hit dictionaries

```python
def build_scifi_hit_dict(hit, Scifi):
    detID = hit.GetDetectorID()
    station = hit.GetStation()
    hit_time = hit.GetTime()
    clock_cycle = hit_time / 6.25
    qdc = non_negative_float(hit.GetSignal(0))
    mat = hit.GetMat()
    sipm = hit.GetSiPM()
    channel = hit.GetSiPMChan()
    layer_channel = channel + sipm * 128 + mat * 4 * 128

    A, B = ROOT.TVector3(), ROOT.TVector3()
    Scifi.GetSiPMPosition(detID, A, B)

    return {
        "detType": 0,
        "station": station,
        "isVertical": hit.isVertical(),
        "layer_channel": layer_channel,
        "qdc": qdc,
        "x": A.x(),
        "y": A.y(),
        "z": A.z(),
        "hitTimeCY": clock_cycle,
    }
```

Use `hit.GetStation()` instead of recomputing the station from detector ID.

### Suggested `process_hits` logic

```python
if is_muon_dis:
    selected_scifi_hits = [hit for hit in event.Digi_ScifiHits if hit.isValid()]
else:
    selected_scifi_hits = selectHits(event, MC=("MC" in args.type))

dens, dens2, dver, dhor = getSumDensity(
    selected_scifi_hits,
    return_2ndhighest=True,
    return_hv=True,
)

SciFi_hits = [
    build_scifi_hit_dict(hit, Scifi)
    for hit in selected_scifi_hits
    if hit.isValid()
]
```

Then run:

- `hitWeightDensity(SciFi_hits, branch_vars)`
- `process_count_and_qdc(SciFi_hits, MuFilter_hits, branch_vars)`
- `process_avgPos(SciFi_hits, MuFilter_hits, branch_vars)`
- `fill_mycode_density(branch_vars)`

No second SciFi time filter should be applied to `SciFi_hits`.

## Validation Plan

For a small representative sample:

1. Run `process_features` before and after the patch on the same input.
2. Confirm output opens with ROOT/uproot.
3. Confirm `sndData` exists and has expected branches.
4. Confirm event counts match selected `cutFlowSummary` entries.
5. Compare selected branches:
   - `density_sndsw_scifi`
   - `density_mycode_scifi`
   - `count_scifi`
   - `qdc_scifi`
   - `avg_scifi_x`
   - `avg_scifi_y`
6. For real data, confirm no errors from `GetCorrectedTime` or
   `GetSiPMPosition` about invalid run numbers.
7. Test one zero-selected input if available, or force a selection that returns
   zero events, and confirm the output `sndData` tree still contains all
   branches.

## Status

- [x] Per-event `InitEvent`
- [x] Unified SciFi hit selection
- [x] Empty-output schema preserved
- [x] `start_z` resolved
- [x] `vetoHits` resolved

## Planned SciFi Topology Feature Extension

### Goal

Add new SciFi-only event-level topology features that describe:

1. where the SciFi activity starts, ends, and peaks;
2. how the shower develops longitudinally through SciFi stations 1 to 5;
3. how wide the hit pattern is in the transverse `x` and `y` views.

These features should be computed from the existing `SciFi_hits` dictionaries
inside `process_hits`, after SND hit selection has already been applied. This
keeps the new branches consistent with existing branches such as
`count_scifi*`, `qdc_scifi*`, `avg_scifi*`, `density_scifi*`, and
`density_sndsw*`.

### Detector/View Convention

Use the current script convention:

- `isVertical == True`: vertical SciFi plane, useful coordinate is `x`.
- `isVertical == False`: horizontal SciFi plane, useful coordinate is `y`.

The first implementation should be SciFi-only. Do not mix US/DS/MuFilter into
these shower-shape variables, because US has only horizontal planes in the
current feature logic and DS has a different station/view structure. US/DS
containment features can be added later as a separate extension.

### Implementation Structure

Add one function:

```python
def process_scifi_topology_features(SciFi_hits, branch_vars):
    ...
```

Call it from `process_hits` after the SciFi hit dictionaries are built and near
the existing SciFi feature functions:

```python
hitWeightDensity(SciFi_hits, branch_vars)
process_count_and_qdc(SciFi_hits, MuFilter_hits, branch_vars)
process_avgPos(SciFi_hits, MuFilter_hits, branch_vars)
process_scifi_topology_features(SciFi_hits, branch_vars)
fill_mycode_density(branch_vars)
```

Inside `process_scifi_topology_features`, loop over `SciFi_hits` once and build
per-station containers:

```text
station 1..5:
  count_total
  qdc_total
  count_x_view
  count_y_view
  qdc_x_view
  qdc_y_view
  x_positions from vertical hits
  y_positions from horizontal hits
```

This prevents duplicated loops and keeps all derived features based on the same
station/view bookkeeping.

### Helper Logic

Use small helpers to avoid `nan` and unclear sentinel behavior.

```python
def safe_fraction(num, den):
    return float(num) / float(den) if den else 0.0
```

Use this for normalized station fractions and peak fractions. A zero denominator
means there is no SciFi activity, so the fraction should be `0.0`.

QDC-like hit signals are used as non-negative detector-signal quantities. If an
individual hit has a negative QDC-like value, use `0.0` before summing or using
it as a topology weight. This keeps raw QDC sums and longitudinal QDC moments
consistent and avoids invalid weighted variance calculations.

```python
def weighted_mean_and_std(values, weights):
    total_weight = sum(weights)
    if total_weight <= 0:
        return -999.0, -999.0
    mean = sum(v * w for v, w in zip(values, weights)) / total_weight
    variance = sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / total_weight
    if variance < 0:
        variance = 0.0
    return mean, math.sqrt(variance)
```

Use this for longitudinal station moments. The values are station numbers
`[1, 2, 3, 4, 5]`; the weights are station hit counts or station QDC sums.

```python
def std_or_sentinel(values):
    if len(values) < 2:
        return -999.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
```

Use this for transverse hit-position spread. This is the standard deviation
around the mean position, not raw RMS from zero. If there are fewer than two
hits in a view, the spread is not physically meaningful, so use `-999.0`.

### Feature Group 1: SciFi Station Hit Profile

These features describe where the event activity appears in the five SciFi
stations. Existing branches already store raw station counts and QDC sums, so
these new branches summarize the profile instead of duplicating it.

#### `scifi_first_station`

Meaning:

- First SciFi station with at least one selected SciFi hit.
- Physically, this approximates where visible SciFi activity begins.

Implementation logic:

- Build a list of active stations where `count_total[station] > 0`.
- If the list is non-empty, fill `min(active_stations)`.
- If there are no SciFi hits, fill `-999`.

#### `scifi_last_station`

Meaning:

- Last SciFi station with at least one selected SciFi hit.
- Physically, this approximates how far downstream the visible activity extends.

Implementation logic:

- Use the same active-station list as `scifi_first_station`.
- If non-empty, fill `max(active_stations)`.
- If no SciFi hits, fill `-999`.

#### `scifi_n_active_stations`

Meaning:

- Number of SciFi stations with at least one selected SciFi hit.
- A compact event may activate few stations; a through-going or extended shower
  may activate many stations.

Implementation logic:

- Count stations where `count_total[station] > 0`.
- Fill `0` if no stations are active.

#### `scifi_n_active_stations_xy`

Meaning:

- Number of SciFi stations with both views active: at least one vertical hit and
  at least one horizontal hit.
- This is more geometrically complete than `scifi_n_active_stations`, because a
  station with only one view has incomplete transverse information.

Implementation logic:

- For each station, require:
  - `count_x_view[station] > 0`
  - `count_y_view[station] > 0`
- Count stations satisfying both.
- Fill `0` if none satisfy both.

#### `scifi_max_count_station`

Meaning:

- SciFi station with the largest number of selected hits.
- This identifies where the shower/activity peaks in hit multiplicity.

Implementation logic:

- If total SciFi count is zero, fill `-999`.
- Otherwise choose the station with the maximum `count_total`.
- If there is a tie, use the first station with the maximum. This is stable and
  matches Python's normal `max` behavior with ordered station scanning.

#### `scifi_max_qdc_station`

Meaning:

- SciFi station with the largest total QDC.
- This identifies where the charge deposition peaks.

Implementation logic:

- If total SciFi QDC is zero, fill `-999`.
- Otherwise choose the station with the maximum `qdc_total`.
- If there is a tie, use the first station with the maximum.

#### `scifi_count_peak_fraction`

Meaning:

- Fraction of all SciFi hits contained in the most populated station.
- High values indicate compact activity; lower values indicate activity spread
  across multiple stations.

Implementation logic:

```python
scifi_count_peak_fraction = max(count_total) / sum(count_total)
```

- Use `0.0` if total count is zero.

#### `scifi_qdc_peak_fraction`

Meaning:

- Fraction of all SciFi QDC contained in the station with the largest QDC.
- High values indicate charge concentrated in one station.

Implementation logic:

```python
scifi_qdc_peak_fraction = max(qdc_total) / sum(qdc_total)
```

- Use `0.0` if total QDC is zero.

### Feature Group 2: Longitudinal Shower Shape

These features use station number as a coarse longitudinal coordinate. They
describe whether activity is upstream, downstream, compact, or spread across
SciFi.

#### `scifi_count_frac1` ... `scifi_count_frac5`

Meaning:

- Fraction of selected SciFi hits in each station.
- These normalize the existing `count_scifi1` ... `count_scifi5` branches by
  total hit count, making the shower profile less dependent on event size.

Implementation logic:

```python
scifi_count_fracN = count_total[N] / sum(count_total)
```

- Use `0.0` for all fractions if total count is zero.

#### `scifi_qdc_frac1` ... `scifi_qdc_frac5`

Meaning:

- Fraction of total SciFi QDC in each station.
- These normalize the existing `qdc_scifi1` ... `qdc_scifi5` branches by total
  SciFi charge.

Implementation logic:

```python
scifi_qdc_fracN = qdc_total[N] / sum(qdc_total)
```

- Use `0.0` for all fractions if total QDC is zero.

#### `scifi_count_mean_station`

Meaning:

- Hit-count-weighted average station number.
- Lower values mean activity is more upstream; higher values mean activity is
  more downstream.

Implementation logic:

```python
stations = [1, 2, 3, 4, 5]
weights = [count_total[1], ..., count_total[5]]
mean, std = weighted_mean_and_std(stations, weights)
```

- Fill the returned mean.
- Use `-999` if total count is zero.

#### `scifi_count_std_station`

Meaning:

- Hit-count-weighted standard deviation of station number.
- Small values mean activity is concentrated in a narrow longitudinal region;
  large values mean activity is spread through many stations.

Implementation logic:

- Use the same count-weighted moment calculation as
  `scifi_count_mean_station`.
- Fill the returned standard deviation.
- Use `-999` if total count is zero.

#### `scifi_qdc_mean_station`

Meaning:

- QDC-weighted average station number.
- This tracks where the charge deposition is centered longitudinally.

Implementation logic:

```python
stations = [1, 2, 3, 4, 5]
weights = [qdc_total[1], ..., qdc_total[5]]
mean, std = weighted_mean_and_std(stations, weights)
```

- Fill the returned mean.
- Use `-999` if total QDC is zero.

#### `scifi_qdc_std_station`

Meaning:

- QDC-weighted standard deviation of station number.
- This measures the longitudinal width of the charge deposition.

Implementation logic:

- Use the same QDC-weighted moment calculation as
  `scifi_qdc_mean_station`.
- Fill the returned standard deviation.
- Use `-999` if total QDC is zero.

### Feature Group 3: Hit-Position Spread

These features measure transverse width of the SciFi hit pattern. They are
standard deviations around the mean position, even though the branch names use
`std`. This is intentionally not raw RMS from the detector coordinate origin.

#### `scifi_std_x`

Meaning:

- Global spread of vertical-view SciFi hits in `x`, using all stations.
- Track-like events tend to be narrow; shower-like events tend to be wider.

Implementation logic:

- Collect `x` values from all SciFi hits with `isVertical == True`.
- Fill `std_or_sentinel(all_x_values)`.
- Use `-999` if fewer than two vertical-view hits exist.

#### `scifi_std_y`

Meaning:

- Global spread of horizontal-view SciFi hits in `y`, using all stations.

Implementation logic:

- Collect `y` values from all SciFi hits with `isVertical == False`.
- Fill `std_or_sentinel(all_y_values)`.
- Use `-999` if fewer than two horizontal-view hits exist.

#### `scifi1_std_x` ... `scifi5_std_x`

Meaning:

- Per-station spread of vertical-view SciFi hits in `x`.
- These describe how broad the event is in each station's x-measuring view.

Implementation logic:

- For station `N`, collect `x` values from hits satisfying:
  - `station == N`
  - `isVertical == True`
- Fill `std_or_sentinel(station_x_values[N])`.
- Use `-999` if fewer than two hits exist in that station/view.

#### `scifi1_std_y` ... `scifi5_std_y`

Meaning:

- Per-station spread of horizontal-view SciFi hits in `y`.
- These describe how broad the event is in each station's y-measuring view.

Implementation logic:

- For station `N`, collect `y` values from hits satisfying:
  - `station == N`
  - `isVertical == False`
- Fill `std_or_sentinel(station_y_values[N])`.
- Use `-999` if fewer than two hits exist in that station/view.

### Branches To Add

Add these scalar branches to the `branches` list.

Integer branches:

```text
scifi_first_station
scifi_last_station
scifi_n_active_stations
scifi_n_active_stations_xy
scifi_max_count_station
scifi_max_qdc_station
```

Double branches:

```text
scifi_count_peak_fraction
scifi_qdc_peak_fraction

scifi_count_frac1
scifi_count_frac2
scifi_count_frac3
scifi_count_frac4
scifi_count_frac5

scifi_qdc_frac1
scifi_qdc_frac2
scifi_qdc_frac3
scifi_qdc_frac4
scifi_qdc_frac5

scifi_count_mean_station
scifi_count_std_station
scifi_qdc_mean_station
scifi_qdc_std_station

scifi_std_x
scifi_std_y
scifi1_std_x
scifi1_std_y
scifi2_std_x
scifi2_std_y
scifi3_std_x
scifi3_std_y
scifi4_std_x
scifi4_std_y
scifi5_std_x
scifi5_std_y
```

### Sentinel And Default Rules

- Station-index branches use `-999` when undefined.
- Count branches use `0` when no stations are active.
- Fraction branches use `0.0` when the denominator is zero.
- Mean/std station branches use `-999.0` when the total weight is zero.
- Position std branches use `-999.0` when fewer than two hits exist in the
  relevant view.

These defaults avoid `nan` values and make undefined physical quantities
explicit.

### Validation For This Extension

After implementation:

1. Compile-check `digi_2_features.py`.
2. Run on a small MC sample and one real-data sample if available.
3. Confirm all new branches exist in `sndData`.
4. Confirm no branch contains `nan` or `inf`.
5. Confirm:
   - `sum(scifi_count_frac1..5)` is `1.0` when `count_scifi > 0`.
   - `sum(scifi_qdc_frac1..5)` is `1.0` when `qdc_scifi > 0`.
   - `scifi_first_station <= scifi_last_station` when both are defined.
   - `scifi_n_active_stations_xy <= scifi_n_active_stations`.
6. Spot-check a few events against existing `count_scifi*`, `qdc_scifi*`, and
   `avg_scifi*` branches.

### Extension Status

- [x] Add helper functions for fractions and std/moments.
- [x] Add `process_scifi_topology_features`.
- [x] Add new branch declarations.
- [x] Call the new function from `process_hits`.
- [x] Compile-check the script.
- [x] Validate on a small representative ROOT sample.
