# MuonDIS SciFi Timing Filter Investigation

## Context

The Snakemake `process_nueAnalysis` rule runs `sndEventFilter` with the
pipeline:

```text
$SNDSW_ROOT/analysis/analyses/snd_analysis_2024_0mu/pipelines/nueFilterMoriondOrder_withEventLevelOutput.h
```

For the MuonDIS sample:

```text
/eos/experiment/sndlhc/users/cvilela/muonDIS_prod_20240826/muonDIS_2001/sndLHC.muonDIS-TGeant4_digCPP.root
```

the output file was:

```text
/eos/experiment/sndlhc/users/zhibin/MC_muonDIS/cvilela/muonDIS_2001/nueAnalysisFilter_MC_muonDIS_cvilela_muonDIS_2001_0.root
```

The event filter ran to completion, but no events passed the `AvgSFChan` cut.

## Observed Cut Behavior

From `cutFlowSummary`:

```text
entries          : 477400
NoVetoHits == 1  : 468078 / 477400
AvgSFChan == 1   :      0 / 477400
SciFiMinHits == 1:      0 / 477400
accepted == 1    :      0 / 477400
```

From cumulative `cutFlow`:

```text
all events       : 477400
after NoVetoHits : 468078
after AvgSFChan  :      0
```

The `AvgSFChan` histograms showed:

```text
cutFlow_-1_AvgSFChan_0: entries 477400, underflow 477400, in-range 0
cutFlow_-1_AvgSFChan_1: entries 477400, underflow 477400, in-range 0
cutFlow_0_AvgSFChan_0 : entries 468078, underflow 468078, in-range 0
cutFlow_0_AvgSFChan_1 : entries 468078, underflow 468078, in-range 0
```

This means the `AvgSFChan` plot variables are `-1` for every event. In
`sndAvgSciFiFiducialCut.cxx`, this happens when the SciFi hit collection seen
by the cut has no valid vertical or horizontal hits.

## Important Pipeline Detail

The first printed pipeline line is blank in the log:

```text
[INFO]
[INFO] No hits in veto
[INFO] Avg SciFi Ver channel ...
```

That blank line is `sciFiHitFilter`, which has no `processName`. The actual
early pipeline order is:

```text
sciFiHitFilter
NoVetoHits
AvgSFChan
...
```

Therefore `AvgSFChan` does not use the raw `Digi_ScifiHits` branch directly.
It uses the filtered global collection created by `sciFiHitFilter`.

## Diagnosis

The MuonDIS sample has a different SciFi hit time convention from the neutrino
sample used to tune the nue pipeline.

For the MuonDIS input, inspecting the first 1000 events showed:

```text
raw_scifi_nonempty_events : 7 / 1000
valid_hit_events          : 7 / 1000
raw_total_valid           : 36
raw_valid_time_min        : 1.39651e+12
raw_valid_time_max        : 1.39654e+12
```

For the successful neutrino sample:

```text
raw_scifi_nonempty_events : 146 / 147
raw_total_valid           : 82892
raw_valid_time_min        : 0.0797778
```

The pipeline constructs:

```cpp
sciFiHitFilter(0.5*6.25, 1.2*6.25, 52, 0, 26)
```

For MC, `sndSciFiTools.cxx` uses:

```cpp
hitTime = hit->GetTime();
```

without the real-data time conversion/correction. The timing peak histogram is
filled in the range `0..26 ns`. MuonDIS valid hit times are around `1.4e12`,
so the timing histogram is effectively empty and the filter keeps no useful
SciFi hits.

The direct consequence is:

```text
MuonDIS absolute/large SciFi time
  -> current SciFi time filter rejects all hits
  -> AvgSFChan receives an empty filtered SciFi collection
  -> AvgSFChan plot variables become -1
  -> AvgSFChan fails every event
```

## Recommended Fix

Adapt the SciFi timing filter to handle absolute-time MC samples by filtering
on relative hit time instead of raw absolute time.

The best approach is not to use a single global first-arrival time for the
whole event. Instead, preserve the current filter design:

```text
for each SciFi station and orientation:
  find the local timing peak
  keep hits near that peak
```

but compute the timing peak using:

```text
relative_time = hit_time - reference_time
```

where `reference_time` is computed from hits in the same station and
orientation.

## Proposed Algorithm

For each `(station, orientation)` group:

1. Collect valid SciFi hits in that station and orientation.
2. Check the time scale.
3. If the hit times are already normal, e.g. within the expected `0..26 ns`
   range, use the current logic unchanged.
4. If the hit times are absolute-scale, e.g. `min_time > 1e6`, compute a local
   reference time.
5. Fill the timing histogram with `hit_time - reference_time`.
6. Find the local timing peak.
7. Keep hits satisfying:

```text
peakTiming - time_lower_range <= hit_time - reference_time <= peakTiming + time_upper_range
```

The simplest reference is:

```text
reference_time = minimum valid hit time in this station/orientation
```

A more robust reference is:

```text
reference_time = low percentile of valid hit times, e.g. 5th or 10th percentile
```

The percentile version is safer against one very early noise hit.

## Sparse Event Handling

MuonDIS appears sparse in SciFi. For sparse groups, a strict timing peak filter
can be unstable. Recommended behavior:

```text
if number of valid hits in station/orientation < 3:
  keep valid hits without timing filtering
else:
  apply relative-time peak filtering
```

This prevents isolated but meaningful MuonDIS hits from being removed simply
because a peak cannot be estimated reliably.

## Why Not A Single First Arrival Time?

A global first-arrival correction would be:

```text
event_reference_time = earliest valid SciFi hit in the event
relative_time = hit_time - event_reference_time
```

This is simple but less robust:

- one early noise hit can shift the whole event;
- different stations or views may have local timing offsets;
- the existing code intentionally finds peaks per station and orientation;
- using one global event reference changes the analysis logic more than needed.

The local station/view reference is the smallest change that fixes the time
convention mismatch while preserving the intent of the existing filter.

## Implementation Target

The most natural implementation point is:

```text
sndsw/analysis/tools/sndSciFiTools.cxx
```

specifically the timing logic used by:

```cpp
peakScifiTiming(...)
selectScifiHits(...)
filterScifiHits(...)
```

A clean implementation would add a helper to compute relative times, or add a
mode in `selectScifiHits` that automatically detects absolute-scale MC timing.

Suggested behavior:

```text
if isMC and selected valid hit times have min_time > 1e6:
  use relative times
else:
  keep current behavior
```

This keeps neutrino MC behavior unchanged and only activates the correction for
samples like MuonDIS.

## Validation Plan

After implementing the change, check the same MuonDIS file:

1. Run `sndEventFilter` on `muonDIS_2001`.
2. Inspect `cutFlowSummary`.
3. Confirm `AvgSFChan == 1` is no longer exactly zero unless physically true.
4. Inspect:

```text
cutFlow_-1_AvgSFChan_0
cutFlow_-1_AvgSFChan_1
cutFlow_0_AvgSFChan_0
cutFlow_0_AvgSFChan_1
```

Expected change:

```text
before: all entries underflow at -1
after : at least some entries in range, or outside fiducial for real geometric reasons
```

5. Compare against the neutrino sample to ensure the cut flow remains consistent
   for normal MC timing.

## Working Conclusion

The best future fix is:

```text
add automatic relative-time handling to the SciFi timing filter for absolute-time MC samples,
using per-station/per-orientation reference times,
with sparse-hit protection.
```

This addresses the source of the `AvgSFChan` failure before the cut runs,
without changing the downstream cut logic or the structure of `cutFlowSummary`.
