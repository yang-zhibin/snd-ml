# digi_2_features Feature Description

This document describes the event-level features written by
`convertData/digi_2_features.py`.

Each row in the output `sndData` tree corresponds to one selected event from the
input digi tree. The selected events are taken from the `cutFlowSummary` tree
produced by the SND analysis filter.

## Part 1: Feature Meaning

### 1. Event Identity Features

```text
runId
eventId
eventIndex
isMC
pdgCode
```

`runId`: run number from the event header.

`eventId`: event number from the event header. For MC, the script falls back to
MC entry number if needed.

`eventIndex`: original entry index in the raw `cbmsim` digi tree. This is useful
because the selected `cutFlowSummary` entry may not always equal the raw tree
entry.

`isMC`: `1` for MC and `0` for real data.

`pdgCode`: truth PDG label for MC. For real data this is `0`.

### 2. MC Truth Position And Momentum

```text
x
y
z
px
py
pz
```

These are MC truth-level quantities.

`x`, `y`, `z`: truth start position from `MCTrack[1]` when at least two MC
tracks exist.

`px`, `py`, `pz`: truth momentum from `MCTrack[0]` when at least one MC track
exists.

For real data these remain at the sentinel value `-999`.

### 3. MuonDIS Truth Features

```text
nPrimary
nSecondaryRaw
nSecondary
```

These are mainly for muonDIS MC.

`nPrimary`: number of primary MC tracks with `mother_id == -1`.

`nSecondaryRaw`: number of raw secondaries with `mother_id == 0`.

`nSecondary`: number of secondaries that are visible in SciFi, meaning they are
linked to at least one valid SciFi digi hit.

Vector branches:

```text
primary_pdg
primary_energy
primary_startX
primary_startY
primary_startZ
primary_px
primary_py
primary_pz

secondary_pdg
secondary_energy
secondary_startX
secondary_startY
secondary_startZ
secondary_px
secondary_py
secondary_pz

secondary_interactionChildTrackID
secondary_interactionChildPdg
secondary_interactionX
secondary_interactionY
secondary_interactionZ
```

These store detailed MC truth information for primary tracks, visible secondary
tracks, and one child track used as a proxy for secondary interaction position.

### 4. Hit Count Features

```text
count_scifi1 ... count_scifi5
count_scifi

count_veto1 ... count_veto3
count_veto

count_us1 ... count_us5
count_us

count_ds1 ... count_ds4
count_ds
```

These count valid detector hits by subsystem and station.

SciFi has five stations. Veto has three planes. US means upstream MuFilter, with
five stations. DS means downstream MuFilter, with four stations.

### 5. QDC Features

```text
qdc_scifi1 ... qdc_scifi5
qdc_scifi

qdc_veto1 ... qdc_veto3
qdc_veto

qdc_us1 ... qdc_us5
qdc_us

qdc_ds1 ... qdc_ds4
qdc_ds
```

These are charge/signal sums. They describe how much detector signal is present
in each station or subsystem. Individual hit signal values are clamped to be
non-negative before summing, so these branches represent non-negative detector
signal.

### 6. Average Position Features

SciFi:

```text
avg_scifi1_x
avg_scifi1_y
...
avg_scifi5_x
avg_scifi5_y
avg_scifi_x
avg_scifi_y
```

Veto:

```text
avg_veto1_y
avg_veto2_y
avg_veto3_x
avg_veto_x
avg_veto_y
```

Upstream MuFilter:

```text
avg_us1_y ... avg_us5_y
avg_us_y
```

Downstream MuFilter:

```text
avg_ds1_x
avg_ds1_y
...
avg_ds4_x
avg_ds4_y
avg_ds_x
avg_ds_y
```

These describe the average detector-hit position.

Detector view convention:

```text
vertical plane   -> useful coordinate is x
horizontal plane -> useful coordinate is y
```

If there are no hits for a given average, the value remains `-999`.

### 7. QDC-Weighted Average Position Features

SciFi:

```text
qdcAvg_scifi1_x
qdcAvg_scifi1_y
...
qdcAvg_scifi5_x
qdcAvg_scifi5_y
qdcAvg_scifi_x
qdcAvg_scifi_y
```

Veto:

```text
qdcAvg_veto1_y
qdcAvg_veto2_y
qdcAvg_veto3_x
qdcAvg_veto_x
qdcAvg_veto_y
```

Upstream MuFilter:

```text
qdcAvg_us1_y ... qdcAvg_us5_y
qdcAvg_us_y
```

Downstream MuFilter:

```text
qdcAvg_ds1_x
qdcAvg_ds1_y
...
qdcAvg_ds4_x
qdcAvg_ds4_y
qdcAvg_ds_x
qdcAvg_ds_y
```

These describe the QDC-weighted detector-hit position:

```text
qdcAvg = sum(position_i * qdc_i) / sum(qdc_i)
```

The detector view convention is the same as for `avg_*`: vertical planes use
`x`, and horizontal planes use `y`. QDC values are clamped to be non-negative.
If the selected hits have zero total QDC, the value remains `-999`.

### 8. SciFi Density Features

Local density features:

```text
density_scifi1 ... density_scifi5
density_scifi

density_scifi1_x ... density_scifi5_x
density_scifi_x

density_scifi1_y ... density_scifi5_y
density_scifi_y
```

These measure how clustered SciFi hits are within each station and view.

SND official density features:

```text
density_sndsw_scifi
density_sndsw_scifi_second
density_sndsw_scifi_hor
density_sndsw_scifi_ver
```

These come from SND `getSumDensity`.

Local summary density features:

```text
density_mycode_scifi
density_mycodescifi_second
density_mycode_scifi_second
density_mycode_scifi_hor
density_mycode_scifi_ver
```

These summarize the local Python density in a way similar to the SND density.
`density_mycodescifi_second` is the older typo branch kept for compatibility;
`density_mycode_scifi_second` is the correctly spelled alias with the same
value.

### 9. Veto Timing Features

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

These describe the earliest and latest veto hit time, globally and per veto
plane.

If no veto hit exists, the value is `-1`.

Note: the branch name is spelled `earlist`, not `earliest`, because that is the
existing output schema.

### 10. SciFi Station Profile Features

```text
scifi_first_station
scifi_last_station
scifi_n_active_stations
scifi_n_active_stations_xy
scifi_max_count_station
scifi_max_qdc_station
scifi_count_peak_fraction
scifi_qdc_peak_fraction
```

These describe where SciFi activity starts, ends, and peaks.

`scifi_first_station`: first station with at least one selected SciFi hit.

`scifi_last_station`: last station with at least one selected SciFi hit.

`scifi_n_active_stations`: number of stations with any selected SciFi hit.

`scifi_n_active_stations_xy`: number of stations with both x-view and y-view
hits.

`scifi_max_count_station`: station with the largest hit count.

`scifi_max_qdc_station`: station with the largest QDC.

`scifi_count_peak_fraction`: fraction of all SciFi hits contained in the most
populated station.

`scifi_qdc_peak_fraction`: fraction of all SciFi QDC contained in the
highest-QDC station.

### 11. Longitudinal Shower Shape Features

```text
scifi_count_frac1 ... scifi_count_frac5
scifi_qdc_frac1 ... scifi_qdc_frac5

scifi_count_mean_station
scifi_count_std_station

scifi_qdc_mean_station
scifi_qdc_std_station
```

These describe the shower development along SciFi stations 1 to 5.

`scifi_count_fracN`: fraction of SciFi hits in station `N`.

`scifi_qdc_fracN`: fraction of SciFi QDC in station `N`.

`scifi_count_mean_station`: hit-count-weighted average station number.

`scifi_count_std_station`: hit-count-weighted longitudinal spread.

`scifi_qdc_mean_station`: QDC-weighted average station number.

`scifi_qdc_std_station`: QDC-weighted longitudinal spread.

### 12. Hit-Position Spread Features

```text
scifi_std_x
scifi_std_y

scifi1_std_x
scifi1_std_y
...
scifi5_std_x
scifi5_std_y
```

These describe the transverse width of the SciFi hit pattern.

`scifi_std_x`: global spread of vertical-view SciFi hits in `x`.

`scifi_std_y`: global spread of horizontal-view SciFi hits in `y`.

`scifiN_std_x`: spread of vertical-view SciFi hits in station `N`.

`scifiN_std_y`: spread of horizontal-view SciFi hits in station `N`.

These are standard deviations around the mean, not raw RMS from zero.

## Part 2: Implementation Logic

### 1. Event Selection Logic

The script reads the preselection tree:

```text
cutFlowSummary
```

The current selection is:

```text
SciFiMinHits == 1
```

For real data it also requires:

```text
StableBeams == 1
IP1 == 1
EventDeltat_m1_100 == 1
```

The selected `cutFlowSummary` entry is mapped back to the raw digi entry using
the `entry` branch if it exists.

### 2. Geometry And Calibration Logic

For every selected event:

```python
raw_tree.GetEntry(raw_entry)
init_event_geometry(snd_geo, raw_tree.EventHeader)
```

This initializes SciFi and MuFilter geometry for the current event.

This is important for:

```text
SciFi corrected time
SciFi position
MuFilter position
run-dependent detector alignment
```

### 3. SciFi Hit Selection Logic

For normal samples:

```python
selected_scifi_hits = selectHits(event, MC=("MC" in args.type))
```

This uses SND's official SciFi timing selection.

For muonDIS samples:

```text
selected_scifi_hits = all valid SciFi hits
```

Then each selected SciFi hit is converted into a dictionary:

```text
detType = 0
station
isVertical
layer_channel
qdc
x, y, z
hitTimeCY
```

This dictionary is used by all later SciFi feature functions.

### 4. MuFilter Hit Logic

For each valid MuFilter hit:

```text
detType = hit.GetSystem()
```

Meaning:

```text
1 = veto
2 = upstream
3 = downstream
```

The script stores:

```text
station
isVertical
qdc
barIndex
hitTimeCY
x, y, z
```

MuFilter positions come from:

```python
MuFilter.GetPosition(detID, A, B)
```

### 5. Count And QDC Implementation

Function:

```python
process_count_and_qdc(SciFi_hits, MuFilter_hits, branch_vars)
```

Logic:

1. Loop over SciFi hits.
2. Count hits per station.
3. Sum QDC per station.
4. Loop over MuFilter hits.
5. Split by detector system: veto, US, DS.
6. Count hits and sum QDC per station.

This fills all `count_*` and `qdc_*` branches.

### 6. Average Position Implementation

Function:

```python
process_avgPos(SciFi_hits, MuFilter_hits, branch_vars)
```

Logic:

1. Combine SciFi and MuFilter hit dictionaries.
2. For each hit, choose coordinate based on detector/view.
3. Accumulate coordinate sums and counts.
4. Fill average as:

```python
average = sum / count
```

If `count == 0`, fill `-999`.

View convention:

```text
vertical   -> x
horizontal -> y
```

### 7. QDC-Weighted Average Position Implementation

Function:

```python
process_qdcAvgPos(SciFi_hits, MuFilter_hits, branch_vars)
```

Logic:

1. Combine SciFi and MuFilter hit dictionaries.
2. For each hit, choose coordinate based on detector/view.
3. Clamp QDC to be non-negative.
4. Accumulate weighted coordinate sums and QDC sums.
5. Fill weighted average as:

```python
qdcAvg = sum(position * qdc) / sum(qdc)
```

If `sum(qdc) == 0`, fill `-999`.

The detector/view mapping is identical to `process_avgPos`, so `avg_*` and
`qdcAvg_*` can be compared directly.

### 8. SciFi Density Implementation

Function:

```python
hitWeightDensity(SciFi_hits, branch_vars)
```

Logic:

1. Group SciFi hit positions by station and view.
2. For vertical hits, use `x`.
3. For horizontal hits, use `y`.
4. For each hit, count neighboring hits within 1 cm.
5. Sum neighbor counts to get density.

Then:

```text
density_scifiN_x = vertical-view density in station N
density_scifiN_y = horizontal-view density in station N
density_scifiN   = x + y density in station N
```

Function:

```python
fill_mycode_density(branch_vars)
```

Logic:

1. Find the station with largest total local density.
2. Store the minimum of x/y density as the balanced density.
3. Also store second-best station density.

SND density is computed separately using:

```python
getSumDensity(selected_scifi_hits)
```

### 9. Veto Timing Implementation

Function:

```python
process_vetoHitTime(MuFilter_hits, branch_vars)
```

Logic:

1. Select only MuFilter hits where `detType == 1`.
2. Collect veto hit times in clock cycles.
3. Fill earliest and latest time globally.
4. Fill earliest and latest time per veto plane.

If no veto hits exist, fill `-1`.

### 10. SciFi Station Profile Implementation

Function:

```python
process_scifi_topology_features(SciFi_hits, branch_vars)
```

This function first builds per-station arrays:

```text
counts[station]
qdcs[station]
x_counts[station]
y_counts[station]
x_positions[station]
y_positions[station]
```

Then:

`scifi_first_station`:

```python
min(stations with count > 0)
```

`scifi_last_station`:

```python
max(stations with count > 0)
```

`scifi_n_active_stations`:

```python
number of stations with count > 0
```

`scifi_n_active_stations_xy`:

```python
number of stations with x_count > 0 and y_count > 0
```

`scifi_max_count_station`:

```python
station with max(counts)
```

`scifi_max_qdc_station`:

```python
station with max(qdcs)
```

`scifi_count_peak_fraction`:

```python
max(counts) / sum(counts)
```

`scifi_qdc_peak_fraction`:

```python
max(qdcs) / sum(qdcs)
```

If undefined, station branches use `-999`, and fractions use `0.0`.

### 11. Longitudinal Shower Shape Implementation

The station number is treated as the longitudinal coordinate:

```python
stations = [1, 2, 3, 4, 5]
```

Fractions:

```python
scifi_count_fracN = count[N] / total_count
scifi_qdc_fracN   = qdc[N] / total_qdc
```

Weighted station moments:

```python
mean = sum(station * weight) / sum(weight)
std  = sqrt(sum(weight * (station - mean)^2) / sum(weight))
```

For count-weighted features:

```text
weight = station hit count
```

For QDC-weighted features:

```text
weight = station QDC
```

All QDC-like hit signals are clamped to non-negative values before feature
sums/moments are computed. This keeps raw QDC sums and topology QDC-weighted
moments consistent as non-negative detector-signal quantities.

If total weight is zero:

```text
mean = -999
std  = -999
```

### 12. Hit-Position Spread Implementation

Position spread uses:

```python
std = sqrt(mean((position - mean_position)^2))
```

For global x spread:

```text
collect x from all vertical SciFi hits
```

For global y spread:

```text
collect y from all horizontal SciFi hits
```

For station-level spread:

```text
collect x/y positions separately for each station and view
```

If fewer than two hits exist in the view:

```text
std = -999
```

This is because one hit does not define a physical width.

### 13. Output Schema Logic

All branches are declared before the event loop.

This means even if zero events pass selection, the output still contains a valid
empty `sndData` tree with the full branch schema.

The removed branches are:

```text
start_z
vetoHits
```

They are no longer written.
