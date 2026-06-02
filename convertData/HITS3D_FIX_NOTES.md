# digi_2_hits3D Follow-Up Notes

## Context

`convertData/digi_2_hits3D.py` has been updated so its event chain is aligned
with `convertData/digi_2_features.py`:

- same broad event selection:
  - `SciFiMinHits == 1`
  - plus real-data beam-quality cuts:
    `StableBeams == 1 && IP1 == 1 && EventDeltat_m1_100 == 1`
- same `cutFlowSummary` to `cbmsim` entry mapping:
  - use `preSelect_tree.entry` when available
  - otherwise fall back to the selected cutflow entry
- same output event key convention:
  - `eventIndex = raw cbmsim entry`
- per-event SciFi/MuFilter geometry initialization is called before hit
  extraction.

This should make the feature output and hit3D output align by:

```text
runId, eventId, eventIndex
```

## Resolved

### 1. SciFi Hit Selection Is Now Feature-Consistent

The event chain is aligned, and the SciFi hit content now follows the same
first-stage convention as `digi_2_features.py`.

Current hit3D behavior:

- normal nue-style samples use SND `selectHits(event, MC=...)`
- muonDIS samples use all valid `Digi_ScifiHits`
- no extra local SciFi time-MPV filter is applied
- no extra local SciFi QDC-threshold filter is applied

This keeps the all-event hit3D intermediate close to the feature-production
detector convention. Any model-specific QDC thinning should be done later when
converting hit3D output into region-specific training NPZ files.

### 2. Negative QDC Values Are Now Clamped

Hit3D QDC extraction now uses `non_negative_float(...)` for both:

- SciFi `aHit.GetSignal(0)`
- MuFilter values from `GetAllSignals()`

This matches the feature-production convention and prevents negative detector
signals from entering crossed-hit or voxel QDC sums.

### 3. All-Event Hit3D Output Is Now ROOT-Streaming

`digi_2_hits3D.py` now writes one event at a time into a ROOT tree named
`hit3D`, instead of accumulating all selected events in a Python list and
writing one large NPZ at the end.

Current scalar branches:

- `runId`
- `eventId`
- `eventIndex`
- `pdgCode`
- `isMC`
- `label`
- `energy`

Current vector branches:

- `hit_x`
- `hit_y`
- `hit_z`
- `hit_qdc`
- `hit_station`
- `hit_detType`
- `hit_source`
- `hit_vertical_id`
- `hit_horizontal_id`
- `hit_barIndex`

The source convention is:

- `1`: SciFi crossed hit
- `2`: DS1/DS2/DS3 crossed hit
- `3`: US/DS4 voxel hit

This resolves the main memory-risk issue and makes the all-event hit3D product
a reusable ROOT intermediate.

### 4. Direct CLI Output Directory Is Now Created

The ROOT output helper creates the output parent directory before opening the
ROOT file, so direct command-line output is safer.

## Findings To Fix Later

### 1. Region/Training NPZ Converter Still Needed

The all-event hit3D product is now ROOT, but the current 3D training code still
expects region/split NPZ files with the old structure:

```python
event["all_3dHits"] = {
    "x": ...,
    "y": ...,
    "z": ...,
    "qdc": ...,
    "station": ...,
    "detType": ...,
}
```

Next step:

- build a ROOT-to-NPZ converter that joins `feature.root:sndData` and
  `hit3D.root:hit3D` by `runId`, `eventId`, and `eventIndex`
- apply region cuts on the feature ROOT
- materialize region-specific train/val/test NPZ files for `train_3d`

## Longer-Term Target Flow

Preferred future structure:

```text
digi ROOT + nueAnalysisFilter ROOT + geo
        |
        +--> digi_2_features.py
        |       output: feature ROOT, tree sndData
        |
        +--> digi_2_hits3D.py
                output: hit3D ROOT, tree hit3D

feature ROOT + hit3D ROOT + region cuts + split metadata
        |
        v
region-specific training NPZ
```

This keeps expensive hit extraction independent of changing region definitions.
