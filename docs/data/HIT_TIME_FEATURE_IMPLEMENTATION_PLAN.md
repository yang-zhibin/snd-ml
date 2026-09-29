# Hit Time Feature Implementation Plan

## Goal

Add hit-time information to both conversion outputs:

- `convertData/digi_2_features.py`: event-level time summary features.
- `convertData/digi_2_hits3D.py`: per-hit `hit_time_ns` vectors.

All new hit-time branches must use **nanoseconds**. Do not save new clock-cycle time branches.

Only these MC samples need first-arrival calibration:

```text
MC_muon
MC_muonDIS
```

For those samples, define:

```text
time0_ns = earliest used detector hit time in the event
calibrated_hit_time_ns = hit_time_ns - time0_ns
```

The first saved/used hit in an `MC_muon` or `MC_muonDIS` event should therefore have time `0 ns`.

## Key `sndsw` Timing Facts

From `/afs/cern.ch/user/z/zhibin/work/public/SndBuild/sndsw`:

- `analysis/analyses/snd_analysis_2024_0mu/sciFiTools.py`
  - MC SciFi: `hit.GetTime()` is already in ns.
  - Real SciFi: `hit.GetTime()` is in clock cycles, then converted to ns and passed to `Scifi.GetCorrectedTime`.
- `shipLHC/sndScifiHit.cxx`
  - MC SciFi digi time is simulated as earliest arrival at the SiPM plus time smearing.
- `shipLHC/MuFilterHit.cxx`
  - MC MuFilter digi channel times are simulated as arrival times at readout channels plus time smearing.
- `shipLHC/MuFilter.cxx`
  - `MuFilter.GetCorrectedTime(detID, channel_side, raw_time_ns, L)` exists.
  - It is implemented only for DS.
  - Veto and US return raw time unchanged.

## Unit Rules

Use one explicit conversion constant:

```python
NS_PER_CLOCK = 6.25
INVALID_TIME = -999.0
```

Time conversion:

```text
MC:
  hit_time_ns = hit.GetTime()

real SciFi:
  raw_time_ns = hit.GetTime() * 6.25
  hit_time_ns = Scifi.GetCorrectedTime(detID, raw_time_ns, 0)

real Veto:
  hit_time_ns = channel_time * 6.25

real US:
  hit_time_ns = channel_time * 6.25

real DS:
  raw_time_ns = channel_time * 6.25
  hit_time_ns = MuFilter.GetCorrectedTime(detID, side, raw_time_ns, 0)
```

For DS, use `L=0` deliberately. The converter has bar endpoints but does not know the real hit impact position along the bar, so it should not invent a propagation length. This applies DS time alignment without an unreliable path-length correction.

## Detector Subsystems

Subsystem meanings:

```text
SciFi: Digi_ScifiHits
Veto:  Digi_MuFilterHits with GetSystem() == 1
US:    Digi_MuFilterHits with GetSystem() == 2
DS:    Digi_MuFilterHits with GetSystem() == 3
```

In `digi_2_features.py`, keep the existing internal dictionary convention:

```text
SciFi detType = 0
Veto  detType = 1
US    detType = 2
DS    detType = 3
```

In `digi_2_hits3D.py`, preserve the existing output `hit_detType` convention. Add time as a new aligned vector rather than changing detector labels.

## Per-Hit Time Definition

### SciFi

Each SciFi digi hit has one time:

```python
hit["hitTimeNS"] = scifi_time_ns(...)
```

### MuFilter

MuFilter hits can have multiple channel times. Use the earliest valid channel time as the bar-level hit time:

```python
hit["hitTimeNS"] = min(valid_channel_times_ns)
```

Use `hit.GetAllTimes()` where available. If a hit has no valid channel time, set:

```python
hit["hitTimeNS"] = -999.0
```

For DS real data, determine side from the channel index:

```python
n_sipms = MuFilter.GetnSiPMs(detID)
side = 0 if channel < n_sipms else 1
```

Only call `MuFilter.GetCorrectedTime` for `GetSystem() == 3`.

## First-Arrival Calibration Strategy

Use a simple extra loop only for `MC_muon` and `MC_muonDIS`. This is acceptable because:

- It only runs for the two sample types that need calibration.
- It is cheap compared with ROOT I/O, geometry lookup, and correction calls.
- It keeps the logic obvious and easy to validate.

Recommended helper:

```python
def needs_first_arrival_time0(args):
    text = " ".join(
        str(getattr(args, name, ""))
        for name in ("type", "digi_path", "out_path", "preSelect_path")
    ).lower()
    return "mc_muon" in text or "muondis" in text
```

After `SciFi_hits` and `MuFilter_hits` have been built and each hit already has `hitTimeNS`:

```python
time0_ns = INVALID_TIME
time0_applied = False

if needs_first_arrival_time0(args):
    valid_times = [
        h["hitTimeNS"]
        for h in SciFi_hits + MuFilter_hits
        if valid_time(h.get("hitTimeNS", INVALID_TIME))
    ]
    time0_ns = min(valid_times) if valid_times else INVALID_TIME

    if valid_time(time0_ns):
        for h in SciFi_hits + MuFilter_hits:
            if valid_time(h.get("hitTimeNS", INVALID_TIME)):
                h["hitTimeRawNS"] = h["hitTimeNS"]
                h["hitTimeNS"] -= time0_ns
        time0_applied = True
```

For all other samples:

```text
time0_ns = -999
time0_applied = 0
```

Important: compute `time0_ns` from the hits used by the converter output:

- Normal samples: SciFi hits after `selectHits(...)`.
- muonDIS: all valid SciFi hits, matching the current code.
- MuFilter: all valid MuFilter hits used by the converter.

This guarantees that for calibrated samples, the minimum saved hit time is `0 ns`, except for empty/no-valid-time events.

## Shared Helper Functions

Add these helpers to both converters. They can be duplicated locally at first. If later maintenance becomes annoying, move them into `convertData/hit_time_utils.py`.

```python
NS_PER_CLOCK = 6.25
INVALID_TIME = -999.0

def is_mc_sample(args):
    return "MC" in str(args.type)

def valid_time(value):
    try:
        value = float(value)
    except Exception:
        return False
    return math.isfinite(value) and value > -900.0

def scifi_time_ns(hit, scifi_det, args):
    if is_mc_sample(args):
        return float(hit.GetTime())
    raw_ns = float(hit.GetTime()) * NS_PER_CLOCK
    return float(scifi_det.GetCorrectedTime(hit.GetDetectorID(), raw_ns, 0))

def mufilter_channel_times_ns(hit, mufilter_det, args):
    det_id = int(hit.GetDetectorID())
    system = int(hit.GetSystem())
    out = []

    try:
        raw_times = hit.GetAllTimes()
    except Exception:
        raw_times = {}

    if not raw_times:
        raw_times = {0: hit.GetTime()}

    n_sipms = None
    if system == 3:
        try:
            n_sipms = int(mufilter_det.GetnSiPMs(det_id))
        except Exception:
            n_sipms = None

    for channel, raw_time in raw_times.items():
        if not valid_time(raw_time):
            continue

        if is_mc_sample(args):
            time_ns = float(raw_time)
        else:
            time_ns = float(raw_time) * NS_PER_CLOCK
            if system == 3 and n_sipms:
                side = 0 if int(channel) < n_sipms else 1
                try:
                    time_ns = float(mufilter_det.GetCorrectedTime(det_id, side, time_ns, 0))
                except Exception:
                    pass

        out.append(time_ns)

    return out

def mufilter_hit_time_ns(hit, mufilter_det, args):
    times = mufilter_channel_times_ns(hit, mufilter_det, args)
    return min(times) if times else INVALID_TIME
```

`digi_2_features.py` already imports `math`. Add `import math` to `digi_2_hits3D.py`.

## `digi_2_features.py` Changes

### Add New Branches

Add event-level timing branches:

```text
hitTime_time0_ns
hitTime_time0_applied

hitTime_min_ns
hitTime_max_ns
hitTime_mean_ns
hitTime_std_ns
hitTime_range_ns

hitTime_scifi_min_ns
hitTime_scifi_max_ns
hitTime_scifi_mean_ns
hitTime_scifi_std_ns
hitTime_scifi_range_ns

hitTime_veto_min_ns
hitTime_veto_max_ns
hitTime_veto_mean_ns
hitTime_veto_std_ns
hitTime_veto_range_ns

hitTime_us_min_ns
hitTime_us_max_ns
hitTime_us_mean_ns
hitTime_us_std_ns
hitTime_us_range_ns

hitTime_ds_min_ns
hitTime_ds_max_ns
hitTime_ds_mean_ns
hitTime_ds_std_ns
hitTime_ds_range_ns
```

Use ROOT type `d` for ns values and `i` for `hitTime_time0_applied`.

### Build Hit Times

Change `build_scifi_hit_dict` so it receives `args` and stores:

```python
"hitTimeNS": scifi_time_ns(hit, Scifi, args)
```

When building `MuFilter_hits`, store:

```python
"hitTimeNS": mufilter_hit_time_ns(aHit, MuFilter, args)
```

Keep existing legacy `hitTimeCY` only for backward compatibility. Do not use it for new branches.

### Add Summary Function

Add:

```python
def fill_time_summary(prefix, times, branch_vars):
    times = [float(t) for t in times if valid_time(t)]
    if not times:
        return
    t_min = min(times)
    t_max = max(times)
    branch_vars[f"{prefix}_min_ns"][0] = t_min
    branch_vars[f"{prefix}_max_ns"][0] = t_max
    branch_vars[f"{prefix}_mean_ns"][0] = sum(times) / len(times)
    branch_vars[f"{prefix}_std_ns"][0] = std_or_sentinel(times)
    branch_vars[f"{prefix}_range_ns"][0] = t_max - t_min
```

Then:

```python
def process_hit_time_features(SciFi_hits, MuFilter_hits, branch_vars, time0_ns, time0_applied):
    scifi_times = [h["hitTimeNS"] for h in SciFi_hits]
    veto_times = [h["hitTimeNS"] for h in MuFilter_hits if h["detType"] == 1]
    us_times = [h["hitTimeNS"] for h in MuFilter_hits if h["detType"] == 2]
    ds_times = [h["hitTimeNS"] for h in MuFilter_hits if h["detType"] == 3]
    all_times = scifi_times + veto_times + us_times + ds_times

    branch_vars["hitTime_time0_ns"][0] = time0_ns
    branch_vars["hitTime_time0_applied"][0] = int(time0_applied)

    fill_time_summary("hitTime", all_times, branch_vars)
    fill_time_summary("hitTime_scifi", scifi_times, branch_vars)
    fill_time_summary("hitTime_veto", veto_times, branch_vars)
    fill_time_summary("hitTime_us", us_times, branch_vars)
    fill_time_summary("hitTime_ds", ds_times, branch_vars)
```

Call this after the optional first-arrival loop.

### Legacy Branches

Keep existing branches such as:

```text
vetoHitTime_earlist
vetoHitTime_latest
```

Do not change their meaning as part of this feature. New analysis should use `hitTime_veto_*_ns`.

## `digi_2_hits3D.py` Changes

### Extend Hit Arrays

Add to `HIT_KEYS`:

```python
"time",
```

Add to `HIT_DTYPES`:

```python
np.float32,
```

Add ROOT vector branch:

```python
"hit_time_ns": ROOT.std.vector("float")()
```

Add scalar event branches:

```text
hitTime_time0_ns       double
hitTime_time0_applied  int
```

### Add Time To Source Hit Dictionaries

For SciFi source hits:

```python
"time": scifi_time_ns(aHit, Scifi, args)
```

For MuFilter source hits:

```python
"time": mufilter_hit_time_ns(aHit, MuFilter, args)
```

Use the same optional first-arrival loop for `MC_muon` and `MC_muonDIS`, but on key `"time"` instead of `"hitTimeNS"`.

### Crossed Hits

For SciFi and DS crossed 3D hits, use the earlier of the two projection times:

```python
time = np.minimum(np.repeat(v_time, nh), np.tile(h_time, nv))
```

### One-Orientation Hits

For vertical-only or horizontal-only hits:

```python
"time": np.array([h["time"] for h in hits], dtype=np.float32)
```

### US/DS4 Voxel Hits

Every voxel from one bar hit inherits that bar's time:

```python
time_all.append(np.full(nvox, float(h["time"]), dtype=np.float32))
```

Do not divide time by the number of voxels. Only QDC is divided.

### Fill Output Vector

In `fill_hit_vectors`, include `time` in the consistency-length check and assign:

```python
assign_vector(vector_vars["hit_time_ns"], time)
```

## Validation Plan

### Static Checks

Use compile-only checks that do not create `__pycache__`:

```bash
python3 -c "compile(open('convertData/digi_2_features.py').read(), 'convertData/digi_2_features.py', 'exec')"
python3 -c "compile(open('convertData/digi_2_hits3D.py').read(), 'convertData/digi_2_hits3D.py', 'exec')"
```

### Runtime Checks

Run representative small samples if available:

```text
MC_muonDIS
MC_muon
MC_neutrino
real_data
```

For `MC_muonDIS` and `MC_muon`:

- `hitTime_time0_applied == 1` when valid hit times exist.
- `hitTime_time0_ns` is finite.
- `hitTime_min_ns == 0` in `sndData`, within floating precision.
- `min(hit_time_ns) == 0` in `hit3D`, if the event has hit3D entries.

For all other samples:

- `hitTime_time0_applied == 0`.
- `hitTime_time0_ns == -999`.
- New hit-time values are in ns.

For real data:

- SciFi times use `Scifi.GetCorrectedTime`.
- Veto and US times are raw clock cycles converted to ns.
- DS times use `MuFilter.GetCorrectedTime(..., L=0)`.

For `hit3D`:

- `len(hit_time_ns)` equals every other hit vector length.
- Crossed hits use the earlier projection time.
- US/DS4 voxels copy bar time unchanged.

## Recommended Implementation Order

1. Add time helper functions to `digi_2_features.py`.
2. Add `hitTimeNS` to SciFi and MuFilter dictionaries.
3. Add the extra `time0_ns` loop only for `MC_muon` and `MC_muonDIS`.
4. Add timing summary branches and fill them.
5. Syntax-check `digi_2_features.py`.
6. Add time helper functions to `digi_2_hits3D.py`.
7. Extend `HIT_KEYS`, `HIT_DTYPES`, and ROOT branches with time.
8. Carry time through crossed, one-orientation, and voxel hit builders.
9. Add `hitTime_time0_ns` and `hitTime_time0_applied`.
10. Syntax-check `digi_2_hits3D.py`.
11. Run small runtime checks.
12. Update `convertData/FEATURE_DESCRIPTION.md` after the final schema is confirmed.
