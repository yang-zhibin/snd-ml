# SciFi clock-cycle selection

This guide describes the fixed timing selection used before shower-feature
calculation on `task/shower-features`. Code verified on 26 November 2026;
implementation commit `a454ef3542fbde1c8c4988c9fe6d22f031f67722`.

## What is a clock cycle?

One cycle is one tick of the electronics clock. SNDSW uses:

```python
clock_period_ns = 1e9 / 160.316e6  # 6.23768058 ns per cycle
```

The timing window is expressed in cycles **relative to the time peak in each
SciFi plane**, not relative to the start of the event.

| Quantity | Meaning |
| --- | --- |
| One clock cycle | Approximately 6.238 ns |
| Histogram bin | 0.5 ns; used to locate the peak |
| `t_peak` | Centre of the bin with the most hits in a plane |
| `LOWER`, `UPPER` | Signed offsets from `t_peak`, in cycles |

A plane is one station/orientation pair: five stations × two orientations.
The histogram bin width and the clock period are different quantities.

## Selection logic

For each already-selected event, the standard SNDSW `selectHits()` function:

1. Collects valid SciFi hits, separately in each of the ten planes.
2. Converts hit times to ns. MC times are already in ns. Data times are
   multiplied by the clock period, then corrected with `Scifi.GetCorrectedTime()`.
3. Fills one time histogram per plane: 200 bins over 0–100 ns.
4. Finds the bin containing the most hits and takes its centre as `t_peak`
   (`peakTime` in the code). Hits have equal weight; QDC is not used.
   Tied maxima select the earliest bin. Underflow/overflow bins do not determine
   the normal histogram peak.
5. Keeps hits inside the inclusive window:

```python
lower_ns = t_peak + LOWER * clock_period_ns
upper_ns = t_peak + UPPER * clock_period_ns
keep = lower_ns <= hit_time_ns <= upper_ns
```

For example, the busiest bin spanning 20.0–20.5 ns gives `t_peak = 20.25 ns`.
The peak is found before applying this window; it is not recalculated after
rejecting hits.

Both data and MC default to `LOWER = -0.5`, `UPPER = 1.2`:

| Window (cycles) | Time relative to peak | Accepted times if peak = 20 ns |
| --- | --- | --- |
| `-0.5 1.2` (default) | −3.12 to +7.49 ns | 16.88–27.49 ns |
| `-0.5 1.8` | −3.12 to +11.23 ns | 16.88–31.23 ns |
| `-0.5 2.0` | −3.12 to +12.48 ns | 16.88–32.48 ns |

These are fixed offsets: the peak varies by event and plane, but the selected
window width does not adapt to hit occupancy or energy.

## Where this sits in production

```text
Digi ROOT + geometry + upstream event-selection indices
    → read selected event
    → select SciFi hits around each plane's time peak
    → apply existing additional MC time cut where applicable
    → calculate baseline features and coworker's plugin features
    → write feature ROOT + timing metadata
    → validate → histograms → PDF plots
```

Counts, QDC, density, clusters and plugin features use the resulting selected
SciFi hits. The timing options do not rebuild events, regenerate selection
indices/cutflows, or change the catalogue's event population. Hit3D conversion
also supports these settings; aligned feature/Hit3D products must use the same
policy and preserve `(runId, eventId, eventIndex)` identity.

### Existing special cases

- **Muon MC and muon-DIS MC:** subtract the earliest valid arrival time across
  SciFi and MuFilter hits, then use the local `select_calibrated_scifi_hits()`.
  It finds the busiest 0.5 ns bin per plane without the fixed 0–100 ns range,
  chooses the earliest tied bin and uses the historical **6.25 ns** period.
- **Additional MC cut:** the converter retains its existing `time <= 25 ns`
  requirement where `applies_mc_25ns_hit_filter()` returns true. MC-neutrino
  production is normally exempt. The legacy exemption checks for `mc_neutrino`
  in the sample type/input/output path strings; it is not purely a type check.
- **Stored times:** the feature converter's data-time helper retains a rounded
  6.25 ns conversion. SNDSW selection uses 6.23768058 ns. Reconstructing selection
  from stored times alone can therefore differ near a boundary.

The 25 ns cut is separate from the clock-cycle window. Neither option configures
an upstream event-builder window.

## How to use it

Run from the repository root using your normal production environment. For batch
profile setup, see the [production guide](../feature_production/README.md#batch-setup-and-full-dataset).

A small testbeam comparison with a wider data window and unchanged MC window:

```bash
./run_shower_features.sh --dataset testbeam --plugin silvia \
  --sample small --batch --jobs 2 --run testbeam-clock20 \
  --scifi-data-window-cycles -0.5 2.0 \
  --scifi-mc-window-cycles -0.5 1.2
```

A small MC-neutrino run with the default MC window explicitly specified:

```bash
./run_shower_features.sh --dataset mc-neutrino --plugin silvia \
  --sample small --run neutrino-clock12 \
  --scifi-mc-window-cycles -0.5 1.2
```

The declared sample type chooses data or MC settings, including within a mixed
testbeam campaign. Omitted options default to `-0.5 1.2`. Each option requires
two finite values with `LOWER < UPPER`. Plugins do not need changes.

**Use a fresh run name when changing the effective window.** Conversion must
reread the original digi files: replotting existing features cannot recover
rejected hits. `--reuse-sources` rejects a changed effective timing policy.
For a controlled comparison, keep source/event limits and plugin settings the
same between runs. Wider windows are comparison settings, not validated new
defaults.

The standalone feature converter, MC-muon wrapper and Hit3D converter accept
the same flags. For the original Snakemake workflow, add these YAML lists to
your campaign config and use a separate output campaign:

```yaml
scifi_data_window_cycles: [-0.5, 2.0]
scifi_mc_window_cycles: [-0.5, 1.2]
```

## Code locations

Repository links below are relative to this guide. Function names are the stable
references if line numbers change.

| File | Function / role |
| --- | --- |
| SNDSW `analysis/analyses/snd_analysis_2024_0mu/sciFiTools.py` | `selectHits()` defines histograms, peak calculation, clock conversion and window cuts |
| [convertData/scifi_timing.py](../convertData/scifi_timing.py) | `add_arguments()`, `effective_policy()`, `configure()`: parse settings, choose data/MC policy and set process-local SNDSW window globals |
| [convertData/digi_2_features.py](../convertData/digi_2_features.py) | `main()` configures timing; `process_hits()` calls selection; `applies_mc_25ns_hit_filter()` controls the extra MC cut |
| [convertData/hit_time_selection.py](../convertData/hit_time_selection.py) | `apply_first_arrival_time0()` and `select_calibrated_scifi_hits()` implement the muon-MC path |
| [convertData/digi_2_hits3D.py](../convertData/digi_2_hits3D.py) | Applies the same timing policy to Hit3D conversion |
| [feature_production/cli.py](../feature_production/cli.py) | Stores timing settings in the production plan and campaign identity |
| [feature_dev/sources.py](sources.py) | `converter_arguments()` forwards settings to the converter |
| [snakemake/rules/unified_shower_features.smk](../snakemake/rules/unified_shower_features.smk) | Forwards YAML settings in the original workflow |
| [convertData/test_scifi_timing_sndsw.py](../convertData/test_scifi_timing_sndsw.py) | Tests actual SNDSW selection with synthetic hits |
| [feature_production/tests/test_scifi_timing.py](../feature_production/tests/test_scifi_timing.py) | Tests boundaries, routing, metadata and reuse protection |

The installed SNDSW file inspected for this implementation is:

```text
/afs/cern.ch/work/z/zhibin/public/SndBuild/sw/rhel9_x86-64/sndsw/master-local14/analysis/analyses/snd_analysis_2024_0mu/sciFiTools.py
```

In that version: `selectHits()` starts at line 9, the data clock conversion is
at line 22, the peak calculation at line 41 and window cuts at lines 57/59.
`configure()` changes module variables only inside the converter process;
it does not edit the shared SNDSW installation. Use the launcher flags to change
windows, rather than editing that installed file.

## Check what was applied

The production `plan.json` contains the requested settings. Each newly converted
feature/Hit3D ROOT file also contains a `TObjString` named `scifi_timing_json`:

- `requested`: separate data and MC windows.
- `effective`: applied offsets, period, sample policy, selector, first-arrival
  flag, additional maximum time, and selector source path/SHA256.

Read it in a configured PyROOT environment:

```python
import json
import ROOT

f = ROOT.TFile.Open("path/to/features.root")
record = json.loads(str(f.Get("scifi_timing_json").GetString().Data()))
print(json.dumps(record, indent=2))
f.Close()
```

Validation receipts also record timing; validation checks the configured policy
and rejects feature/Hit3D mismatches. Shard merging rejects inconsistent timing
records. Older files may lack this metadata. See [data format](DATA_FORMAT.md#scifi-timing-provenance).

The implementation passed unit checks and small real-source batch comparisons;
those previews do not establish the best physics window. See
[validation results](TEST_RESULTS.md) for evidence and limits.
