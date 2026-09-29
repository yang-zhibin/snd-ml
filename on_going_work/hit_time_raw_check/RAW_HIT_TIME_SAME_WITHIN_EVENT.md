# Raw Hit Time Is Identical Within Each Event

## Context

This check was done after adding raw per-hit timing vectors to
`convertData/digi_2_features.py`:

- `hitTime_raw_ns`
- `hitTime_raw_detType`

The goal was to verify whether the `MC_muonDIS` digi files contain useful
per-hit timing information before applying first-arrival calibration and the
25 ns hit-time filter.

## Checked Files

```text
/eos/experiment/sndlhc/users/zhibin/MC_muonDIS/cvilela/muonDIS_2001/feature_MC_muonDIS_cvilela_muonDIS_2001_0.root
/eos/experiment/sndlhc/users/zhibin/MC_muonDIS/cvilela/muonDIS_2002/feature_MC_muonDIS_cvilela_muonDIS_2002_1.root
```

Plots and summary CSV were produced with:

```text
on_going_work/hit_time_raw_check/plot_raw_hit_time.py
```

Output directory:

```text
on_going_work/hit_time_raw_check/plots/
```

## Observation

For every checked event, all saved raw hit times are identical within that
event.

Summary from `raw_hit_time_summary.csv`:

| sample | entries | events with times | same-time events | same-time fraction | raw hit times | event range min/max |
|---|---:|---:|---:|---:|---:|---|
| `feature_MC_muonDIS_cvilela_muonDIS_2001_0` | 472 | 472 | 472 | 1.0 | 54270 | `0.0 / 0.0 ns` |
| `feature_MC_muonDIS_cvilela_muonDIS_2002_1` | 505 | 505 | 505 | 1.0 | 50837 | `0.0 / 0.0 ns` |

Example debug output from the feature converter:

```text
raw_range_ns:        (1396390887424.0, 1396390887424.0)
calibrated_range_ns: (0.0, 0.0)
kept_range_ns:       (0.0, 0.0)
```

The raw values vary between events, but within one event every hit has the
same raw timestamp.

## Interpretation

For these `MC_muonDIS` files, `Digi_ScifiHits.GetTime()` and the MuFilter
bar-level time used by the converter appear to behave like an event-level
timestamp, not a per-hit detector arrival time.

Therefore, after first-arrival calibration:

```text
calibrated_hit_time_ns = hit_time_ns - earliest_hit_time_ns
```

all hits in the event become:

```text
0 ns
```

This explains why:

- `hitTime_min_ns == 0`
- `hitTime_max_ns == 0`
- `hitTime_range_ns == 0`
- no hits are removed by the 25 ns filter

The 25 ns filter is working logically, but it has no discriminating power on
these samples because all hits already have the same event-level raw time.

## Consequence For Current Feature Files

The current timing branches are internally consistent, but they should not be
interpreted as measuring intra-event hit timing for these `MC_muonDIS`
samples.

For these files:

```text
hitTime_raw_ns
```

is useful for confirming the event timestamp behavior, but not for studying
hit-by-hit arrival time spread.

## Possible Next Checks

1. Check how the `cvilela` `MC_muonDIS` digi files were produced.
2. Inspect whether per-hit timing exists in linked MC points rather than digi
   hit times.
3. Compare against EventBuilder-produced `newDigi` samples.
4. Compare against another MC sample where digitized hit times are expected to
   contain detector arrival timing.

## Relevant Files

```text
convertData/digi_2_features.py
on_going_work/hit_time_raw_check/plot_raw_hit_time.py
on_going_work/hit_time_raw_check/plots/raw_hit_time_summary.csv
```
