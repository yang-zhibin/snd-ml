# SciFi hit density: calculation and code reference

Verified from source on **16 November 2026**. Intended for collaborators reading
or extending the shower features.

SciFi density measures how closely hits are packed **within one station and one
view**. It is the sum of nearby-hit counts, not charge density. QDC does not
enter the calculation. Positions are in **cm**; the result is a **dimensionless
count**, not hits/cm.

## 1. Where is the code?

| Implementation | File and function | Purpose |
| --- | --- | --- |
| Local Python | [digi_2_features.py](../convertData/digi_2_features.py), `hitWeightDensity()` (line 1043) | Calculates density separately for each station/view. |
| Local Python | Same file, `fill_mycode_density()` (line 1007) | Chooses the best/second station and fills event summaries. |
| SNDSW C++ | `analysis/tools/sndSciFiTools.cxx`, `hitWeightComputation()` (line 635) | Sliding-window neighbour sum reproduced by the local Python helper. |
| SNDSW C++ | Same file, `hitDensityPerStation()` (line 706) | Returns the two view densities for a station. |
| SNDSW Python | `analysis/analyses/snd_analysis_2024_0mu/sciFiTools.py`, `getSumDensity()` (line 72) | Calculates view densities, ranks stations and returns event summaries. |

**The converter calls SNDSW Python, not the C++ helper**, for its
`density_sndsw_scifi` branches (local converter line 1355). It also calls the
local calculation (line 1363) and local summary (line 1403).

SNDSW locations inspected on CERN AFS:

```text
C++ source root:
/afs/cern.ch/work/z/zhibin/public/SndBuild/sndsw/

Configured Python release root:
/afs/cern.ch/work/z/zhibin/public/SndBuild/sw/rhel9_x86-64/sndsw/master-local14/
```

The release is selected by [env_sndsw_nue.sh](../snakemake/envs/env_sndsw_nue.sh):
`latest-event_level_info_output-release` resolved to `master-local14` at the
verification date. These external locations are machine-specific, mutable
references. Line numbers are navigation hints for this snapshot.
An older local copy also exists in
[digi_2_nueAnalysis.py](../convertData/digi_2_nueAnalysis.py), `hitWeightDensity()`
(line 247).

## 2. Calculate one view's density

1. Group hits by station (1–5) and orientation.
2. Use **x** for vertical hits and **y** for horizontal hits.
3. For each hit, count other hits within 1 cm along that coordinate.
4. Sum those counts.

For the local/C++ rule:

```text
weight(i) = number of j != i for which |position[j] - position[i]| <= 1 cm
D         = sum of weight(i) over the hits in this station/view
```

Example, away from the boundary where implementations differ:

| Hit position (cm) | Neighbours (cm) | Weight |
| ---: | --- | ---: |
| 0.0 | 0.4 | 1 |
| 0.4 | 0.0, 1.2 | 2 |
| 1.2 | 0.4 | 1 |

**Density = 1 + 2 + 1 = 4.** Each neighbouring pair contributes twice.
Empty or single-hit views give zero. If all N hits are mutually within the
window, the result is N × (N − 1).

The local and C++ implementations sort positions and move two pointers around
each hit. The window contains `right - left` hits; subtract one to exclude
itself. Sorting costs O(N log N), followed by an O(N) scan.

## 3. Calculate the event summary

For station s, let `Dx[s]` and `Dy[s]` be the two view densities:

```text
station score[s] = Dx[s] + Dy[s]
best station     = station with the largest score
reported density = min(Dx[best], Dy[best])
```

| Station | Dx | Dy | Score used to select station | Minimum |
| --- | ---: | ---: | ---: | ---: |
| 1 | 20 | 4 | 24 | 4 |
| 2 | 10 | 10 | 20 | 10 |

Station 1 wins, so **the reported density is 4**, not 10. The station is selected
by the sum; the returned value is its smaller view density. A station populated
in only one view can therefore win the ranking but return zero.

Both summary implementations visit stations in order 1–5. The first maximum
wins a tie. The second station uses the largest strictly smaller positive score;
a tied maximum is not second. Its output is again the minimum of its two views.
If there is no second candidate, the second-density output is zero.

## 4. Which output branch should I read?

| Branch | Meaning |
| --- | --- |
| `density_scifiN_x`, `density_scifiN_y` | Local Dx/Dy for station N, N = 1–5. |
| `density_scifiN` | Local Dx + Dy for station N. |
| `density_scifi_x`, `density_scifi_y` | Local view sums over all five stations. |
| `density_scifi` | Local sum over every station and both views. |
| `density_mycode_scifi` | Local minimum view density of the best station. |
| `density_mycode_scifi_second` | Local minimum view density of the second station. |
| `density_mycode_scifi_ver`, `density_mycode_scifi_hor` | Local Dx/Dy of the best station. |
| `density_sndsw_scifi` and its `_second`, `_ver`, `_hor` branches | Corresponding summaries from SNDSW Python. |

`density_mycodescifi_second` is an older spelling retained as an alias of
`density_mycode_scifi_second`. Do not confuse the all-station sum
`density_scifi` with the best-station summary `density_mycode_scifi`.

## 5. Differences that affect comparison

| Detail | Local Python | SNDSW C++ | SNDSW Python |
| --- | --- | --- | --- |
| Neighbour distance | ≤ 1 cm | ≤ 1 cm | < 1 cm |
| Coordinate | Endpoint A | Midpoint of A and B | Midpoint of A and B |
| Counting method | Sliding window | Sliding window | Scan neighbours in sorted positions |

Two hits at exactly **0 and 1 cm** give density **2** locally/in C++, but **0**
in SNDSW Python. Midpoint and endpoint coordinates coincide in the measured
coordinate when the endpoints share that coordinate; do not assume this for
all geometries/alignment configurations.

The comparison above describes the converter's call with `digi=True`,
`return_position=False`, no x/y limits, and second/highest-view outputs enabled.
Other modes of `getSumDensity()` are outside this report. In particular,
`return_position=True` changes self-counting for position weighting.

For a numerical comparison, use identical selected hits and geometry. The
converter passes raw objects and local records from its selected-hit entries to
the two paths. The C++ station wrapper independently skips invalid hits. The
local helper defaults missing x/y to zero; callers should supply coordinates
rather than relying on that default.

## 6. Minimal calculation you can run

This small reference uses direct pair counting for clarity, not production
performance. It demonstrates both boundary conventions:

```python
def view_density(positions_cm, *, inclusive=True):
    """Count directed neighbouring pairs within 1 cm; exclude self."""
    return sum(
        (abs(a - b) <= 1.0 if inclusive else abs(a - b) < 1.0)
        for i, a in enumerate(positions_cm)
        for j, b in enumerate(positions_cm)
        if i != j
    )

assert view_density([0.0, 0.4, 1.2]) == 4
assert view_density([0.0, 1.0]) == 2
assert view_density([0.0, 1.0], inclusive=False) == 0
assert view_density([]) == 0
assert view_density([0.4]) == 0
```

For development-plugin hits, use `x_cm` / `y_cm`; the converter's internal
records use `x` / `y`. Check that these optional exported fields exist before
calculating density. See the [plugin field reference](README.md#reading-the-values-passed-to-your-plugin).

**Validation scope:** code inspection and worked-example checks. This report
does not establish compiled-library provenance or event-by-event equality of
the local and SNDSW density branches. No density implementation was changed.
