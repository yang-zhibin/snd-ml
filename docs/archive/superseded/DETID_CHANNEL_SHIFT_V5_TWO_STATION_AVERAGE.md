# V5 two-station-average detector-ID transform (superseded)

> This is the historical first-two-active-station specification. It was
> replaced on 2026-09-23 because two points were too sensitive to longitudinal
> shower-occupancy changes. Current v5 products use the all-active-plane fit
> documented in `DETID_CHANNEL_SHIFT_V5_ALL_PLANE_AVERAGE.md`.

Status: implemented and technically validated on ten Run 8329 events on
2026-09-23; physics equivalence to the frozen-track v4 method is not established.

## Purpose

V5 tests a reconstruction-free alternative to the accepted v4 detector-ID
channel transform. It derives an activity line from raw SciFi digi positions,
then applies the same one-axis, proximity-constrained detector-ID mapping used
by v4. The accepted v4 implementation and products are not modified.

The ten-event output compares three official SNDSW displays on every page:

1. original raw digis;
2. the accepted frozen-SciFi-track v4 transform;
3. the v5 first-two-station-average transform.

No display-time Simple or Hough tracking runs. The original column overlays the
frozen reconstructed line in blue and the two-station activity line as a dashed
magenta line. The v4 and v5 columns overlay their corresponding translated
frozen lines.

## Reconstruction-free line definition

Travel is explicitly assumed to be toward positive z. Scan SciFi stations 1
through 5 and call a station active only when it contains at least one vertical
and one horizontal raw digi. Select the first two active stations `i` and `j`.

For each selected station, enumerate real detector IDs through initialized
SNDSW geometry. Vertical channels contribute their channel-centre x coordinate;
horizontal channels contribute their channel-centre y coordinate. Compute the
unweighted arithmetic averages

```text
x_i = mean(x positions of vertical digis in station i)
y_i = mean(y positions of horizontal digis in station i)
x_j = mean(x positions of vertical digis in station j)
y_j = mean(y positions of horizontal digis in station j)
```

This matches the arithmetic convention of the existing
`avg_scifiN_x`/`avg_scifiN_y` features. Duplicate detector IDs, non-finite
coordinates, or fewer than two stations active in both views are errors.

Use the exact view-plane z coordinates from geometry; the x- and y-measuring
planes in one station need not have identical z:

```text
sx = (x_j - x_i) / (z_xj - z_xi)
sy = (y_j - y_i) / (z_yj - z_yi)

x(z) = x_i + sx * (z - z_xi)
y(z) = y_i + sy * (z - z_yi)
```

The stored common reference z is the midpoint of the first station's x- and
y-plane z coordinates; both projection lines are evaluated at that z. This is
only a representation change and does not alter either fitted slope.

This object is an activity-centre line, not a reconstructed particle track.
V5 never reads reconstructed momentum or track measurements when creating the
simple transform. Its direction audit must be `assumed_positive_z`.

## Transformation policy

Choose x when `abs(sx) > abs(sy)` and y for the converse. V5 uses a minimum
dominance ratio of 1.0, so only an exact absolute-slope tie is rejected. It then
uses the accepted v4 rules without modification:

- transform exactly one coordinate;
- at the earliest selected-view SciFi plane with hits, project the activity
  line to the exact plane z;
- when the projected coordinate is within an inclusive 5.0 cm of its closest
  active edge-channel centre, target that closest edge;
- otherwise choose the entry edge from the selected slope and the assumed
  positive-z direction;
- derive one SciFi ordinal channel offset by mapping an occupied reference hit
  to the selected real edge channel;
- reuse that offset for all selected-view SciFi stations;
- independently quantize the same requested physical displacement for every
  compatible MuFilter plane;
- copy all orthogonal detector IDs and payloads unchanged;
- drop and audit destinations beyond an edge; never clamp, wrap, merge, or
  populate a hardware gap.

The x/y detector-view compatibility, gap policy, synthetic DAQ warning, output
branch schema, and requirement to rerun all downstream reconstruction/features
are identical to v4.

## Validation

The independent validator reconstructs the station/view arithmetic averages
from every SciFi row in `mapping_audit.csv`, selects the first two common active
stations again, and recomputes the line, dominant axis, target edge, requested
shift, detector/view preservation, mapping equations, count closure, payload
equality, and ROOT contents.

The ROOT-independent suite contains 25 tests, including average calculation,
positive/negative travel order, and missing-view rejection. In the full sample:

- all 10 v4 track transforms passed;
- all 10 v5 simple transforms passed;
- all 10 three-way official displays completed;
- all 53 Snakemake jobs completed;
- the composed PDF contains 10 pages with three equal-size columns and no
  scaling of an individual display.

Technical validation proves that the specified transformation was executed
correctly. It does not prove that the two-station activity line estimates the
same physical trajectory as reconstruction.

## Ten-event comparison result

All ten events used stations 1 and 2 as their first two common active stations.
Relative to the accepted frozen-track v4 decision:

- selected-axis agreement: 4/10;
- target-edge agreement: 5/10;
- simultaneous selected-axis and target-edge agreement: 2/10;
- total dropped SciFi digis: 3,613 for v4 and 3,844 for v5;
- none of the ten v5 events activated the 5 cm closest-edge override.

Several two-station averages have much larger slopes than the reconstructed
track because shower occupancy changes strongly between stations 1 and 2. V5
is therefore a working reconstruction-free event-shape transformation, but the
current ten-event result does not support treating it as an interchangeable
replacement for the v4 muon-track transform without further physics review.

## Files and command

Configuration:

```text
snakemake/metadata/configs/region_evaluation/
passing_muon_detid_channel_shift_v5.yaml
```

Workflow:

```text
snakemake/rules/passing_muon_detid_channel_shift_v5.smk
```

Products:

```text
evaluation_region_partitions/passing_muon_detid_channel_shift_v5/
  selected_events.csv
  rejected_events.csv
  three_way_comparison.pdf
  three_way_audit.json
  events/event_<eventId>/
    track_transform/{transformed.root,mapping_audit.csv,event_audit.json,validation.ok}
    simple_transform/{transformed.root,mapping_audit.csv,event_audit.json,validation.ok}
    comparison_three_way.pdf
```

Reproduce from `snakemake/`:

```bash
source /afs/cern.ch/user/z/zhibin/env_snakemake.sh
snakemake -s snakefile passing_muon_detid_channel_shift_v5 \
  --cores 2 --printshellcmds
```
