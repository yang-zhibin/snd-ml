# V5 all-plane-average detector-ID transform

Status: implemented and technically validated on ten Run 8329 events on
2026-09-23. Physics equivalence to the frozen-track v4 method is not assumed.

## Purpose and comparison

V5 is a reconstruction-free alternative to the accepted v4 transform. It
derives an activity line from raw SciFi digis and then applies the same
one-axis, proximity-constrained detector-ID mapping as v4. It never reads a
reconstructed track or reconstructed momentum to make the simple-transform
decision. The accepted v4 implementation is unchanged.

Each page of `three_way_comparison.pdf` contains equal-size official SNDSW
displays:

1. original raw digis, with the frozen reconstructed SciFi track in solid blue
   and the all-plane activity fit in dashed magenta;
2. the accepted frozen-track v4 transform and its translated blue line;
3. the all-plane v5 transform and its translated magenta line.

Display-time Simple and Hough reconstruction are disabled.

## Activity-line definition

Travel is explicitly assumed to be toward positive z. For each nonempty SciFi
station/orientation plane, enumerate the real detector IDs through initialized
SNDSW geometry and obtain each fired channel centre. View 1 contributes x and
view 0 contributes y. For plane p, form one unweighted digi-position average:

```text
c_p = sum(channel-centre coordinates in p) / number of digis in p
```

This matches the arithmetic convention of the existing `avg_scifiN_x/y`
features. The x and y projections are independent: a missing y plane does not
discard an active x plane at the same station. Each projection needs at least
two active planes with distinct geometry z coordinates.

Fit every active plane-average point with equal weight using ordinary least
squares:

```text
s = sum((z_p - mean(z)) * (c_p - mean(c)))
    / sum((z_p - mean(z))^2)
c(z) = mean(c) + s * (z - mean(z))
```

Plane averages are equally weighted regardless of their hit counts. This
prevents a high-occupancy shower plane from dominating solely because it has
more digis. The stored common reference z is halfway between the mean x-plane
z and mean y-plane z; each fitted projection is evaluated there. This changes
only the line representation, not either slope.

The audit stores every input point (`station`, `view`, `z_cm`, `coordinate_cm`,
and `n_hits`), both slopes, and the common reference. It records the source as
`all_active_scifi_plane_average_linear_fit` and direction as assumed positive
z.

## Transformation policy

Choose x when `abs(slope_xz) > abs(slope_yz)` and y for the converse; an exact
tie is rejected. Then reuse the accepted v4 policy:

- move exactly one coordinate and preserve the orthogonal detector IDs and
  payloads unchanged;
- project the fitted line to the earliest hit-bearing SciFi plane of the
  selected view;
- if that coordinate is within an inclusive 5 cm of its nearest active edge
  channel, target that edge; otherwise use the slope and positive-z direction
  to select the entry edge;
- derive one SciFi ordinal-channel offset and reuse it in all selected-view
  SciFi stations;
- independently quantize the same physical displacement in compatible
  MuFilter planes;
- map only to real destination detector IDs, preserving hardware gaps through
  geometry ordering;
- drop and audit destinations outside an edge; never clamp, wrap, merge, or
  create a channel inside a hardware gap.

All downstream reconstruction/features must be rerun on the transformed raw
digi file.

## Validation and ten-event result

The independent validator reconstructs every plane average from
`mapping_audit.csv`, refits both projections, and reproduces the dominant axis,
edge decision, requested translation, detector/view preservation, ordinal
mapping, count closure, payload equality, and ROOT contents.

Validation completed successfully:

- 28 ROOT-independent synthetic tests passed, including all-point OLS,
  unequal active-plane sets in x and y, and insufficient-plane rejection;
- all 10 v4 track transforms and all 10 v5 all-plane transforms passed;
- all 10 official three-column displays and all 53 Snakemake jobs passed;
- all ten selected events had five active x planes and five active y planes;
- the final PDF has 10 pages, each 2721.6 x 567 points.

Relative to the frozen-track v4 decisions:

- selected-axis agreement: 7/10;
- target-edge agreement: 6/10;
- simultaneous axis and target-edge agreement: 3/10;
- total dropped SciFi digis: 3,613 for v4 and 3,564 for all-plane v5;
- the v5 5 cm closest-edge override activated in 2/10 events.

This is better agreement than the superseded first-two-station prototype
(axis 4/10, edge 5/10, both 2/10, and 3,844 v5 SciFi drops). Technical
validation proves that the specified all-plane transformation was executed
correctly; it does not prove that the activity-centre line is a reconstructed
muon trajectory or that the two methods are interchangeable for physics.

## Files and command

```text
snakemake/metadata/configs/region_evaluation/
  passing_muon_detid_channel_shift_v5.yaml
snakemake/rules/passing_muon_detid_channel_shift_v5.smk
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
