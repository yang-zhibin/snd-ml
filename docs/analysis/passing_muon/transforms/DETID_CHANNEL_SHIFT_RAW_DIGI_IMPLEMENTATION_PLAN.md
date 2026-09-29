# Detector-ID channel-shift raw-digi method and implementation record

Status: proximity-constrained, single-axis v4 implemented, validated, and
accepted for use on 2026-09-23; v3 and the earlier two-axis v2 products remain
preserved as separate workflows
Created: 2026-09-22
Initial sample: Run 8329 veto-tagged Muon-DIS candidates
Supersedes: `SCIFI_EDGE_TRANSFORM_VALIDATION_IMPLEMENTATION_PLAN.md`

Canonical analysis documentation:
`PASSING_MUON_DETID_TRANSFORM_METHOD.md`. This file retains implementation
history and development details; use the canonical document when applying or
citing the accepted v4 method.

## Objective

Create a new SNDSW-format digi ROOT file in which the detector occupancy is
shifted toward a selected detector edge by remapping real detector IDs. The
output must contain ordinary `sndScifiHit` and `MuFilterHit` objects attached to
real destination channels, so the unmodified SNDSW display and downstream digi
workflow can read it.

This is a discrete channel transformation, not the earlier continuous-position
overlay. The original file is read-only. Reconstructed tracks, clusters, and
features from the original detector IDs are not valid for the transformed file
and must be regenerated if needed.

## Accepted v4 method: authoritative specification

This section is the concise specification of the accepted method. Later v2 and
v3 sections are retained as design history and must not override this section.
The canonical parameters are in
`snakemake/metadata/configs/region_evaluation/passing_muon_detid_channel_shift_v4.yaml`.

### Input event and track selection

The validation sample starts from the Run 8329, veto-tagged, passing-muon
topology-failure population in the frozen v7 event manifest. A separate
`sndData` track-input file supplies reconstructed-track measurements. Tracks
must have a true fit flag, at least one degree of freedom, finite non-negative
chi2/NDF, and fitted SciFi points spanning at least two distinct z planes in
both x-z and y-z. Enabled track types are considered in priority order
`[11, 1, 15]`, then by lower chi2/NDF and collection index. Type 11 is the
preferred Hough SciFi track. The exact selected track index, type, fitted
points, slopes, momentum, source file, and tree entry are frozen in
`selected_events.csv`; the raw-digi writer does not silently select a different
track.

### Per-event transformation

For each accepted event, apply these steps in order:

1. Fit straight lines `x(z)` and `y(z)` to the selected track's SciFi fitted
   points. Select x when `abs(dx/dz) > abs(dy/dz)`, otherwise select y. Require
   a dominance ratio of at least 1.2; reject exact ties or ambiguous cases.
   Exactly one axis is selected and transformed.
2. Obtain the travel direction from the sign of the track momentum `p_z`. If a
   finite non-zero `p_z` is unavailable, explicitly assume positive z as
   configured and record that fallback in the audit.
3. In travel order, find the first SciFi station containing hits in the view
   that measures the selected axis. Project the track to that plane's geometry
   z coordinate. Its first and last active channel centres define the low and
   high edges.
4. Compute the signed-slope entry side. If the projected track coordinate is
   within the inclusive 5.0 cm threshold of its closest active edge-channel
   centre, override the signed-slope side with that closest edge. Otherwise use
   the signed-slope entry side. A distance tie deterministically selects low.
5. Compute one requested physical displacement from the projected track to the
   selected edge-channel centre. In the reference SciFi plane, choose the
   occupied source anchor whose translated centre is closest to the target and
   map that anchor exactly to the edge channel. The resulting signed ordinal
   channel offset is reused for all five SciFi stations in the selected view.
6. Remap only detector views that measure the selected coordinate. For x these
   are vertical SciFi, vertical Veto, and vertical DS. For y these are
   horizontal SciFi, horizontal Veto, all US, and horizontal DS. Every
   orthogonal hit is copied with the identical detector ID, payload, and
   collection order.
7. Because MuFilter planes have different channel counts and pitches, each
   compatible plane independently quantizes the same requested physical
   displacement to its own valid ordered-bar offset. A SciFi channel offset is
   never reused as a MuFilter bar offset.
8. Copy each retained digi and change only its detector ID. Destination
   channels are real SNDSW geometry channels ordered by measured position, not
   numerical detector ID. Hits whose ordinal destinations fall beyond a
   detector edge are dropped and counted; hits are never clamped, wrapped,
   merged, or placed in a physical gap.

Empty channels in the original pattern remain empty because every source hit
receives the same ordinal offset within its applicable layout. Die, 128-channel
SiPM-array, mat, and bar gaps remain fixed gaps of the destination hardware.
This is consequently a channel-space translation, not an exactly rigid shift
in centimetres.

### Output meaning and required downstream treatment

The output is a new SNDSW-readable ROOT file containing a transformed
`Digi_ScifiHits` collection and transformed `Digi_MuFilterHits` collection,
with the event identity and approved raw metadata retained. The input ROOT file
is never modified. Original reconstructed tracks, clusters, Hit3D objects, and
features describe the old detector IDs and must not be reused. Reconstruction,
Hit3D building, feature generation, and selection must be rerun from the new
digi file before the transformed events enter a background estimate.

The digi signal, time, validity, mask, and other copied payload fields are
preserved. The original DAQ identifier is also retained while the detector ID
changes, so this is a synthetic detector-level sample and not an
electronics-consistent raw-data emulation. Every file and JSON audit carries a
`synthetic_daq_warning`; it must not be used for studies that depend on the
destination channel's electronics address without a separately validated DAQ
remapping.

### Validation and visual review

Every event must produce all of the following before it is accepted:

- `transformed.root`: reopened successfully with valid SNDSW branch classes;
- `mapping_audit.csv`: one row per input SciFi or MuFilter digi, including
  source/destination IDs, ordinals, action, offset, and realized displacement;
- `event_audit.json`: the complete track, axis, edge, reference-plane, mapping,
  count-closure, geometry, and DAQ-warning decisions;
- `validation.ok`: proof that the independent validator recomputed the policy,
  checked detector/view preservation, payload equality, injectivity, overflow
  accounting, orthogonal-view identity, and output ROOT contents;
- `comparison.pdf`: original and transformed events rendered independently by
  the official SNDSW display.

The display uses `withTrack=0`, `withHoughTrack=2`, and `minimumTracks=0`.
Therefore its blue track is a new display-time SciFi-only Hough fit, not a
copied input track. A page remains valid when this optional display fit does not
converge and the blue line is absent; transformation validity comes from the
numerical validator, not from display-track availability.

### Reproducible workflow targets

Run from `snakemake/` after loading the Snakemake environment:

```bash
source /afs/cern.ch/user/z/zhibin/env_snakemake.sh

# Reviewed ten-event sample with feature-level SciFi occupancy >= 500.
snakemake -s snakefile \
  passing_muon_detid_channel_shift_v4_high_density \
  --cores 2 --printshellcmds

# Extended 100-event density-ranked visual-review sample.
snakemake -s snakefile \
  passing_muon_detid_channel_shift_v4_display100 \
  --cores 2 --printshellcmds
```

The reviewed ten-event products are under
`evaluation_region_partitions/passing_muon_detid_channel_shift_v4/high_density_sample/`.
The 100-event products are isolated under
`evaluation_region_partitions/passing_muon_detid_channel_shift_v4/display_sample_100/`.
For the latter, events are still ranked by descending SciFi occupancy, but the
hard 500-hit cutoff is disabled because only 85 manifest candidates meet it
before track-quality rejection. The completed sample selected 100 usable events
after scanning 151 candidates and spans 1,717 to 326 feature-level SciFi hits.

The present workflow is a validated Run 8329 sample-production workflow. A
future full signal-region production should reuse this transformation and
validator but add a dataset-scale manifest and batch target; it must not bypass
per-event validation or treat the 100-event display sample as the complete
background dataset.

## Policy revision for v3

The v2 sample moved both measured coordinates and selected the nearest edge in
each view. That is no longer the intended physics model. Version 3 chooses
exactly one transverse axis from the SciFi track direction and moves only hits
whose readout orientation measures that coordinate. It chooses the entry edge
from the signed track direction, not from the distance to an edge.

Existing files under
`evaluation_region_partitions/passing_muon_detid_channel_shift_v2/` remain a
useful test of detector-ID remapping and ROOT I/O, but they are products of the
superseded two-axis policy and must not be used to validate the v3 background
model. New products go in
`evaluation_region_partitions/passing_muon_detid_channel_shift_v3/`; v2 files
must not be overwritten.

## Near-edge direction constraint for v4

Version 4 retains the v3 rule that exactly one coordinate is transformed. The
selected axis is still the projection with the larger absolute SciFi-track
slope. The v4 change affects only which edge is the translation target.

At the earliest selected-view SciFi plane with hits, project the chosen track
and measure its distance from the centres of the first and last active SciFi
channels. These are the same edge-channel centres used as discrete mapping
destinations. With the configured threshold `near_edge_threshold_cm: 5.0`, use

```text
closest_side = low if distance_to_low <= distance_to_high else high

if min(distance_to_low, distance_to_high) <= near_edge_threshold_cm:
    target_side = closest_side
    edge_selection_policy = nearest_edge_within_threshold
else:
    target_side = slope_entry_side
    edge_selection_policy = slope_entry_side
```

The threshold is inclusive. A deterministic tie selects the low edge. The
signed-slope side remains in the audit as `slope_entry_side`; it is not
overwritten when the proximity rule activates. This distinction is necessary
because a near-edge override is intended to preserve the already recorded part
of a shower rather than assert that the fitted track enters from that edge.

The requested physical shift is

```text
requested_shift = target_edge_channel_centre - track_coordinate(z_reference)
```

That one shift is applied only to the selected coordinate across SciFi and the
compatible Veto/US/DS views. The orthogonal coordinate is copied unchanged.
The existing v3 constant-offset MuFilter mapping, real-channel gap handling,
payload preservation, and boundary dropping are unchanged.

Every v4 audit records both edge distances, the threshold, closest edge,
slope-derived edge, final target edge, whether the override activated, and the
requested shift. New products are written under
`evaluation_region_partitions/passing_muon_detid_channel_shift_v4/`; neither v3
code products nor v3 output files are relabelled as v4.

## Agreed transformation definition

Treat every plane/view as an ordered sequence of real readout channels. Sort by
the measured coordinate obtained from initialized SNDSW geometry:

- vertical channel or bar: order by x;
- horizontal channel or bar: order by y.

Do not order by the numerical detector ID: mat and SiPM encodings are not a
guaranteed physical coordinate order.

For an orientation, choose a source anchor channel and the destination edge
channel. Define a signed ordinal offset:

```text
channel_offset = destination_edge_index - source_anchor_index
destination_index = source_index + channel_offset
```

Apply that offset to every hit in the applicable channel layout. Copy the
original digi object, replace its detector ID with the destination detector ID,
and omit the original detector ID from the output.

The sequence is the complete detector-channel sequence, not the sequence of
observed hits. For example, source hit indices `[100, 101, 104]` shifted by
`-90` become `[10, 11, 14]`, never `[10, 11, 12]`. This preserves empty
electronic channels in the occupancy pattern.

## Gap policy

The ordered destination sequence contains real channels only. Physical gaps
between the two SciFi dies, adjacent 128-channel SiPM arrays, and adjacent mats
have no detector ID and are never assigned a hit. The destination detector's
fixed gaps therefore appear naturally when SNDSW draws the remapped detector
IDs.

This policy deliberately shifts occupancy in channel space:

- retained hits always have valid detector IDs;
- no hit is snapped to a fictional gap location;
- a hit is dropped only when its destination ordinal is outside the first or
  last channel;
- the source gap phase need not coincide with the destination gap phase.

Consequently, a single channel offset is not an exact rigid displacement in
centimetres. Crossing a die, SiPM, or mat boundary changes the physical spacing.
For every mapping, record both the requested/anchor displacement and the actual
centre-to-centre displacement. Never label the result an exact spatial
translation.

## Slope-selected axis and entry-side contract

Let the fitted SciFi track be

```text
x(z) = x0 + sx * (z - z0), where sx = dx/dz
y(z) = y0 + sy * (z - z0), where sy = dy/dz
```

Choose one and only one transform axis:

```text
selected_axis = x  if abs(sx) > abs(sy)
selected_axis = y  if abs(sy) > abs(sx)
```

Reject rather than guess when the dominant absolute slope is below a configured
minimum or when the two absolute slopes are too similar. The production
configuration therefore includes `min_abs_slope` and
`min_axis_dominance_ratio`; the initial ratio will be 1.2 and will be retained
in the audit. Exact ties are always rejected.

A geometrical fitted line has no travel direction. Obtain `pz_sign` from the
selected reconstructed track momentum when a finite non-zero `p_z` is stored.
If that quantity is unavailable, use an explicit configuration value
`assumed_pz_sign` (initially `+1` for downstream-going Run 8329 tracks) and
record `direction_source=assumed_positive_z`. Never infer travel direction
silently from the slope alone.

For the selected coordinate, define

```text
travel_slope = selected_slope * pz_sign
entry_side   = low  if travel_slope > 0
entry_side   = high if travel_slope < 0
shift_sign   = negative for the low edge, positive for the high edge
```

This makes the transformed track start at the selected side and move inward as
it travels through increasing path length. If the fitted track has `sx > 0`
and is downstream-going, for example, x is moved to the low-x edge; it then
increases toward the detector interior.

Use the earliest SciFi plane along the track travel direction that has valid
hits in the selected readout view as the reference plane. Project the fitted
track to that plane's exact geometry z coordinate. The continuous requested
translation is

```text
requested_shift = entry_edge_channel_centre - track_coordinate(z_reference)
```

Inspect the real hits of that reference plane and select the source anchor whose
translated centre is closest to the entry-edge channel centre. Map that anchor
exactly onto the edge channel and use the resulting signed ordinal offset for
every SciFi plane of the selected view. Reject the event if no reference-plane
hit exists. Explicit source detector IDs remain an allowed debugging override.

Only readout views measuring the selected coordinate are remapped:

| Selected axis | Remapped hits | Hits copied unchanged |
|---|---|---|
| x | vertical SciFi, vertical Veto, vertical DS | horizontal SciFi, horizontal Veto, all US, horizontal DS |
| y | horizontal SciFi, horizontal Veto, all US, horizontal DS | vertical SciFi, vertical Veto, vertical DS |

US contains horizontal bars, so it can represent a y displacement but cannot
represent an x displacement. Leaving an orthogonal view unchanged is required,
not a fallback.

SciFi planes share the same 1,536-channel layout, so one ordinal offset is used
across stations for the selected view. MuFilter systems have different bar
counts and pitches. Each applicable MuFilter plane/view uses the same requested
physical displacement and entry side, but chooses and audits its own anchor and
ordinal offset. This avoids treating a SciFi channel offset as a MuFilter bar
offset.

## Geometry model

Add a ROOT-independent model:

```text
PlaneKey(detector, system, plane, view)
ChannelAddress(det_id, plane_key, ordinal, centre_cm, low_cm, high_cm,
               endpoint_a, endpoint_b, hierarchy)
ChannelLayout(plane_key, channels_in_coordinate_order, det_id_to_ordinal)
ChannelShift(side, anchor_det_id, edge_det_id, offset)
MappedHit(source_det_id, destination_det_id, source_ordinal,
          destination_ordinal, source_centre_cm, destination_centre_cm,
          realized_shift_cm, status)
```

Build SciFi layouts by enumerating every
`(station, view, mat, SiPM, channel)` detector ID and calling
`Scifi.GetSiPMPosition()`. Initialize `Scifi` with the event header before
geometry access. Validate:

- exactly 1,536 channels per station/view;
- detector-ID encode/decode round trips;
- unique detector IDs and ordinals;
- strict monotonic centre order;
- expected die, SiPM, and mat boundary discontinuities;
- identical channel count and hierarchy pattern across compatible SciFi planes.

Build future MuFilter layouts from `MuFilter.GetPosition()` and configuration
bar counts. Keep Veto, US, DS, and horizontal/vertical layout families separate.

## Pure channel mapper

Create `evaluation_region_partitions/detid_channel_shift.py` with no ROOT
dependency. It will:

- validate channel layouts;
- select/validate explicit and automatic anchors;
- calculate signed channel offsets;
- map source detector IDs to destination detector IDs;
- classify `mapped`, `outside_low`, and `outside_high`;
- prove one-to-one destination mapping;
- calculate realized physical shifts and their spread;
- reject duplicate source IDs and destination collisions.

The mapper must not merge, clamp, wrap, or silently overwrite hits.

## SNDSW geometry and digi adapter

Create `evaluation_region_partitions/detid_channel_shift_sndsw.py`. It will:

- load the Run 8329 geofile once;
- initialize geometry for every event header;
- build/cache compatible channel layouts;
- extract hit detector IDs and validity;
- clone `sndScifiHit` and `MuFilterHit` objects with their C++ copy
  constructors;
- call `SetDetectorID(destination_det_id)` on each copied hit;
- construct output `TClonesArray` branches;
- verify that every output detector ID resolves through SNDSW geometry.

The copy constructor preserves signal, time, validity, masks, and DAQ fields.
Changing the detector ID while retaining the original DAQ address makes this a
synthetic digi product, not genuine raw detector data. Record that fact in file
metadata and audit all `fDaqID` values. Do not claim electronics-level
consistency unless a separately validated destination detector-ID-to-DAQ map is
implemented.

## ROOT writer

Create `evaluation_region_partitions/write_detid_channel_shift_raw.py`.

The writer must create a new file atomically and never modify its input. The
output tree keeps the original tree name (`rawConv` or `cbmsim`) and includes:

- unchanged event header;
- transformed `Digi_ScifiHits`;
- transformed `Digi_MuFilterHits` for Veto, US, and DS;
- only explicitly approved raw metadata branches.

Do not copy original reconstructed tracks, clusters, reconstructed vertices,
features, or other branches derived from the old detector IDs. Implement the
branch allowlist explicitly rather than cloning every branch by default.

For each event, build a fresh output `TClonesArray`. Use the C++ copy
constructor for each retained hit, set its destination detector ID, and fill the
output tree once. Input objects are never mutated on disk.

Write file-level metadata containing at least:

```text
transform_version
source_file
source_tree
geometry_file
run_id
axis_policy
selected_axis, selected_view
slope_xz, slope_yz, axis_dominance_ratio
pz, pz_sign, direction_source
travel_slope, entry_side
reference_plane, reference_z_cm
requested_shift_cm, selected_channel_offset
anchor detector IDs
configuration checksum
synthetic_daq_warning
```

## Output and audit contract

Use a new versioned directory; never overwrite v1 or the superseded v2 policy:

```text
evaluation_region_partitions/passing_muon_detid_channel_shift_v3/
```

Products:

```text
selected_events.csv
transformed_digis/run_008329_event_<eventId>.root
mapping_audit.csv
event_audit.json
comparison.pdf
validation.ok
```

One CSV row per input digi includes:

```text
runId,eventId,detector,system,plane,view,
source_det_id,destination_det_id,
source_ordinal,destination_ordinal,status,
source_coordinate_cm,destination_coordinate_cm,realized_shift_cm,
anchor_det_id,edge_det_id,channel_offset,
source_mat,source_sipm,source_channel,
destination_mat,destination_sipm,destination_channel
```

Event JSON also records requested displacement, anchor error, retained/dropped
counts, shift distributions, boundary crossings, duplicate checks, branch
schema checks, all axis/direction decision inputs, the transformed view list,
and the unchanged view list. For every event it must contain enough information
to reproduce why x or y and low or high were selected.

## Official display validation

Do not draw transformed hits manually. Run the unmodified official
`2dEventDisplay.py` twice:

1. on the original raw event;
2. on the transformed raw-digi event.

Merge the two official pages side by side or consecutively without replacing
their detector rendering. Because transformed hits now carry real detector IDs,
SNDSW will fill the physical destination channels itself.

Add separate diagnostic plots only for the detector-ID mapping table and shift
residuals. These diagnostics are not substitutes for the official event display.

## Validation gates

The validator must enforce:

1. output run/event identity equals the input identity;
2. every retained output detector ID is valid in initialized geometry;
3. every mapped output detector ID belongs to the same detector, plane, and
   view as its source;
4. `destination_ordinal = source_ordinal + offset`;
5. mapping is injective within each event/plane/view;
6. retained plus outside-low plus outside-high equals the input hit count;
7. copied signal/time arrays, validity, and masks equal the source object;
8. output files contain no stale reconstructed branches;
9. reopening the output ROOT file reproduces the audited detector IDs and digi
   payloads;
10. official SNDSW display completes on both original and transformed files.
11. exactly one axis and its matching readout views were remapped;
12. every orthogonal-view detector ID, digi payload, and collection order is
    identical to the input;
13. the transformed track coordinate is on the chosen entry edge at the
    reference plane within the audited discrete-channel residual;
14. the signed track motion after transformation points from that edge toward
    the detector interior;
15. the selected axis, edge, and shift agree with the stored slopes, momentum
    direction, thresholds, and formulas in this plan.

Treat DAQ-ID inconsistency as an explicit warning for the synthetic product,
not as a hidden validation success.

## Tests

Add `evaluation_region_partitions/test_detid_channel_shift.py` with synthetic
layouts covering:

- low-edge and high-edge offsets;
- sparse hits preserving empty channel indices;
- ordering independent of numerical detector ID;
- source hits spanning die, SiPM, and mat boundaries;
- destination gaps remaining fixed in geometry;
- low/high overflow drops;
- no wraparound or clamping;
- collision detection;
- realized shift changes across physical gaps;
- deterministic anchor tie-breaking;
- missing reference-plane anchor rejection.

Add policy tests covering:

- x selected for `abs(sx) > abs(sy)` and y for the converse;
- low-edge selection for positive travel slope and high-edge selection for
  negative travel slope;
- reversal of the entry side when `pz_sign` reverses;
- rejection of zero/small slopes, exact ties, and insufficient dominance;
- remapping only coordinate-compatible readout views;
- all US hits remaining unchanged for an x transform;
- orthogonal detector IDs, order, and payload remaining exactly unchanged;
- transformed reference point landing at the selected edge within the channel
  residual and the downstream track moving inward.

For the currently selected Run 8329 event 6000152, the recorded fit slopes are
`sx = +0.00401119825513222` and `sy = -0.0024016601493382496`. With an assumed
positive pz this is an explicit regression case. The selected track is track
index 0, type 11 (Hough SciFi), and its slope-dominance ratio is approximately
1.67. Therefore select x, choose the low-x edge, apply a negative x channel
shift, and leave every y-measuring view unchanged.

Add a one-event SNDSW integration test that writes, reopens, validates, and
displays a transformed file before running the ten-event sample.

## Snakemake workflow

Add `snakemake/rules/passing_muon_detid_channel_shift_v3.smk` with isolated
targets:

```text
select_passing_muon_detid_channel_shift_v3
write_passing_muon_detid_channel_shift_v3
display_passing_muon_detid_channel_shift_v3
validate_passing_muon_detid_channel_shift_v3
passing_muon_detid_channel_shift_v3
```

The writer and display rules use `env_sndsw_nue.sh`. The pure unit tests and
final audit may run with the lightweight Python environment. The v3 selector
may reuse the frozen Run 8329 event identity, but it must rebuild the manifest
with track slopes, direction metadata, and track provenance. No v1 or v2 output
is a dependency.

The v3 configuration replaces `x_side`/`y_side` with:

```yaml
transform:
  axis_policy: largest_absolute_scifi_track_slope
  # Prototype value: reject an exactly flat dominant projection. Tune before
  # production using the Run 8329 slope distribution.
  min_abs_slope: 0.0
  min_axis_dominance_ratio: 1.2
  ambiguous_axis_policy: reject
  direction:
    use_track_pz: true
    assumed_pz_sign: 1
  reference_plane_policy: earliest_selected_view_plane_with_hits
```

## Implementation sequence

### Phase 1: geometry audit

1. Enumerate one Run 8329 SciFi plane from real geometry.
2. Print ordered detector IDs around every die, SiPM, and mat boundary.
3. Verify physical ordering and measure every gap.
4. Compare hierarchy patterns across all ten SciFi planes.

### Phase 2: pure mapper

1. Implement layout, anchor, offset, and mapping data classes.
2. Add the synthetic gap/boundary tests.
3. Require injection and count closure.

### Phase 3: one-event transformed ROOT file

1. Transform SciFi digis for one selected Run 8329 event.
2. Transform MuFilter digis with plane/view-specific ordered-bar mappings.
3. Reopen the output and compare all digi payloads.
4. Draw original and transformed files with official SNDSW.
5. Review anchor choice, direction, gaps, overflow, and realized shifts.

### Phase 4: ten-event SciFi validation

1. Produce versioned transformed files and mapping audits.
2. Run official displays and numerical validation.
3. Do not proceed if any collision, invalid detector ID, stale branch, or
   unexplained count mismatch occurs.

### Phase 5: MuFilter extension (implemented)

1. Audit ordered Veto, US, and DS bar layouts.
2. Define compatible layout families and their anchor/offset policy.
3. Enable MuFilter remapping using the same generic mapper.
4. Repeat one-event and ten-event ROOT/display validation.

### Phase 6: slope-selected single-axis v3 migration (implemented)

1. Extend the selected-event manifest with the selected track's intercept,
   `reference_z_cm`, `track_slope_xz`, `track_slope_yz`, track type/index,
   track input path/entry, and momentum components when available.
2. Make the pure policy decision before building any channel mapping. Reject an
   ambiguous axis or unavailable direction according to configuration.
3. Build mappings only for compatible views. Copy every orthogonal hit without
   changing its detector ID or payload.
4. Write new v3 files and audit records; never mutate or relabel v2 files.
5. Run the one-event numerical validator, official event display, and V7-style
   SciFi channel comparison before expanding the sample.

### Phase 7: downstream workflow

Only after visual approval, feed transformed digi files into reconstruction,
Hit3D, feature generation, and background-template steps. The transform belongs
before every derived reconstruction/feature step.

### Phase 8: near-edge constrained v4 migration (implemented)

1. Add a ROOT-independent edge-decision function with an inclusive 5 cm
   closest-edge override and signed-slope fallback.
2. Keep axis selection independent from edge selection, guaranteeing that only
   one coordinate is transformed.
3. Persist and independently recompute the complete edge decision during raw
   Digi validation.
4. Write v4 products and the ten-event high-density comparison to an isolated
   output directory.

## File-by-file code implementation

### `detid_channel_shift.py`

Add ROOT-independent `AxisDecision` and `TrackLine` data classes plus pure
functions such as `choose_transform_axis`, `choose_entry_side`, and
`project_track`. `AxisDecision` contains both slopes, dominance ratio, selected
axis/view, `pz_sign`, direction source, travel slope, and entry side. Keep the
existing `ChannelLayout`, `choose_anchor`, and `map_detector_ids` primitives;
the new policy decides which layouts are allowed to call them. Unit-test every
sign and ambiguity branch without SNDSW.

### `detid_channel_shift_sndsw.py`

Add adapters that classify every SciFi and MuFilter `PlaneKey` as measuring x
or y and return whether it is compatible with the selected axis. Add a track
metadata reader that prefers a finite non-zero `reco_track_mom_z`/track momentum
from the selected track and otherwise invokes the explicit configured pz
assumption. Keep geometry enumeration and digi cloning unchanged.

### selected-event manifest producer

Extend the v3 selector rather than relying on the small frozen v2 CSV. Persist
the exact track used by the current event-display/selection workflow: track
type, collection index, source file and tree entry, fit intercept/reference z,
`dx/dz`, `dy/dz`, and momentum components. The selector must fail if the track
cannot be joined unambiguously to the raw event. It must not silently select a
different track during writing.

### `write_detid_channel_shift_raw.py`

Refactor `transform_event` into four explicit stages:

1. read and validate the frozen track metadata;
2. compute one `AxisDecision`;
3. construct SciFi and MuFilter mappings only for compatible views;
4. clone compatible hits with new detector IDs and clone orthogonal hits with
   their original detector IDs.

Remove the production use of `closest_side` and the independent x/y loop.
Determine the reference plane from travel direction and selected-view hit
availability, project the track to its geometry z, compute the requested shift,
and use the existing real-channel anchor/ordinal mapper. Add `mapping_action`
values `remapped`, `outside_low`, `outside_high`, and
`unchanged_orthogonal`. Include unchanged rows in the audit so count closure is
checked over the complete input collection.

### `validate_detid_channel_shift_raw.py`

Recompute the axis and side decision from audited inputs rather than trusting
the writer's labels. Partition hits by coordinate view. Apply existing ordinal,
geometry, injectivity, and payload checks to remapped views; require exact
detector-ID, ordering, and payload equality for orthogonal views. Project the
translated track at the reference and subsequent SciFi planes and check that it
starts at the chosen side and moves inward.

### display and diagnostics

Keep `render_detid_channel_shift_comparison.py` based on the unmodified SNDSW
display. Update `render_detid_channel_shift_scifi_channels.py` to annotate the
selected axis, both slopes, pz-direction source, entry side, requested shift,
and realized ordinal offset. The original and transformed pages must retain the
same V7 channel-display style, with the orthogonal projection visibly
unchanged.

### Snakemake/configuration

Create v3 rule and configuration files by wiring the new manifest, transformed
ROOT, full audit, official display, V7-style SciFi channel display, and
validation marker into one isolated target. Add the v3 rule include and target
without changing the existing v2 target, so the two policies remain directly
comparable and cached v2 products are not mistaken for v3 results.

Additional reviewed events are configured as `eventId: source_category` pairs
under `input.additional_display_events`. The wildcard rules write each event to
`passing_muon_detid_channel_shift_v3/extra_events/run_008329_event_<eventId>/`.
The aggregate target `passing_muon_detid_channel_shift_v3_more_events` produces
and validates both official and V7-style displays for every configured event.

## Acceptance criteria

- The original ROOT files and v1 products are untouched.
- Every retained transformed hit has a real destination detector ID.
- Sparse channel occupancy is preserved by ordinal offset.
- Detector gaps remain those of the destination hardware geometry.
- No collision, clamping, wrapping, or silent overwrite occurs.
- Dropped edge hits and realized physical shifts are fully audited.
- Exactly one coordinate is transformed for each accepted event.
- Axis choice follows the dominant absolute SciFi track slope and passes the
  configured dominance/absolute-slope gates.
- In v3, the target side is the signed-travel-slope entry side. In v4, a track
  within 5 cm of an active edge targets its closest edge; all other tracks use
  the signed-travel-slope side.
- The v4 validator independently reproduces edge distances, threshold state,
  closest side, slope side, and final target side from the audit inputs.
- Orthogonal readout views retain identical detector IDs, order, and digi
  payloads.
- The new ROOT file reopens with the expected SNDSW branch classes.
- The official SNDSW display renders the transformed file without a custom hit
  overlay.
- Derived products are regenerated rather than copied from the source event.

## Implementation and validation log

- 2026-09-23: recorded user acceptance of v4 and added the canonical method
  reference `PASSING_MUON_DETID_TRANSFORM_METHOD.md`. It fixes the authoritative
  equations, coordinate/view conventions, 5 cm proximity rule, channel and gap
  behavior, synthetic-DAQ limitation, validation contract, reviewed-sample
  statistics, commands, version lineage, and conditions requiring revalidation.
- 2026-09-23: replaced the experimental v5 first-two-station estimator with
  the method documented in `DETID_CHANNEL_SHIFT_V5_ALL_PLANE_AVERAGE.md`.
  Each nonempty SciFi plane contributes one unweighted channel-centre average;
  equal-weight ordinary least squares uses every available plane independently
  in x-z and y-z. The fit assumes positive-z travel and otherwise reuses the v4
  one-axis/proximity transform. The ten-page output has original, frozen-track
  v4, and all-plane v5 columns with frozen line overlays; no display-time
  tracking runs. All 20 transformed files, all 10 displays, 28 unit tests, and
  all 53 workflow jobs passed. Relative to v4, all-plane v5 agrees on axis in
  7/10 events, edge in 6/10, and both in 3/10; SciFi edge drops total 3,564
  versus 3,613 for v4. This improves on the superseded two-station result but
  remains an activity-axis method, not a documented physics-equivalent track.
- 2026-09-23: initially implemented the reconstruction-free first-two-active-
  station prototype. It passed technical validation but agreed with v4 on axis
  in only 4/10 events and axis-plus-edge in 2/10, motivating the all-plane fit.
- 2026-09-23: added and completed the isolated
  `passing_muon_detid_channel_shift_v4_display100` target. It selects 100
  track-valid events in descending feature-level SciFi occupancy, writes and
  validates one transformed raw-Digi file per event, renders original and
  transformed events with the official SNDSW display, and composes a 100-page
  side-by-side PDF without changing either source page's aspect ratio. The
  original `>500`-hit high-density condition contains only 85 eligible events,
  so this extended display uses no hard occupancy cutoff and preserves density
  ranking; the selected sample spans 1,717 to 326 SciFi hits. Selection scanned
  151 ranked candidates and rejected 51 without a usable input SciFi track.
  All 100 per-event validators and all 302 Snakemake jobs passed. Outputs are
  isolated under
  `evaluation_region_partitions/passing_muon_detid_channel_shift_v4/display_sample_100`;
  the reviewed ten-event v4 sample is unchanged.
- 2026-09-22: completed the isolated v4 ten-event high-density workflow with
  all 33 Snakemake jobs and all ten per-event raw-Digi validators passing. Three
  events activated the inclusive 5 cm proximity branch: 4038619 at 1.155 cm,
  6018903 at 0.014 cm, and 7004982 at 4.615 cm. Event 4038619 is the regression
  case in which the signed slope selected low-y but proximity selected high-y;
  v4 retained 1062 of 1135 SciFi digis, compared with 746 in v3. The final
  official-SNDSW comparison has ten side-by-side pages with aspect ratios
  preserved. All 22 ROOT-independent policy/mapping tests pass.
- 2026-09-22: implemented the v4 proximity-constrained edge policy. The pure
  policy keeps the v3 dominant-slope axis selection, overrides only the target
  side when the reference-plane track lies within an inclusive 5 cm of its
  closest active edge-channel centre, and otherwise retains the signed-slope
  entry side. Added isolated v4 configuration/Snakemake targets, full edge
  decision audit, independent validator reconstruction, and tests for both
  edges, conflicting slope signs, the exact threshold, the just-outside case,
  one-axis preservation, and invalid inputs.

- 2026-09-22: added reusable wildcard rules and ran three more displays:
  central/high-occupancy event 6000210, largest-slope event 5004148, and the
  highest-SciFi-density reviewed event 7000202. All three transformed ROOT
  files, official displays, V7-style channel displays, and numerical validators
  passed. Event 7000202 contains 905 raw SciFi digis (the earlier feature-level
  manifest recorded occupancy 659), selects y/high with channel offset `+528`,
  retains all 451 orthogonal x-view hits, and drops 208 y-view hits beyond the
  high edge, closing at 905 to 697 SciFi digis.
- 2026-09-22: implemented the v3 exact-event manifest producer, pure
  `AxisDecision`/`TrackLine` policy, version-aware raw-digi writer, orthogonal
  view preservation, independent compatible MuFilter remapping, expanded CSV/
  JSON audit, recomputing validator, official SNDSW comparison, V7-style SciFi
  channel comparison, isolated configuration, and six-job Snakemake target.
- 2026-09-22: all 11 ROOT-independent tests passed. The full one-event SNDSW
  target passed selection, ROOT writing/reopening, detector/view and ordinal
  checks, complete digi payload equality, orthogonal detector-ID/order checks,
  continuous edge placement, inward-motion checks, and both display jobs.
- 2026-09-22: event 6000152 selected its type-11 Hough SciFi track with
  `pz=99.9989070961`, `sx=+0.00401119826`, `sy=-0.00240166015`, chose x and the
  low-x entry edge, and used SciFi station 1 as its first hit-bearing reference
  plane. The track was already 0.0262 cm from the edge and the first vertical
  hit was already the edge channel, so the discrete SciFi offset is zero. The
  70 horizontal SciFi hits and 11 horizontal MuFilter hits were copied exactly;
  the 63 vertical SciFi hits were retained, and compatible vertical DS planes
  used audited offsets from -11 to -14 bars. Totals closed at 133 to 133 SciFi
  and 19 to 19 MuFilter hits with no drops.
- 2026-09-22: revised the next implementation to v3. The authoritative policy
  now selects exactly one axis from the larger absolute SciFi track slope,
  obtains travel direction from track pz or an explicit audited assumption,
  chooses the corresponding entry edge, remaps only coordinate-compatible
  views, and preserves all orthogonal hits unchanged. Existing v2 products are
  explicitly classified as outputs of the superseded two-axis/nearest-edge
  policy.

- 2026-09-22: implemented the ROOT-independent ordinal channel mapper and seven
  synthetic tests covering non-numeric physical ordering, sparse occupancy,
  both edge directions, overflow, gap-dependent realized shifts, deterministic
  anchors, duplicate rejection, and no wrapping.
- 2026-09-22: implemented Run 8329 SNDSW geometry enumeration for all ten SciFi
  planes. Each plane contains 1,536 unique channels ordered by its measured
  x/y coordinate. The geometry audit identifies 12 central-die, nine SiPM, and
  two mat boundaries per plane.
- 2026-09-22: implemented an atomic one-event raw-digi writer. It retains the
  three original raw branches and required FairRoot file objects, remaps copied
  SciFi and MuFilter digi detector IDs, and stores a synthetic DAQ-warning
  metadata object. No reconstruction branch is copied.
- 2026-09-22: event 6000152 uses SciFi station 3 as the reference. The vertical
  x mapping selected offset `-3`; the horizontal y mapping selected offset
  `-644`. The writer mapped 133 input SciFi digis to 93 output digis and audited
  40 low-edge overflow drops. All 19 MuFilter digis map to valid Veto/US/DS
  destination bars with no edge drops or collisions.
- 2026-09-22: reopen validation passed event identity, exact branch schema,
  ordinal equations, injectivity, station/view preservation, signal/time/
  validity/DAQ payload equality, MuFilter equality, FairRoot metadata, and
  count closure.
- 2026-09-22: the unmodified official SNDSW `2dEventDisplay.py` successfully
  rendered both source and transformed ROOT files. The retained comparison has
  two official pages and no custom transformed-hit overlay.
- 2026-09-22: reused the V7 `scifi_sipm_event_display.py` renderer to produce a
  four-page channel comparison: original horizontal/vertical followed by
  transformed horizontal/vertical. It retains the V7 physical channel
  rectangles, die/SiPM/mat separators, QDC colors, and per-station layout.
- 2026-09-22: added and executed the isolated
  `passing_muon_detid_channel_shift_v2` Snakemake target. All four jobs completed
  successfully. The reviewed event identity is stored as a frozen v2 manifest,
  so the superseded v1 selector is not scheduled.
