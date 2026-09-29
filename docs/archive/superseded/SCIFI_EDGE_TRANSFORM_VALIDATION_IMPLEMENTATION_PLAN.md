# SciFi edge-transform validation and event-display implementation plan

Status: superseded on 2026-09-22 by
`DETID_CHANNEL_SHIFT_RAW_DIGI_IMPLEMENTATION_PLAN.md`. The files and outputs
described here are retained only as a record of the rejected continuous-position
prototype and must not be used for the new transformed-digi production.
Created: 2026-09-22
Scope: Run 8329 veto-tagged Muon-DIS candidates, before background-template production

## Objective

Validate the proposed detector geometry transformation visually and numerically
on a small, reproducible Run 8329 sample. Use the reconstructed SciFi track at
each detector plane, and move SciFi and MuFilter hit coordinates by the distance
between that track and the chosen active detector edge. Preserve the hit pattern
relative to the extrapolated track and render the original and transformed event
side by side.

This is an isolated validation product. It must not modify source ROOT files,
rewrite `Digi_ScifiHits`, alter frozen V7 products, or enter the production
Hit3D/feature workflow until the transformation has passed review.

## Physics sample

Use the V8 `muon_dis_candidate` definition:

```text
count_veto > 0
and the V7 coverage_loose / V8 passing-muon detector requirements fail
```

The first validation sample contains 5--10 events with a converged track that
has SciFi measurements. Select events deterministically to cover useful edge
cases rather than taking only the first valid rows:

- tracks closest to the low and high x edges;
- tracks closest to the low and high y edges;
- a central track;
- large absolute x-z or y-z slope;
- high SciFi occupancy;
- an event for which the transformation sends at least one hit into a gap or
  beyond the active boundary, if one exists.

The prototype may draw its event pool from the frozen V7 Run 8329 display
manifest, filtered to failure events with `count_veto > 0`. This provides
already-audited raw-event identities. A later production study must build the
Muon-DIS population from the complete V8 feature sample, not from this small
display manifest.

## Coordinate convention and transformation

SciFi view 1 is vertical and measures x. SciFi view 0 is horizontal and
measures y. Define the measured coordinate `u` as:

```text
vertical plane:   u = x
horizontal plane: u = y
```

For station/view plane `p`, obtain its exact z position and active lower and
upper measured-coordinate boundaries from the initialized Run 8329 geometry.
Let the selected track position at that plane be `u_track(p)` and the selected
target boundary be `u_edge(p)`. With an optional positive inward margin `m`:

```text
lower edge target: u_target(p) = u_low(p)  + m
upper edge target: u_target(p) = u_high(p) - m
delta_u(p) = u_target(p) - u_track(p)
u_hit_transformed = u_hit_original + delta_u(p)
u_track_transformed = u_track(p) + delta_u(p) = u_target(p)
```

The initial validation uses `m = 0 cm`. The boundaries are the outside edges
of the first and last active channel rectangles, not merely the channel-centre
extrema.

### Edge-side policy

Choose the x side and y side independently. By default, lock one x side across
all vertical planes and one y side across all horizontal planes. For each
orientation, compare the summed track-to-low-edge and track-to-high-edge
distances across the available planes and choose the smaller total. The edge
coordinate and shift remain plane-dependent because of detector alignment and
track slope.

Record a diagnostic `plane_local_closest_side` as well. If it differs from the
locked event side, show that in the audit, but do not silently switch sides.
An optional unlocked mode may be implemented for comparison only and must not
be the default.

This operation is a plane-by-plane morphology transform, not one rigid 3D
translation. It intentionally makes the transformed reference track follow
the selected detector edge while preserving each plane's measured residuals.

## Geometry construction

Load the event geometry with `SndlhcGeo.GeoInterface`. After loading each event,
call `Scifi.InitEvent(EventHeader)` before requesting positions so Run-dependent
alignment is applied.

Construct all ten SciFi planes from real channel geometry:

1. enumerate every `(station, view, mat, SiPM, channel)` detector ID;
2. call `Scifi.GetSiPMPosition(detID, A, B)`;
3. use x for vertical planes and y for horizontal planes;
4. obtain the plane z from the channel endpoints and validate its spread;
5. combine channel centres with the `ChannelVol` half-width to determine the
   active intervals and the outer active boundaries;
6. preserve central-die, inter-SiPM, and mat gaps as inactive intervals.

The detector-ID encoder must be checked at runtime against every observed
`Digi_ScifiHit`: decoding the generated/observed ID must reproduce station,
view, mat, SiPM, and channel. Stop on any mismatch. Geometry values must never
be hard-coded from the approximate Run 8329 ranges measured during planning.

Refactor public geometry helpers from
`evaluation_region_partitions/scifi_sipm_event_display.py` where practical,
but keep the existing V7 renderer's behaviour unchanged.

For MuFilter, initialize `MuFilter` with the same event header and enumerate
the complete Veto, US, and DS bar geometry with `MuFilter.GetPosition()`. Build
one measured-coordinate boundary per physical plane/view. The selected x/y
side is inherited from the SciFi decision, while the shift remains specific to
each MuFilter plane and uses the extrapolated SciFi track at that plane's z.

## Reconstructed-track policy

Read `Reco_MuonTracks` from the same raw event shown in the display. A usable
track must have:

- `getTrackFlag()` true;
- positive NDF and finite, non-negative chi2/NDF;
- aligned `getTrackPoints()` and `getRawMeasDetIDs()` arrays;
- enough SciFi measurements to determine x(z) and y(z).

Use a configurable, explicit track-type priority. The proposed initial order is
Hough SciFi (11), Simple SciFi (1), then Combined Hough (15). Within the first
available type, choose the usable track with the smallest chi2/NDF and use its
collection index as a final deterministic tie-breaker. Record all eligible
tracks and the reason for the choice in the audit.

Collapse repeated fitted points within a station/view plane, then fit straight
`x(z)` and `y(z)` track models from the SciFi fitted points. Evaluate those
models at the exact geometry z of every plane. Record fit residuals and compare
the predictions with the fitted points belonging to each plane. Do not use a
hit coordinate itself as the track coordinate.

If a track cannot support both projections, reject the event with a documented
reason rather than inventing a coordinate. Track selection and extrapolation
thresholds belong in YAML configuration.

## Event-display product

Produce one four-page vector-PDF block per event:

1. the unchanged official SNDSW full-detector overview;
2. an official-SNDSW-style full-event comparison with original/transformed
   `z-x` and `z-y` panels, transformed Veto/SciFi/US/DS activity, the selected
   SciFi track, and the transformed plane-target reference;
3. all horizontal SciFi planes, original and transformed side by side;
4. all vertical SciFi planes, original and transformed side by side.

Reuse the V7 SNDSW overview runner and geometry-faithful SciFi channel layout.
The comparison pages show, per station:

- active channel rectangles and physical gaps;
- original hit coordinates and QDC;
- original track intersection;
- lower and upper active boundaries;
- selected target boundary and inward margin;
- transformed track intersection;
- transformed hit coordinates with the same QDC scale;
- the plane's `delta_u`;
- transformed hits inside active channels, in gaps, and outside the detector
  with visibly different marker styles.

Use continuous transformed coordinates in phase 1. Do not colour a physical
channel as if a transformed coordinate were a valid detector ID. The official
SNDSW overview remains the original event because the virtual transformed hits
are not written to a ROOT event. The transformed full-event panels reuse the
official `2dEventDisplay.py` axes, detector geometry, colors, annotations, and
canvas style, then overlay the virtual continuous coordinates as ROOT graphs.
This is necessary because the unmodified SNDSW renderer fills physical geometry
nodes by detector ID and cannot represent an arbitrary coordinate without
inventing a false detector ID.

## Numerical validation

For every event, plane, and hit, enforce or report the following invariants:

1. **Track target:**
   `abs(u_track_transformed - u_target) <= track_edge_tolerance_cm`.
2. **Residual preservation:**
   `abs((u_hit_transformed - u_track_transformed) -
   (u_hit_original - u_track_original)) <= residual_tolerance_cm`.
3. **Count closure:** original valid hits equal transformed hits, and
   `inside_active + inside_gap + outside_geometry` equals transformed hits.
4. **Side locking:** all planes of one orientation use the configured event
   side when locking is enabled.
5. **Finite values:** geometry, track, shift, and hit coordinates are finite.
6. **Identity:** manifest run/event and raw-tree run/event agree.
7. **Geometry:** observed digis map to exactly one original physical channel.

The continuous residual-preservation tolerance can be tight (for example
`1e-9 cm`) because both translated quantities use the same floating-point
shift. Geometry/track-edge tolerances should be separately configurable.

Do not discard gap or outside hits in the validator. They are an output of the
test and an important input to the later remapping design.

## Files to add

### Core transformation

`evaluation_region_partitions/scifi_edge_transform.py`

Suggested data classes and pure functions:

```text
PlaneKey(station, view)
ActiveInterval(low_cm, high_cm, mat, sipm, channel)
PlaneGeometry(key, z_cm, low_edge_cm, high_edge_cm, intervals)
TrackProjection(...)
PlaneTransform(...)
TransformedHit(...)

choose_locked_side(...)
fit_track_projections(...)
evaluate_track_at_planes(...)
build_plane_transforms(...)
classify_transformed_coordinate(...)
transform_hits(...)
validate_transform(...)
```

Keep the arithmetic and classification functions independent of ROOT so they
can be unit-tested with synthetic geometry.

### ROOT/SNDSW adapter and renderer

`evaluation_region_partitions/plot_passing_muon_scifi_edge_transform.py`

Responsibilities:

- load and validate the selected manifest;
- batch original SNDSW overview rendering using
  `sndsw_event_display_common.py`;
- open each raw event and verify identity;
- initialize event geometry;
- extract SciFi digis and reconstructed tracks;
- build the geometry and transformation inputs;
- call the pure transform module;
- draw the two before/after SciFi pages;
- merge the four pages per event;
- atomically write PDF, CSV, and JSON outputs.

### Deterministic sample selector

`evaluation_region_partitions/select_passing_muon_scifi_edge_transform_events.py`

Responsibilities:

- filter the source pool to V8 Muon-DIS candidates;
- inspect track usability without changing reconstruction;
- calculate ranking variables needed for the edge-case sample;
- deduplicate events selected for more than one reason;
- write selected and rejected rows with selection/rejection reasons.

### Tests

`evaluation_region_partitions/test_scifi_edge_transform.py`

Cover at least:

- straight and sloped tracks;
- low- and high-edge transforms in x and y;
- plane-dependent aligned boundaries;
- event-side locking where a local closest side changes;
- inward margin sign;
- active-channel, gap, and outside classification;
- exact residual and hit-count closure;
- missing/degenerate track projections;
- deterministic track and event tie-breaking.

### Configuration

`snakemake/metadata/configs/region_evaluation/passing_muon_scifi_edge_transform_v1.yaml`

Suggested configuration blocks:

```yaml
version: passing_muon_scifi_edge_transform_v1
run: 8329
selection:
  population: muon_dis_candidate
  count_veto_expression: count_veto > 0
  sample_size: 10
track:
  type_priority: [11, 1, 15]
  require_flag: true
  min_ndf: 1
  min_scifi_planes_per_projection: 2
transform:
  lock_side_per_orientation: true
  inward_margin_cm: 0.0
  residual_tolerance_cm: 1.0e-9
  track_edge_tolerance_cm: 1.0e-8
display:
  page_types: [overview, scifi_horizontal_comparison, scifi_vertical_comparison]
  output_mode: vector_pdf
  plane_page_width: 66
output:
  directory: evaluation_region_partitions/passing_muon_scifi_edge_transform_v1
```

Paths to the frozen source manifest, event tree candidates, and Run 8329
geofile should also be explicit in the actual YAML.

### Snakemake integration

Add `snakemake/rules/passing_muon_scifi_edge_transform_v1.smk` and include it
from `snakemake/snakefile`. Use separate rules and targets:

```text
select_passing_muon_scifi_edge_transform_events_v1
plot_passing_muon_scifi_edge_transform_events_v1
validate_passing_muon_scifi_edge_transform_v1
passing_muon_scifi_edge_transform_v1
```

The selection and rendering rules run under `env_sndsw_nue.sh`; a lightweight
final audit/merge rule may use `env_lcg.sh` if it does not import SNDSW classes.
Start with one rendering job because the sample is small. Parallelize per event
only if runtime measurements justify the extra merging complexity.

## Output contract

Write only to:

```text
evaluation_region_partitions/passing_muon_scifi_edge_transform_v1/
```

Expected products:

```text
selected_events.csv
rejected_events.csv
comparison.pdf
rendered_pages.csv
transform_audit.csv
transform_audit.json
validation.ok
```

The per-plane CSV includes at least:

```text
runId, eventId, source_file, source_event_index,
track_index, track_type, track_chi2ndf,
station, view, orientation, plane_z_cm,
locked_side, plane_local_closest_side,
low_edge_cm, high_edge_cm, target_cm,
track_original_cm, track_transformed_cm, delta_cm,
n_original_hits, n_transformed_hits,
n_inside_active, n_inside_gap, n_outside,
max_residual_difference_cm, validation_passed
```

The JSON retains event identity, all eligible-track metadata, track-fit
diagnostics, geometry validation, individual plane transforms, and every
failed assertion. Write outputs atomically so a failed batch cannot look
complete.

## Implementation sequence

### Phase 1: pure transform and synthetic tests

1. Implement data classes and continuous-coordinate arithmetic.
2. Implement locked-side selection and interval classification.
3. Add synthetic unit tests and run them outside SNDSW where possible.
4. Require complete residual and count closure before ROOT integration.

### Phase 2: event/geometry/track extraction

1. Export or add reusable public geometry helpers without changing V7 output.
2. Implement event identity and `InitEvent` handling.
3. Implement strict detector-ID round-trip validation.
4. Implement deterministic track selection and plane-z projection.
5. Run an extraction-only audit on one event before rendering.

### Phase 3: one-event visual prototype

1. Choose one valid veto-tagged Muon-DIS candidate.
2. Render the original overview, full-event comparison, and horizontal and
   vertical plane comparisons.
3. Check transformation annotations, sign, side, gaps, and outside hits.
4. Compare CSV/JSON values to plotted markers.

### Phase 4: frozen 5--10 event validation sample

1. Run the deterministic edge-case selector.
2. Render the full sample with one shared display contract.
3. Produce merged numerical audits and `validation.ok` only if all hard
   invariants pass.
4. Record elapsed time, peak memory, and per-event page counts.

### Phase 5: review gate

Review the PDF and audits before implementing channel remapping or production
background templates. Any change to edge policy, track policy, or margins
requires a new configuration version and regenerated validation outputs.

### Phase 6: later channel remapping, not part of the first implementation

If continuous-coordinate validation succeeds, add a separate mapping stage:

- locate the active channel interval containing each transformed coordinate;
- explicitly classify coordinates in gaps or outside the detector;
- define and audit collision behaviour when multiple source hits map to one
  destination channel;
- only then consider generating transformed detector IDs or transformed ROOT
  products.

Do not clamp to the closest channel silently. The remapping policy affects the
physics template and requires a separate decision.

## Acceptance criteria for phase 1--4

- The configured sample contains only Run 8329 events with `count_veto > 0`
  classified as Muon-DIS candidates.
- Source event identity and selected track identity are reproducible.
- Geometry is loaded at runtime and initialized for every event.
- The track lands on the configured target in every transformed plane.
- All hit-to-track measured-coordinate residuals are preserved.
- Hit counts close, including gap and outside categories.
- Every event contributes exactly four vector-PDF pages.
- Original and transformed panels use identical geometry and QDC scales.
- Existing V7 and V8 products remain untouched.
- Unit tests, Snakemake dry-run, and the final validation rule pass.

## Current review decisions

Agreed for the first implementation:

- validate with veto-tagged Muon-DIS candidates, not the no-Veto signal region;
- compute the track position at each exact SciFi plane z;
- translate the measured coordinate plane by plane;
- derive active edges from geometry rather than constants;
- lock the selected edge side per orientation across stations;
- display continuous transformed coordinates before channel remapping;
- preserve V7 displays and source ROOT files unchanged.

## Implementation and validation log

- 2026-09-22: added the ROOT-independent transform module and five synthetic
  tests covering fits, locked sides, margins, residual closure, and
  active/gap/outside classification.
- 2026-09-22: found that the raw Run 8329 display files do not contain
  `Reco_MuonTracks`. The implementation therefore joins each raw event to the
  existing V7 flat-v2 track product by `(runId, eventId)`. Row numbers are not
  assumed to align; a checked ROOT tree index supplies `track_entry`.
- 2026-09-22: verified the SciFi detector-ID encoder against observed digis
  and implemented all-channel geometry enumeration through
  `Scifi.GetSiPMPosition()` after `Scifi.InitEvent(EventHeader)`.
- 2026-09-22: measured a maximum 0.487 cm endpoint-z spread within one
  station/view due to the staggered SciFi mat construction. The transform uses
  the median physical plane z and records the spread in the audit.
- 2026-09-22: selected ten deterministic veto-tagged Muon-DIS candidates with
  usable Hough SciFi tracks and rendered the initial 30-page PDF.
- 2026-09-22: the final audit contains 100 planes and 1,800 input digis. All
  counts close: 949 transformed coordinates are in active channels, 65 are in
  physical gaps, and 786 are outside the active detector. The largest
  hit-to-track residual difference is `7.11e-15 cm`.
- 2026-09-22: the numerical validator passed all ten events and wrote
  `validation.ok`. A Snakemake dry-run contains only the new selector, plotter,
  validator, and aggregate targets; no V7 producer is scheduled.
- 2026-09-22: replaced the custom full-event page with an adapter around the
  official SNDSW `2dEventDisplay.py`. The adapter reuses its detector outlines,
  axes, colors, event annotation, and ROOT canvas style for side-by-side original
  and transformed `z-x`/`z-y` views. The original and transformed track
  references use the SNDSW Hough-track color; the transformed reference is
  dashed and is not presented as a new reconstruction.
- 2026-09-22: extended the plane-specific transformation to every observed
  Veto, SciFi, US, and DS hit. MuFilter boundaries come from initialized SNDSW
  geometry, and the same locked x/y edge side selected from SciFi is used when
  extrapolating the SciFi track to each MuFilter plane. Across ten events, all
  232 MuFilter hits close exactly (123 inside active geometry, 109 outside and
  zero in gaps), with a maximum residual change of `7.11e-15 cm`. The output is
  40 pages, all ten selected tracks are Hough SciFi type 11 at collection index
  zero, and both the SciFi and MuFilter validators pass every event.
