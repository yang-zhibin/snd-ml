# Passing-muon detector-ID transformation method

Status: **accepted method**
Workflow version: `passing_muon_detid_channel_shift_v4`
Accepted: 2026-09-23
Scope: Run 8329 side-entering muon-background template studies

## Purpose

This method constructs a synthetic SNDSW digi event in which the observed hit
pattern is translated toward one detector side. It is intended to study
side-entering passing-muon and muon-DIS backgrounds in the no-Veto signal
region, starting from real veto-tagged control events.

The transformation changes real detector IDs, not just plotted coordinates.
The output can therefore be read by the standard SNDSW display and downstream
reconstruction. Exactly one transverse coordinate is transformed per event;
the orthogonal coordinate is preserved exactly.

This produces a detector-level synthetic sample. It is not itself a background
normalization or proof that the transformed control sample has the same
distribution as the signal region.

## Canonical implementation

The accepted implementation is defined by:

- `snakemake/metadata/configs/region_evaluation/passing_muon_detid_channel_shift_v4.yaml`;
- `snakemake/rules/passing_muon_detid_channel_shift_v4.smk`;
- `evaluation_region_partitions/detid_channel_shift.py`;
- `evaluation_region_partitions/detid_channel_shift_sndsw.py`;
- `evaluation_region_partitions/write_detid_channel_shift_raw.py`;
- `evaluation_region_partitions/validate_detid_channel_shift_raw.py`.

The configuration is part of the method. The accepted edge threshold is 5.0
cm and the slope-dominance ratio is 1.2. Changing either creates a new method
variant and must not be described as the validated v4 configuration.

## Inputs and frozen provenance

Each event needs the raw-digi event, geometry, Run/Event identity, raw-tree
entry, and one selected reconstructed SciFi track. Track types are considered
in priority order `[11, 1, 15]`, then by lower chi2/NDF and collection index.
Type 11 is Hough SciFi, type 1 is Simple SciFi, and type 15 is combined Hough.

A track must have a valid fit, at least one degree of freedom, finite
non-negative chi2/NDF, and SciFi fitted points spanning at least two distinct z
positions in both projections. The exact selected track, slopes, momentum,
source files, and entries are frozen in `selected_events.csv`. The writer does
not silently select another track.

## Transformation definition

### 1. Select one transverse axis

```text
x(z) = x0 + sx (z - z0),   sx = dx/dz
y(z) = y0 + sy (z - z0),   sy = dy/dz
```

Select x when `|sx| > |sy|` and y when `|sy| > |sx|`. The ratio of the larger
to smaller absolute slope must be at least 1.2. Exact ties, non-finite slopes,
a zero dominant slope, and insufficient dominance are rejected. Axis selection
never depends on the closest detector edge.

### 2. Establish travel direction

A fitted line has no direction. Use the sign of the selected track's finite,
non-zero `p_z`. If it is unavailable, explicitly assume `p_z > 0` as configured
and record the fallback.

```text
travel_slope = selected_slope * sign(p_z)
slope_entry_side = low,  if travel_slope > 0
slope_entry_side = high, if travel_slope < 0
```

### 3. Choose the reference plane

Order SciFi stations along the travel direction. The first station with a
valid hit in the selected view is the reference plane. Project the frozen track
to that plane's geometry z. The low and high edges are the centres of the first
and last active channels, not the outer mechanical boundaries.

### 4. Choose the target edge

Let `d_low` and `d_high` be the distances to the edge-channel centres:

```text
closest_side = low if d_low <= d_high else high

if min(d_low, d_high) <= 5.0 cm:
    target_side = closest_side
    policy = nearest_edge_within_threshold
else:
    target_side = slope_entry_side
    policy = slope_entry_side
```

The threshold is inclusive and a distance tie selects low. This prevents an
event already near one edge from being moved across the detector only because
of its slope sign. The audit retains both the slope-derived and final sides.

### 5. Convert the displacement into detector IDs

```text
requested_shift = target_edge_channel_centre
                  - track_coordinate(z_reference)
```

Channels are ordered by measured coordinate from initialized SNDSW geometry:
vertical channels/bars by x and horizontal channels/bars by y. Numerical
detector-ID order is not used.

Choose the observed reference-plane anchor whose centre plus
`requested_shift` is closest to the target edge centre, then map it exactly to
the edge channel:

```text
channel_offset = target_edge_ordinal - source_anchor_ordinal
destination_ordinal = source_ordinal + channel_offset
```

The same SciFi offset is used in all five stations of the selected view. Each
compatible MuFilter plane independently quantizes the same requested physical
shift in its own real bar layout because its channel count and pitch differ.

## Detector views

| Selected axis | Remapped views | Copied unchanged |
|---|---|---|
| x | vertical SciFi, vertical Veto, vertical DS | horizontal SciFi, horizontal Veto, all US, horizontal DS |
| y | horizontal SciFi, horizontal Veto, all US, horizontal DS | vertical SciFi, vertical Veto, vertical DS |

US has horizontal bars only, so it represents y but not x. Every orthogonal hit
keeps the identical detector ID, digi payload, and relative collection order.

## Gaps, sparse occupancy, and boundaries

Layouts contain every real channel, including empty ones. Sparse patterns stay
sparse: source ordinals `[100, 101, 104]` shifted by `-90` become
`[10, 11, 14]`, not `[10, 11, 12]`.

Gaps between SciFi dies, 128-channel SiPM arrays, mats, or bars have no detector
ID, so no hit is placed in a gap. Crossing a gap changes the physical
centre-to-centre displacement even with a constant ordinal offset. This is a
discrete channel-space translation, not an exact rigid shift in centimetres.

An out-of-range destination is recorded as `outside_low` or `outside_high` and
dropped. Hits are never clamped, wrapped, merged, or assigned to a fictional
channel. Retained and dropped counts must close to the input count.

## Output and limitation

The writer atomically creates a new ROOT file with the event header,
transformed `Digi_ScifiHits`, transformed `Digi_MuFilterHits`, required FairRoot
metadata, and transformation metadata. It does not copy tracks, clusters,
vertices, Hit3D objects, or features derived from old detector IDs. Those must
be regenerated from the transformed digis.

Signal, time, validity, mask, and other digi payloads are preserved. The
original DAQ identifier is also preserved when the detector ID changes, so the
file is not electronics-consistent raw data. Every output records a
`synthetic_daq_warning`; do not use it for destination-electronics studies
without a separately validated DAQ remapping.

## Audit and validation contract

Each event directory contains:

```text
transformed.root       synthetic SNDSW digi event
mapping_audit.csv      one row per input digi
event_audit.json       complete policy and geometry decisions
comparison.pdf         official original/transformed displays
validation.ok          written only after all hard checks pass
```

The validator independently recomputes the policy and checks Run/Event
identity, output branches, destination IDs, detector/plane/view preservation,
the ordinal equation, injectivity, count closure, payload equality, exact
orthogonal-view identity and order, target-edge placement, travel direction,
ROOT reopening, FairRoot metadata, and the DAQ warning.

Official SNDSW displays provide visual review but do not replace numerical
validation. Their optional blue line is a new display-time Hough fit, not the
frozen track used by the transform.

## Validated Run 8329 samples

| Sample | Events | x / y | Near-edge / slope-edge | SciFi retained | MuFilter retained |
|---|---:|---:|---:|---:|---:|
| High-density review | 10 | 6 / 4 | 3 / 7 | 12,601 / 16,214 | 224 / 233 |
| Density-ranked display review | 100 | 53 / 47 | 27 / 73 | 62,465 / 79,954 | 2,152 / 2,215 |

All 110 outputs have `validation.ok`. The 10-event sample requires
feature-level SciFi occupancy of at least 500. The 100-event sample is
density-ranked without that cutoff and selected 100 track-valid events after
scanning 151 candidates. These are implementation and visual-review samples,
not the complete Run 8329 background production.

## Reproducible commands

```bash
cd /afs/cern.ch/work/z/zhibin/snd-ml
source /afs/cern.ch/user/z/zhibin/env_snakemake.sh

snakemake -s snakemake/snakefile \
  passing_muon_detid_channel_shift_v4_high_density \
  --cores 2 --printshellcmds

snakemake -s snakemake/snakefile \
  passing_muon_detid_channel_shift_v4_display100 \
  --cores 2 --printshellcmds
```

Products are in
`evaluation_region_partitions/passing_muon_detid_channel_shift_v4/`, separated
into `high_density_sample/` and `display_sample_100/`. Each contains selected
and rejected manifests, a combined comparison PDF and audit, and individual
`events/event_<eventId>/` directories.

## Version lineage

- v2 moved both transverse coordinates and is retained only as history.
- v3 introduced dominant-slope single-axis movement and signed-slope entry.
- **v4 is accepted:** it adds the inclusive 5 cm closest-edge override.
- v5 is an experimental comparison and must not replace v4 without a new
  review and explicit acceptance.

## Changes requiring revalidation

Create a new version and repeat numerical and visual validation after changing
any slope/edge threshold, track priority or quality requirement, reference-plane
or target-edge policy, channel ordering, anchor or overflow policy, transformed
detector views, geometry/data-taking period, digi/DAQ mapping, or downstream
use that assumes more than synthetic detector occupancy.
