# SciFi SiPM event-display implementation plan

Status: one-event prototype implemented and rendered
Created: 2026-09-08
Scope: one-event prototype before workflow integration

## Objective

Keep the existing SNDSW full-detector event display and add drill-down pages
where individual SciFi SiPM channels are readable. Channel placement must come
from the event's geometry so ordinary channel pitch, the central die gap,
inter-SiPM gaps, and mat boundaries remain visibly different.

## Prototype event

Use the first event from the existing signal-region CC-nue display:

- particle group: `CC_nue`;
- region: `signal_no_veto_has_us_fiducial_inside`;
- displayed MC run/event: `1752482844 / 1`;
- historical display annotation: `count_scifi = 1078`,
  `density_sndsw_scifi = 32396`;
- reference: `evaluation_region_partitions/event_display/events__Signal__signal_no_veto_has_us_fiducial_inside__CC_nue.pdf`.

This high-occupancy event is a useful stress test for channel readability.
Before rendering, resolve and record its partition ID, feature entry,
`original_entry`, `eventIndex`, source feature file, source DIGI file, event
tree, and geometry file using the current `SourceFeatureMapper` workflow.

## Prototype output

Produce one vector PDF plus a JSON identity/validation sidecar.

1. **Full-detector overview:** reuse the current SNDSW `z-x` and `z-y` page
   without changing its rendering.
2. **Station summary:** automatically choose the SciFi station with the most
   valid hits and show its horizontal and vertical planes separately.
3. **SiPM details:** make one enlarged 128-channel page for each fired SiPM in
   that station, subject to a configurable page limit.

The station page uses one continuous physical-coordinate strip across all
three mats and twelve SiPMs. Its height is enlarged for visibility, but its
horizontal distances remain physical. The detailed pages include channel
status, QDC versus channel, and time versus channel.

Channel styling:

- white: no recorded hit;
- Viridis colour: valid hit, using `GetSignal()` QDC;
- grey/red hatch: recorded hit with `isValid() == false`;
- optional marker: unusable time.

An absent digi is described as `no recorded hit`, not `healthy`; a full
healthy/masked/dead classification requires run-specific conditions data.

## Geometry and gap handling

Do not assign equally spaced channel coordinates. Load the event geometry and:

1. construct `SndlhcGeo.GeoInterface(geofile)` and retain the Scifi module;
2. call `Scifi.SiPMOverlap()`;
3. read channel nodes from `ROOT.gGeoManager.FindVolumeFast("SiPMmapVol")`;
4. decode each node number into local mat, SiPM, and channel;
5. use the node translation as the channel centre;
6. use the `ChannelVol` box half-width as the rectangle half-width;
7. rotate the measured coordinate appropriately for vertical planes.

For the current TI18 geometry, validation should observe approximately:

| Adjacency | Centre distance | Empty gap |
|---|---:|---:|
| ordinary neighbours | 0.25 mm | 0.00 mm |
| channel 63 to 64 | 0.45 mm | 0.20 mm |
| channel 127 to next SiPM channel 0 | 0.65 mm | 0.40 mm |

The fibre-mat gap is a separate 0.50 mm feature. Use a thin dashed separator at
the channel 63/64 boundary, a medium separator between SiPMs, and a thick
labelled separator between mats. The values above are validation expectations,
not production placement constants.

## Hit mapping

For each `Digi_ScifiHits` object, extract:

```python
station = hit.GetStation()
vertical = hit.isVertical()
mat = hit.GetMat()
sipm = hit.GetSiPM()
channel = hit.GetSiPMChan()
detector_id = hit.GetDetectorID()
qdc = hit.GetSignal()
time = hit.GetTime()
valid = hit.isValid()
```

Map by `(mat, sipm, channel)`. Independently call
`Scifi.GetSiPMPosition(detector_id, A, B)` and require its measured coordinate
to agree with the `SiPMmapVol` channel centre within a documented tolerance.

## Code organization

Add `evaluation_region_partitions/scifi_sipm_event_display.py` with separated
functions:

- `load_sipm_geometry(...)`;
- `validate_sipm_geometry(...)`;
- `extract_event_sipm_hits(...)`;
- `select_detail_station(...)`;
- `draw_station_summary(...)`;
- `draw_sipm_detail(...)`;
- `write_event_identity(...)`.

Use ROOT/SNDSW for event and geometry access and Matplotlib for precise vector
rectangle rendering. Do not modify upstream SNDSW for the prototype. Generate
the overview with the existing runner, generate detail pages independently in
the same SNDSW environment, and merge them afterward.

Proposed direct CLI:

```text
--input ROOT_FILE --tree TREE --event-index INDEX --geo GEO_FILE
--output-pdf OUTPUT --identity-output OUTPUT_JSON
--station auto|1|2|3|4|5 --orientation both|horizontal|vertical
--detail-mode fired-sipms|all-sipms --max-detail-pages N
```

## Validation criteria

- The overview matches the existing page for the chosen event.
- The complete event identity is recorded and reproducible.
- Every recorded SciFi hit maps to exactly one channel rectangle.
- Detector-ID decoding and geometry-node decoding agree.
- `GetSiPMPosition()` and the channel-node position agree.
- Ordinary, central-die, inter-SiPM, and mat boundaries are distinct.
- Hit counts close globally and by station/orientation.
- Invalid recorded hits remain distinct from absent hits.
- Inputs are never modified; output is vector PDF plus an audit sidecar.

## Integration after prototype review

After visual approval, add an optional `scifi_detail` YAML block, integrate the
renderer with `sndsw_event_display_common.py`, interleave overview/detail pages,
extend rendered-event CSV rows with page and channel identity, and expose a
separate Snakemake target so normal event-display production is unaffected.

## Work log

- 2026-09-08: inspected the region-event-display and SNDSW SiPM geometry code.
- 2026-09-08: chose CC-nue run/event `1752482844 / 1` as the prototype.
- 2026-09-08: recorded this plan; implementation intentionally not started.
- 2026-09-09: resolved the current partition identity to part 001, feature
  entry 3, original/source event entry 15, run/event `1752482844 / 1`.
- 2026-09-09: added `evaluation_region_partitions/scifi_sipm_event_display.py`.
- 2026-09-09: corrected the native SciFi mat convention to zero-based 0--2
  after the first closure test caught the original 1--3 assumption.
- 2026-09-09: rendered a six-page vector PDF (overview, station summary, four
  SiPM details) and a JSON identity/geometry audit. All 1,354 recorded SciFi
  digis map to a geometry channel; station 5 is selected automatically.
- 2026-09-09: geometry measured from this geofile gives 0.25 mm ordinary
  pitch, 0.45 mm across channels 63/64, and 0.65 mm across SiPM and mat
  boundaries. The corresponding empty gaps are 0.00, 0.20, and 0.40 mm.
  The fibre-mat construction gap is not an additional spacing in SiPMmapVol.
- 2026-09-10: replaced the single selected-station summary with two wide
  orientation summaries. Page 2 contains every horizontal plane with recorded
  digis and page 3 every vertical plane with recorded digis; each station is
  one row containing its three mats. The four enlarged SiPM pages remain after
  these overview pages.
- 2026-09-10: doubled the two all-plane page widths from 22 to 44 inches to
  improve individual-channel readability. Added `--plane-page-width` so the
  physical PDF width can be adjusted directly (for example, 66 for 3x).
- 2026-09-10: increased the default all-plane width to 66 inches (3x the
  original) and added optional standalone PNG output via `--plane-png-dir`
  and `--plane-png-dpi`. The event PDF remains vector; the PNGs provide
  predictable full-size rendering outside PDF viewers that downsample previews.
- 2026-09-10: removed individual-SiPM detail pages from the default document.
  The normal output is now exactly three pages: full event, all active
  horizontal planes, and all active vertical planes. Detail rendering remains
  available only when `--max-detail-pages` is set above zero.
- 2026-09-10: exposed `render_plane_pages()` and a reusable geometry context,
  then integrated the renderer into an isolated Run-8329 V7 workflow. The new
  target interleaves `overview`, `scifi_horizontal`, and `scifi_vertical` for
  each event in a separate output directory.
- 2026-09-10: expanded the workflow to mirror the complete current
  `event_display_loose_coverage/` hierarchy: four base categories plus three
  veto categories for each of five SciFi criteria. Each of the 19 existing
  per-category summaries supplies its exact first 25 event identities, for 475
  category/event entries and 1,425 validated output pages in total. All 475
  identities were matched against `selected_events.csv`; a real additional-
  criterion event produced the required three pages with zero unmapped hits;
  the 21-job Snakemake DAG and synthetic 1,425-page merger closure both passed.
- 2026-09-10: added a QDC-versus-channel-position panel directly beneath each
  active station's channel-status strip. Both panels use the same transformed
  geometry coordinate, so every QDC bar aligns with its physical channel and
  the unequal die, SiPM, and mat gaps remain visible. Valid hits use the shared
  event QDC colour scale; recorded invalid hits are red crosses.
