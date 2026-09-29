# Reconstructed-track feature schemas

`digi_2_features.py --extract-reco-tracks` reads existing `sndRecoTrack`
objects from `Reco_MuonTracks`; it does not rerun track reconstruction.

## Schema selection

`flat_v1` is the unchanged default. `flat_v2` is selected with:

```bash
--extract-reco-tracks --reco-track-schema flat_v2
```

The selected schema is recorded in `metadata_reco_track_features`. The
passing-muon v1 and v2 productions use separate targets and output directories.

## flat_v2 track vectors

These additions contain one element per track and align with existing track
vectors:

```text
reco_track_n_measurements
reco_track_n_unique_planes
reco_track_n_scifi_measurements
reco_track_n_veto_measurements
reco_track_n_us_measurements
reco_track_n_ds_measurements
reco_track_n_unknown_measurements
reco_track_n_scifi_planes
reco_track_n_veto_planes
reco_track_n_us_planes
reco_track_n_ds_planes
reco_track_measurement_alignment_ok
reco_track_first_measurement_detid
reco_track_last_measurement_detid
```

`n_measurements` is the size of `sndRecoTrack::getRawMeasDetIDs()`. It is an
associated-hit count for Hough tracks but an associated-cluster count for
simple tracks. It is not the exact number of individual hits inside simple
track clusters.

`n_unique_planes` counts planes that supplied track measurements. It does not
count physical detector planes crossed without an associated measurement.
Unknown IDs contribute to `n_unknown_measurements`, but not to plane counts.

The same detail fields are written for each selected best track as
`<track-type-prefix>_best_<field>`. Best-track selection remains unchanged.

## flat_v2 measurement vectors

Measurements from every event track are flattened into:

```text
reco_track_measurement_detid
reco_track_measurement_point_x
reco_track_measurement_point_y
reco_track_measurement_point_z
reco_track_measurement_subsystem
reco_track_measurement_station
reco_track_measurement_plane
reco_track_measurement_view
reco_track_measurement_plane_key
```

Use `reco_track_measurement_offset` to obtain measurements for track `i`:

```python
begin = event.reco_track_measurement_offset[i]
end = event.reco_track_measurement_offset[i + 1]
detids = event.reco_track_measurement_detid[begin:end]
point_z = event.reco_track_measurement_point_z[begin:end]
```

The invariants are:

```text
len(reco_track_measurement_offset) == reco_track_n_all + 1
offset[0] == 0
offset[-1] == number of flattened measurements
```

Point and detector-ID vectors must have identical sizes. This follows the
`sndRecoTrack` constructor, which appends both in the same loop. A mismatch
stops flat_v2 extraction instead of silently misaligning physics data. This
check is not run by flat_v1.

## Detector decoding

`reco_track_measurement_subsystem` uses:

| Code | Meaning |
| ---: | --- |
| 0 | unknown |
| 1 | SciFi |
| 2 | Veto |
| 3 | upstream MuFilter (US) |
| 4 | downstream MuFilter (DS) |

For a confirmed SciFi ID:

```text
station = detID // 1,000,000
view = (detID // 100,000) % 10
plane identity = (station, view)
plane_key = 100 + 10 * station + view
```

View 0 is horizontal and view 1 is vertical. `plane` is `-1` for SciFi.

For a confirmed MuFilter ID:

```text
system = detID // 10,000       # 1 Veto, 2 US, 3 DS
plane = (detID // 1,000) % 10
plane identity = (subsystem, plane)
plane_key = 100 * subsystem_code + plane
```

`station` and `view` are `-1` for MuFilter. Unrecognized IDs retain their raw
ID and use decoded sentinels `0, -1, -1, -1, -1`.

## Validation and production

Validate a v2 file with:

```bash
python validate_reco_track_feature_file.py output.root --schema flat_v2
```

The validator checks base v1 invariants plus offsets, measurement-vector
alignment, count closure, plane-count closure, and first/last ID consistency.

The dedicated passing-muon v2 target is:

```bash
snakemake -s snakemake/snakefile passing_muon_track_features_v2 ...
```
