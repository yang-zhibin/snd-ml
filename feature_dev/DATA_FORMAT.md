# Data formats and structure

This guide follows data through the current shower-feature production workflow.
It distinguishes **measured detector information**, **MC truth**, and
**features calculated from hits**.

## 1. Files and trees at a glance

| Product | Main structure | One entry represents | Purpose |
| --- | --- | --- | --- |
| Input digi ROOT | Catalogue-selected tree, e.g. `cbmsim` for inspected neutrino MC | One input event, containing collections of hits/tracks | Detector input and, for MC, simulation truth |
| Geometry ROOT | Detector geometry/configuration | Detector description | Convert detector IDs into positions |
| Selection/cutflow ROOT | Selected raw-entry indices and selection markers | Selection record | Choose input events to process |
| `sources/<index>/features.root` | `sndData` tree + metadata | One saved selected event | Standard and coworker features, selected truth summaries |
| `sources/<index>/hits.jsonl` (small runs) | JSON object per line | One selected event and its SciFi hits | Inspect plugin inputs and validate baseline parity |
| Separate Hit3D ROOT | `hit3D` tree | One event with variable-length hit vectors | Derived spatial representation for other production/ML workflows |
| Histogram ROOT | `hist`/`hist2d`, optional weighted copies, metadata | Aggregate distribution | Plotting and normalization checks |

The current `run_shower_features.sh` writes feature ROOT files. It does **not**
also run the separate Hit3D converter. Small runs additionally export selected
hits; full production omits that debugging export.

Source paths and input tree names come from catalogue rows in `plan.json` →
`jobs`. Catalogue locations are configured in [datasets.yaml](datasets.yaml).
Use that recorded tree name rather than assuming every input has `cbmsim`.

## 2. What information does raw data contain?

Here “raw input” means the **digitized ROOT file used by the converter**, not
the original DAQ byte stream.

| Object | Useful information | Notes |
| --- | --- | --- |
| `EventHeader` (stored as `EventHeader.` in the inspected MC file) | Run/event identity | Converter uses header accessors, with an MC-entry fallback |
| `Digi_ScifiHits` | Detector ID, station, orientation, mat/SiPM/channel, signal, time, validity | One-dimensional fibre readout |
| `Digi_MuFilterHits` | Detector ID, subsystem/bar, orientation, per-channel signals/times, validity | Veto, upstream and downstream detector hits |
| `Reco_MuonTracks`, when present | Reconstructed tracks | Optional; extraction requires converter configuration |
| Geometry file | Fibre/bar endpoints and detector calibration/configuration | Positions are obtained using the geometry, not read as particle truth |

The converter selects valid hits and applies the configured timing selection.
Plugin input is the resulting **selected SciFi hits**, not every raw hit.

QDC is detector signal/charge in QDC units, **not GeV**. SciFi QDC is taken from
`GetSignal(0)` and sanitized to a nonnegative value; MuFilter summaries add
nonnegative channel signals. Time conventions differ at input: the current
converter takes MC SciFi times in ns and converts/corrects real-data SciFi
clock values. The plugin receives the converter's time in ns.

Real collision data have no MC truth. Testbeam beam energy/particle labels can
come from source metadata; those are beam settings, not event-by-event truth.

## 3. What truth information does MC have?

### In the input MC ROOT file

| Collection | Information |
| --- | --- |
| `MCTrack` | Particle PDG code, mother-track index, momentum, mass/energy accessors, production/start position and time; also stored weight/process fields |
| `ScifiPoint`, `MuFilterPoint`, `vetoPoint` when present | Simulated detector points, including the track association; distinct from digitized hits |
| `Digi_ScifiHits2MCPoints`, `Digi_MuFilterHits2MCPoints` when present | Associations between digitized hits and simulated points |

A mother ID refers to another track in the **same event**. Hit-to-point links
must be followed to the point's track ID; do not assume digi-hit index equals
MCTrack index. These collections can have different lengths.

The inspected neutrino file contains all three point collections and both
hit-to-point maps. Other datasets/releases may differ. Raw `MCTrack` weight
fields are not automatically used as luminosity weights by this plotter; see
[normalization](NORMALIZATION.md) for the audited exposure calculation.

### Truth retained in the feature tree

| `sndData` field | Current converter meaning | Unit |
| --- | --- | --- |
| `isMC` | 1 for simulation, 0 for real data | Flag |
| `mc_incoming_energy` | `MCTrack[0].GetEnergy()` when available | GeV |
| `px`, `py`, `pz` | Momentum of `MCTrack[0]` | GeV/c |
| `x`, `y`, `z` | Start position of `MCTrack[1]` when available | cm |
| `pdgCode` | Incoming particle code with the neutrino CC/NC encoding below | Integer label |
| `source_energy_gev`, `source_pdg_code` | Source/beam information resolved from converter arguments and MC where applicable | GeV / PDG code |
| `source_domain_id` | Source-domain identifier from the manifest | Integer label |

`x/y/z` here are truth-derived coordinates, **not a reconstructed shower
position**. The mapping to track 0/1 is the converter's convention; the feature
file does not copy the complete MCTrack collection.

For neutrino plots, the converter uses:

| `abs(pdgCode)` | Plot class |
| --- | --- |
| 12 | `CC_nue` |
| 14 | `CC_numu` |
| 112 | `NC_nue` |
| 114 | `NC_numu` |

NC is encoded when the first two MC tracks have the same neutrino PDG code:
100 is added to its magnitude, preserving sign. Thus 112/114 are **project
labels**, not standard PDG particle codes. The plotter groups particles and
antiparticles using the absolute value. The current neutrino plotting map
supports these four classes.

For muon-DIS processing, additional `primary_*` and `secondary_*` vectors save
PDG, energy, start position and momentum. Primaries have mother ID −1;
`nSecondaryRaw` counts tracks with mother ID 0, while saved secondaries must
also be linked to a valid SciFi digi hit. `secondary_interaction*` stores a
child-track start position as an interaction proxy. These vector branches are
not a full truth record and may be empty outside that processing mode.

For real data, `pdgCode` is 0 and unavailable MC quantities remain missing.
Never use truth labels or truth energy as detector-derived input features.

## 4. Hit features and cluster structure

A **hit feature** describes one hit. An event has a variable number of hits.
The coworker plugin receives `hits`, a list of dictionaries:

| Key | Meaning | Type / unit |
| --- | --- | --- |
| `station` | SciFi station | Integer, 1–5 |
| `isVertical` | Fibre orientation | Boolean |
| `layer_channel` | Channel index within that station/view | Integer, 0–1535 |
| `qdc` | Nonnegative signal | Float, QDC units |
| `x_cm`, `y_cm`, `z_cm` | Geometry position, when available | Float, cm |
| `time_ns` | Converter-selected/corrected time, when available | Float, ns |

Channel calculation: `SiPMChan + 128 × SiPM + 512 × mat`.
Use `(station, isVertical, layer_channel)` to identify a channel within an event;
a channel number alone is not unique across detector planes.

Positions currently come from endpoint A returned by `GetSiPMPosition`.
A vertical fibre constrains x; a horizontal fibre constrains y. The full endpoint
coordinate tuple is **not a measured 3D crossing point**. Check optional fields
with `hit.get(...)`; missing coordinates do not mean zero.

The supplied gap-clustering example returns:

```text
clusters
  planes[(station, is_vertical)]
    clusters[]
  orientations['vertical' or 'horizontal']
    clusters[]
    ranked_clusters[]
```

Each cluster has `station`, `is_vertical`, aligned member `channels`/`qdcs`,
`nhits`, summed `qdc`, and `orientation_rank` (1 is leading in that view).
Clusters are algorithm-defined objects; other plugins can use another structure.
These Python objects are not automatically serialized into the event tree.

The plugin receives selected SciFi hits and parameters, **not MC truth, raw ROOT
objects or MuFilter hit lists**. It returns the scalar outputs declared in its
schema. See the [interface reference](../feature_plugins/README.md) and
[annotated example](algorithms/gap_clusters.py).

## 5. Event features

An **event feature** summarizes hits/clusters for one event. Each scalar branch
in `sndData` has one value per saved event.

| Family | Examples | Meaning / unit |
| --- | --- | --- |
| Identity | `runId`, `eventId`, `eventIndex` | Integer identifiers |
| Hit counts | `count_scifi`, `count_scifi1`, `count_us`, `count_ds` | Selected detector-hit counts |
| Signal sums | `qdc_scifi`, `qdc_scifi1`, `qdc_us` | Summed QDC |
| Spatial summaries | `avg_scifi_x`, `qdcAvg_scifi_x`, `scifi_std_x` | Mean, charge-weighted mean, spread; cm |
| Shower development | `scifi_first_station`, `scifi_n_active_stations`, `scifi_count_frac1` | Station indices/counts and dimensionless fractions |
| Density | `density_sndsw_scifi`, `density_mycode_scifi` | Defined hit-neighbour statistics; see [density guide](SCIFI_DENSITY.md) |
| Timing summaries | `hitTime_scifi_mean_ns`, `hitTime_scifi_std_ns` | ns |
| Plugin outputs | `silvia__leading_charge_fraction`, `silvia__scifi_vertical_cluster1_density` | Units and missing values declared in plugin schema |

Most baseline scalar branches use integer/float/double storage. Plugin scalars
are `int64` or `float64`. Many baseline fields use −999 for missing values;
plugin sentinels are specified in [schema.yaml](../feature_plugins/silvia/schema.yaml).
Zero can be a valid count or signal and should not be treated as missing.

The tree also has variable-length branches such as `hitTime_raw_ns`,
`hitTime_raw_detType` and the truth vectors above. Their presence does not make
`sndData` a full copy of the raw detector event. Optional reconstructed-track
branches depend on extraction settings. Use the file's branch list for its
exact schema rather than assuming every historical output is identical.

## 6. Separate Hit3D format

[convertData/digi_2_hits3D.py](../convertData/digi_2_hits3D.py) creates a `hit3D`
tree with event identifiers and aligned variable-length arrays:

| Branch family | Content |
| --- | --- |
| `hit_x`, `hit_y`, `hit_z` | Derived positions in cm |
| `hit_qdc`, `hit_time_ns` | Signal and time |
| `hit_station`, `hit_detType` | Station and subsystem |
| `hit_ix`, `hit_iy`, `hit_iz`, `hit_index_valid`, `hit_index_type` | Discrete representation indices and validity/type |
| `hit_v_channel`, `hit_h_channel`, `hit_v_qdc`, `hit_h_qdc` | Associated vertical/horizontal channels and signals |
| `scifi_native_*` | Native selected SciFi station, orientation, channel, detector ID, position, QDC and time |

`hit_*` arrays have matching lengths within an event. Native SciFi arrays form
a separate aligned collection and need not have the same length as `hit_*`.
Native orientation uses **0 = vertical, 1 = horizontal**, unlike the plugin's
`isVertical` boolean. Detector type codes follow 0 = SciFi, 1 = veto,
2 = upstream, 3 = downstream; actual output contents depend on the conversion.

Crossed-view points and voxelized detector representations are derived from
readout geometry. One derived point is not necessarily one measured hit or one
truth energy deposit.

## 7. Event identity and file metadata

`eventIndex` is the **original raw-tree entry index**, not the row number in the
filtered feature tree. Join aligned feature and Hit3D events using
`(runId, eventId, eventIndex)` and verify identity/selection consistency. When
combining independent sources, retain source identity as well; matching row
numbers alone do not establish an event match.

Source feature ROOT files also retain source/geometry metadata, plugin schema
and hashes (`feature_plugin_json`), and exposure information (`source_exposure`,
`production_exposure_json`). These describe a source or configuration, not an
event. See [normalization](NORMALIZATION.md) for luminosity fields and weights.

## 8. Inspect a feature file

Use the documented [LCG environment](../docs/project/ENVIRONMENT_AND_SNAKEMAKE.md):

```python
import uproot

with uproot.open('/path/to/sources/0/features.root') as root:
    print(root.keys())
    events = root['sndData']
    print(events.num_entries)
    print(events.keys())
    rows = events.arrays(
        ['runId', 'eventId', 'eventIndex', 'count_scifi',
         'mc_incoming_energy', 'silvia__leading_charge_fraction'],
        entry_stop=5, library='np',
    )
    print(rows)
    print(root['source_exposure'].arrays(library='np'))
```

For variable-length branches use `library='ak'`. Reading raw detector objects
through their methods requires the separate SNDSW environment; do not mix it
with the LCG analysis environment in one process.

Verified on 19 November 2026 against converter code and one raw neutrino MC
source used by `larger-mc-20261119`. Its feature file has 70 selected events and
451 branches, including 13 Silvia branches. Those counts describe that output,
not a fixed schema requirement. Real-data/testbeam and Hit3D details above are
code/interface references, not newly validated files from those datasets.
