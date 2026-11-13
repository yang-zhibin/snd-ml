# Shower-feature branch: a guide for collaborators

This guide describes `task/shower-features`, checked on 13 November 2026.
Start here to understand the physics workflow and develop a new feature.
The detailed command reference is in [feature_dev/README.md](feature_dev/README.md).

## 1. What this branch does

The aim is to describe the shape of a shower using simple quantities calculated
from SciFi detector hits, then compare those quantities between testbeam data,
collision data, and Monte Carlo (MC).

For example: is the charge concentrated in one cluster or spread across many?
Does most of the shower appear in early or late stations? Do the two detector
orientations give similar amounts of charge?

There are two connected workflows:

```text
Physics workflow
  Digitized ROOT events + detector geometry + event-selection information
      → select/calibrate hits
      → calculate event features
      → validate feature ROOT files and source identities
      → group samples and apply comparison cuts
      → save raw histograms
      → normalize histogram copies and make comparison plots

Algorithm-development workflow
  A fixed export of the selected SciFi hits
      → run baseline and candidate algorithms on exactly the same events
      → validate the calculated features
      → make histograms and a standalone HTML comparison report
```

The second workflow lets a collaborator work on the algorithm without rewriting
ROOT input handling, detector selection, normalization, or plotting. Exporting
real selected hits is a separate preparation step. The supplied development
sample is synthetic; it does not automatically load the physics datasets below.

The existing production converter still uses the baseline algorithm. A candidate
plugin changes the development calculation; promoting it to the production
campaign requires an explicit converter/schema update and physics review.

## 2. Datasets, locations, and quantities

### Where to find things

For the current checkout and campaign:

```bash
cd /afs/cern.ch/work/z/zhibin/worktrees/snd-ml-shower-features

# Existing campaign metadata, plots, and development runs:
CAMPAIGN=/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1

# Existing large production feature files and histogram bundles:
EOS_CAMPAIGN=/eos/experiment/sndlhc/users/zhibin/snd-ml/unified_shower_features_v1
```

These are Zhibin's current locations. A collaborator needs permission to read
them. The development command defaults to the collaborator's own AFS work output
location; `--output-root` can select another writable directory.

The source configuration is
[unified_shower_features_v1.yaml](snakemake/metadata/configs/unified_shower_features_v1.yaml).
Each source manifest records the exact `digi_path`, `geo_path`, `cutflow_path`,
`source_uid`, and `feature_scifi_cluster_v1_path`; use these columns rather than
constructing filenames by hand.

### Source datasets

The following counts are from the saved manifests/configurations, not a fresh
scan of every ROOT event. Input events, selected events, and plotted events are
different quantities.

| Dataset | Catalogued sources and event counts | Raw input location |
| --- | --- | --- |
| 2024 neutrino MC, electron and muon flavours | 1,866 sources; 186,091 input events recorded in the manifest; selected-event column is unfilled | `/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/2024/`, including `nu12/volume_volTarget/`; exact flavour paths in manifest |
| Upward-going muon MC | 15 sources; 3,099,404 input events; direct-input selection | `/eos/experiment/sndlhc/MonteCarlo/MuonBackground/muons_up/` |
| 2024 collision-data skim catalog | 11,368 source files; 11,359 have validated selection records in the saved selection coverage | `/eos/experiment/sndlhc/convertedData/physics/2024/` |
| 2024 testbeam MC, electrons and pions | 75 sources; 95,766 input events | `/eos/experiment/sndlhc/MonteCarlo/testbeam2024/` |
| 2024 H4 testbeam data | 15 sources; 5,399,810 input events; manifest records 430,240 selected events | `/eos/experiment/sndlhc/convertedData/commissioning/testbeam_24/` |
| 2023 H8 pion testbeam data | 385 sources; 370,929,095 input events; 200,000 selected for conversion, 50,000 per particle/energy cell | `/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/` |

Manifest locations relative to `$CAMPAIGN/metadata/`:

- `manifests/mc_neutrino.csv`, `manifests/mc_muon_up.csv`.
- `manifests/testbeam_mc_2024.csv`, `manifests/testbeam_data_2024.csv`.
- `testbeam_2023_h8_v2/testbeam_data_2023_h8.csv`.
- `real_data_selection/real_data_2024_selection_sources.csv`.
- Real-data comparison manifests:
  `real_data_distributions/unified_real_data_distributions_v1/source_manifests/`.

The 2024 testbeam source catalog is
`/eos/user/z/zhibin/TestBeam/binary_full_v1/catalog/master_sources.csv`.
The collision-data source catalog is
`/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/real_data_2024_skim_runs_metadata.csv`.

### What has actually been processed for the saved comparisons?

These are the **saved comparison snapshots**, inspected on 13 November 2026.
A source may produce multiple shard files. Do not sum input-event columns across
shard rows: source-level counts can be repeated.

The size column is the sum of file lengths from `stat` for the referenced,
currently accessible **feature ROOT files**, in GiB (2^30 bytes). It excludes raw
inputs, geometry, selection files, validation receipts, histograms, and plots.
It is not the total storage of the raw datasets.

| Comparison snapshot | Sources / feature files | Feature ROOT size |
| --- | --- | ---: |
| Neutrino MC, partial-distribution snapshot | 1,865 of 1,866 sources; 1,865 files | 1.051 GiB |
| Upward muon MC, partial-distribution snapshot | 10 of 15 sources; 10 files; manifest records 912,922 input events | 0.121 GiB |
| 2024 testbeam MC, V3 source snapshot used by V4 histograms | 75 of 75 sources; 75 files | 0.186 GiB |
| 2024 testbeam data, same snapshot | 14 of 15 sources; 22 files including shards | 1.520 GiB |
| 2023 H8 testbeam data, same snapshot | Snapshot lists 385 sources/files; 384 files currently accessible | 0.127 GiB for accessible files |
| 2024 no-veto collision-data comparison | 1,109 feature files; manifest records 76,972 selected events before plotting cuts | 0.223 GiB |
| 2024 veto-hit 10% comparison | 1,109 feature files; manifest records 3,921,536 selected events before plotting cuts | 8.080 GiB |

The H8 snapshot's coverage JSON says all 385 files were available when recorded.
The current filesystem check found this feature file missing:

```text
$EOS_CAMPAIGN/features/testbeam_data_2023_h8/
  unified_tb23_src_e596bffdd9b7f0e203ef/
  feature_scifi_cluster_v1_unified_tb23_src_e596bffdd9b7f0e203ef.root
```

The saved 180 GeV pion histogram bundle records 88 source files, whereas the
source catalog lists 89 for that cell. Regenerating comparisons should check
coverage again; this guide does not claim the entire catalog is complete.

Coverage evidence lives under `$CAMPAIGN/metadata/`:

- `partial_distributions/unified_partial_distributions_20261007_v1/coverage.json`.
- `testbeam_distributions_v3/feature_snapshots/coverage.json`.
- `real_data_distributions/unified_real_data_distributions_v1/feature_coverage.json`.
- `real_data_distributions/unified_real_data_distributions_v1/source_coverage.json`.

The full validated collision-data selection records 773,927 no-veto events and
39,527,344 events in the deterministic 10% veto sample. Those totals cover the
selection catalog, **not** just the 1,109 processed feature files.

### How many events are in the existing plots?

The common physics comparison cut is `count_scifi > 35`. Counts below are the
`sampled_entries` stored in existing histogram-bundle metadata, after the common
and group cuts and any group cap. Individual features can have additional cuts
or lose entries outside their histogram range.

| Saved plot group | Events |
| --- | ---: |
| Real data, no veto | 71,568 |
| Real data, veto-hit 10% sample | 3,777,757 |
| H4 data electrons, 100 / 150 / 200 / 300 GeV | 14,075 / 43,471 / 50,000 / 36,009 |
| H8 data pions, pi+ 100 / 140 / 180 GeV; pi− 300 GeV | 3,076 / 1,702 / 2,380 / 1,832 |
| MC electron comparison, each of 50 / 100 / 150 / 200 / 250 / 300 GeV | 2,412 each |
| MC pion comparison, 50 / 100 / 150 / 200 / 250 / 300 GeV | 1,115 / 1,183 / 1,211 / 1,210 / 1,217 / 1,260 |

Testbeam configuration contains additional energy groups; configuration alone
does not mean a completed histogram bundle exists for each one. The H4 setup is
**tungsten, two walls / 28 plates**; H8 is **iron, three walls**. Cross-geometry
plots and unequal-energy comparisons are explicitly diagnostic, not controlled
particle-only comparisons.

### The development fixture

[feature_dev/samples/small.jsonl](feature_dev/samples/small.jsonl) contains
**24 synthetic events**, eight per artificial source label, including one empty
hit event. Its JSONL is 32,675 bytes; a completion manifest accompanies it.
The labels illustrate testbeam/collision/MC grouping but are not real data.
Real selected-hit exports for this interface are still a preparation and
acceptance step; the existing production feature ROOT files are not hit exports.

## 3. Hit inputs, clustering, and feature definitions

### What is in a hit?

The raw digitized ROOT input supplies detector identifiers, signals, times, and
validity flags. The converter uses the geometry to obtain positions. The
algorithm-development interface receives **selected SciFi hits**, after detector
selection and timing handling, with these fields:

| Field in development input | Meaning | Production construction |
| --- | --- | --- |
| `station` | SciFi station 1–5 | `GetStation()` |
| `isVertical` | Plane orientation, boolean | `isVertical()` |
| `layer_channel` | Channel coordinate 0–1535 within a plane | `channel + 128*sipm + 512*mat` |
| `qdc` | Nonnegative detector QDC; not energy in GeV | Signal channel 0, sanitized to nonnegative charge |
| `x_cm`, `y_cm`, `z_cm` | Optional geometry coordinates in cm | Renamed from converter `x`, `y`, `z`, using endpoint A from `GetSiPMPosition()` |
| `time_ns` | Optional selected/calibrated time in ns | Renamed from `hitTimeNS` |

The position is the geometry endpoint used by the converter, not a reconstructed
3D shower-space point. The baseline clustering uses station, orientation,
channel, and charge; it does not cluster by time or position.

Code: [build_scifi_hit_dict and process_hits](convertData/digi_2_features.py),
[SelectedHitWriter](feature_dev/selected_hits.py), and
[load_events](feature_dev/engine.py).

Event identity `(runId, eventId, eventIndex)`, `source_uid`, sample group, and
geometry label are stored outside the hit list. They support joining and plotting
but are not passed to the feature plugin. MuFilter/Veto/US/DS hits are processed
by the production converter, but the current development export contains SciFi
hits only. MC truth labels are not algorithm inputs.

### How are hits selected?

`process_hits()` in [digi_2_features.py](convertData/digi_2_features.py):

1. Uses the external SND `selectHits()` for ordinary samples; muon MC starts
   from valid SciFi hits for its dedicated timing treatment.
2. Builds geometry-aware hit dictionaries and rejects invalid detector hits.
3. For muon MC, subtracts the earliest valid detector arrival and selects the
   main calibrated-time group separately in each station/orientation plane.
4. Applies the configured MC late-hit filter. The current helper keeps valid
   times up to 25 ns for MC other than the neutrino path; invalid times are not
   removed by this particular filter. Its sample/path classification logic is
   in `applies_mc_25ns_hit_filter()`.
5. Exports this final SciFi list, if requested, and calculates the features.

The shared muon timing implementation is
[hit_time_selection.py](convertData/hit_time_selection.py). Event selection is
separate: neutrino/real samples use their configured cutflow or selected-entry
indices; testbeam sources use their configured identity/index selection.
See [sample_selection.py](convertData/sample_selection.py).

### Baseline clustering

Implementation: `cluster_scifi_hits()` in
[scifi_plane_clustering.py](convertData/scifi_plane_clustering.py).

1. Split hits into ten planes: five stations × two orientations.
2. Sort each plane by `layer_channel`.
3. Consecutive occupied channels form one cluster. For example,
   `[10, 11, 12, 16, 17]` becomes `[10, 11, 12]` and `[16, 17]`.
4. Never join hits across stations or orientations. Adjacent flattened channels
   can cross a SiPM/mat boundary. Duplicate channels within a plane are rejected.
5. Rank all vertical clusters across stations by descending total QDC, then
   descending hit count, ascending station, and ascending first channel. Do the
   same independently for horizontal clusters.
6. Store the leading two clusters per orientation in fixed feature slots.
   Remaining clusters contribute to the “other clusters” summaries.

Thus `cluster1` means the highest-charge cluster in an orientation, not the
first station. The top two clusters may be in the same station. This is
one-dimensional channel clustering, not a reconstruction of a connected 3D shower.

### What features are calculated?

`_make_cluster()`, `_orientation()`, and `payload_from_analysis()` in
[scifi_plane_clustering.py](convertData/scifi_plane_clustering.py) define the
baseline. `scalar_feature_dtypes()` is the complete list of scalar names/types.

For a cluster with N hits and charges q_i:

| Quantity | Definition |
| --- | --- |
| `nhits`, `qdc` | N and Q = sum(q_i) |
| `qdc_per_hit` | Q/N |
| `qdc_std` | Population standard deviation: sqrt(sum((q_i − Q/N)^2)/N) |
| `qdc_cv` | Standard deviation divided by mean QDC |
| `qdc_roughness` | Mean absolute QDC difference between adjacent ordered hits, divided by mean QDC; zero for a single hit or zero mean |
| Plane/orientation hit and QDC fractions | Cluster contribution divided by the corresponding plane/orientation total |

Each orientation also has:

- Leading/second-cluster ratios, signed station difference, absolute separation,
  same-station flag, and upstream/downstream charge/hit asymmetries.
- Counts, hit/charge totals, mean/std/CV across clusters, and totals outside the
  leading two clusters.
- Effective cluster multiplicity `1 / sum(f_i^2)`, where f_i is each cluster's
  hit or charge fraction. It is zero when the total is zero.
- Five station fractions for hits, QDC, and cluster count; weighted mean station,
  weighted spread, and the station containing the largest contribution.
- Profile slope `sum(x_s*f_s)/10` and curvature
  `sum((x_s^2−2)*f_s)/14`, with `x_s = s−3`, for hit and QDC fractions.
  These summarize the station profile; they are not a track fit in cm.

Event summaries combine orientations, including total clusters/hits/QDC,
QDC per hit, charge outside the top two clusters of **each** orientation, and
`(Q_vertical − Q_horizontal)/(Q_vertical + Q_horizontal)`.

Empty ratios generally return zero. Undefined shape/station quantities use
`-999`; an absent leading/second cluster has `exists=0`, zero count/charge,
and sentinels for undefined fields. Always respect the existence flag or schema
sentinel when plotting.

The production converter additionally calculates selected hit counts/charges,
MuFilter summaries, time summaries, spatial summaries, and
`density_sndsw_scifi` using SND's external `getSumDensity()`. The density is not
calculated by the cluster plugin. `scifi_qdc_per_hit_direct` independently divides
selected SciFi charge by hit count, providing a useful consistency check against
cluster-based QDC/hit. These extra production branches are not automatically
included in the current development plugin's schema.

## 4. What happens after feature calculation?

### Existing physics comparison

1. Write scalar features to ROOT tree `sndData`, with event identity and source
   provenance. Sharded jobs are validated and merged by the production workflow.
2. Validate schema, source information, and event identity. Feature/Hit3D products
   must preserve their `(runId, eventId, eventIndex)` join when used together.
   Code: [validate_unified_shower_feature_file.py](convertData/validate_unified_shower_feature_file.py).
3. Freeze a source snapshot and check required branches. This records which
   sources actually contribute, including missing-source coverage.
4. Apply group cuts and the common `count_scifi > 35` cut. Apply per-group event
   caps and feature-specific cuts, such as `cluster1_exists == 1`.
5. Save unnormalized histogram bundles: **30 one-dimensional and seven
   two-dimensional histograms** for testbeam V4 and real-data V2.
6. Read the bundles, normalize copies, and produce comparison PDFs/manifests.
   Plot styling can be changed without reconverting events.

Key code and configuration:

| Responsibility | File |
| --- | --- |
| Histogram building; unweighted ROOT fills; metadata | [unified_testbeam_histogram_bundles.py](evaluation_region_partitions/unified_testbeam_histogram_bundles.py), shared by the bundle workflows |
| Bundle plotting and normalization | [plot_unified_testbeam_histogram_bundles.py](evaluation_region_partitions/plot_unified_testbeam_histogram_bundles.py) |
| Physics histogram definitions | [unified_testbeam_histograms_v4.yaml](snakemake/metadata/configs/unified_testbeam_histograms_v4.yaml) |
| Real-data definitions inherit the same histograms | [unified_real_data_histograms_v2.yaml](snakemake/metadata/configs/unified_real_data_histograms_v2.yaml) |
| Testbeam group/energy/geometry selection | [unified_testbeam_distributions_v3.yaml](snakemake/metadata/configs/unified_testbeam_distributions_v3.yaml) |
| Real-data cohorts and sampling metadata | [unified_real_data_distributions_v1.yaml](snakemake/metadata/configs/unified_real_data_distributions_v1.yaml) |
| Testbeam / real-data plotting orchestration | [unified_testbeam_distributions_v4.smk](snakemake/unified_testbeam_distributions_v4.smk), [unified_real_data_distributions_v2.smk](snakemake/unified_real_data_distributions_v2.smk) |

Bundles are under `$EOS_CAMPAIGN/histogram_bundles/`; plots are under
`$CAMPAIGN/plots/testbeam_distributions/` and
`$CAMPAIGN/plots/real_data_distributions/`.

### Normalization: shapes versus yields

**Current bundle shape plots:** for bin count n_i,

```text
p_i = n_i / sum(in-range bin counts)
```

Bin fractions sum to one. This compares shapes, not absolute numbers of events.
There is no division by bin width in the bundle plotter's `_unit_area()`.
Two-dimensional distributions are likewise normalized over their in-range bins.
Under/overflow handling and displayed axis ranges must be checked for the chosen
plotter; a zoomed display is not a new event selection. Do not assume a legacy
`fold_overflow_for_plot` setting is honored by every plotter.

**Older distribution plots:**
[unified_feature_distributions.py](evaluation_region_partitions/unified_feature_distributions.py)
supports probability density `n_i/(N*bin_width)`, as well as explicit bin-fraction
plot sets. These have different y-axis units despite both being called
`unit_area` in configuration.

**Neutrino MC expected yields:** the older distribution workflow also supports
`common_exposure_19fb`:

```text
scaled bin = raw bin × 19 / sum(lumi_per_file for unique contributing source_uid)
```

Exposure is summed for the relevant group sources, not once per event or shard.
Upward muon MC has unvalidated equivalent exposure and is excluded from this
normalization. Testbeam samples are event-count comparisons, not luminosity
normalizations. See
[unified_shower_distributions_v1.yaml](snakemake/metadata/configs/unified_shower_distributions_v1.yaml).

**Collision data sampling:** the veto-hit comparison uses a deterministic 10%
sample. Metadata records `event_weight=10`, but the current histogram-bundle
builder fills unweighted counts. A constant factor cancels in unit-normalized
shapes. For absolute yields, the 10% sampling correction and incomplete source
coverage must be handled explicitly; the plotted veto counts are not a full-data
yield estimate.

### Development workflow after the plugin

[feature_dev/Snakefile](feature_dev/Snakefile) runs these functions from
[feature_dev/engine.py](feature_dev/engine.py):

```text
calculate → validate → histograms → report
```

It saves baseline/candidate feature ROOT trees and JSON, checks schema/types,
finite values, bounds, identity/order, ROOT/JSON equality, and optional
algorithm invariants, then plots features that have histogram definitions.

The development report compares baseline versus candidate in separate panels
for each `(group, geometry)`. It uses in-range bin fractions, reports missing
values and under/overflow, and includes the rightmost edge in the final NumPy
bin. It applies **no extra `count_scifi > 35` cut**: the exported sample defines
the population. It does not automatically reproduce the production's 30+7 plots
or apply luminosity normalization. The supplied baseline schema plots six
features; the gap example adds a seventh.

## 5. Develop an algorithm and see its plots

### Step A: run the existing example

```bash
cd /afs/cern.ch/work/z/zhibin/worktrees/snd-ml-shower-features

# Preview the DAG without creating result directories:
./run_feature_dev.sh --algorithm gap_clusters --sample small \
  --run tutorial-gap1 --dry-run

# Execute, or reuse the already completed identical run:
./run_feature_dev.sh --algorithm gap_clusters --sample small \
  --run tutorial-gap1
```

The command uses the repository's LCG environment and an existing Snakemake
installation. For a different machine/runtime, see the environment overrides in
[feature_dev/README.md](feature_dev/README.md).

The example is [gap_clusters.py](feature_dev/algorithms/gap_clusters.py), with
[gap_clusters.yaml](feature_dev/schemas/gap_clusters.yaml). It joins runs separated
by at most `max_missing_channels` (default 1). Thus channels `[1, 3, 10]`
become `[1, 3]` and `[10]`. It adds no hit or charge for the missing channel.
It also calculates:

```text
charge_fragmentation_v1 = 1 − largest_cluster_charge / total_event_charge
```

For charges `[2, 3, 5]`, the baseline has three clusters with charges `[2, 3, 5]`.
The gap example has two with charges `[5, 5]`, giving fragmentation 0.5.
For zero total charge it returns the schema sentinel `-999`.

To retain strict baseline clustering:

```bash
./run_feature_dev.sh --algorithm gap_clusters --sample small \
  --parameters '{"max_missing_channels":0}' --run tutorial-gap0
```

These two tutorial runs have already been generated in Zhibin's output area.

### Step B: understand the interface

A plugin implements two functions:

```python
def cluster_hits(hits, parameters):
    return clusters

def calculate_features(hits, clusters, parameters):
    return {"feature_name": value}
```

`clusters` can be any structure understood by your second function. Reuse the
baseline clustering if you only want a new feature. For a different clustering
algorithm, implement the first function too. If you reuse
`payload_from_analysis()`, preserve its plane/orientation structure and totals;
otherwise return your own features with your own matching schema.

An optional `validate_algorithm(hits, clusters, features)` can raise an exception
if an invariant fails. Return ordinary Python `int`/`float` values, exactly the
schema's feature names, and no NaN/inf. Do not mutate the hits. Keep the algorithm
deterministic and declare adjustable settings through `parameters`.

### Step C: add a feature while keeping baseline clustering

Create `feature_dev/algorithms/my_features.py` with this complete example:

```python
from baseline import cluster_hits, validate_algorithm
from scifi_plane_clustering import payload_from_analysis


def calculate_features(hits, clusters, parameters):
    values = payload_from_analysis(clusters)["scalars"]
    all_clusters = [
        cluster
        for view in clusters["orientations"].values()
        for cluster in view["clusters"]
    ]
    total = sum(cluster["qdc"] for cluster in all_clusters)
    leading = max((cluster["qdc"] for cluster in all_clusters), default=0.0)
    values["leading_charge_fraction"] = leading / total if total else -999.0
    return values
```

Create `feature_dev/schemas/my_features.yaml`:

```yaml
version: my-features-v1
extends: baseline.yaml
features:
  leading_charge_fraction:
    dtype: float64
    description: Fraction of event SciFi QDC in the highest-charge cluster across both orientations.
    missing_value: -999.0
    valid_range: [0, 1]
    histogram: {bins: 25, range: [0, 1]}
```

The inherited baseline schema matches the baseline values returned above; the
new entry describes the extra feature. `histogram` determines whether and how it
is plotted. A candidate-only feature has no baseline curve for that feature.

Run:

```bash
./run_feature_dev.sh \
  --algorithm feature_dev/algorithms/my_features.py \
  --schema feature_dev/schemas/my_features.yaml \
  --sample small --run my-features-v1
```

To develop clustering instead, copy `gap_clusters.py` and its schema as a starting
point, edit `cluster_hits()`, and run your file with `--algorithm` and `--schema`.
Keep charge/hit closure checks when your algorithm is intended to partition all
hits. The supplied gap example's roughness uses consecutive observed channels,
including across a permitted gap; decide whether that definition fits your work.

### Step D: find and share outputs

For Zhibin, the last command writes:

```text
/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/feature_dev/
  my-features-v1/
    manifest.json                 parameters, schemas, hashes, input provenance
    plan.json                     stage plan
    code/                         source snapshots
    calculate/features.json       event identities and both algorithms' features
    calculate/features.root       baseline and candidate ROOT trees
    validate/validation.json      validation result
    histogram/histograms.json     counts, bin fractions, missing/flow counts
    plot/*.png                    individual plots
    plot/report.html              standalone report with embedded images
```

Open `plot/report.html` in a browser or HTML preview. Copy that single HTML file
to share the report. Stage directories are links into the output root's `.cache`;
keep the cache with retained runs, or dereference links when archiving a whole run.

An already generated report is:

```text
/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/feature_dev/tutorial-gap1/plot/report.html
```

Use `--output-root /your/writable/path` to override the default. Use a new `--run`
after changing settings or code. Reusing a run name with different inputs is
rejected to protect previous results.

### Step E: use real events

Prepare a small fixed export once per source using the existing production
converter command and its matching SNDSW environment. Append:

```text
--max-events 100
--feature-dev-export /your/afs/work/samples/testbeam-electron100.jsonl
--feature-dev-group testbeam-electron-100GeV
```

Keep the source's original input, geometry, selection arguments, distinct
`--source-uid`, and correct `--source-geometry-id`. Both
[digi_2_features.py](convertData/digi_2_features.py) and
[digi_2_features_mc_muon.py](convertData/digi_2_features_mc_muon.py) support export.
A completed `.manifest.json` is required; incomplete or modified exports are
rejected. An existing export cannot be overwritten.

After exporting real testbeam, collision, and MC samples, merge them in LCG
(replace the example absolute paths with your exports):

```bash
set +u
source snakemake/envs/env_lcg.sh
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

python3 feature_dev/selected_hits.py /your/afs/work/samples/comparison.jsonl \
  /your/afs/work/samples/testbeam.jsonl \
  /your/afs/work/samples/collision.jsonl \
  /your/afs/work/samples/mc.jsonl

./run_feature_dev.sh \
  --algorithm feature_dev/algorithms/my_features.py \
  --schema feature_dev/schemas/my_features.yaml \
  --sample /your/afs/work/samples/comparison.jsonl --run real-comparison-v1
```

The merger preserves source/group/geometry labels and checks duplicate identities.
Once these selected-hit exports exist, algorithm iterations do not reread raw
ROOT or rerun hit selection. Validate the exported baseline against the original
production feature branches before using the results for physics. This real
SNDSW export/acceptance step has not yet been exercised by the interface tests.

### What must rerun after an edit?

| Edit | Development work required |
| --- | --- |
| Same code, sample, parameters, schema | Reuse completed results |
| Histogram bins/ranges or descriptions only | Histograms and report |
| Feature formula, clustering, parameters, or value/type contract | Feature calculation onward |
| Selected hits or their completion manifest | Feature calculation onward |
| Timing/event selection in raw conversion | Export a new sample first, then calculate onward |

The cache is based on hashes and runtime versions. External helper source files
must be declared with `--dependency /path/to/helper.py`; see the detailed guide.
Changing shared framework code conservatively invalidates calculation too.

### How to test your change

In the repository's LCG environment:

```bash
set +u
source snakemake/envs/env_lcg.sh
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

python3 -m unittest discover -s feature_dev/tests -v
python3 -m unittest discover -s convertData -p 'test_scifi_plane_clustering.py' -v
python3 feature_dev/tests/integration.py
```

Add a small known-answer test for your own algorithm, including empty/zero-charge
input and any new boundary cases. Then run the synthetic fixture, inspect the
plots and validation result, and finally use an approved fixed real sample.

The interface implementation passed 41 unit/regression tests, 10 end-to-end
acceptance checks, and 100 randomized comparisons with the previous baseline.
Evidence and limitations: [feature_dev/TEST_RESULTS.md](feature_dev/TEST_RESULTS.md).
Software validation checks consistency; it does not establish that a new feature
is physically useful or that different detector geometries are directly comparable.
