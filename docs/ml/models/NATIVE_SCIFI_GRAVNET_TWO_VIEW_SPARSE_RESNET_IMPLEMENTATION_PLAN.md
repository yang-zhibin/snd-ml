# Native SciFi GravNet and two-view sparse ResNet implementation plan

Status: implemented and canary-tested; production data regeneration pending
Date: 2026-09-22
Scope: replace Cartesian-product SciFi candidate inputs with detector-native
fiber hits while preserving the existing seven-class sampling, folds,
evaluation, W&B, and HTCondor training workflow

## 1. Decision summary

Implement two new, independently selectable model families:

1. `native_scifi_gravnet`: one graph node for every selected original SciFi
   fiber hit.
2. `two_view_sparse_resnet`: one sparse 2D image for vertical fibers and one
   sparse 2D image for horizontal fibers, followed by late feature fusion.

Do not replace or silently change the existing `pyg_gravnet`,
`clustered_pyg_gravnet`, or `sparse_resnet` families. They remain frozen
candidate-crossing baselines.

The production data contract will be written directly from individual SciFi
digis after the existing QDC/time selection and before
`combine_station_hits_to_3d`. This is necessary because the current hit3D
candidate representation has already discarded information:

```text
vertical hits: Nv       horizontal hits: Nh
current hit3D candidates per station: Nv * Nh
new native nodes/sites per station:    Nv + Nh
```

The current file can support an approximate pilot by deduplicating repeated
`hit_v_channel` and `hit_h_channel` values. It cannot exactly recover separate
vertical/horizontal time or Z, because `digi_2_hits3D.py` currently stores:

```text
candidate z    = (vertical z + horizontal z) / 2
candidate time = min(vertical time, horizontal time)
```

Therefore reconstructed inputs are for software bring-up and speed/memory
tests only. Physics comparisons must use regenerated files with native
branches.

## 2. Why this representation preserves detector resolution

The present candidate input invents a possible three-dimensional crossing for
every vertical-horizontal pair. It does not represent an observed 3D point.
The product can become very large, and 1 cm Cartesian voxelization can merge
several distinct candidates.

The proposed representation keeps the actual detector measurement:

- station/layer identity;
- fiber orientation;
- exact detector channel;
- measured transverse coordinate;
- fiber-plane Z coordinate;
- QDC;
- calibrated hit time, retained in the file for controlled ablations;
- detector ID for validation and traceability.

No Cartesian crossing is created. The sparse ResNet uses integer detector
indices, not centimetre voxels. The GravNet receives continuous measured
coordinates and exact channel identity. If multiple digis occupy the same
station/orientation/channel, GravNet retains separate nodes; the ResNet
explicitly reduces them at the same physical detector cell and records the
occupancy.

## 3. Target ROOT schema

Keep all existing `hit_*` branches and add aligned event-vector branches:

| Branch | Type | Meaning |
|---|---:|---|
| `scifi_native_station` | `vector<short>` | SciFi station, expected 1 through 5 |
| `scifi_native_orientation` | `vector<short>` | 0 = vertical, 1 = horizontal |
| `scifi_native_channel` | `vector<int>` | `layer_channel` from `scifi_channel_info` |
| `scifi_native_det_id` | `vector<int>` or `vector<long long>` | original detector ID, selected after checking its actual range |
| `scifi_native_x` | `vector<float>` | fiber midpoint X from geometry |
| `scifi_native_y` | `vector<float>` | fiber midpoint Y from geometry |
| `scifi_native_z` | `vector<float>` | fiber midpoint Z from geometry |
| `scifi_native_qdc` | `vector<float>` | non-negative QDC for this digi |
| `scifi_native_time_ns` | `vector<float>` | selected/calibrated per-hit time |

All vectors must have exactly the same length and preserve the stable order of
the filtered `SciFi_hits` list. In this project, a vertical fiber is treated as
measuring X and a horizontal fiber as measuring Y, consistent with
`build_one_orientation_hits(..., known_axis="x"/"y")`.

`scifi_native_x` and `scifi_native_y` are both stored for auditing geometry,
but models use only the measured coordinate:

```text
u = x for a vertical fiber
u = y for a horizontal fiber
```

The unmeasured along-fiber midpoint coordinate must not be presented to the
model as if it were a measured particle position.

Add file metadata such as:

```text
metadata_scifi_native_schema = scifi_native_hits_v1
metadata_scifi_native_orientation = 0_vertical_1_horizontal
metadata_scifi_native_selection = post_time_and_qdc_selection
```

## 4. Shared in-memory event contract

Add a native event representation independent of the model:

```python
{
    "station":     int16[N],
    "orientation": int16[N],
    "channel":     int32[N],
    "det_id":      int64[N],
    "x":           float32[N],
    "y":           float32[N],
    "z":           float32[N],
    "qdc":         float32[N],
    "time_ns":     float32[N],
}
```

Validation must reject:

- misaligned vectors;
- non-finite geometry, QDC, or time;
- negative QDC;
- station or orientation outside the defined range;
- an invalid channel;
- a disagreement between orientation and which transverse coordinate is used.

Return event metadata:

```text
n_native_hits
n_vertical_hits
n_horizontal_hits
n_vertical_sites
n_horizontal_sites
```

The first three count digis. The last two count unique sparse detector cells
after same-cell collision reduction.

## 5. Native-hit GravNet logic

### 5.1 Nodes and features

Use one node per native SciFi hit. Version 1 node features are:

```text
u_norm
z_norm
log1p(qdc)_norm
channel_norm
station_norm
is_vertical
is_horizontal
```

This gives `in_channels: 7`. Orientation is one-hot even though one bit is
mathematically sufficient; the explicit representation is easier to audit.
Do not include the unmeasured coordinate. Keep time in the ROOT schema but set
`include_time: false` in v1. Add time only as a named ablation after checking
the MC/data distributions and reference-time convention.

Compute `u`, Z, and `log1p(QDC)` mean/scale from training folds only. Record
the channel and station normalization definitions in the normalization JSON.
Fixed detector-range scaling is preferred for channel and station if geometry
validation confirms stable ranges; otherwise compute training-fold constants
and record that choice.

### 5.2 Message passing and pooling

Use the existing `PyGGravNetClassifier` architecture initially:

```text
7 native features
  -> input MLP
  -> residual GravNetConv blocks
  -> event mean || event max
  -> classification head
  -> seven logits
```

The `batch` vector still prevents edges between different events. Each
GravNet layer learns its own latent coordinates and k-nearest-neighbour graph.
The model may connect vertical and horizontal nodes when their learned
features make that useful; it is not forced to accept all possible crossings.

Start with `k: 16`, the existing hidden size, and three blocks so the input
representation is the main changed variable. Native hit counts are expected
to be much smaller than candidate counts. Measure the maximum real event
before deciding whether a clustered native variant is needed; do not enable
Cluster-GCN partitioning by default.

### 5.3 Batch product

The collator returns:

```text
x:              float32[sum(N), 7]
batch:          int64[sum(N)]
n_nodes:        int64[events]       # native hits, not candidates
n_vertical:     int64[events]
n_horizontal:   int64[events]
model_family:   native_scifi_gravnet
```

## 6. Two-view sparse ResNet logic

### 6.1 Sparse detector coordinates

Build separate integer-coordinate tensors:

```text
vertical view:   [batch_id, station_index, channel]
horizontal view: [batch_id, station_index, channel]
```

This is a detector grid, not a physical-centimetre voxel grid. It therefore
has no `voxel_size`. A fiber channel remains one distinct sparse site.

Initial per-site features are:

```text
log1p(qdc_sum)_norm
occupancy
```

If duplicate digis occur at one detector cell, sum QDC before `log1p` and set
occupancy to the number of contributing digis. This does not merge different
fibers; it only resolves repeated measurements at the exact same detector
cell. Record duplicate rates in validation. Keep time disabled initially; a
later ablation can add QDC-weighted mean time and earliest time.

### 6.2 Two independent encoders

Add two independent `dimension=2` MinkowskiEngine residual towers:

```text
vertical sparse sites   -> vertical ResNet -> mean || max -> vertical embedding
horizontal sparse sites -> horizontal ResNet -> mean || max -> horizontal embedding

[vertical embedding,
 horizontal embedding,
 vertical_present,
 horizontal_present]
    -> fusion MLP
    -> seven logits
```

Use independent towers in v1 because the two orientations can have different
response and topology. Weight sharing can be tested later as a parameter
reduction ablation.

Mean and max pooling must be computed per original event. If an event has no
hits in one orientation, use a zero embedding plus a zero presence flag. Do
not create a fake hit. Events with no native SciFi hits in either view remain
excluded by the current positive-node sampler condition.

### 6.3 Batch product

The collator returns:

```text
vertical_coords, vertical_features
horizontal_coords, horizontal_features
vertical_present, horizontal_present
n_nodes             # native digi count
n_vertical_sites
n_horizontal_sites
model_family: two_view_sparse_resnet
```

Coordinates stay on CPU until MinkowskiEngine constructs each sparse tensor,
matching the current safe device-transfer convention.

## 7. Code implementation plan

### Phase 0: contract tests and a reconstruction audit

Add tests before changing production output.

Files:

- add `test/test_native_scifi_input.py`;
- add `train_3d/audit_native_scifi_reconstruction.py`;
- optionally add a pure helper module
  `train_3d/src/train_3d/data/native_scifi.py`.

Implement a pure validator and a temporary reconstruction function for old
hit3D files. For SciFi rows, reconstruct vertical records from
`(hit_station, hit_v_channel, hit_v_qdc, hit_x)` and horizontal records from
`(hit_station, hit_h_channel, hit_h_qdc, hit_y)`. Include one-orientation rows
where the opposite channel is `-1`.

The audit must report, per file and class:

- candidate count versus reconstructed `Nv + Nh`;
- compression factor;
- duplicate station/orientation/channel keys;
- contradictory QDC or coordinate values for one key;
- empty-orientation and empty-event rates;
- maximum and percentile native-hit counts.

Abort reconstruction on contradictory keys. Exclude time, and label its Z as
approximate. Do not use reconstructed data for final physics results.

Acceptance gate: tests cover crossed, vertical-only, horizontal-only, empty,
duplicate-consistent, and duplicate-contradictory examples, and an audit runs
successfully on a small real file sample.

### Phase 1: write lossless native branches at conversion time

Modify `convertData/digi_2_hits3D.py`:

1. Add `NATIVE_SCIFI_KEYS` and dtypes.
2. Preserve `det_id` when constructing each `SciFi_hits` dictionary.
3. Add `build_native_scifi_hits(SciFi_hits)` after all existing time/QDC
   selection and before station crossing.
4. Return native hits from `process_hits_numpy` along with `all_3dHits`.
5. Add the `scifi_native_*` ROOT vectors in `create_output_root`.
6. Add `fill_native_scifi_vectors` with strict length/range checks.
7. Fill native and legacy branches in the same event, preserving legacy
   output byte semantics as far as practical.
8. Write native schema metadata.

Add converter tests using synthetic hit dictionaries. Validate that the native
count is exactly the number of filtered SciFi digis and is independent of the
Cartesian-product candidate count.

Acceptance gate: for a small regenerated MC and real-data sample, direct
source-digi inspection and native branches agree exactly in station,
orientation, channel, detector ID, QDC, time, and geometry.

### Phase 2: reader and normalization contract

Modify `train_3d/src/train_3d/data/scifi_hit_event_dataset.py`:

- define `NATIVE_SCIFI_BRANCHES`;
- add a `representation`/input-contract selection instead of accumulating
  boolean flags;
- implement native PyROOT and uproot reads;
- add `validate_native_scifi_hits`;
- return native arrays and counts without candidate arrays for native models;
- keep all old readers unchanged for legacy families.

Add a new normalization dataclass rather than changing the meaning of
`SciFiNormalization`, for example `NativeSciFiNormalization`. Version its JSON
as `scifi_native_hit_normalization_v1`.

Update the metadata pipeline:

- `train_3d/src/train_3d/data/scifi_metadata_v2.py`;
- `train_3d/build_scifi_metadata_shard.py`;
- `train_3d/build_scifi_event_size_group.py`;
- `train_3d/reduce_scifi_event_size_files.py`;
- validation and staging helpers.

Add products:

```text
n_native_hits.npy
n_vertical_sites.npy
n_horizontal_sites.npy
fold_N/native_normalization.json
```

Use a new metadata version/root so old arrays and checksums remain valid. Add
`native` and `two_view_sparse` representations to `EventSizeIndex`; both batch
on native workload, with the two-view option using total occupied sites.

Acceptance gate: PyROOT and uproot produce equal arrays for all native fields,
normalization uses training folds only, and binary metadata lookup matches a
direct ROOT scan.

### Phase 3: native GravNet data path and model registration

Modify `train_3d/src/train_3d/data/classification_collate.py`:

- add `NativeGravNetClassificationCollator`;
- construct the seven documented features;
- emit orientation counts and native `n_nodes`.

Modify:

- `train_3d/src/train_3d/data/classification_datamodule.py` to select the
  native reader, collator, normalization, and size representation;
- `train_3d/src/train_3d/models/classifier_factory.py` to register
  `native_scifi_gravnet` using `PyGGravNetClassifier` initially;
- `train_3d/src/train_3d/classification_module.py` to route its forward pass
  and diagnostics;
- `train_3d/src/train_3d/data/classification_labels.py` to validate the new
  exact branch list, representation version, and `in_channels: 7`.

Add `train_3d/configs/classifier_native_scifi_gravnet_v1.yaml`. Give it a new
model and representation version, native metadata root, `include_time: false`,
and W&B tags that cannot be confused with the candidate model.

Acceptance gate: CPU unit forward/backward has finite loss; GPU canary trains
several batches; batch isolation holds; saved checkpoint reload gives the same
evaluation logits; logs clearly report native hit counts.

### Phase 4: two-view sparse collation and model

Modify `classification_collate.py`:

- add `TwoViewSparseClassificationCollator`;
- split by orientation;
- build exact `[station_index, channel]` sites;
- perform deterministic same-site QDC/occupancy reduction;
- return separate coordinates, features, presence flags, and site counts.

Add
`train_3d/src/train_3d/models/two_view_sparse_resnet_classifier.py`:

- factor the current sparse stem/stages into a reusable 2D view encoder;
- create independent vertical and horizontal encoders;
- implement event-wise mean and max pooling with explicit expected event IDs;
- insert zero embeddings for a missing view;
- fuse both view embeddings and presence flags in an MLP head.

Do not silently alter `SparseResNetClassifier`; shared low-level residual
blocks may be reused only if its existing tests remain unchanged.

Register `two_view_sparse_resnet` in the factory, Lightning forward routing,
DataModule, and config validator. Add
`train_3d/configs/classifier_two_view_sparse_resnet_v1.yaml` with
`dimension: 2`, no `voxel_size`, and explicit collision reduction.

Acceptance gate: tests cover both views, one missing view, same-cell
duplicates, multiple events, and event-order permutations; CPU/GPU outputs
have shape `[events, 7]` and finite backward gradients.

### Phase 5: Snakemake and workflow products

Update `snakemake/rules/data_driven_3d_classifier_training.smk` to build and
validate versioned native metadata and to expose both new configs. Update
`snakemake/metadata/configs/training/data_driven_3d_classifiers_v1.yaml` only
by adding new model keys or create a `v2` training manifest if the current
manifest is already treated as frozen.

Recommended names:

```yaml
models:
  native_scifi_gravnet: classifier_native_scifi_gravnet_v1.yaml
  two_view_sparse_resnet: classifier_two_view_sparse_resnet_v1.yaml
```

Make native ROOT schema validation an upstream prerequisite. Do not let a
training job fall back silently to hit3D reconstruction when native branches
are missing.

Acceptance gate: `snakemake -n` resolves metadata, one-fold training,
evaluation shards, and merge products for both models without changing old
targets.

### Phase 6: staged performance and physics validation

Run in this order:

1. synthetic unit and contract tests;
2. one real event through reader, collator, forward, and backward;
3. 100-event loader/profile test with W&B disabled;
4. 2,000-event fold-0 HTCondor canary with W&B offline;
5. the same canary with W&B online;
6. one complete fold with fixed epoch size and timing/memory diagnostics;
7. five-fold comparison only after all gates pass.

Compare against candidate GravNet, clustered candidate GravNet, and 3D sparse
ResNet using the same folds, sampled event references, class probabilities,
objective, and evaluation code. Report:

- events/s, native hits/s, loader wait, step time, and peak GPU memory;
- native hits and sparse sites by class/domain/density bin;
- validation macro F1 and per-class precision/recall/F1;
- confusion matrix;
- performance versus event size;
- reconstructed-pilot versus lossless-native differences, clearly labelled.

## 8. Test matrix

Extend the existing files where appropriate:

- `test/test_scifi_classification_pipeline.py`: reader, validation, collation,
  metadata, missing views, and deterministic reduction;
- `test/test_scifi_classification_models.py`: both model forward/backward paths,
  pooling, batch isolation, and checkpoint compatibility;
- `test/test_initial_training_policy.py`: exact new schemas, representation
  versions, channels, no centimetre voxel size, and time disabled in v1;
- `test/test_train_classifier_smoke.py`: one tiny end-to-end configuration;
- `test/test_training_container_3d_ci.py`: imports and GPU operators in the
  pinned container;
- new converter/native-contract tests for exact digi preservation.

Critical invariants:

1. Every selected SciFi digi appears exactly once in the native ROOT arrays.
2. GravNet creates exactly one node per native array row.
3. The sparse representation creates no site spanning different channels,
   stations, orientations, or events.
4. Labels remain one per event, never one per hit or view.
5. Missing views cannot shift event indices.
6. No validation-fold statistic influences training-fold normalization.
7. Legacy candidate models remain bitwise or numerically reproducible within
   their existing tolerance.

## 9. Initial configuration sketches

Native GravNet:

```yaml
representation:
  version: scifi_native_hits_v1
  nodes: original_fiber_hits
  qdc_transform: log1p
  include_time: false
  include_unmeasured_coordinate: false
model:
  family: native_scifi_gravnet
  in_channels: 7
  hidden_channels: 64
  blocks: 3
  space_dimensions: 4
  propagate_dimensions: 64
  k: 16
  dropout: 0.1
```

Two-view sparse ResNet:

```yaml
representation:
  version: scifi_two_view_detector_grid_v1
  coordinates: [station_index, channel]
  include_time: false
  collision_reduction: qdc_sum_then_log1p_and_occupancy
model:
  family: two_view_sparse_resnet
  in_channels: 2
  channels: [32, 64, 128, 256]
  dimension: 2
  independent_view_encoders: true
  pooling: mean_max
  dropout: 0.1
```

Exact batch budgets should be chosen from measured native-hit/site
percentiles, not copied from the old `max_nodes_per_batch: 20000` candidate
budget.

## 10. Risks and mitigations

- **Old files cannot reproduce native time/Z.** Use reconstruction only for a
  labelled pilot; regenerate before physics training.
- **Duplicate channel digis.** Preserve all nodes in GravNet; use documented
  same-cell reduction and occupancy for the sparse model; quantify the rate.
- **Geometry convention mistakes.** Validate orientation, detector ID,
  channel, midpoint, and station directly against sndsw on MC and real data.
- **Sparse downsampling across a small station axis.** Test active-coordinate
  survival after every stage. If four stride stages collapse too aggressively,
  use anisotropic strides or avoid station-axis downsampling.
- **A missing orientation.** Use zero embedding plus presence flag, never a
  dummy coordinate.
- **Accidental baseline mutation.** New family names, representation versions,
  metadata root, configs, and output groups are mandatory.
- **W&B comparison ambiguity.** Log schema/model versions and counts, and use
  separate groups/tags for native and candidate inputs.

## 11. Recommended implementation order

The shortest safe path is:

```text
pure native contract + reconstruction audit
  -> production native converter branches
  -> PyROOT/uproot native reader parity
  -> native metadata and batch sizes
  -> native GravNet
  -> two-view collator and sparse ResNet
  -> Snakemake targets
  -> GPU/W&B canaries
  -> complete fold comparison
```

The converter and reader contract should be completed before optimizing either
model. Otherwise both models could train successfully on an approximation that
cannot answer the intended detector-resolution question.

## 12. Implementation and test record

Implemented on 2026-09-22:

- lossless `scifi_native_*` branches alongside all legacy `hit_*` branches;
- strict native PyROOT/uproot reader contract and training-only normalization;
- native-hit and sparse-site binary batch-size metadata;
- `native_scifi_gravnet` with seven input features;
- `two_view_sparse_resnet` with independent 2D encoders, mean/max pooling,
  missing-view flags, and channel-only downsampling;
- versioned configs, factory/Lightning/DataModule routing, Snakemake target,
  reconstruction audit, tests, and a GPU benchmark canary.

Validation results:

- 9 native contract/reader/collator/metadata tests passed;
- 9 existing classification-pipeline regressions passed;
- 5 training-policy tests passed;
- 7 PyG/MinkowskiEngine forward/backward model tests passed;
- 6 existing clustered-GravNet regressions passed;
- both new families completed a one-epoch Lightning CPU smoke test and wrote
  best/last checkpoints;
- native converter vectors and metadata passed a ROOT round-trip in the pinned
  sndsw environment;
- HTCondor GPU canary `16236930.0` passed on an NVIDIA A100-PCIE-40GB.

GPU canary workload and result:

| Model | Synthetic workload | Peak allocated | Forward + backward |
|---|---:|---:|---:|
| native GravNet | 20,000 nodes, 2 events | 991.8 MiB | 6.75 s |
| two-view sparse ResNet | 6,000 sites/view, 2 events | 107.5 MiB | 1.91 s |

Both paths returned `[2, 7]` logits and finite gradients. The benchmark is an
operator/memory canary, not a throughput estimate for the final dataset.

The current fold files were inspected after implementation. They contain only
`hit_x`, `hit_y`, `hit_z`, `hit_qdc`, `hit_station`, and `hit_detType`; they do
not contain either the new native branches or the later legacy
`hit_v_channel`/`hit_h_channel` reconstruction fields. Consequently a real
native-input training canary correctly remains gated on regenerating the
hit3D products with the updated converter. No silent fallback is implemented.
