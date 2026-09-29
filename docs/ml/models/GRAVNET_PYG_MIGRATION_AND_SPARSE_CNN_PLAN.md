# Data-driven seven-class GravNet and sparse-CNN implementation plan

## 1. Objective

Build two event classifiers on the `data_driven_3d_cnn_v1` dataset:

1. a sparse 3D residual CNN using MinkowskiEngine;
2. a GravNet classifier using the maintained PyTorch Geometric
   `GravNetConv` implementation.

Both models must use the same physical events, five-fold partitions, training
sampling policy, seven-class label order, validation events, and evaluation
metrics. This makes their comparison meaningful.

The initial input study is now deliberately restricted to SciFi. For every
event, select only rows with `hit_detType == 1` and expose only the 3D
position, QDC, and detector type. Do not use MuFilter hits, station, timing,
event energy, reconstructed tracks, occupancy, or other engineered detector
features in the initial models.

The first GravNet implementation should not extend the legacy code under
`training/models/GravNet/`. That code remains available as a historical
reference and, if useful, as a separately named legacy baseline. The new model
should import `torch_geometric.nn.GravNetConv` from a pinned, tested PyG
release and put only the SND-specific encoder, stacking, pooling, and
classification head in this repository.

The software and batch deployment are specified separately in
`document/TRAINING_CONTAINER_AND_HTCONDOR_ENVIRONMENT_PLAN.md`. The production
target is one pinned container that supports GPU training and CPU evaluation;
LCG remains useful for dataset production and as a possible build base, but
stock LCG 110 is not sufficient for both training backends.

### 1.1 Frozen initial decisions

- task: seven-class event classification with `CC_nutau` excluded;
- selected detector: SciFi only (`hit_detType == 1`);
- allowed raw inputs: `hit_x`, `hit_y`, `hit_z`, `hit_qdc`, and
  `hit_detType`;
- GravNet representation: one node per selected SciFi 3D candidate crossing;
- sparse-CNN representation: the same selected candidates quantized into
  sparse voxels, with QDC summed on coordinate collisions;
- training: GPU, including the natural validation pass used for checkpointing;
- final exhaustive evaluation: shardable across CPU HTCondor jobs and merged
  from additive statistics;
- experiment tracking: new W&B project `snd-ml-data-driven-3d`.

## 2. Authoritative references

- Original GravNet/GarNet paper:
  <https://arxiv.org/abs/1902.07987>
- Maintained PyG layer documentation:
  <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GravNetConv.html>
- Maintained PyG source:
  <https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gravnet_conv.py>
- PyG releases:
  <https://github.com/pyg-team/pytorch_geometric/releases>
- Minkowski CNN paper:
  <https://arxiv.org/abs/1904.08755>
- MinkowskiEngine source:
  <https://github.com/NVIDIA/MinkowskiEngine>

The paper does not supply a currently maintained, canonical PyTorch event
classifier that can simply be copied into this repository. PyG supplies a
maintained implementation of the GravNet *operator*. We still need to build
the event-level network around that operator.

Do not install directly from the PyG `master` branch for production. Select
the newest stable release compatible with the chosen PyTorch and CUDA build,
pin all three versions, and record them in every W&B run. The current PyG
source requires `pyg-lib>=0.6.0` for the k-nearest-neighbour operation.

## 3. Current dataset contract

The completed Snakemake dataset will expose the training input through:

```text
snakemake/metadata/ml_dataset/data_driven_3d_cnn_v1/
  five_fold_train_validation_views.csv
  dataset_catalog_five_fold.csv
  five_fold_summary.csv
  density_index_manifest.csv
  validation_report.json
```

Training must not start unless `validation_report.json` exists and contains
`"valid": true`.

Each event is addressed without copying it by:

```text
(dataset_file_uid, hit3d_path, source_entry)
```

The `hit3D` tree supplies variable-length vectors:

```text
hit_x, hit_y, hit_z, hit_qdc, hit_station, hit_detType
```

The initial loader must read the aligned vectors, check their lengths, apply
`hit_detType == 1`, and then discard `hit_station` and every non-selected
detector row. The selected event contract is:

```text
position:      [N, 3] float32  # hit_x, hit_y, hit_z
qdc:           [N]    float32  # hit_qdc
detector_type: [N]    int16    # always 1 after the SciFi selection
```

`detector_type` is intentionally retained because it is part of the requested
input schema, although it is constant in a SciFi-only sample and therefore
contains no discriminating information. Record this explicitly in the run
configuration so it is not mistaken for a learned detector-composition
feature.

The current SciFi 3D rows are candidate crossings: within a station, every
selected vertical hit is combined with every selected horizontal hit and the
candidate QDC is their sum. They are not guaranteed to be unique physical
space points. Log both the number of selected candidates and the number of
occupied sparse voxels so combinatorial growth remains visible.

Reuse these existing components:

- fold and sampling logic:
  `train_3d/src/train_3d/data/data_driven_fold_dataset.py`;
- ROOT event reader and sparse collator:
  `train_3d/src/train_3d/data/data_driven_root_dataset.py`;
- voxelizer:
  `train_3d/src/train_3d/data/voxelize.py`;
- current minimal Minkowski model as a smoke-test reference:
  `train_3d/src/train_3d/models/simple_cnn.py`.

## 4. Seven-class contract

Keep the permanent dataset IDs unchanged, but remap labels by class name for
training:

```python
TRAINING_LABELS = {
    "CC_numu": 0,
    "CC_nue": 1,
    "NC": 2,
    "neutral_kaon": 3,
    "neutron": 4,
    "passing_muon": 5,
    "muon_DIS": 6,
}
```

`CC_nutau` is excluded explicitly. A sample should retain both identifiers:

```python
{
    "label": TRAINING_LABELS[reference.class_name],
    "source_class_id": reference.class_id,
    "class_name": reference.class_name,
}
```

This is required because permanent source ID 7 (`muon_DIS`) cannot be passed
to a seven-logit cross-entropy loss.

## 5. Difference between the legacy and maintained GravNet

### 5.1 Legacy repository implementation

The legacy implementation consists mainly of:

```text
training/models/GravNet/Models/gravnet.py
training/models/GravNet/Models/gravconv.py
training/models/GravNet/gnn_base.py
training/dataset/torchGeoDataset.py
```

Its `GravConv`:

- learns coordinates with a configurable multi-layer `spatial_network`;
- appends the mean of each node's hidden vector as another input feature;
- can construct radius, k-nearest-neighbour, and random-neighbour edges;
- can combine multiple edge sets and remove duplicates;
- supports configurable add, max, min, standard-deviation, and mean scatter
  reductions;
- uses `exp(-grav_weight * squared_distance / r**2)` edge weights;
- normalizes propagated hidden features optionally;
- runs the concatenated self and aggregated features through another MLP;
- supports multiple GravNet heads, pyramid channel schedules, and optional
  concatenation of intermediate layers;
- performs graph-level sum, mean, and max pooling in the event classifier.

The checked `GravNet_v4.yml` does not use fixed-k GravNet: `knn` is null and
`r: 1.0`, so it builds a learned-space radius graph capped by `max_knn`. It is
therefore a useful custom dynamic graph network, but it is not a clean
implementation of the paper's fixed-k GravNet layer.

The legacy data and training code also do not match the new task:

- it uses bar endpoint/orientation features rather than the current Hit3D
  voxel representation;
- its PDG mapping has a tau class and one combined muon class, whereas the new
  task excludes tau and separates passing muons from muon-DIS;
- `TrainGeoDataset` materializes all events as an `InMemoryDataset`;
- the older entrypoint can force preprocessing on every launch;
- batches are limited by event count rather than total nodes;
- the `use_weights` branches in `GNNBase` do not express the new hierarchical
  training-sampling policy clearly;
- metrics and labels are coupled to the old task;
- W&B is initialized manually and then wrapped by a second logger setup.

### 5.2 Maintained PyG operator

`torch_geometric.nn.GravNetConv` has the interface:

```python
GravNetConv(
    in_channels,
    out_channels,
    space_dimensions,
    propagate_dimensions,
    k,
)
```

For every layer it:

1. projects input features into learned spatial coordinates with `lin_s`;
2. projects features to propagate with `lin_h`;
3. constructs a fixed-k graph in the learned space, independently for every
   event using the `batch` vector;
4. weights messages with `exp(-10 * squared_distance)`;
5. aggregates neighbour messages with mean and max;
6. combines the aggregation with a direct linear projection of the input.

The PyG implementation is much smaller and less configurable than the legacy
code, but it is maintained, documented, tested in a widely used library, and
closer to the operator described by the paper. It is a layer, not a complete
seven-class event classifier.

### 5.3 Migration decision

Use PyG `GravNetConv` for the new primary model. Do not copy its source into
the repository. Pin PyG and `pyg-lib` and import the layer normally, so fixes
and compatibility behavior remain owned by the upstream package.

Keep the legacy implementation unchanged. If it is compared later, label it
`legacy_radius_gravnet`; do not call it the paper-faithful baseline.

## 6. Shared SciFi input and model-specific representation

### 6.1 Common raw event

Both paths must start from the exact same ordered arrays after the SciFi mask:

```text
position:      [N, 3] float32
qdc:           [N, 1] float32
detector_type: [N, 1] int16, value 1
```

Use normalized physical coordinates and a training-derived QDC transform such
as `log1p(qdc)`. Calculate coordinate and QDC normalization constants using
only the four training folds for the selected cross-validation fold. Save the
constants in a versioned JSON file and never derive them from validation.

Do not add station, occupancy, timing, detector fractions, reconstructed
quantities, or event energy to this first experiment.

### 6.2 GravNet nodes

Use one node per selected SciFi 3D candidate crossing:

```text
pos: [N, 3] = x_norm, y_norm, z_norm
x:   [N, 2] = transformed_qdc, detector_type
```

The implementation may concatenate `pos` and `x` before the input MLP because
PyG `GravNetConv` learns its own spatial projection from node features. Keep
the physical position separately in the batch for diagnostics. Do not create
edges in the data loader; `GravNetConv` constructs the fixed-k graph inside
each event using the `batch` vector.

### 6.3 Sparse-CNN voxels

Quantize the same selected SciFi candidates into sparse coordinates:

```text
coordinates: [M, 3] int32
features:    [M, 2] float32  # transformed qdc_sum, detector_type
```

Spatial position is already represented by the sparse coordinate and must not
be duplicated as three dense feature channels. When multiple candidates map
to one voxel, sum their QDC and retain detector type 1. Occupancy is useful as
diagnostic metadata but is not an input channel in the initial experiment.

The two networks therefore share the same filtered raw candidates, labels,
folds, and sampling but perform architecture-required representation steps:
continuous candidate nodes for GravNet and quantized sparse voxels for the
sparse CNN. Report this difference when comparing their results.

### 6.4 Later controlled representations

Only after both baselines are stable should a separate experiment attempt to
resolve candidate-crossing ghosts, use compact physical detector hits, add
MuFilter detectors, or add station/time/occupancy features. Each such change
must receive a new representation version and W&B group.

## 7. Dataset and loader refactor

Create a shared, model-independent event dataset:

```text
train_3d/src/train_3d/data/classification_labels.py
train_3d/src/train_3d/data/scifi_hit_event_dataset.py
train_3d/src/train_3d/data/classification_collate.py
train_3d/src/train_3d/data/classification_datamodule.py
```

The event dataset should accept an `EventReference` directly:

```python
sample = dataset[reference]
```

It must return the filtered raw SciFi candidate arrays before either model's
representation is assembled. Filtering must happen once in shared code, not
independently in the two collators.

The main-process sampler, rather than worker-local dataset state, should emit
the epoch's deterministic references. This allows `persistent_workers=True`
without stale epoch reference lists.

Training policy:

- draw classes uniformly for the initial baseline;
- make domain probabilities explicit in configuration;
- retain uniform populated-density-bin selection for training muon-DIS;
- draw with replacement for the configured epoch size;
- seed by `(base_seed, cv_fold, epoch)`;
- under DDP, split one deterministic global reference sequence into disjoint
  rank slices.

Validation policy:

- iterate every natural held-out event once;
- do not rebalance classes, domains, or density bins;
- do not resample validation;
- preserve source and domain metadata for sliced metrics.

### 7.1 Sparse-CNN collator

Return:

```text
coords:   [total_voxels, 4] int32  # batch, x, y, z
features: [total_voxels, F] float32
label:    [events] long
metadata: event-level identity fields
```

The current `classification_sparse_collate_fn` can be refactored rather than
rewritten.

### 7.2 GravNet collator

Return either a PyG `Batch` or an equivalent dictionary:

```text
x:       [total_nodes, 5] float32  # xyz, transformed QDC, detector type
pos:     [total_nodes, 3] float32  # normalized physical xyz for diagnostics
batch:   [total_nodes] long
label:   [events] long
metadata: event-level identity fields
```

There must be no event padding. The `batch` vector ensures that kNN edges do
not connect different events.

### 7.3 Size-aware batches

Fixed event-count batches are unsafe because event sizes vary strongly. Add a
batch sampler constrained by both:

```yaml
max_events_per_batch: 16
max_nodes_per_batch: 20000  # initial GravNet value; tune by profiling
```

Use gradient accumulation for the requested effective batch size. Log actual
events, nodes, and edges per batch.

For the initial ROOT-streaming loader, store both selected-candidate and voxel
counts in a lightweight manifest. For production, create medium-sized caches
without duplicating events across folds. The GravNet cache stores filtered
candidate arrays and the sparse cache may store derived voxel arrays:

```text
raw position/qdc/detector_type [all_candidates, ...]
raw offsets                    [events + 1]
sparse coordinates/features    [all_voxels, ...]  # optional derived cache
sparse offsets                 [events + 1]
labels and identity metadata
```

Use medium-sized shards keyed by `(dataset_file_uid, source_entry)`. Do not
write one `.pt` file per event, do not duplicate events for five folds, and do
not build a single all-events `InMemoryDataset`.

## 8. New PyG GravNet classifier

Add:

```text
train_3d/src/train_3d/models/pyg_gravnet_classifier.py
```

Initial architecture:

```text
node input
  -> input MLP, F -> 64
  -> GravNetConv(64, 64, space_dimensions=4,
                 propagate_dimensions=64, k=16)
  -> normalization + activation + residual
  -> GravNetConv(64, 64, 4, 64, 16)
  -> normalization + activation + residual
  -> GravNetConv(64, 64, 4, 64, 16)
  -> concatenate global mean and global max pools
  -> MLP(128 -> 64 -> 7)
```

Start with one head and three blocks. Multiple heads, sum pooling, deeper
models, and legacy radius graphs are later ablations, not initial defaults.

Recommended initial search after the baseline is stable:

```text
k:                       8, 16, 32
space_dimensions:        3, 4, 8
propagate_dimensions:    32, 64
blocks:                   2, 3, 4
hidden size:              64, 128
```

Change one family at a time and compare all choices on the same development
fold.

## 9. Sparse residual CNN

Add:

```text
train_3d/src/train_3d/models/sparse_resnet_classifier.py
```

Initial architecture:

```text
sparse stem, 32 channels
residual stage 32
residual stage 64, stride 2
residual stage 128, stride 2
residual stage 256, stride 2
Minkowski global average pooling
dropout
linear 256 -> 7
```

Adapt the minimum residual-block structure from the MIT-licensed official
MinkowskiEngine examples and retain attribution. Do not vendor the full
MinkowskiEngine source.

## 10. Common training module and metrics

Add:

```text
train_3d/src/train_3d/classification_module.py
train_3d/train_classifier.py
```

Select the model from configuration. Both models must use the same loss and
metric code.

Initial loss:

```python
loss = cross_entropy(logits, label)
```

Do not combine uniform class sampling with inverse-frequency class weights in
the initial experiment. That would correct class imbalance twice.

Log:

- training and natural-validation cross entropy;
- accuracy and balanced accuracy;
- macro and weighted F1;
- per-class precision, recall, and F1;
- absolute and row-normalized confusion matrices;
- metrics split by domain when meaningful;
- node-count and occupancy distributions;
- events/second, nodes/second, data time, step time, and peak GPU memory;
- fraction and identity of skipped empty or invalid events.

Use natural `val_macro_f1` as the primary checkpoint monitor, with mode `max`.
Also retain `val_loss` for diagnosing calibration and overfitting.

## 11. Weights & Biases workflow

### 11.1 Lessons from the existing repository

There are two W&B patterns in the current code.

The older GravNet entrypoints in `training/train.py` and `src/train_model.py`:

1. call `wandb.init(...)` directly;
2. construct a human-readable fixed ID such as `GravNet_v2_v1`;
3. create a `WandbLogger` afterwards;
4. save selected source files with `run.save(...)`;
5. call `wandb.save("*ckpt*")` after training.

The reusable ideas are configuration tracking, Lightning scalar logging,
best/last checkpoints, and automatic machine/environment metadata. The new
workflow should not copy the initialization and checkpoint-upload details:

- initializing manually and then constructing a logger creates two possible
  owners of run configuration and lifecycle;
- a name such as `GravNet_v2_v1` is suitable as a display name but not as a
  globally reused run ID;
- the local logs contain many directories with the same old fixed run ID,
  demonstrating how retries can become attached to one remote run;
- `wandb.save("*ckpt*")` overlaps with `WandbLogger` model-artifact support;
- metrics such as `val_acc0` require an external class map to interpret;
- the old W&B SDK recorded in local runs is `0.15.12` and must not determine
  the version of the new environment;
- previous regression summaries contain `Infinity` and `NaN`, so the new
  trainer must fail clearly on non-finite losses and metrics.

The newer `train_3d/train.py` pattern is the starting point:

- construct one `WandbLogger`;
- pass it to Lightning's `Trainer`;
- log the fully resolved YAML configuration;
- use `LightningModule.log(...)` for scalar metrics;
- use W&B Tables and plots for epoch-level diagnostics;
- use `ModelCheckpoint` for local best/last recovery.

Avoid mixing `pytorch_lightning` and `lightning.pytorch` imports in the new
code. Select one pinned Lightning package and import path for the entire
pipeline.

### 11.2 New project and run hierarchy

Create a new W&B project:

```text
snd-ml-data-driven-3d
```

Do not reuse the historical `snd-ml-GravNet` or `snd-ml-train-3d` projects.
The project will be created by W&B on the first online run if the authenticated
account has permission.

Use the following hierarchy:

```text
Project: snd-ml-data-driven-3d
|
|-- Group: pyg_gravnet_v1
|   |-- pyg-gravnet-v1__fold0__seed42
|   |-- pyg-gravnet-v1__fold1__seed42
|   |-- ...
|   `-- pyg-gravnet-v1__fold4__seed42
|
`-- Group: sparse_resnet_v1
    |-- sparse-resnet-v1__fold0__seed42
    |-- ...
    `-- sparse-resnet-v1__fold4__seed42
```

The meanings are:

- **project**: the complete data-driven 3D classification study;
- **group**: all folds and seeds for one frozen model configuration;
- **run**: one model, one fold, and one seed;
- **job type**: `train`, `evaluation`, `benchmark`, or later `sweep`;
- **tags**: searchable dataset, representation, fold, and model attributes.

Do not put both architectures in one group if the W&B group summary will be
used to estimate cross-validation performance. Separate model groups allow
fold means and spreads to represent one architecture rather than a mixture of
architectures.

Initial naming:

```text
project: snd-ml-data-driven-3d
group:   pyg_gravnet_v1 or sparse_resnet_v1
job_type: train
name:    <model-version>__fold<fold>__seed<seed>
tags:
  - data_driven_3d_cnn_v1
  - seven_class
  - scifi_only
  - scifi_xyz_qdc_dettype_v1
  - sparse_resnet or pyg_gravnet
  - fold_0 ... fold_4
```

### 11.3 One initialization path

Let `WandbLogger` own creation and finalization of the run. Do not call
`wandb.init()` separately in the normal Lightning training path.

Conceptual implementation:

```python
logger = WandbLogger(
    project="snd-ml-data-driven-3d",
    entity=wandb_config.get("entity"),
    name=f"{model_version}__fold{cv_fold}__seed{seed}",
    group=model_version,
    job_type="train",
    tags=[
        "data_driven_3d_cnn_v1",
        "seven_class",
        "scifi_only",
        "scifi_xyz_qdc_dettype_v1",
        model_family,
        f"fold_{cv_fold}",
    ],
    save_dir=wandb_config["save_dir"],
    log_model=wandb_config.get("log_model", True),
)

trainer = Trainer(
    logger=logger,
    callbacks=[checkpoint_callback, early_stopping_callback],
    ...,
)
```

The `entity` value should be either `null`, so W&B uses the logged-in default,
or the real W&B username/team slug. Do not reuse the numeric account value
from the legacy configuration as an entity slug. Authentication comes from an
existing W&B login or an environment-provided credential; no API key belongs
in YAML, source code, job arguments, or logs.

Let W&B generate a unique run ID for every genuinely new run. A readable name
and a W&B run ID have different purposes:

```text
display name: pyg-gravnet-v1__fold0__seed42
run ID:       automatically generated opaque identifier
```

### 11.4 Resolved configuration and provenance

Log exactly one fully resolved configuration after CLI overrides and defaults
have been applied. `LightningModule.save_hyperparameters()` may still be used
for checkpoint reconstruction, but it must not create a second inconsistent
view of the experiment.

Log the complete resolved configuration, including:

- dataset and fold-view paths plus checksums;
- dataset version, dataset-manifest version, and validation-report contents;
- class map and excluded classes;
- fold, seed, epoch size, and sampling probabilities;
- domain probabilities and muon-DIS density-bin sampling policy;
- voxel definition and normalization constants;
- model hyperparameters;
- optimizer, scheduler, early stopping, precision, and gradient accumulation;
- maximum events and nodes per batch;
- PyTorch, CUDA, PyG, `pyg-lib`, MinkowskiEngine, Lightning, ROOT, and Python
  versions;
- git commit plus an explicit dirty-worktree flag;
- hostname, GPU model, and batch resource limits.

The run should also contain the path and checksum of
`validation_report.json`, and training should stop before W&B initialization
if that report is absent or invalid. Failed data validation should not create
a misleading training run.

Do not upload the whole repository blindly. It currently contains a large
dirty worktree and generated files. Record the git commit, dirty flag, resolved
config, and a deliberately scoped code snapshot or diff for the files used by
the run.

### 11.5 Scalar metric policy

Use `self.log(...)` for all high-frequency and epoch scalar metrics. Do not
mix Lightning scalar logging with repeated direct `wandb.log(..., step=...)`
calls, because two independent step-management paths can produce inconsistent
x-axes.

Suggested keys:

```text
train/loss_step
train/loss
train/accuracy
train/macro_f1

val/loss
val/accuracy
val/balanced_accuracy
val/macro_f1
val/weighted_f1

val/precision/CC_numu
val/recall/CC_numu
val/f1/CC_numu
...
val/precision/muon_DIS
val/recall/muon_DIS
val/f1/muon_DIS

throughput/events_s
throughput/nodes_s
timing/data_s
timing/step_s
memory/gpu_peak_mb
batch/events
batch/nodes
```

Use class names in metric keys, not numeric labels such as `val_acc0`.

Training and validation metrics have different statistical meanings:

- training metrics describe the deliberately resampled training population;
- validation metrics describe the complete natural held-out fold.

Do not interpret a training/validation difference as ordinary overfitting
without also considering this population difference. Checkpoint and early
stopping decisions must use natural validation.

Set W&B summaries explicitly:

```text
val/loss                 min
val/macro_f1             max
val/balanced_accuracy    max
throughput/nodes_s       max
memory/gpu_peak_mb       min
```

The main checkpoint criterion is:

```yaml
checkpoint:
  monitor: val/macro_f1
  mode: max
  save_top_k: 1
  save_last: true
```

Before logging or reducing a loss or metric, require it to be finite. A
non-finite loss must raise an error containing the fold, epoch, batch index,
event identities, node counts, and feature ranges. Do not let `NaN` or
`Infinity` become the recorded run result.

### 11.6 Tables, confusion matrices, and plots

At validation epoch end, aggregate predictions across the whole natural
validation fold before computing classification metrics. Log:

- an absolute confusion matrix;
- a row-normalized confusion matrix;
- a per-class precision/recall/F1/support table;
- data/MC domain slices where meaningful;
- muon-DIS performance by density bin;
- performance versus event node count and occupancy;
- class and domain populations actually seen by the loader.

Use a bounded validation prediction table, initially at most 2,000 events per
logged table, with deterministic selection. Suggested columns:

```text
dataset_file_uid
source_entry
run_id
event_id
event_index
domain
density_bin
n_nodes
true_class
predicted_class
prob_CC_numu
prob_CC_nue
prob_NC
prob_neutral_kaon
prob_neutron
prob_passing_muon
prob_muon_DIS
```

Do not upload every validation event after every epoch. That would make table
artifacts and network traffic grow unnecessarily. During training, log
confusion and per-class summary tables at a configurable interval. Always log
a complete bounded evaluation table for the best/final model.

Scalar metrics continue to use `self.log(...)`. Epoch tables and W&B custom
charts may use `self.logger.experiment` on global rank zero only, without
manually forcing a conflicting step.

### 11.7 Artifact policy

Use three artifact categories:

1. `dataset-manifest`: small catalogs, fold views, validation report, class
   map, normalizations, configuration, and checksums; never upload the full
   ROOT dataset or voxel cache;
2. `model`: best checkpoint, last checkpoint, resolved config, class map, and
   normalization JSON;
3. `evaluation`: final metrics, confusion matrices, per-class table, and a
   bounded prediction table containing identities, truth, prediction, and
   seven probabilities.

The training run should declare the manifest artifact as an input, creating
lineage from the validated dataset description to the model. Large EOS ROOT
files remain external data referenced by checksummed metadata.

`WandbLogger(log_model=True)` uploads `ModelCheckpoint` outputs at the end of
training. `log_model="all"` uploads them while training. Use:

- `true` for normal runs where local AFS/EOS checkpoints are reliable and
  lower network traffic is preferred;
- `"all"` for preemptible jobs only when remote checkpoint availability is
  worth the upload cost;
- `false` for offline/disabled runs, followed by an explicit later sync if
  required.

Keep `save_top_k: 1` and `save_last: true`. Remove the legacy
`wandb.save("*ckpt*")` path so checkpoints are not handled twice.

### 11.8 Resume and retry behavior

Checkpoint state and W&B identity must resume together. Store:

```text
<run-output>/last.ckpt
<run-output>/wandb_run_id.txt
<run-output>/resolved_config.yaml
```

For a new run:

1. do not provide a W&B ID;
2. allow W&B to generate one;
3. write `logger.experiment.id` next to the checkpoints;
4. train without a checkpoint path.

For an intentional resume:

1. require both `last.ckpt` and `wandb_run_id.txt`;
2. pass the checkpoint to `trainer.fit(..., ckpt_path=...)`;
3. pass the stored ID to `WandbLogger(id=run_id, resume="must", ...)`;
4. validate that model version, fold, seed, dataset checksum, and class map
   match the stored resolved configuration.

If the checkpoint exists but the W&B ID does not, resume the model into a new
run tagged `resume_without_original_run` and record the parent checkpoint. Do
not guess the old run ID from a display name.

A scheduler retry that starts from scratch must create a new run. A retry that
resumes `last.ckpt` must resume the same W&B run. This distinction prevents
several independent trainings from silently sharing one history.

### 11.9 Online, offline, disabled, and distributed operation

Support:

```yaml
wandb:
  mode: online  # online, offline, or disabled
```

- `online`: stream metrics and configured artifacts normally;
- `offline`: write a local W&B run and sync it later; disable automatic model
  artifact upload in the logger configuration;
- `disabled`: use no remote experiment tracking while keeping local
  checkpoints and final CSV/JSON evaluation outputs.

Only global rank zero may initialize W&B or log tables/artifacts under DDP.
Lightning scalar reduction should use the appropriate `sync_dist` setting.
Do not create one W&B run per GPU.

Model gradient watching should be optional and off by default because it can
be expensive for sparse and graph workloads. If enabled for debugging, use a
large logging interval and disable it again for production throughput tests.

### 11.10 W&B implementation acceptance checks

Before full training, verify:

1. a disabled-mode smoke run completes without network access;
2. an offline run writes metrics locally without attempting model upload;
3. a short online run appears in `snd-ml-data-driven-3d` under the correct
   model group;
4. the run config contains fold, class order, dataset checksum, node budget,
   environment versions, and dirty-worktree status;
5. scalar curves share the expected Lightning global-step axis;
6. readable per-class keys and both confusion matrices appear;
7. best and last checkpoints exist locally;
8. model artifacts do not duplicate checkpoints;
9. a stopped run resumes from both `last.ckpt` and the same W&B run ID;
10. a fresh retry creates a new ID even if its display name is similar;
11. DDP creates one run, not one run per device;
12. injected non-finite loss fails the smoke test instead of logging a corrupt
    summary.

## 12. Configuration layout

Add:

```text
train_3d/configs/classifier_sparse_resnet_v1.yaml
train_3d/configs/classifier_pyg_gravnet_v1.yaml
```

Keep a shared schema:

```yaml
task: classification
dataset_version: data_driven_3d_cnn_v1
num_classes: 7
class_names:
  - CC_numu
  - CC_nue
  - NC
  - neutral_kaon
  - neutron
  - passing_muon
  - muon_DIS
excluded_classes: [CC_nutau]

data:
  cv_fold: 0
  epoch_size: 100000
  class_sampling: uniform
  allowed_detector_types: [1]
  raw_input_fields: [hit_x, hit_y, hit_z, hit_qdc, hit_detType]
  num_workers: 4
  persistent_workers: true
  prefetch_factor: 2
  max_events_per_batch: 16
  max_nodes_per_batch: 20000

representation:
  version: scifi_xyz_qdc_dettype_v1
  qdc_transform: log1p
  include_station: false
  include_occupancy_as_input: false
  include_time: false
  gravnet_nodes: candidate_crossings
  sparse_collision_reduction: qdc_sum

wandb:
  project: snd-ml-data-driven-3d
  entity: null
  group: pyg_gravnet_v1  # sparse_resnet_v1 in the sparse config
  job_type: train
  mode: online
  save_dir: /afs/cern.ch/work/z/zhibin/snd-ml/train_3d/wandb_logs
  log_model: true
  watch_model: false
  prediction_table_max_rows: 2000
  diagnostic_table_every_n_epochs: 5
  tags:
    - data_driven_3d_cnn_v1
    - seven_class
    - scifi_only
    - scifi_xyz_qdc_dettype_v1
```

Model-only sections differ between the two files. Resolve all paths and
defaults before logging the configuration.

## 13. Tests and acceptance checks

Add tests for:

1. source-to-training label remapping and explicit tau rejection;
2. deterministic epoch sampling;
3. natural, exhaustive validation iteration;
4. preservation of uniform populated muon-DIS density-bin sampling;
5. strict `hit_detType == 1` filtering and exclusion of every MuFilter row;
6. rejection of misaligned `x/y/z/qdc/detType` vectors;
7. proof that station, occupancy, timing, energy, and reconstructed features
   cannot enter the initial model inputs;
8. sparse collation and GravNet collation of variable-size events;
9. no graph connection across event boundaries;
10. identical filtered raw SciFi candidate arrays before model-specific
    representation assembly;
11. sparse QDC collision summation and GravNet one-node-per-candidate behavior;
12. node-budget batch limits, including a single oversized event;
13. one CPU forward/backward smoke test for each model;
14. one CUDA forward/backward container test for each model on an assigned GPU;
15. W&B disabled/offline smoke tests without network access;
16. checkpoint resume preserving sampler epoch and W&B run identity;
17. ROOT-streaming and cached representations producing identical tensors;
18. validation metrics computed across a whole epoch, not averaged incorrectly
    from unequal batch sizes.

Before a long run, overfit 20--100 events with each model. Failure to overfit
that subset indicates a label, batch, feature, loss, or architecture bug.

## 14. Implementation sequence

### Phase A: freeze the data contract

1. Wait for `validation_report.json` with `valid: true`.
2. Record population, fold, domain, selected-SciFi-candidate, and sparse-voxel
   distributions.
3. Freeze the seven-class order, SciFi mask, allowed raw fields, voxel
   definition, coordinate ranges, and training-only normalization constants.

### Phase B: shared loader

4. Add the label map and explicit exclusions.
5. Refactor the ROOT-backed dataset to accept sampled references and return
   only aligned, filtered SciFi `xyz/qdc/detType` arrays.
6. Implement sparse and GravNet collators.
7. Implement size-aware batching and loader benchmarks.
8. Keep ROOT streaming for the first correctness tests.

### Phase C: models and training

9. Add the PyG `GravNetConv` classifier without copying legacy `GravConv`.
10. Add the sparse residual classifier.
11. Add shared classification loss, metrics, checkpoints, and prediction
    collection.
12. Add the single-init W&B workflow and the new project configuration.
13. Build and validate the pinned CPU/GPU training container described in
    `TRAINING_CONTAINER_AND_HTCONDOR_ENVIRONMENT_PLAN.md`.

### Phase D: validation and performance

14. Overfit a tiny subset with each model.
15. Run short fold-0 jobs and confirm label/sample parity.
16. Profile data time, GPU time, kNN time, and peak memory.
17. Create sharded candidate/voxel caches if ROOT reading plus representation
    assembly is a meaningful fraction of step time.
18. Tune only node budgets and loader workers before model searches.
19. Validate CPU inference, shard outputs, and global metric merging without
    averaging per-shard F1 scores.

### Phase E: experiment program

20. Train the fixed initial configuration on fold 0 for both models.
21. Perform a small GravNet search on fold 0.
22. Freeze each architecture.
23. Run all five folds for sparse ResNet and PyG GravNet on GPU.
24. Run final exhaustive evaluation as CPU shards where benchmarking shows it
    is efficient, then merge additive statistics and predictions.
25. Aggregate fold results in W&B and in a versioned local CSV/Markdown report.
26. Only then test de-ghosted hits, additional detectors/features, or the
    legacy radius GravNet as separately versioned ablations.

## 15. Definition of done

The initial implementation is complete when:

- both models train from the same fold-view file and shared event loader;
- both models start from identical SciFi-only `xyz/qdc/detType` candidate
  arrays and no prohibited feature reaches either network;
- the tau class cannot enter training accidentally;
- all labels lie in `[0, 6]` and retain their source IDs;
- muon-DIS sampling is balanced only in training and validation remains
  natural;
- batches obey a node budget and do not pad events;
- the same pinned container passes CPU and assigned-GPU forward/backward
  tests for both architectures;
- best and last checkpoints resume reproducibly;
- W&B records full provenance and comparable metrics in the new project;
- fold-0 smoke runs complete without invalid samples, cross-event edges,
  memory instability, or hidden dataset rebuilding;
- five-fold results can be aggregated without manual relabeling;
- sharded CPU evaluation reproduces single-process global metrics by merging
  counts and loss sums rather than averaging shard-level metrics.
