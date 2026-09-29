# Clustered GravNet implementation plan

Status: proposed
Date: 2026-09-22
Scope: memory-safe seven-class SciFi event classification without dropping or
quantizing candidate crossings

## 1. Objective

Implement a Cluster-GCN-inspired GravNet classifier that can process every
candidate in events too large for one GPU GravNet operation.

Requirements:

- retain every selected SciFi candidate and its continuous XYZ coordinates;
- never reject, truncate, sample, or voxelize a large event;
- bound each GravNet kNN operation by a configurable cluster size;
- produce exactly one prediction and loss contribution per event;
- remain compatible with the fold sampler, evaluation, W&B, container, and
  Snakemake workflow;
- preserve the existing `pyg_gravnet` as an unchanged baseline.

This is a memory-safety experiment for the present candidate-crossing input.
It does not eliminate the total Cartesian-product cost. A native two-view
representation remains a separate longer-term option.

## 2. Papers and code references

### Cluster-GCN

Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, and Cho-Jui Hsieh,
"Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph
Convolutional Networks," KDD 2019.

- Paper: <https://arxiv.org/abs/1905.07953>
- DOI: <https://doi.org/10.1145/3292500.3330925>
- Authors' TensorFlow implementation:
  <https://github.com/google-research/google-research/tree/master/cluster_gcn>
- PyG `ClusterData`/`ClusterLoader` API:
  <https://pytorch-geometric.readthedocs.io/en/2.4.0/modules/loader.html>
- Official PyG example:
  <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/cluster_gcn_reddit.py>

Cluster-GCN partitions a fixed graph into densely connected subgraphs and
restricts message passing to sampled subgraphs. This bounds the GPU working set
and avoids layer-by-layer neighbourhood expansion. The paper reports, for
example, 2.2 GB rather than 11.2 GB for VR-GCN in its three-layer Amazon2M
comparison.

The authors' code uses METIS on an existing graph and targets node-level
classification. It is a scientific reference, not code to copy here.

### GravNet

Shah Rukh Qasim et al., "Learning representations of irregular particle-
detector geometry with distance-weighted graph networks."

- Paper: <https://arxiv.org/abs/1902.07987>
- PyG `GravNetConv`:
  <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GravNetConv.html>

`GravNetConv` projects current node features into a learned coordinate space
and constructs a new kNN graph inside every layer. There is no fixed
`edge_index` that METIS can partition before the forward pass.

## 3. Current project behaviour

`convertData/digi_2_hits3D.py::combine_station_hits_to_3d` forms all
vertical-horizontal pairs in each station:

```text
N_candidates(station) = N_vertical(station) * N_horizontal(station)
```

The ROOT output contains `hit_station`, `hit_v_channel`, `hit_h_channel`,
`hit_v_qdc`, and `hit_h_qdc`, but the current classifier reader loads only
XYZ, QDC, and detector type. The clustered model will additionally read
`hit_station` as partition metadata; it will not be a learned feature in v1.

The current model is:

```text
normalized XYZ + transformed QDC + detector type
  -> input MLP
  -> three residual GravNetConv blocks
  -> event mean || event max
  -> MLP -> seven logits
```

Its single `batch` vector both restricts kNN to an event and defines final
pooling. `NodeBudgetBatchSampler` correctly yields an oversized event alone,
but that still makes GravNet run kNN over the entire oversized event.

## 4. Adaptation, not literal Cluster-GCN

PyG `ClusterData` cannot be inserted directly because:

1. GravNet has no fixed graph before its forward pass.
2. This task has one graph-level event label, not node labels.
3. ROOT supplies many independent streamed graphs, not one global graph.

The proposed model transfers Cluster-GCN's central constraint—message passing
inside bounded spatial subgraphs—to a dynamic graph-level classifier. Code and
plots must call it `clustered_pyg_gravnet`; scientific text should call it a
"Cluster-GCN-inspired clustered GravNet", not a Cluster-GCN reproduction.

## 5. Model logic

### 5.1 Separate cluster and event grouping

Each node has two logical groupings:

- `cluster_id` restricts neighbours inside every `GravNetConv`;
- `event_id` is used only for final event pooling.

```text
event 0: clusters 0, 1, 2
event 1: cluster 3
event 2: clusters 4, 5
```

Cluster IDs are unique within a collated batch, and clusters never mix events.

### 5.2 Deterministic spatial partition

Version 1 uses station-aware recursive bisection:

1. Group candidates by `hit_station`.
2. Emit groups no larger than `max_cluster_nodes`.
3. For an overfull group, choose X or Y according to its largest physical
   span, stable-sort on that coordinate, and split at the median.
4. Recurse until all leaves are bounded.
5. Order leaves deterministically by station and spatial bounds.

Coordinates are used only for membership. The coordinates passed to the model
are neither rounded nor replaced. Every candidate occurs exactly once.

Initial settings:

```yaml
algorithm: station_xy_recursive_bisection
max_cluster_nodes: 8192
cluster_by_station: true
partition_axes: [x, y]
halo: null
```

`8192` is a safe profiling start, not an optimum. Benchmark 4096, 8192, 16384,
and, if safe, 32768.

### 5.3 Cluster-local encoder

Reuse the existing input MLP and `GravNetBlock`. Encode one cluster or a small
micro-batch of clusters at a time. Construct local batch IDs so PyG kNN never
connects distinct clusters.

Two limits have different meanings:

```yaml
max_cluster_nodes: 8192      # maximum size of one kNN graph
max_nodes_per_forward: 8192  # total nodes in one encoder call initially
```

After the safe point is measured, increase `max_nodes_per_forward` to encode
several independent clusters together for better GPU utilization.

### 5.4 Exact event pooling after local encoding

For each cluster `c`, retain final-embedding sum `S_c`, elementwise maximum
`M_c`, and core count `N_c`. For event `e`:

```text
mean_e = sum(S_c for c in e) / sum(N_c for c in e)
max_e  = elementwise max(M_c for c in e)
logits = head(mean_e || max_e)
```

This exactly reconstructs global mean/max pooling over the encoded nodes. The
approximation is only that learned edges cannot cross a cluster boundary.

Never replicate the event label and optimize clusters independently. That
would overweight large events and change the classification task.

### 5.5 Activation checkpointing

Sequential encoding bounds kNN workspace, but autograd otherwise retains all
cluster activations until the event loss. Wrap cluster micro-batch encoding in
non-reentrant `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.
Then only checkpoint inputs and compact summaries remain live. Make this
configurable and measure the recomputation cost.

### 5.6 Boundary treatment

Version 1 uses disjoint clusters without halos. A later ablation may duplicate
neighbouring nodes as halos:

- core nodes occur once and contribute to pooling;
- halo nodes participate in message passing only;
- diagnostics report core and encoded counts separately.

Do not add epoch-dependent random partitions initially; persistent loader
workers make that state easy to mishandle.

## 6. Data implementation

### Reader

Modify `train_3d/src/train_3d/data/scifi_hit_event_dataset.py`:

- separate common required branches from model-specific branches;
- request `hit_station` only for `clustered_pyg_gravnet`;
- apply the same `hit_detType == 1` mask to station and other arrays;
- validate alignment, integer type, and station range;
- preserve existing GravNet and sparse-ResNet contracts.

Test PyROOT/uproot parity for station.

### Pure partition module

Add `train_3d/src/train_3d/data/gravnet_partition.py`:

```python
@dataclass(frozen=True)
class GravNetPartitionConfig:
    max_cluster_nodes: int
    cluster_by_station: bool = True
    partition_axes: tuple[int, ...] = (0, 1)

def partition_event(position, station, config) -> list[np.ndarray]:
    """Return deterministic core-index arrays, one per cluster."""
```

Required invariants:

- all indices are valid;
- clusters are non-empty and bounded;
- every input node appears exactly once;
- station groups do not mix when requested;
- identical inputs produce identical partitions.

### Collator

Add `ClusteredGravNetClassificationCollator` to
`classification_collate.py`. Reorder nodes into contiguous cluster order:

```text
x:               [total_nodes, 5] float32
pos:             [total_nodes, 3] float32
cluster_ptr:     [num_clusters + 1] int64
cluster_event:   [num_clusters] int64
cluster_station: [num_clusters] integer
n_nodes:         [num_events] int64
n_clusters:      [num_events] int64
label:            [num_events] int64
event metadata:   existing fields unchanged
model_family:     "clustered_pyg_gravnet"
```

`cluster_ptr[c]:cluster_ptr[c+1]` addresses one cluster. Retain a debugging
permutation path so coverage can be verified.

## 7. Model implementation

Add:

```text
train_3d/src/train_3d/models/clustered_pyg_gravnet_classifier.py
```

Suggested structure:

```text
ClusteredPyGGravNetClassifier
├── input_mlp
├── blocks (existing GravNetBlock semantics)
├── encode_microbatch()
├── summarize_clusters()
├── aggregate_events()
└── head
```

Forward contract:

```python
forward(x, cluster_ptr, cluster_event, num_events) -> logits
```

Validate empty clusters, pointer coverage/monotonicity, cluster bounds, event
indices, and events without clusters. Use PyG/PyTorch APIs; do not copy Google
or PyG implementation source.

Keep `input_mlp`, block, and head parameter names/shapes aligned with the
baseline where practical. With one cluster per event and dropout disabled,
copied baseline weights must reproduce baseline logits and gradients within
numerical tolerance.

Register `clustered_pyg_gravnet` in:

```text
train_3d/src/train_3d/models/classifier_factory.py
train_3d/src/train_3d/classification_module.py
train_3d/src/train_3d/data/classification_datamodule.py
```

The raw full-event five-column tensor can transfer normally to GPU. Even 1.8M
nodes occupy only about 34 MiB as five float32 columns; create hidden features
and kNN workspace only per micro-batch. Add a custom Lightning transfer hook
only if profiling shows the raw transfer is problematic.

## 8. Batching and configuration

Continue using candidate counts in `EventSizeIndex` and current
`NodeBudgetBatchSampler` semantics. Oversized events remain present and are
yielded alone.

`max_nodes_per_batch` is an input-event packing budget;
`max_nodes_per_forward` is a GPU encoder-call limit. Log both clearly.

Add:

```text
train_3d/configs/canary_clustered_pyg_gravnet_fold0_v1.yaml
train_3d/configs/classifier_clustered_pyg_gravnet_v1.yaml
```

Initial canary core:

```yaml
model_version: clustered_pyg_gravnet_canary_v1

data:
  raw_input_fields:
    [hit_x, hit_y, hit_z, hit_qdc, hit_detType, hit_station]
  max_events_per_batch: 16
  max_nodes_per_batch: 20000

representation:
  version: scifi_xyz_qdc_dettype_station_partition_v1
  gravnet_nodes: candidate_crossings
  partition:
    algorithm: station_xy_recursive_bisection
    max_cluster_nodes: 8192
    cluster_by_station: true
    partition_axes: [x, y]
    halo: null

model:
  family: clustered_pyg_gravnet
  in_channels: 5
  hidden_channels: 64
  blocks: 3
  space_dimensions: 4
  propagate_dimensions: 64
  k: 16
  dropout: 0.1
  max_nodes_per_forward: 8192
  activation_checkpointing: true

trainer:
  precision: 32-true
```

Start at full precision to isolate partition/model correctness; test
`16-mixed` separately after passing. Use distinct checkpoint, output, and W&B
identifiers.

## 9. Logging

Add metrics:

```text
batch/events
batch/candidates
batch/clusters
batch/max_cluster_nodes
batch/mean_cluster_nodes
batch/max_clusters_per_event
throughput/candidates_s
throughput/clusters_s
timing/data_s
timing/step_s
memory/gpu_peak_allocated_mb
memory/gpu_peak_reserved_mb
```

Record the partition version, both node limits, checkpointing setting,
cluster-size summary, container digest, and PyTorch/PyG versions in the
training manifest. Include file UID and source entry in partition, non-finite,
and OOM diagnostics.

## 10. Tests

### Partition and collator

- Below-limit and limit-plus-one cases.
- Every node appears once; no empty or over-limit cluster.
- Stations do not mix; repeated positions remain distinct.
- Deterministic output and unmodified inputs.
- Correct five-feature `x`; station is not a model feature.
- Valid pointers for mixed-size, multi-event batches.
- Labels and metadata remain one row per event.
- PyROOT/uproot station parity.
- Synthetic million-node partition uses bounded host memory.

### Model

- CPU forward/backward with multiple clusters and events.
- Finite gradients with checkpointing on and off.
- Exactly one logit row per event.
- No message passing between clusters.
- Correct sum/max/count event reconstruction.
- Baseline parity in one-cluster mode with copied weights.
- Invariance to cluster order.
- Explicit handling of `cluster_size <= k`.

First probe the installed `GravNetConv` for small clusters. If `k` cannot vary
per call, merge undersized same-station leaves during partitioning rather than
depending on undefined backend behaviour.

### GPU staircase

Test synthetic and real events:

```text
cluster nodes: 4096, 8192, 16384, 32768
event nodes:   20k, 100k, 400k, 1.8M
k:             16
blocks:        3
precision:     32-true, then 16-mixed
```

Fail on OOM, non-finite loss, missing gradients, dropped nodes, or wrong output
shape. Record peak allocated/reserved memory and candidates/s.

## 11. Snakemake and HTCondor

The generic `train_classifier_fold` rule already dispatches model configs.
Proceed as follows:

1. Run isolated operator/model tests in the pinned container.
2. Submit only the fold-0 canary with a small epoch and limited validation.
3. Verify it passes the previous large-event failure point.
4. Inspect W&B cluster, timing, and memory metrics.
5. Run one complete fold-0 epoch.
6. Add the production config to
   `snakemake/metadata/configs/training/data_driven_3d_classifiers_v1.yaml`
   only after acceptance gates pass.
7. Verify checkpoint reconstruction in `evaluate_classifier_shard.py`.

Profile one CPU evaluation shard before retaining its current time request;
clustered evaluation may require more runtime or a GPU based on measurement.

## 12. Implementation phases

### Phase 0: feasibility

- Probe small-cluster `k` behaviour.
- Prototype checkpointed sequential encoding.
- Verify backward through cluster summaries.
- Measure whether kNN temporary memory is released between calls.

### Phase 1: reader and partition

- Add model-specific station reading in both backends.
- Implement deterministic recursive bisection.
- Add reader, partition, and collator tests.
- Preserve baseline contracts.

### Phase 2: model

- Implement the clustered classifier and event aggregation.
- Add checkpointing.
- Register the family and Lightning forward branch.
- Add parity and gradient tests.

### Phase 3: workflow

- Add the canary config, logging, manifests, and smoke tests.
- Validate checkpoint save/resume and reload.

### Phase 4: memory tuning

- Run the cluster-size and forward-budget staircase.
- Select the largest safe values with at least 20% GPU memory headroom.

### Phase 5: fold-0 comparison

Compare baseline and clustered GravNet on the same sampled population using
loss, macro F1, per-class efficiency/purity, confusion matrices, node-count
slices, runtime, and GPU memory. Focus on the large-event-heavy `CC_nue`,
`CC_numu`, and `NC` classes. Test halos only after disjoint v1 is stable.

## 13. Acceptance criteria

1. Every selected candidate occurs exactly once as a core node.
2. No cluster exceeds its limit and no event is rejected for size.
3. Each event produces one logit and one loss contribution.
4. One-cluster mode matches the current GravNet numerically.
5. Forward/backward completes for a real event near 1.8M candidates on the
   target 40 GB A100.
6. Peak allocated and reserved memory are recorded.
7. The selected setting leaves at least 20% GPU memory headroom.
8. Checkpoint resume and sharded evaluation work.
9. Existing GravNet and sparse-ResNet tests/configs remain valid.

## 14. Limitations and deferred alternatives

- Candidate coordinate resolution is preserved, but cluster boundaries remove
  possible learned cross-boundary edges.
- Clustering bounds peak memory; it does not eliminate total work, so the
  largest events may be slow.
- Candidate crossings remain combinatorial hypotheses rather than independent
  raw measurements.
- Physics accuracy must be measured before production adoption.

An exact static Cluster-GCN baseline would construct physical-space kNN edges,
use PyG `ClusterData`/METIS, replace GravNet by GraphSAGE/GCN, and pool by
event. For 1.8M nodes and `k=16`, approximately 28.8M directed edges require
about 461 MB for a two-row int64 edge index alone, before reverse edges,
partition data, features, and activations. It is not the first target.

A detector-native two-view model would reduce the representation from roughly
`N_vertical * N_horizontal` to `N_vertical + N_horizontal`. It is likely the
stronger long-term solution but is outside this implementation plan.
