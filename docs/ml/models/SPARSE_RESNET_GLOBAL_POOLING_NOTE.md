# Sparse ResNet global-pooling edge case

## Decision

The SciFi `SparseResNetClassifier` uses an explicit PyTorch mean over each
event's encoded sparse features. It does not use
`MinkowskiGlobalAvgPooling`.

This is a correctness adaptation for low-occupancy detector events. It keeps
the intended operation—one average feature vector per event—but also works
when an event has already collapsed to one active coordinate.

## Failure observed

The original implementation used three stride-2 residual stages followed by
`MinkowskiGlobalAvgPooling`. A four-voxel test event could therefore become
one active coordinate. With the pinned MinkowskiEngine CPU build, global
pooling then failed with:

```text
coordinate_map_cpu.hpp:737, assertion (in_size > out_size) failed.
Invalid out_coordinate_map
```

This was not a memory, container, or data-loader failure. The global pooling
operator attempted to construct one output coordinate per event when there
was already only one input coordinate per event.

## Relation to the original author implementation

This repository's classifier is not an exact published model. Its residual
blocks follow the official MinkowskiEngine examples, while its event head and
SciFi input contract are specific to SND@LHC.

The official [ModelNet40 classification example](https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/classification_modelnet40.py)
uses global maximum and average pooling after multiple spatial reductions. It
contains no explicit one-coordinate guard. ModelNet40 starts from much larger,
finely voxelized point clouds, so the normal benchmark inputs retain multiple
active coordinates at the pooling stage. The likely reason the authors did
not report this edge case is therefore their input regime, not a special fix
in the example. This is an inference from the example and its preprocessing.

Using both maximum and average pooling would not fix our edge case: both are
Minkowski global-pooling operations with the same coordinate-map construction.
The [pooling implementation](https://github.com/NVIDIA/MinkowskiEngine/blob/master/MinkowskiEngine/MinkowskiPooling.py)
also has no Python-level guard for an already-global coordinate map.

## Implemented behavior

After the sparse encoder, the classifier groups rows of the encoded feature
matrix (`encoded.F`) using the batch column in the corresponding coordinates
(`encoded.C[:, 0]`). Vectorized PyTorch `index_add` and `bincount` operations
compute the event sums and counts, followed by:

```python
event_vector = event_features.mean(dim=0)
```

The event vectors are stacked in batch order and passed to the classification
head. The implementation additionally verifies that:

- the input batch is nonempty;
- batch IDs are contiguous and start at zero;
- the encoder preserves every input event; and
- no encoded event is empty.

This reduction remains differentiable and gives exactly the identity result
for a one-coordinate event. It has no event-by-event Python loop and can run
on either CPU or GPU.

The implementation deliberately does not use `encoded.decomposed_features`.
In the pinned MinkowskiEngine version that convenience property calls
`origin_map`, which triggers the same assertion as global pooling for an
already-global sparse tensor.

## Regression coverage

`test/test_scifi_classification_models.py` covers:

1. forward and backward propagation when small events collapse during the
   encoder;
2. forward and backward propagation for an event containing exactly one input
   voxel; and
3. a mixed one-voxel/multi-voxel batch, checking that the first event's result
   is unchanged when another event is added.

Changing voxel size or removing downsampling can reduce how often events
collapse, but those are physics/model-capacity choices rather than a complete
correctness fix. The explicit event-wise reduction remains necessary.

## Verification

Verified on 2026-09-18 inside
`snd-ml-training-eca898623c12ae80.sif` on CPU:

- model tests: 5 passed, including both new sparse edge-case tests;
- shared SciFi collation/pipeline tests: 9 passed;
- fold dataset and sampling tests: 11 passed; and
- seven-class contract tests: 4 passed.

The container emits an upstream MinkowskiEngine autograd deprecation warning,
but there are no test failures or non-finite gradients. This source-code fix
does not change the container dependencies, so rebuilding the SIF is not
required as long as the checkout is bind-mounted in the training job as
planned.
