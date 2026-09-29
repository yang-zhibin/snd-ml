# SciFi Training Metadata v2

## Purpose

This implementation changes metadata construction and lookup without changing
the training data definition. Raw ROOT files remain the source of model input.
The selected hits are still exactly `hit_detType == 1`, normalization uses all
eligible hits in the four training folds, and sparse voxel counts use each
held-out fold's own training-derived coordinate origin.

The old `build_scifi_statistics.py` pipeline remains available as a reference.
The v2 products are written to a different directory:

```text
train_3d/metadata/scifi_xyz_qdc_dettype_metadata_v2/
```

## Workflow

The v2 Snakemake target is:

```bash
snakemake data_driven_3d_cnn_scifi_training_metadata_v2 -j 2
```

Use the repository HTCondor profile for production. The two ROOT-scanning
worker rules request the training container because LCG_107 contains uproot
5.3.11, which is known to hang on some `std::vector` branches. The tested
training container contains uproot 5.7.6. Planner, reducer and validator jobs
remain CPU-only LCG jobs.

The stages are:

1. deterministically assign complete ROOT files to 50 balanced shards;
2. build physical-fold statistics in parallel;
3. reduce them in shard order into five leave-one-fold-out normalizations;
4. scan each event once more and calculate one candidate count plus five exact
   fold-specific voxel counts;
5. assemble memory-mappable arrays and validate checksums, identities, shapes,
   dtypes and count invariants.

`complete.json` is written last. Training additionally requires
`validation.json`, so incomplete metadata cannot be used accidentally.

## Training lookup

Training no longer reads a 43-million-row CSV into pandas and a Python
dictionary. `EventSizeIndex.from_binary` keeps only the per-file table in
memory and memory-maps:

- `source_entries.npy` plus `n_candidates.npy` for GravNet; or
- `source_entries.npy` plus `n_voxels_fold_<cv_fold>.npy` for sparse ResNet.

With `metadata_local_cache: auto`, these required arrays are copied atomically
to `$TMPDIR` once per job/container namespace. A file lock prevents concurrent
ranks from using partial copies, and copied files are checked against the
checksums in `complete.json`.

Sampling order and `NodeBudgetBatchSampler` packing logic are unchanged.

## Tests

`test/test_scifi_metadata_v2.py` covers:

- accumulator serialization and deterministic merging;
- binary lookup, missing identities and unchanged batch boundaries;
- chunked SciFi selection;
- two-shard statistics and size production;
- reduction, validation and job-local staging;
- fold-specific normalization and voxel-count expectations.

The focused model, collator, sampler, resume-safety and legacy-builder tests
must also pass before production metadata is launched.
