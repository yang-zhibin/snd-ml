# Region-Aware ML Workflow Plan

## Goal

Build the downstream ML workflow on top of the implemented region/particle
partition stage.

The current design is intentionally simpler than the original catalog-ROOT
proposal. The partition metadata CSV is the central file-level catalog. It
records the produced feature and hit3D partition ROOT files, their region,
particle group, selected event count, lumi, source files, and provenance.

The full workflow is:

```text
production metadata
  + feature.root
  + hit3d.root
        |
        v
region/particle partitions
        |
        +--> feature partition ROOT files
        +--> hit3D partition ROOT files
        +--> partition metadata CSV
        |
        v
split partition metadata
        |
        v
ML dataset builder / streaming inference
        |
        v
model prediction ROOT
        |
        v
feature-with-prediction ROOT
```

## Current Starting Point

The existing production workflow creates aligned source files:

```text
feature.root:sndData
hit3d.root:hit3D
```

`feature.root` contains flat analysis variables used for selections:

```text
runId
eventId
eventIndex
pdgCode
isMC
energy
count_scifi
count_veto
count_us
count_ds
density_sndsw_scifi
avg_scifi*_x
avg_scifi*_y
```

`hit3d.root` contains event-level detector-hit arrays used as ML input:

```text
runId
eventId
eventIndex
pdgCode
label
energy
hit_x
hit_y
hit_z
hit_qdc
hit_station
hit_detType
```

The required assumption is entrywise alignment:

```text
feature.root entry i == hit3d.root entry i
```

The partition workflow preserves that alignment in the produced partition
pairs.

## Implemented Partition Layer

The implemented first stage is documented in
`REGION_PARTITION_WORKFLOW_PLAN.md`.

It produces:

```text
/eos/experiment/sndlhc/users/zhibin/snd-ml/region_partitions/partition_v1/region_v1/<region>/<particle_group>/feature_*.root
/eos/experiment/sndlhc/users/zhibin/snd-ml/region_partitions/partition_v1/region_v1/<region>/<particle_group>/hit3d_*.root
snakemake/metadata/region_partitions/partition_v1_region_v1.csv
```

Important implemented choices:

- no separate event catalog ROOT is produced;
- metadata `lumi_per_file` defines whole-file partition boundaries;
- ROOT RDataFrame applies the final region and particle selections;
- one Snakemake job builds one `region + particle_group + part_index`;
- metadata row CSVs are Snakemake completion markers;
- feature and hit3D ROOT files are side outputs copied to EOS;
- event identity branches are copied/added to support traceability;
- hit3D output is slimmed to the configured ML-relevant branches.

## Partition Metadata As Catalog

The combined partition metadata CSV is now the central catalog for downstream
steps:

```text
snakemake/metadata/region_partitions/partition_v1_region_v1.csv
```

Each row corresponds to one produced partition pair and contains:

```text
partition_version
region_version
region
particle_group
particle_family
particle_id
partition_id
feature_partition_path
hit3d_partition_path
n_events
lumi_per_partition
fraction
max_events
n_partitions_requested
n_partitions_written
selection_expression
particle_filter_expression
source_metadata_csv
source_feature_paths
source_hit3d_paths
source_file_indices
first_original_entry
last_original_entry
shuffle
split_method
seed
```

This replaces the earlier plan's separate `eventCatalog` ROOT for the current
workflow. Event-level traceability is preserved through partition identity,
source paths, source row indices, `original_entry`, and `event_uid` branches in
produced ROOT files.

## Split Metadata Stage

The next stage should create a split metadata CSV from the partition metadata.

Planned files:

```text
snakemake/metadata/configs/splits/split_v1.yaml
convertData/split_region_partitions.py
snakemake/metadata/region_partitions/partition_v1_region_v1_split_v1.csv
```

The split config should describe:

```yaml
split_version: split_v1
seed: 123
fractions:
  train: 0.70
  val: 0.15
  test: 0.15
group_by:
  - region
  - particle_group
inference_regions: []
```

The split script should:

1. read `partition_v1_region_v1.csv`;
2. group partition rows by the configured keys;
3. shuffle whole partition rows with a fixed seed;
4. assign rows to train/val/test;
5. optionally mark selected regions as inference-only;
6. write a new metadata CSV with split columns.

Suggested added columns:

```text
split_version
split
usage
split_seed
split_group
split_strategy
```

Allowed `split` values:

```text
train
val
test
none
```

Allowed `usage` values:

```text
train
val
test
inference
excluded
```

Splitting by whole partition files keeps the workflow simple and prevents train
and validation from sharing events from the same produced partition.

## ML Dataset Builder

The ML dataset builder should consume split partition metadata, not define
regions or partition boundaries itself.

For training, it reads rows such as:

```text
split == train
split == val
split == test
```

For inference, it reads rows such as:

```text
usage == inference
```

The builder opens `hit3d_partition_path` files and materializes model inputs.
It may also read `feature_partition_path` for labels, analysis variables, or
auxiliary selections when needed.

The partition files should stay general-purpose. Detector-view choices belong
in the ML dataset builder.

Example input views:

```text
full_detector
scifi_only
ds_only
scifi_plus_ds
```

For the current hit3D convention:

```text
hit_detType == 1 -> SciFi
hit_detType == 2 -> DS crossed hits
hit_detType == 3 -> US/DS4 voxelized hits
```

An input-view config can select detector hits and hit features, for example:

```yaml
input_view: scifi_only
input_view_version: v1
allowed_detType:
  - 1
selected_hit_features:
  - hit_x
  - hit_y
  - hit_z
  - hit_qdc
  - hit_station
```

The dataset manifest should record:

```text
input_view
input_view_version
allowed_detType
selected_hit_features
split_version
split
usage
region
particle_group
partition_id
source hit3D partition path
```

## Model Prediction

Prediction should be written as separate ROOT files. It should not overwrite
feature partitions.

A prediction file should contain enough identity and model metadata to merge
back later:

```text
event_uid
partition_id
local_entry
runId
eventId
eventIndex
pdgCode
label
energy
region
particle_group
split_version
split
usage
model_name
model_version
prediction_version
input_view
input_view_version
Prediction_0
Prediction_1
Prediction_2
PredClass
PredScore
```

The natural merge key is:

```text
event_uid
```

Fallback keys, if needed, are:

```text
partition_id + local_entry
runId + eventId + eventIndex + source_file_index
```

For MC, `runId` and `eventId` alone are not globally unique, so source or
partition identity must be included.

## Merge Prediction Back Into Features

The final analysis product should merge:

```text
feature_partition.root + prediction.root
```

and write:

```text
feature_with_prediction_<region>__<particle_group>__partNNN.root
```

Added branches should include:

```text
ml_has_prediction
ml_model_name
ml_model_version
ml_prediction_version
ml_input_view
ml_input_view_version
ml_Prediction_0
ml_Prediction_1
ml_Prediction_2
ml_PredClass
ml_PredScore
```

Then final analysis can use both hand-made features and model outputs:

```text
count_scifi
count_veto
count_us
density_sndsw_scifi
ml_PredClass
ml_PredScore
```

## Luminosity Handling

Luminosity is handled at partition-file level in the current workflow.

The partition metadata records:

```text
lumi_per_partition
source_metadata_csv
source_file_indices
source_feature_paths
source_hit3d_paths
```

This is sufficient for later normalization by:

```text
region
particle_group
split
usage
```

Do not bake final analysis weights into the partition files. Keep lumi and
event counts available so weights can be computed later per analysis choice.

Useful summaries after split generation:

```text
sum lumi by region and particle_group
sum lumi by split and particle_group
event count by region and split
event count by usage
```

## Traceability

Traceability is preserved through both metadata and ROOT branches.

Important fields are:

```text
event_uid
partition_id
original_entry
source_file_index
source_feature_paths
source_hit3d_paths
source_metadata_csv
source_file_indices
runId
eventId
eventIndex
pdgCode
```

The intended chain is:

```text
prediction
  -> partition_id + event_uid/local_entry
  -> feature/hit3D partition
  -> partition metadata row
  -> source metadata row and source ROOT file
```

## Snakemake Structure

Implemented:

```text
build_region_partition
combine_region_partition_metadata
region_partitions
```

Planned next rules:

```text
split_region_partitions
build_ml_dataset_partition
run_model_prediction_partition
merge_prediction_to_feature_partition
summarize_region_partitions
summarize_split_metadata
```

The future dependency chain should be:

```text
feature_with_prediction_partition
    needs feature_partition
    needs prediction_partition

prediction_partition
    needs model checkpoint
    needs split metadata or partition metadata query
    needs hit3D partition

ml_dataset_partition
    needs split metadata
    needs hit3D partition
    needs input-view config

split metadata
    needs partition metadata
    needs split config

partition metadata
    needs source metadata
    needs region config
    needs partition config
    needs build_region_partitions.py
```

## Recommended Next Implementation Order

1. Finish a clean full `region_partitions` run.
2. Validate the combined partition metadata and ROOT entry counts.
3. Implement `split_region_partitions.py`.
4. Add `split_v1.yaml`.
5. Add a Snakemake rule for split metadata.
6. Update the ML dataset builder to read split metadata and hit3D partition
   files.
7. Add input-view configs for full-detector and SciFi-only training.
8. Add prediction ROOT output.
9. Add merge-to-feature output.
10. Add summary scripts for lumi and event counts.

## Final Logic Summary

The current best workflow is:

```text
1. Produce aligned source feature.root and hit3d.root.
2. Build lumi-based region/particle partitions with RDF selection.
3. Store produced partition paths, lumi, counts, and provenance in a metadata CSV.
4. Split whole partition rows into train/val/test/inference metadata.
5. Build ML datasets from hit3D partitions according to input-view configs.
6. Apply models on split data or inference regions.
7. Save predictions with full identity and model metadata.
8. Merge predictions back into feature partitions for final analysis.
```

This keeps partitioning, regions, splits, inference, luminosity, and
traceability explicit while avoiding a heavy event-catalog pre-scan.
