# Region-Aware ML Workflow Plan

## Goal

Define the downstream region-aware ML workflow after region/particle partition
production is finished.

The partition stage produces aligned feature and hit3D ROOT partition files and
a combined partition metadata CSV. This post-partition workflow uses that CSV
as the central catalog for splitting data, building ML datasets, training
models, producing predictions for all analysis files, merging predictions back
into feature files, and evaluating results with both hand-made and ML-derived
features.

The intended workflow is:

```text
partition_v1_region_v1.csv
        |
        v
fold-aware split partition metadata
        |
        v
ML dataset builder, one job per fold + partition
        |
        v
model training, one model per fold
        |
        v
prediction for fold test partitions
        |
        v
out-of-fold prediction catalog
        |
        v
feature + prediction merge
        |
        v
evaluation / histograms / feature studies
```

## Starting Point

The input catalog is:

```text
snakemake/metadata/region_partitions/partition_v1_region_v1.csv
```

Each row represents one produced partition pair:

```text
feature_partition_path
hit3d_partition_path
region
particle_group
particle_family
particle_id
partition_id
n_events
lumi_per_partition
source_feature_paths
source_hit3d_paths
source_metadata_csv
```

The key invariant from the partition stage is:

```text
feature_partition.root:sndData entry i == hit3d_partition.root:hit3D entry i
```

Downstream steps should preserve this alignment and should not redefine region
cuts or partition boundaries.

## Stage 1: Split Region Partitions

### Purpose

Assign every whole partition file to a clear downstream role.

This stage splits metadata rows, not individual ROOT events. Keeping whole
partition files together avoids leakage where events from the same produced
partition appear in both training and validation.

The split stage should also define the ML study configuration attached to this
split version:

```text
training-eligible regions
training-eligible particle groups
optional real-data sideband training samples
classification classes
default behavior for non-training partitions
input view used by the ML conversion
```

The split config is therefore more than a train/validation/test fraction file.
It is the contract for one complete ML dataset version.

### Planned Files

```text
snakemake/metadata/configs/splits/split_v1.yaml
convertData/split_region_partitions.py
snakemake/metadata/region_partitions/partition_v1_region_v1_split_v1.csv
```

### Input

```text
snakemake/metadata/region_partitions/partition_v1_region_v1.csv
```

### Example Split Config

```yaml
split_version: split_v1
seed: 123

description: >
  First split for a five-class classifier using one real-data no-veto/no-US
  sideband proxy class and the four neutrino truth classes. The input view uses
  SciFi hits only.

fractions:
  train: 0.40
  val: 0.10
  test: 0.50

cross_validation:
  enabled: true
  strategy: two_fold_cross_fit
  n_folds: 2
  test_fraction_per_fold: 0.50
  val_fraction_within_train_pool: 0.20

group_by:
  - region
  - particle_group

split_strategy: two_fold_cross_fit_by_partition_rows

default_unassigned:
  split: test
  usage: test

real_data_default:
  split: test
  usage: inference

training_selection:
  include_regions:
    # Real-data sideband: no veto and no US, both fiducial categories.
    - sideband_no_veto_no_us_fiducial_inside
    - sideband_no_veto_no_us_fiducial_outside

    # Neutrino training region.
    - signal_no_veto_has_us_fiducial_inside

  include_particle_groups:
    - real_data
    - CC_nue
    - CC_numu
    - NC_nue
    - NC_numu

  allow_real_data_training: true

exclude:
  regions: []
  particle_groups: []

classification:
  task: particle_classification
  n_classes: 5
  classes:
    - class_id: 0
      class_name: real_data_no_veto_no_us_sideband
      class_source: proxy_label
      particle_groups:
        - real_data
      regions:
        - sideband_no_veto_no_us_fiducial_inside
        - sideband_no_veto_no_us_fiducial_outside

    - class_id: 1
      class_name: CC_nue
      class_source: truth_label
      particle_groups:
        - CC_nue
      regions:
        - signal_no_veto_has_us_fiducial_inside

    - class_id: 2
      class_name: CC_numu
      class_source: truth_label
      particle_groups:
        - CC_numu
      regions:
        - signal_no_veto_has_us_fiducial_inside

    - class_id: 3
      class_name: NC_nue
      class_source: truth_label
      particle_groups:
        - NC_nue
      regions:
        - signal_no_veto_has_us_fiducial_inside

    - class_id: 4
      class_name: NC_numu
      class_source: truth_label
      particle_groups:
        - NC_numu
      regions:
        - signal_no_veto_has_us_fiducial_inside

input_view:
  name: scifi_only
  version: v1
  allowed_detType:
    - 1
  selected_hit_features:
    - hit_x
    - hit_y
    - hit_z
    - hit_qdc
    - hit_station
```

This structure supports different studies without changing code:

```text
MC-only training
MC plus real-data sideband training
signal-region-only training
signal plus sideband training
binary classification
three-class classification
many-class particle classification
different detector input views
```

Real-data training should be treated carefully. If real-data sidebands are used
for training, the class name should describe the selection/source, for example:

```text
sideband_real_data
```

Do not label real data as a truth particle type unless the study intentionally
uses proxy labels or weak supervision.

### Split Logic

The script should first classify every row into one of three broad categories:

```text
excluded
training-eligible
not training-eligible but still useful for testing/evaluation/inference
```

Then it should assign split and usage values.

The split script should:

1. read the combined partition metadata CSV;
2. read the split config;
3. validate required metadata columns and config fields;
4. determine whether each row is excluded;
5. determine whether each row is training-eligible;
6. assign a configured class to eligible rows;
7. group eligible rows by the configured `group_by` keys;
8. shuffle each group with the configured seed;
9. assign whole partition rows to `train`, `val`, and `test`;
10. assign non-eligible but usable rows to the configured default;
11. attach input-view information to every usable row;
12. write a new split metadata CSV;
13. print and optionally write split summaries.

The recommended default split grouping is:

```text
region + particle_group
```

This keeps each region and particle group represented across train,
validation, and test when enough partitions exist. Class assignment should be
config-defined, but the split grouping should usually remain more granular than
the class name.

For example, if there are 10 partition files for:

```text
signal_no_veto_has_us_fiducial_inside + CC_nue
```

and the fractions are:

```text
train = 0.40
val = 0.10
test = 0.50
```

the split could assign:

```text
4 partitions -> train
1 partition  -> val
5 partitions -> test
```

Each partition remains intact.

### Cross-Validation Logic

For low-statistics regions, the split should support two-fold cross-fitting.
For each configured split group, usually:

```text
region + particle_group
```

the script should divide partition rows into two halves:

```text
half_0
half_1
```

Then it should create two fold assignments:

```text
fold_0:
  test = half_0
  train/val pool = half_1
  train = 80% of half_1
  val = 20% of half_1

fold_1:
  test = half_1
  train/val pool = half_0
  train = 80% of half_0
  val = 20% of half_0
```

This gives approximately:

```text
train = 40%
val = 10%
test = 50%
```

per fold. Across both folds, every partition appears in a test set once. Final
unbiased evaluation should use only out-of-fold predictions:

```text
split == test
```

from each fold.

The split output should therefore contain one row per:

```text
partition_id + fold
```

not just one row per partition.

### Added Columns

The split output should preserve all partition metadata columns and add:

```text
split_version
cv_enabled
cv_strategy
fold
fold_index
n_folds
split
usage
is_train_eligible
is_real_data
class_id
class_name
class_source
split_seed
split_group
split_strategy
input_view
input_view_version
allowed_detType
selected_hit_features
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

Recommended interpretation:

```text
split == train       partition can be used for training
split == val         partition can be used for validation
split == test        partition is available for testing/evaluation
split == none        partition is intentionally not part of a data split

usage == train       use in the training loss
usage == val         use for validation during training
usage == test        labeled test/evaluation sample
usage == inference   prediction/evaluation sample without truth-label training assumptions
usage == excluded    do not process unless explicitly requested
```

### Expected Usage

MC training samples:

```text
split = train / val / test
usage = train / val / test
```

Real data:

```text
split = test
usage = inference
```

Real-data sidebands used for training:

```text
split = train / val / test
usage = train / val / test
class_name = sideband_real_data
```

This keeps the workflow explicit: real data can be used for training only when
the config allows it and assigns it a clear proxy class.

### Default-To-Test Policy

Every row should receive an explicit assignment. Rows should not be left blank.

The recommended default is:

```text
If a row is not selected for training/validation and is not explicitly
excluded, assign it to test.
```

For MC rows:

```text
split = test
usage = test
```

For real-data rows not used in training:

```text
split = test
usage = inference
```

For intentionally ignored rows:

```text
split = none
usage = excluded
```

This makes later evaluation simple:

```text
select split == test
```

while still allowing code to distinguish labeled MC test samples from real-data
inference samples through `usage`.

### Validation

The split script should check:

```text
input CSV exists
required columns exist
fractions sum to 1.0
group_by columns exist
class definitions are valid
each train-eligible row matches exactly one class
excluded rows are explicitly marked
all output rows have split and usage
all usable rows have input_view and input_view_version
partition_id values are unique unless duplicates are explicitly allowed
```

Useful summaries:

```text
row count by split
row count by usage
row count by class_name
row count by region + particle_group + split
sum n_events by split and usage
sum lumi_per_partition by split and usage
```

## Stage 2: Build ML Dataset

### Purpose

Convert hit3D partition ROOT files into ML-ready files while preserving the
partition structure, split assignment, fold assignment, and class labels.

The ML dataset builder should consume the split partition metadata. It should
not define regions, particle groups, or partition boundaries itself.

The recommended design is:

```text
one fold-aware split metadata row -> one ML conversion job -> one ML output file
```

This mirrors the existing partition workflow:

```text
one Snakemake job per fold + region + particle_group + part_index
one row CSV as completion marker
one combined metadata catalog
```

This stage is where sparse detector hits become model inputs. The hit3D files
should remain a reusable, detector-level representation. The ML converter
should decide the exact tensor view, feature planes, output format, and
normalization policy for one training study.

### Inputs

```text
snakemake/metadata/region_partitions/partition_v1_region_v1_split_v1.csv
hit3d_partition_path
feature_partition_path
```

For training, select rows with:

```text
usage == train
usage == val
usage == test
```

For inference-only dataset materialization, select rows with:

```text
usage == inference
```

Rows with:

```text
usage == excluded
```

should not be converted unless a special debug or override flag is used.

### Current hit3D Contract

The current `convertData/digi_2_hits3D.py` output is sparse: it stores only
valid detector hits. Empty detector cells are not written to ROOT. Dense,
zero-filled CNN tensors must therefore be created by the ML converter.

The common coordinate branches are:

```text
hit_x
hit_y
hit_z
hit_qdc
hit_station
hit_detType
```

The index-rich branches are:

```text
hit_ix
hit_iy
hit_iz
hit_index_valid
hit_index_type
hit_v_channel
hit_h_channel
hit_v_qdc
hit_h_qdc
```

Detector type convention:

```text
hit_detType == 1 -> SciFi
hit_detType == 2 -> upstream MuFilter / US
hit_detType == 3 -> downstream MuFilter / DS
```

Index type convention:

```text
0 -> unknown
1 -> SciFi crossed vertical-horizontal hit
2 -> US voxel hit
3 -> DS crossed vertical-horizontal hit
4 -> DS station-4 single-orientation voxel hit
5 -> SciFi vertical-only hit
6 -> SciFi horizontal-only hit
7 -> DS vertical-only hit
8 -> DS horizontal-only hit
```

For crossed SciFi hits:

```text
hit_detType == 1
hit_index_type == 1
hit_index_valid == 1
hit_ix == vertical SciFi layer channel
hit_iy == horizontal SciFi layer channel
hit_iz == station - 1
hit_v_channel == vertical SciFi layer channel
hit_h_channel == horizontal SciFi layer channel
hit_v_qdc == vertical-hit QDC
hit_h_qdc == horizontal-hit QDC
hit_qdc == hit_v_qdc + hit_h_qdc
```

The SciFi layer channel saved by hit3D is:

```text
layer_channel = sipm_channel + sipm * 128 + mat * 4 * 128
```

For crossed DS MuFilter hits:

```text
hit_detType == 3
hit_index_type == 3
hit_index_valid == 1
hit_ix == vertical MuFilter channel
hit_iy == horizontal MuFilter channel
hit_iz == station - 1
```

For US and DS station-4 voxel hits:

```text
hit_detType == 2 and hit_index_type == 2 -> US voxel
hit_detType == 3 and hit_index_type == 4 -> DS station-4 voxel
hit_index_valid == 1
hit_ix, hit_iy, hit_iz are local voxel indices inside the expanded bar volume
```

For one-orientation hits:

```text
hit_index_type in [5, 6, 7, 8]
hit_index_valid == 0
the missing transverse axis is saved as -1
```

One-orientation hits should not be filled directly into the main 2D crossed-hit
grid because one transverse index is genuinely unknown. They can be used later
as optional projection tensors, mask features, or auxiliary inputs.

### Recommended First ML Input

The recommended first converter target is a SciFi-only dense CNN tensor. This
is the cleanest first model because SciFi crossed hits already provide a stable
2D channel pair per station.

For the first version:

```text
input view: scifi_only
hit selection:
  hit_detType == 1
  hit_index_type == 1
  hit_index_valid == 1
tensor axes:
  ix, iy, station, feature
first feature planes:
  qdc_sum
  v_qdc
  h_qdc
  occupancy
```

The per-event tensor should have fixed shape:

```text
(N_x, N_y, N_station, N_feature)
```

For file-based storage, one partition output should store:

```text
X_scifi shape = (N_event, N_x, N_y, N_station, N_feature)
y shape       = (N_event,)
```

The simplest robust output format for this stage is `NPZ`. It is easy to
inspect, easy to load from PyTorch, and does not commit the workflow to one
training framework. A later PyTorch `Dataset` can load the NPZ files and apply
training-time transforms.

Recommended NPZ keys:

```text
X_scifi
y
runId
eventId
eventIndex
local_entry
class_id
partition_id
fold_index
split
usage
```

### Channel and Grid Convention

The raw hit3D indices are detector indices, not yet a guaranteed
physics-oriented image convention. The converter should therefore support two
grid modes:

```text
raw_index
coordinate_ordered
```

In `raw_index` mode, the converter uses:

```text
ix = hit_ix
iy = hit_iy
iz = hit_iz
```

This is the fastest first implementation and is already useful for training
because the mapping is deterministic.

In `coordinate_ordered` mode, the converter builds or reads a lookup table:

```text
detector, station, orientation, raw_channel -> dense_index
```

This lookup should be derived from the combination of:

```text
raw channel index
hit_x
hit_y
hit_z
station
orientation
```

This avoids assuming whether increasing channel number means increasing or
decreasing `x` or `y`. Since hit3D already stores both raw channel indices and
aligned coordinates, this mapping can be handled entirely in the converter
without changing the hit3D format again.

The converter should record the mapping choice in the output manifest:

```text
grid_mode
grid_version
channel_map_path
channel_map_version
```

### Input-View Configuration

Detector-view choices should be configured by the ML dataset config and applied
by the dataset builder. They should not be baked into the partition ROOT files.

Example input views:

```text
scifi_only
scifi_plus_ds
ds_only
full_detector
```

Recommended first `scifi_only` view:

```yaml
input_view:
  name: scifi_only
  version: v1
  tensor_format: dense_cnn
  output_format: npz
  grid_mode: raw_index
  detectors:
    scifi:
      enabled: true
      detType: 1
      index_types:
        - 1
      require_index_valid: true
      axes:
        x: hit_ix
        y: hit_iy
        station: hit_iz
      feature_planes:
        - qdc_sum
        - v_qdc
        - h_qdc
        - occupancy
      aggregation:
        qdc_sum: sum
        v_qdc: sum
        h_qdc: sum
        occupancy: count
```

A later multi-detector view should keep detector systems separate:

```text
X_scifi for SciFi crossed hits
X_us    for upstream MuFilter voxel hits
X_ds    for downstream MuFilter crossed and voxel hits
```

SciFi channel indices and MuFilter voxel indices should not be mixed into one
shared image grid. They have different detector meanings and different natural
granularities. The neural network can combine them after separate detector
encoders, for example by concatenating latent features.

### Planned Files

```text
convertData/convert_hit3d_partition_to_ml.py
snakemake/rules/ml_dataset.smk
snakemake/metadata/configs/ml_datasets/<dataset_version>.yaml
snakemake/metadata/ml_datasets/<dataset_version>/rows/<fold>__<partition_id>.csv
snakemake/metadata/ml_datasets/<dataset_version>.csv
```

The exact dataset version can be built from:

```text
split_version
input_view
input_view_version
output_format
grid_mode
```

For example:

```text
split_v1__scifi_only_v1__raw_index__npz
```

### Conversion Logic

For one partition row, the conversion script should:

1. read the split metadata CSV;
2. select exactly one row by `partition_id` and `fold`;
3. skip or write an empty marker for `usage == excluded`;
4. open `hit3d_partition_path`;
5. optionally open `feature_partition_path` for auxiliary variables;
6. validate that required branches exist;
7. loop over events in the partition;
8. read all hit vectors for the event;
9. validate that hit vector branches have equal length;
10. filter hits using the configured detector and `hit_index_type` rules;
11. map hit indices to dense grid indices using `raw_index` or a channel map;
12. initialize zero-filled tensors for the event;
13. fill configured feature planes;
14. aggregate repeated hits in the same dense cell using the configured reducer;
15. attach labels and event identifiers from the split metadata and ROOT event;
16. write one ML output file;
17. write one dataset metadata row.

The conversion should not decide whether a file is train, validation, test, or
inference. It should only obey the split metadata.

### Filling Rules

For the first SciFi-only tensor:

```text
select hit_detType == 1
select hit_index_type == 1
require hit_index_valid == 1
require hit_ix >= 0
require hit_iy >= 0
require hit_iz >= 0
```

Then for each selected hit:

```text
X_scifi[event, ix, iy, iz, qdc_sum]   += hit_qdc
X_scifi[event, ix, iy, iz, v_qdc]     += hit_v_qdc
X_scifi[event, ix, iy, iz, h_qdc]     += hit_h_qdc
X_scifi[event, ix, iy, iz, occupancy] += 1
```

The converter should not silently drop out-of-bounds hits. It should count them
and either fail or write the count to metadata depending on a strictness flag.

Recommended default:

```text
strict_validation: true
```

Normalization should be explicit. For the first implementation, store raw
summed QDC and occupancy. Apply clipping, log transforms, and standardization
in the training pipeline or in a later explicitly versioned converter config.

### Validation

At minimum, the converter should validate:

```text
required hit3D branches are present
all hit vector branches have the same length per event
crossed and voxel hits have hit_index_valid == 1
crossed and voxel hits have non-negative hit_ix, hit_iy, hit_iz
one-orientation hits have hit_index_valid == 0
dense indices are within configured tensor bounds
the selected split metadata row is unique
usage == train/val/test rows have valid class_id and class_name
usage == inference rows may have proxy or placeholder labels
the number of output events matches the number of input ROOT events selected
```

Validation counters should be written into the row metadata:

```text
n_events_read
n_events_written
n_hits_read
n_hits_selected
n_hits_filled
n_hits_out_of_bounds
n_hits_missing_index
n_empty_events
```

### Dataset Output and Manifest

Each fold + partition row should produce one ML output file, for example:

```text
ml_dataset_<fold>__<partition_id>.npz
```

The dataset builder should also write a manifest row that records:

```text
dataset_version
dataset_format
input_view
input_view_version
tensor_format
tensor_keys
tensor_shapes
feature_planes
grid_mode
grid_version
channel_map_path
split_version
fold
fold_index
split
usage
is_train_eligible
class_id
class_name
region
particle_group
partition_id
hit3d_partition_path
feature_partition_path
ml_output_path
n_events_read
n_events_written
n_hits_selected
n_hits_filled
n_empty_events
```

This manifest is important because the same partition files can be reused with
different detector views, grid conventions, feature planes, or output formats.

### Snakemake Shape

The conversion rule should be one job per fold + partition. Conceptually:

```text
convert_ml_dataset_partition
    input:
        split metadata CSV
        hit3D partition path from the selected row
        optional feature partition path from the selected row
    output:
        one ML-format file
        one dataset metadata row CSV
```

The rule should load the split metadata at parse time, like the partition
workflow does, and create one target per usable row. The row CSV is the
completion marker and the combined catalog is the source of truth for training.

Then:

```text
combine_ml_dataset_metadata
```

should combine the row CSV files into one dataset catalog.

### Stage 2 Implementation Order

Recommended order:

1. create the ML dataset config for `scifi_only_v1`;
2. implement the converter in `raw_index` mode;
3. write NPZ output with `X_scifi`, labels, and event identifiers;
4. write one row metadata CSV per fold-aware partition row;
5. add Snakemake rules for per-row conversion and manifest combination;
6. run one small partition as a validation sample;
7. inspect tensor shape, nonzero occupancy, label arrays, and manifest counters;
8. only then run all partition rows.

## Stage 3: Train Model

### Purpose

Train one ML model per fold using the partition-derived datasets.

Training should use only rows assigned to training and validation. Real data
and inference-only rows should not be included in the loss unless the split
config explicitly allowed those rows as training samples and assigned them a
training class.

### Inputs

```text
train dataset
validation dataset
test dataset
dataset manifest
model config
```

### Outputs

```text
model checkpoint
training metrics
model metadata
```

The model metadata should record:

```text
model_name
model_version
checkpoint_path
dataset_version
split_version
input_view
input_view_version
classification_task
class_map
training_config
training_timestamp
```

### Important Rule

The training workflow should no longer depend on old production metadata split
files such as rows with:

```text
output_base_path
npz_hit_path
split
```

Instead, training should depend on:

```text
split partition metadata
ML dataset manifest
```

For each fold, training should select dataset rows using:

```text
fold == current_fold
usage == train
usage == val
```

Testing after training should select:

```text
fold == current_fold
usage == test
```

Inference-only rows can be converted and predicted, but should not contribute
to training metrics that assume truth labels.

The final analysis should use the model prediction for the fold where each
partition was held out:

```text
split == test
```

This is the out-of-fold prediction and avoids evaluating on partitions used to
train that model.

## Stage 4: Predict For Partition Files

### Purpose

Run each fold's trained model over the partition files that should be
predicted.

For unbiased evaluation, prediction must be produced for each fold's held-out
test rows:

```text
fold == current_fold
split == test
```

Prediction may also be produced for train/validation/inference rows for
debugging or monitoring, but final model-performance evaluation should use only
out-of-fold test predictions.

### Inputs

```text
split partition metadata
fold
hit3d_partition_path
model checkpoint
input-view config
```

### Output

Prediction files should be separate ROOT files. They should not overwrite or
modify feature partition files.

Example layout:

```text
prediction_<fold>__<region>__<particle_group>__partNNN.root
```

Suggested prediction tree branches:

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
fold
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
Prediction_3
Prediction_4
PredClass
PredScore
```

### Prediction Catalog

The prediction stage should write or update metadata with:

```text
partition_id
fold
prediction_path
model_name
model_version
prediction_version
input_view
input_view_version
```

This allows multiple models or detector views to be compared without replacing
old prediction outputs.

The out-of-fold prediction catalog should select only:

```text
split == test
```

from all folds, giving one unbiased prediction source for each partition.

## Stage 5: Merge Prediction Back Into Feature Files

### Purpose

Create analysis-ready ROOT files that contain both standard feature branches
and ML prediction branches.

The merge stage combines:

```text
feature_partition.root
prediction.root
```

and writes:

```text
feature_with_prediction_<region>__<particle_group>__partNNN.root
```

### Merge Keys

Preferred key:

```text
event_uid
```

Fallback keys:

```text
partition_id + local_entry
runId + eventId + eventIndex + source_file_index
```

For MC, `runId` and `eventId` alone are not globally unique, so partition or
source identity must be included.

### Added Branches

The merged feature file should add:

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
ml_Prediction_3
ml_Prediction_4
ml_PredClass
ml_PredScore
ml_fold
```

The original feature branches should be preserved unchanged.

## Stage 6: Evaluation With Different Features

### Purpose

Evaluate signal and background behavior using both hand-made variables and ML
outputs.

Evaluation should read:

```text
feature_with_prediction_<region>__<particle_group>__partNNN.root
split partition metadata
prediction metadata
```

### Feature Families

Hand-made feature examples:

```text
count_scifi
count_veto
count_us
count_ds
density_sndsw_scifi
avg_scifi*_x
avg_scifi*_y
```

ML feature examples:

```text
ml_Prediction_0
ml_Prediction_1
ml_Prediction_2
ml_PredClass
ml_PredScore
```

### Evaluation Modes

Useful studies include:

1. region-only selections;
2. region plus ML class selections;
3. region plus ML score threshold selections;
4. feature distributions before and after ML cuts;
5. signal/background comparison by particle group;
6. real-data sideband comparisons;
7. stability checks across regions;
8. model comparison across input views or model versions.

Example selections:

```text
region == signal_no_veto_has_us_fiducial_inside
ml_PredClass == signal_class
ml_PredScore > 0.80
```

or:

```text
count_scifi > 35
count_veto == 0
count_us > 0
ml_PredScore > 0.90
```

### Luminosity And Weights

Luminosity should remain metadata-level information:

```text
lumi_per_partition
region
particle_group
split
usage
```

Do not bake final analysis weights into partition ROOT files or merged ROOT
files. Compute weights during evaluation based on the analysis choice.

Useful summaries:

```text
sum lumi by region and particle_group
sum lumi by split and particle_group
event count by region and split
event count by usage
event count after each ML cut
```

## Snakemake Rule Roadmap

The partition workflow already has:

```text
build_region_feature_partition
build_region_hit3d_partition
combine_region_partition_metadata
region_feature_partitions
region_hit3d_partitions
region_partitions
```

The post-partition workflow should add:

```text
split_region_partitions
convert_ml_dataset_partition
combine_ml_dataset_metadata
train_ml_model_fold
run_model_prediction_partition_fold
combine_prediction_metadata
combine_out_of_fold_predictions
merge_prediction_to_feature_partition
summarize_split_metadata
summarize_prediction_outputs
evaluate_feature_with_prediction
```

The dependency chain should be:

```text
evaluate_feature_with_prediction
    needs feature_with_prediction files
    needs split metadata
    needs evaluation config

feature_with_prediction partition
    needs feature partition
    needs prediction partition

prediction partition
    needs fold model checkpoint
    needs fold split metadata row
    needs hit3D partition
    needs input-view config

fold model checkpoint
    needs ML training dataset
    needs ML validation dataset
    needs model config

ML dataset
    needs split metadata
    needs hit3D partitions
    needs input-view settings from split config

ML dataset partition
    needs one fold split metadata row
    needs one hit3D partition
    optionally needs one feature partition

split metadata
    needs partition metadata
    needs split config

partition metadata
    needs region/particle partition production
```

## Implementation Order

Recommended order:

1. Validate the existing combined partition metadata.
2. Fix or complete partition-level traceability fields if needed.
3. Ensure all samples needed for ML have hit3D partitions. In particular,
   real-data sideband training needs `hit3d_available == true`.
4. Add or update `split_v1.yaml` with training selection, class definitions,
   two-fold cross-fitting, default assignment policy, and SciFi-only input-view
   settings.
5. Implement `split_region_partitions.py`.
6. Add a Snakemake rule for fold-aware split metadata.
7. Implement `convert_hit3d_partition_to_ml.py`.
8. Add a Snakemake rule that converts one fold + partition per job.
9. Add dataset row metadata and combined dataset manifests.
10. Update training rules to use train/validation rows from the new dataset
   catalog.
11. Train one model per fold.
12. Implement prediction ROOT output for fold test partition rows.
13. Combine out-of-fold prediction metadata.
14. Implement merge-back into feature-with-prediction ROOT files.
15. Update evaluation scripts to read merged files and compare feature sets.

## Current Important Caveat

The current partition builder adds identity-related branches, but the
traceability fields should be checked before relying on merge-back by
`event_uid`.

In particular, verify:

```text
event_uid
original_entry
source_file_index
partition_id
local_entry
```

If `event_uid` is not fully populated, use:

```text
partition_id + local_entry
```

as the first practical merge key for prediction-to-feature matching.
