# Data-Driven 3D-CNN Dataset Implementation

## Scope

This implementation prepares the natural, source-traceable dataset for the
eight-class 3D-hit classifier. It does not launch the large production and does
not choose final class or data/MC training weights.

The configured classes are:

1. `CC_numu`
2. `CC_nue`
3. `CC_nutau`
4. `NC`
5. `neutral_kaon`
6. `neutron`
7. `passing_muon`
8. `muon_DIS`

There is currently no configured tau-neutrino source, so `CC_nutau` is defined
but marked non-required. Adding its metadata source later does not require a
change to the fold or sampling logic.

## Implemented data flow

```text
existing metadata CSV files
        |
        v
50 existing feature/Hit3D files selected across each of 22 data runs
        |
        v
source_catalog.csv  (one row per original file)
        |
        +-- neutrino feature/Hit3D pairs
        |     `-- physical CC_numu / CC_nue / CC_nutau / NC pairs
        +-- existing kaon / neutron Hit3D products
        |
        `-- data skim + MC muon
              |
              v
        exact V7 coverage_loose expression
              |
              +-- data count_veto == 0 -> sealed signal-region pair
              +-- pass -> passing_muon
              `-- fail -> muon_DIS
        |
        v
dataset_catalog.csv  (natural counts and original luminosity lineage)
        |
        v
dataset_catalog_five_fold.csv  (whole source groups assigned to one fold)
        |
        +-- density indices for every muon_DIS file
        |
        v
five_fold_train_validation_views.csv
        |
        +-- training muon_DIS: uniform populated density bins
        `-- validation: complete held-out fold, natural distribution
```

Data files are grouped by complete run. MC files are grouped by original source
file. Pass/fail samples from one muon source therefore cannot cross folds, and
CC/NC events derived from one neutrino source cannot cross folds.

The real-data input is frozen to 50 files per run (1,100 files total). Files
are sorted by source-file number, divided into 50 position strata within each
run, and one file is chosen from each stratum using a stable seeded hash. This
covers the complete time/file range of every run and is reproducible if the
input metadata row order changes. Only named feature/Hit3D pairs that exist are
eligible; the subset builder deliberately does not scan ROOT-tree integrity.

## Exact loose-selection behavior

`convertData/passing_muon_selection.py` constructs the selection directly from
`passing_muon_binary_region_study_v7.yaml`. The production code applies that
same expression to data and MC. For real data, `count_veto == 0` is the blinded
signal region. It is written to a separate sealed feature/Hit3D pair, but never
inserted into the dataset catalog, folds, sampler, or result summaries. The
proxy training parent is exactly `count_veto > 0`. For MC only, the existing V7
muon-parent baseline is applied first. Within the corresponding parent:

```text
coverage_loose       -> passing_muon
not coverage_loose   -> muon_DIS
```

Feature and Hit3D trees must have equal entry counts and matching `runId`,
`eventId`, and `eventIndex`. Each output event records its original entry,
source identifier, class, domain, selection result, selection fingerprint, and
SciFi density. Every input source has audited parent and pass/fail closure.

This split deliberately runs single-threaded. Independent ROOT snapshots made
with implicit multithreading are not guaranteed to preserve the same row order
in the feature and Hit3D outputs. After all four snapshots are written, the
splitter scans each feature/Hit3D output pair and verifies both pairwise event
identity and the `ml_source_entry` link back to the original Hit3D tree.

## Density sampling and five folds

The complete natural dataset is stored once. It is not physically resampled
when folds are created.

For cross-validation view `k`:

- Fold `k` is validation and always uses every event at natural frequency.
- The other four folds are training.
- Only training rows of class `muon_DIS` use uniform populated-density-bin
  sampling, with replacement.
- All other training classes remain marked `natural`; class balancing is a
  separate training decision.

Compact `.npz` files contain ROOT entry numbers for each SciFi-density bin. The
sampler first chooses a populated bin uniformly and then chooses uniformly from
all events in that bin. File choice is proportional to the number of eligible
events, so small files are not accidentally over-represented.
Non-finite or below-range density values are retained in the audit/index closure
but excluded from uniform training sampling.

The configured SciFi-density bins are `[0,500)`, `[500,2000)`, `[2000,5000)`,
`[5000,10000)`, and `[10000,infinity)`.

Neutrino inputs are physically split using truth `pdgCode`: CC numu uses
`+/-14`, CC nue uses `+/-12`, CC nutau uses `+/-16`, and NC combines
`+/-112`, `+/-114`, and `+/-116`. These partitions are mutually exclusive and
must close to the input. The current 2024 metadata has electron- and
muon-neutrino productions but no tau-neutrino production, so CC nutau remains
defined and optional until that source is added.

`HierarchicalTrainingSampler` implements class -> domain -> event sampling. For
`muon_DIS`, the last step becomes populated density bin -> event. Class and
domain probabilities remain runtime inputs so no balance choice is hidden in
dataset preparation. `NaturalValidationEvents` visits every held-out event
exactly once and performs no class, domain, or density balancing.

Optional data/MC domain probabilities are supported by
`UniformDensityBinSampler`, but none are frozen in the dataset configuration.
This is intentional: domain and class balance will be chosen after inspecting
the produced event populations.

## Luminosity rule

`lumi_per_file` is copied from each original metadata row and is associated with
`lumi_source_uid`. It is never converted to an event weight during dataset
preparation.

When calculating a physical yield or total integrated luminosity, deduplicate
by `lumi_source_uid` before summing. In particular, do not add the luminosity a
second time because one source contributes to both pass/fail or CC/NC logical
rows. Training resampling does not change physical normalization.

## Important files

- Configuration: `snakemake/metadata/configs/ml_dataset/data_driven_3d_cnn_v1.yaml`
- Workflow: `snakemake/rules/data_driven_3d_cnn_dataset.smk`
- Source discovery: `convertData/build_3d_cnn_source_catalog.py`
- V7 splitting: `convertData/build_muon_loose_partitions.py`
- Neutrino splitting: `convertData/build_neutrino_class_partitions.py`
- Natural catalog: `convertData/build_3d_cnn_dataset_catalog.py`
- Fold assignment: `convertData/assign_3d_cnn_folds.py`
- Density indices: `convertData/build_muondis_density_indices.py`
- V7 run-8329/MC pilot: `convertData/summarize_muondis_density_sampling.py`
- Train/validation views: `convertData/build_3d_cnn_fold_views.py`
- Final checks: `convertData/validate_3d_cnn_dataset.py`
- Runtime sampler: `train_3d/src/train_3d/data/muondis_density_sampler.py`
- Full fold dataset/sampler: `train_3d/src/train_3d/data/data_driven_fold_dataset.py`
- ROOT-to-sparse-CNN adapter: `train_3d/src/train_3d/data/data_driven_root_dataset.py`

## Using one fold in training

Load `fold_views.csv` and call `build_root_fold_datasets(...)`. It returns a
sampled `train` dataset and an untouched `validation` dataset. Pass
`classification_sparse_collate_fn` to the PyTorch `DataLoader`. The coordinate
ranges, voxel size, epoch size, and any class/domain probabilities are explicit
training hyperparameters; the dataset metadata does not silently choose them.

Before constructing the training loader for each epoch, call:

```python
datasets["train"].set_epoch(epoch)
```

This regenerates a deterministic sample using `(seed, cv_fold, epoch)`. The
same fold, seed, and epoch always produce the same physical event references,
independent of earlier calls. Do not call `set_epoch` on validation. Keep
`persistent_workers=False` (the PyTorch default), or recreate workers after
`set_epoch`, so worker copies receive the new epoch references.

## Running it

The two physical partition stages can be submitted independently, in order:

```bash
snakemake --snakefile snakemake/snakefile \
  --profile snakemake/.config/snakemake/htcondor \
  data_driven_3d_cnn_neutrino_partitions_v1

snakemake --snakefile snakemake/snakefile \
  --profile snakemake/.config/snakemake/htcondor \
  data_driven_3d_cnn_muon_partitions_v1
```

The second target creates passing-muon, muon-DIS, and sealed signal-region
feature/Hit3D pairs for data sources. The sealed partition is not a dependency
of the training catalog.

First inspect the source discovery stage locally:

```bash
cd /afs/cern.ch/work/z/zhibin/snd-ml
source /afs/cern.ch/user/z/zhibin/env_lcg.sh
snakemake -n --cores 1 --snakefile snakemake/snakefile \
  build_data_driven_3d_cnn_source_catalog_v1
```

Dry-run the complete workflow before submitting it:

```bash
snakemake -n --cores 1 --snakefile snakemake/snakefile \
  data_driven_3d_cnn_dataset_v1
```

After reviewing the job count and paths, submit with the repository's HTCondor
profile:

```bash
snakemake --snakefile snakemake/snakefile \
  --profile snakemake/.config/snakemake/htcondor \
  data_driven_3d_cnn_dataset_v1
```

The final target is complete only after `validation_report.json` is written with
`"valid": true`. The validator rejects missing classes, leakage of a source
group across folds, inconsistent luminosity lineage, entry-identity failures,
pass/fail non-closure, incomplete density indices, or resampled validation rows.
It also writes `muon_DIS_density_sampling_run8329_vs_mc.csv`, showing each
pilot cohort's natural bin fractions and the multipliers implied by uniform
training sampling. The pilot is read from the completed V7 run-8329 and pooled
MC cumulative-density histograms; run 8329 is not part of the ordinary 2024
skim source catalog.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s test -p 'test_*3d*py' -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s test -p 'test_muon_loose_partitions.py' -v
```

The tests cover the exact V7 expression, density boundaries, aligned ROOT
pass/fail outputs and original-entry lineage, rejection of unequal trees and
unsafe multithreading, source-grouped deterministic folds, natural validation,
uniform-bin training sampling, optional domain probabilities, and final
metadata invariants. A synthetic ROOT integration test also checks physical
entry lookup, sparse voxel features, labels, event identity, batching, and the
fold-level training/validation factory.
