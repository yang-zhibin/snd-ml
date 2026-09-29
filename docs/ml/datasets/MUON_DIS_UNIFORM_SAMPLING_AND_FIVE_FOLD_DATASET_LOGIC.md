# Muon-DIS Uniform Sampling and Five-Fold Dataset Logic

## Purpose

This note defines how to construct the muon-DIS training sample and the five
cross-validation datasets for the data-driven 3D-hit CNN.

The physical classes are:

1. `CC_numu`
2. `CC_nue`
3. `CC_nutau` when a tau-neutrino production becomes available
4. `NC`
5. `neutral_kaon`
6. `neutron`
7. `passing_muon`
8. `muon_DIS`

Real-data events with `count_veto == 0` form the sealed signal-region
partition. They must not enter training, validation, sampling audits, plots,
event counts, or yield calculations before explicit unblinding.

## 1. Natural dataset first

Keep one natural, unresampled catalog of all allowed events. Each row records:

- physical ROOT file and tree;
- class and domain (`data` or `mc`);
- original source and fold-group identifiers;
- number of events;
- original luminosity-source identifier and file luminosity;
- for muon-DIS, the SciFi-density index file.

Training balance is not written into this catalog. The same natural catalog is
used to construct all five cross-validation views.

The real-data proxy split is:

```text
count_veto == 0                         -> sealed signal region
count_veto > 0 and coverage_loose       -> passing_muon
count_veto > 0 and not coverage_loose   -> muon_DIS
```

Only the final two partitions can enter the machine-learning catalog.

## 2. Muon-DIS density pools

Use `density_sndsw_scifi` and the following half-open bins:

```text
bin 0: [0, 500)
bin 1: [500, 2000)
bin 2: [2000, 5000)
bin 3: [5000, 10000)
bin 4: [10000, infinity)
```

For every muon-DIS ROOT file, store the physical ROOT entry numbers belonging
to each bin in a compact index file. Non-finite or negative densities are
recorded as invalid for closure checks. They are excluded from uniform training
sampling, but remain in natural held-out validation.

Density indices are created before training and separately for every physical
source file. This avoids repeatedly scanning the ROOT density branch during
each epoch.

## 3. Uniform-bin muon-DIS training sampler

For one training draw of class `muon_DIS`:

1. Choose the representation domain (`data` or `mc`) using the explicitly
   configured domain probabilities. If no probabilities have been decided,
   use the natural valid-event fractions in the training folds.
2. Within the chosen domain, find the density bins that contain at least one
   event in the current four training folds.
3. Choose one populated density bin uniformly. If four bins are populated,
   each has probability `1/4`; an empty bin is not selected.
4. Choose one event uniformly from all eligible events in the chosen
   domain/bin, pooled across all source files.
5. Load that event using its physical ROOT entry number.

Sampling is with replacement. A rare high-density bin can therefore appear as
often as a large low-density bin without exhausting or duplicating physical
files on disk.

Pooling all entries before choosing the event is important. It makes the
probability of selecting a file proportional to the number of eligible events
in that file. Choosing a file uniformly first would incorrectly over-represent
small files.

The uniform rule applies only to muon-DIS training events. It must not be
implemented simultaneously as both a uniform sampler and an activity-dependent
loss weight, because that would apply the same correction twice.

## 4. Sampling all training classes

The complete training hierarchy is:

```text
choose class
    -> choose data/MC domain available for that class
        -> muon_DIS: choose populated density bin uniformly -> choose event
        -> other classes: choose event naturally within class/domain
```

Class probabilities and data/MC probabilities are explicit training
hyperparameters. They should be fixed only after auditing the available event
populations. Reasonable comparisons include natural class frequencies and
uniform class frequencies, but the dataset preparation must not silently choose
one of them.

The number of draws in one epoch is also a training choice. It does not change
the natural catalog or the physics normalization.

Use a deterministic random seed derived from the experiment seed, fold number,
and epoch number. This makes every sampled epoch reproducible while allowing a
new sample in the next epoch.

## 5. Assigning five source-safe folds

Assign folds once, before any training sampling:

- Group real data by complete run.
- Group MC by original source file or production source identifier.
- Keep passing-muon, muon-DIS, and the sealed partition derived from one data
  source in the same group.
- Keep all neutrino flavor/NC partitions derived from one original neutrino
  file in the same group.

Assign each complete group to exactly one of folds `0, 1, 2, 3, 4`. The
assignment should approximately balance class counts, domains, muon-DIS density
populations, neutral-hadron energy bins, productions, and luminosity, while
treating group integrity as a hard constraint.

This prevents events from the same run or original MC source appearing in both
training and validation.

## 6. Constructing the five train/validation datasets

For cross-validation view `k`:

```text
validation = fold k
training   = the other four folds
```

This produces:

| Model | Training folds | Validation fold |
|---|---|---|
| 0 | 1, 2, 3, 4 | 0 |
| 1 | 0, 2, 3, 4 | 1 |
| 2 | 0, 1, 3, 4 | 2 |
| 3 | 0, 1, 2, 4 | 3 |
| 4 | 0, 1, 2, 3 | 4 |

The training side uses the hierarchical sampler described above. The
validation side is always natural:

- every held-out event appears exactly once;
- no uniform density sampling;
- no class or domain resampling;
- invalid-density muon-DIS events remain present;
- no event from the other four folds is included.

Training sampling must construct its density pools using only the four training
folds. Counts or distributions from the validation fold must not influence the
sampling decisions.

## 7. Evaluation and physical yields

After training the five models, predict only their held-out folds. Concatenate
the five held-out prediction tables. This is the out-of-fold result: every
allowed event is evaluated once by a model that did not train on its source
group.

Never use the resampled training frequency to calculate a physical yield.
Yields come from the natural validation or combined out-of-fold events.

For luminosity:

- `lumi_per_file` describes the complete original source file;
- sum it once per unique `lumi_source_uid`, not once per event or derived class;
- pass/fail partitions and CC/NC partitions inherit the same source luminosity;
- MC event normalization is applied per explicitly defined physics production
  group using its total equivalent luminosity;
- data proxy and MC proxy yields are alternative domain representations and
  must not be added as independent backgrounds without a separate physics
  model.

The sealed `count_veto == 0` real-data partition remains outside this entire
procedure until explicit unblinding.

## 8. Required checks

Before training each fold, require:

- no fold-group overlap between training and validation;
- no sealed signal-region file or event in either split;
- all catalogued ROOT files contain only their declared class;
- density-index counts close to the corresponding muon-DIS file counts;
- sampled muon-DIS bins are approximately uniform over populated bins;
- sampled files within a bin follow their eligible event counts;
- validation contains each held-out event exactly once;
- luminosity is unique and consistent for every `lumi_source_uid`.

## Implementation references

- Dataset configuration:
  `snakemake/metadata/configs/ml_dataset/data_driven_3d_cnn_v1.yaml`
- Fold assignment: `convertData/assign_3d_cnn_folds.py`
- Fold views: `convertData/build_3d_cnn_fold_views.py`
- Density indices: `convertData/build_muondis_density_indices.py`
- Training/validation sampler:
  `train_3d/src/train_3d/data/data_driven_fold_dataset.py`
- ROOT-backed sparse CNN datasets and fold factory:
  `train_3d/src/train_3d/data/data_driven_root_dataset.py`
- Final validation: `convertData/validate_3d_cnn_dataset.py`
