# Data-Driven 3D-CNN Diagnostics and Seven-Class Training Plan

## Scope

This plan covers two additions to the data-driven 3D-CNN workflow:

1. Produce population, fold-balance, and muon-DIS density-sampling diagnostics
   after `data_driven_3d_cnn_dataset_v1` completes.
2. Train a seven-class classifier that excludes `CC_nutau` while preserving the
   permanent eight-class dataset schema and provenance.

Diagnostics must use the natural event distribution. Training resampling must
never be presented as a physical population or used for physics yields.

## Implemented files

```text
evaluation_data_driven_3d_cnn/
|-- plot_population_and_folds.py
|-- plot_muondis_density_sampling.py
|-- build_dataset_report.py
`-- plotting_common.py

test/
|-- test_data_driven_population_plots.py
`-- test_data_driven_density_plots.py

snakemake/rules/
`-- data_driven_3d_cnn_diagnostics.smk

train_3d/
|-- configs/classifier_sparse_cnn_v1.yaml
|-- train_data_driven_classifier.py
`-- src/train_3d/
    |-- classification_module.py
    `-- data/
        |-- classification_contract.py
        `-- data_driven_classification_datamodule.py
```

Suggested outputs:

```text
snakemake/metadata/ml_dataset/data_driven_3d_cnn_v1/diagnostics/
|-- dataset_report.md
|-- population_table.csv
|-- population_table.md
|-- fold_balance_metrics.csv
|-- population_by_class_and_fold.png
|-- class_fraction_by_fold.png
|-- fold_balance_ratio.png
|-- domain_composition.png
|-- source_groups_by_fold.png
|-- muondis_density_table.csv
|-- muondis_density_table.md
|-- muondis_density_counts.png
|-- muondis_natural_vs_uniform.png
|-- muondis_sampling_multiplier.png
`-- muondis_density_by_fold.png
```

CSV files remain machine-readable products. Markdown tables and a combined
report provide the human-readable view. PDF copies of plots can also be
produced for presentations.

## 1. Population and fold-balance diagnostics

### Inputs and counting rule

Use:

```text
dataset_catalog_five_fold.csv
five_fold_summary.csv
validation_report.json
data_driven_3d_cnn_v1.yaml
```

Use `dataset_catalog_five_fold.csv` as the primary source because every logical
ROOT dataset appears once. Use `five_fold_summary.csv` as an aggregate
cross-check.

Do not derive natural populations by summing
`five_fold_train_validation_views.csv`: each dataset row appears in all five CV
views, so a naive sum counts the population five times.

### Seven-class order

Use this order consistently in plots, training, evaluation, and confusion
matrices:

```python
CLASS_ORDER = [
    "CC_numu",
    "CC_nue",
    "NC",
    "neutral_kaon",
    "neutron",
    "passing_muon",
    "muon_DIS",
]
```

Report `CC_nutau` separately as a reserved class excluded from seven-class
training. Its absence is expected. If it unexpectedly contains events, issue a
prominent warning.

### Population aggregation

For every `fold x class_name x domain` combination, calculate:

- natural events;
- logical dataset rows;
- original source files;
- independent fold groups;
- luminosity sources;
- deduplicated luminosity.

Reindex onto the complete expected grid and fill missing combinations with
zero so absent class/fold combinations cannot silently disappear.

### Population heatmap

Use classes as rows and folds 0-4 as columns. Annotate exact event counts and
color by `log10(n_events + 1)` because populations may span orders of
magnitude. This shows whether every fold contains every class and whether
absolute populations are comparable.

### Per-class fold fraction

For class `c` and fold `k`, calculate:

```text
f(c,k) = N(c,k) / sum_j N(c,j)
```

The equal-share reference is `1/5 = 0.20`. Show it on grouped bars or a
heatmap. This prevents large classes from hiding imbalances in small classes.

### Balance ratio

Calculate:

```text
R(c,k) = N(c,k) / (N(c) / 5)
```

`R=1` is ideal, `R=1.2` is 20% high, and `R=0.8` is 20% low. Display a
diverging heatmap centered on 1.

### Domain composition

For `passing_muon` and `muon_DIS`, plot stacked data/MC counts per fold and
calculate:

```text
data_fraction = N_data / (N_data + N_mc)
```

The other active classes are currently MC-only. This diagnostic informs the
later choice of training-time data/MC probabilities.

### Independent-source coverage

Plot source-group and source-file counts per class and fold. Event balance is
not enough: many events from one large source provide little independent-source
diversity.

### Numerical balance table

For each class report total events, minimum/maximum fold count,
maximum/minimum ratio, mean, standard deviation, coefficient of variation,
largest deviation from the ideal 20% share, and number of empty folds.

Suggested presentation flags:

```text
OK       maximum relative deviation <= 20%
CHECK    20% < deviation <= 50%
WARNING  deviation > 50% or an empty fold
```

These are diagnostics, not automatic physics-quality decisions.

## 2. Muon-DIS density-sampling diagnostics

Use the complete produced dataset as the authoritative view.

### Retired V7 pilot

The old `muon_DIS_density_sampling_run8329_vs_mc.csv` pilot and its V7 ROOT
inputs are not part of this dataset or its training workflow. Diagnostics must
not depend on them. The complete produced density-index manifest replaces the
pilot for all reported counts, fractions, and sampling multipliers.

### Complete produced dataset

Join `dataset_catalog_five_fold.csv` and `density_index_manifest.csv` on
`dataset_file_uid`. Sum counts for every `domain x fold x density_bin`.

Valid bins are:

```text
[0, 500)
[500, 2000)
[2000, 5000)
[5000, 10000)
[10000, infinity)
```

Show `invalid` separately. Invalid-density events remain in natural validation
but are excluded from uniform training sampling.

For domain `d`, fold `k`, and bin `b`:

```text
p_natural(d,k,b) = N(d,k,b) / sum_b' N(d,k,b')

p_target(d,k,b) = 1/B if the valid bin is populated
                  0   otherwise

M(d,k,b) = p_target(d,k,b) / p_natural(d,k,b)
```

Here `B` is the number of populated valid bins. `M>1` means oversampling,
`M<1` means undersampling, and `M=1` means unchanged. `M` is a training
exposure multiplier, not a physics weight.

### Cross-validation-specific training pools

For the model holding out fold `k`, use only the other four folds:

```text
N_train(k,d,b) = sum over j != k of N(d,j,b)
```

Recalculate populated bins, uniform targets, and multipliers for every held-out
fold. A globally populated bin can become sparse or empty after one fold is
removed.

Produce a heatmap whose rows are `domain x density bin`, columns are held-out
folds, and values are the relative sampling multipliers.

Use these warnings:

```text
multiplier > 5     notable repetition
multiplier > 10    strong repetition risk
multiplier > 50    severe repetition/overfitting risk
zero training bin  unavailable for that model/domain
```

Once epoch size and class/domain probabilities are configured, also estimate
draws and average event reuse per bin. Until then, report relative multipliers.

## 3. Combined report

`build_dataset_report.py` should combine validation status, classes present and
excluded, population and fold metrics, domain composition, source coverage,
pilot and full density results, plots, and automatically generated warnings.

Example header:

```markdown
# Data-driven 3D-CNN dataset diagnostics

Validation: PASS
Training classes: 7
Excluded class: CC_nutau
Folds: 5

## Warnings
- MC muon-DIS density_ge5000 has a sampling multiplier of 74.8.
- Fold 3 contains only two independent neutron source groups.
- Fold 1 has 31% fewer passing-muon data events than the ideal share.
```

## 4. Snakemake integration

Add a separate target:

```text
data_driven_3d_cnn_diagnostics_v1
```

It depends on the validation report, folded catalog, fold summary, and
density-index manifest. Requiring
`validation_report.json` prevents diagnostics from running on an invalid or
incomplete dataset.

```text
data_driven_3d_cnn_dataset_v1
              |
              v
validation_report.json with valid=true
              |
              v
data_driven_3d_cnn_diagnostics_v1
              |
              v
tables + plots + dataset_report.md
```

Keep diagnostics separate from expensive ROOT production so plotting failures
do not make completed production appear to have failed.

## 5. Seven-class training contract

Keep permanent eight-class dataset IDs unchanged. Do not rewrite ROOT files or
natural catalogs just to exclude `CC_nutau`.

Use this contiguous training map:

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

Permanent source IDs are `0,1,3,4,5,6,7`, which cannot be used directly with
seven output logits because source ID 7 is outside `[0,6]`. Remap from
`class_name`, not from the stored ID. Preserve both identities:

```python
{
    "label": 6,
    "source_class_id": 7,
    "class_name": "muon_DIS",
}
```

Classifier configuration:

```yaml
task: classification
num_classes: 7
class_names:
  - CC_numu
  - CC_nue
  - NC
  - neutral_kaon
  - neutron
  - passing_muon
  - muon_DIS
excluded_classes:
  - CC_nutau
```

Fail clearly for a class outside `class_names` unless it is explicitly in
`excluded_classes`. With current sources, `CC_nutau` should have zero events.
Use seven output logits, cross-entropy loss, and the frozen class order for all
metrics.

## 6. Implementation sequence

1. Add shared class order, density labels, table formatting, and plot style.
2. Implement population aggregation and CSV/Markdown tables.
3. Add population, fold-fraction, balance-ratio, domain, and source plots.
4. Implement complete-dataset density aggregation.
5. Add CV-specific training-pool density multipliers.
6. Generate the combined Markdown report.
7. Test missing tau, missing class/fold combinations, domain splits, empty
   density bins, invalid densities, severe oversampling, and prevention of
   five-times overcounting from fold views.
8. Add the independent Snakemake diagnostics target.
9. Add the contiguous seven-class map to the ROOT-backed dataset.
10. Build the seven-output sparse classifier and five-fold training entrypoint.

## 7. Verification and test plan

Testing is split into fast numerical contracts, synthetic end-to-end tests,
ROOT integration, workflow validation, and real-dataset acceptance.

### Fast numerical contracts

- Reindex population summaries onto every `class x fold x domain` cell and
  confirm missing cells are explicit zeroes.
- Verify class totals equal the sum of the five natural folds and are never
  derived by summing the five repeated CV views.
- Check balanced, moderately imbalanced, severely imbalanced, and empty-fold
  status thresholds.
- Check luminosity, source-file, and fold-group deduplication.
- Enforce density-count closure, natural-fraction closure, populated-bin
  uniform targets, invalid-density exclusion, and held-out-fold recalculation.
- Require labels `0..6` in the frozen order, map permanent `muon_DIS=7` to
  training label 6, reject `CC_nutau`, and reject unknown classes.
- Require logits with shape `[batch, 7]` and reject labels outside `[0, 6]`.

### Synthetic integration

Build a small seven-class, five-fold catalog with data/MC muon rows, all five
density bins, and invalid events. Generate every CSV, Markdown table, and PNG
through `build_dataset_report.py`. A failed validation JSON must stop report
generation before plots are produced.

### ROOT-backed integration

- Read exact physical entries and variable-length hit vectors.
- Voxelize and collate sparse batches while preserving source identity.
- Preserve both contiguous `label` and permanent `source_class_id`.
- Keep sampled training reproducible by fold and epoch.
- Iterate held-out validation naturally and exhaustively.

### Snakemake and real-data acceptance

- Dry-run `data_driven_3d_cnn_diagnostics_v1` in the full workflow.
- Require `validation_report.json` with `valid=true`.
- Require one density-manifest row per catalogued `muon_DIS` logical file.
- Require every dataset file to be validation once and training four times.
- Require seven active classes, zero `CC_nutau` events, five folds, density
  closure, and no source group shared across folds.
- Generate all diagnostics on the final real metadata and review automatic
  balance and oversampling warnings before training.

### Training smoke gate

Before a long run, freeze coordinate ranges, run one short disabled-W&B fold,
verify finite forward/backward loss, save/reload a checkpoint, and overfit a
20--100 event subset. Then verify one offline W&B run before enabling online
logging in project `snd-ml-data-driven-3d`.

## 8. Implementation status

Implemented and covered by focused tests:

- natural population aggregation and balance metrics;
- population, fraction, ratio, domain, and source-group plots;
- full-dataset and per-fold muon-DIS density tables and plots;
- held-out-fold training-pool multipliers and repetition warnings;
- validation-gated combined Markdown report;
- independent `data_driven_3d_cnn_diagnostics_v1` target without V7 inputs;
- contiguous seven-class label contract with permanent source IDs preserved;
- ROOT-streamed fold datamodule, seven-output sparse classifier, fold training
  entrypoint, and W&B project configuration.

Pending real-data gates:

- completion of all density-index jobs and `validation_report.json`;
- generation and review of final real-data diagnostics;
- freezing validated coordinate ranges in the sparse-CNN config;
- CPU/GPU forward-backward and overfit smoke tests in an environment with
  MinkowskiEngine before full five-fold training.

### Muon-DIS feature sampling report

Implemented as the independent Snakemake target
`data_driven_3d_cnn_muondis_feature_sampling_v1`. For each configured
fold/epoch, it:

1. reproduces the exact deterministic hierarchical training draw from the
   GravNet training configuration;
2. builds one ROOT histogram shard per non-held-out Muon-DIS source;
3. merges the shards with source, event-count, and sampled-draw closure checks;
4. writes a PDF with natural and sampled raw event counts in separate
   side-by-side pads for all 46 coverage features.

The natural panel contains every valid-density Muon-DIS event in the four
training folds with unit weight. The sampled panel counts the exact epoch-zero
draws, including repeated-event multiplicity. The report is therefore a
training-sampler diagnostic, not a unit-area shape comparison. Its default
output is
`snakemake/metadata/ml_dataset/data_driven_3d_cnn_v1/diagnostics/muondis_feature_sampling/fold_0/epoch_0/natural_vs_sampled.pdf`.

## 9. Central rules

- use natural data for population reporting;
- derive each model's sampling policy from its four training folds only;
- keep validation natural and complete;
- preserve permanent source IDs while using contiguous training labels;
- keep training sampling separate from luminosity and physics normalization.
