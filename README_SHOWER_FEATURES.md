# Shower features: coworker guide

Develop your features in `feature_plugins/silvia/`, then run
`run_shower_features.sh`. The existing production converter calculates your
features alongside the standard features and writes them into the same ROOT tree.

```text
Existing EOS data + geometry + event selection
    → production converter + your feature plugin
    → features.root (standard features + silvia__* branches)
    → existing ROOT plotting backend
    → per-feature PDFs and CSV summaries
```

## Get the branch from GitHub and start working

Use Bash on lxplus, your own account and your own checkout.

First checkout:

```bash
mkdir -p /afs/cern.ch/work/${USER:0:1}/$USER/worktrees
cd /afs/cern.ch/work/${USER:0:1}/$USER/worktrees
git clone --branch task/shower-features \
  https://github.com/yang-zhibin/snd-ml.git snd-ml-shower-features
cd snd-ml-shower-features
./run_shower_features.sh setup
```

To update an existing checkout of the shared branch, first commit or stash any
unfinished edits, then:

```bash
git status
git fetch origin
git switch task/shower-features
git pull --ff-only origin task/shower-features
```

For development, create your own branch once:

```bash
git switch -c silvia/my-feature
```

Commit your changes on that branch and open a pull request into
`task/shower-features`. To bring later shared changes into your development
branch, use `git fetch origin` followed by `git merge origin/task/shower-features`.

Setup installs a private Snakemake environment. The launcher activates the
appropriate converter and plotting environments for each step. You do not need
to activate Conda manually. See the [setup reference](feature_dev/README.md#environment-setup-reference)
if setup fails.

## Run on the MC neutrino data

Check access using the same sample settings you intend to run:

```bash
./run_shower_features.sh --dataset mc-neutrino --plugin silvia \
  --sample small --files-per-group 6 --events-per-file 200 \
  --run trial01 --check-access
```

Then process the sample:

```bash
./run_shower_features.sh --dataset mc-neutrino --plugin silvia \
  --sample small --files-per-group 6 --events-per-file 200 \
  --run trial01
```

| Option | Meaning | Default |
| --- | --- | --- |
| `--dataset` | Catalogue to read; see choices below | Required |
| `--plugin` | Feature folder name, or an absolute plugin directory | `silvia` |
| `--sample small` | Process a bounded selection of source files/events | `small` |
| `--files-per-group 6` | Up to six source files per dataset group **and geometry** | `1` |
| `--events-per-file 200` | Up to 200 selected events from each file | `20` |
| `--run trial01` | Output run name | Required |
| `--parameters '{"max_missing_channels":1}'` | Settings passed to the plugin | `{}` |
| `--check-access` | Check selected inputs, output access and SNDSW; do not convert | Off |
| `--dry-run` | Preview the Snakemake tasks; do not execute them | Off |

Both sampling parameters are retained. Small runs allow at most 20 files per
group/geometry and 1,000 selected events per file. Files with fewer selected
events contribute fewer events. The number of files is **not** a quota per
plotted CC/NC class: truth classes are assigned from events after reading.
File selection is deterministic; each file contributes its first selected
events. This is a development sample, not an unbiased yield sample.

Use a **new run name after changing code, settings or Git HEAD**, for example
`trial02`. An unchanged command reuses verified completed products.

### Choose another dataset

Change only `--dataset` in the command above:

| Dataset | Input |
| --- | --- |
| `mc-neutrino` | MC neutrino catalogue |
| `testbeam` | Testbeam catalogue |
| `real-data` | Real-data catalogue |
| `mc-muon` | MC muon catalogue |

Paths come from [feature_dev/datasets.yaml](feature_dev/datasets.yaml) and its
referenced catalogues. The workflow reads the existing EOS files directly;
you do not need to copy the dataset or create a separate sample directory.
Your account needs access to the digi, geometry and selection files. Run
`--check-access` for each dataset; missing inputs fail explicitly.

## Develop a feature

| File | What to change or read |
| --- | --- |
| [feature_plugins/silvia/features.py](feature_plugins/silvia/features.py) | Implement feature calculations |
| [feature_plugins/silvia/schema.yaml](feature_plugins/silvia/schema.yaml) | Declare output names, types, units and missing values |
| [feature_plugins/README.md](feature_plugins/README.md) | Interface reference: hits, clusters, parameters and outputs |
| [feature_dev/algorithms/gap_clusters.py](feature_dev/algorithms/gap_clusters.py) | Previous annotated example, with NumPy-style docstrings and worked calculations |
| [feature_dev/SCIFI_DENSITY.md](feature_dev/SCIFI_DENSITY.md) | SciFi density code locations, logic and formulas |

**The previous example is kept.** Silvia's plugin imports its `cluster_hits`
function. Edit the plugin folder for your feature work; changes to the shared
gap-clustering example would also affect plugins that import it.

The production converter calls these functions once per event:

```python
clusters = cluster_hits(hits, parameters)
values = calculate_features(hits, clusters, parameters)
```

`hits` is a list of selected SciFi hit dictionaries. Each has `station`,
`isVertical`, `layer_channel` and `qdc`. Optional positions `x_cm`, `y_cm`,
`z_cm` are in cm; optional `time_ns` is in ns. Check for missing coordinates.
`clusters` is the object returned by your clustering function. `values` must
contain the scalar outputs declared in `schema.yaml`.

For example, the current plugin calculates `leading_charge_fraction` and
cluster densities. Its declared outputs appear as branches such as
`silvia__leading_charge_fraction`. Standard production branches remain present.
See the interface reference for cluster fields, sentinels and validation rules.

## Open the results

Outputs belong to the account that runs the command:

```text
/afs/cern.ch/work/<initial>/<username>/snd-ml-outputs/shower-features/
  <dataset>/<run>/
    sources/<index>/features.root
    sources/<index>/validation.json
    logs/source-<index>.log
    config/                         # saved settings, resolved ranges, coverage
    histograms/1d/<feature>/<selection>/<class>.root
    histograms/2d/<pair>/<selection>/<class>.root
    plots/1d/<feature>/<selection>/plot.pdf
    plots/1d/<feature>/<selection>/summary.csv
    plots/2d/<pair>/<selection>/plot.pdf
    plots/2d/<pair>/<selection>/summary.csv
```

For example, open `plots/1d/count_scifi/selected/plot.pdf`.
Each 2D PDF contains a density page and a profile page. Geometries are combined
in the plots; original geometry information remains in the metadata.
Override the base path with `--output-root /your/writable/path`.

### Change plotting settings

| Setting | File |
| --- | --- |
| Features, selections, bins, axis ranges and 2D pairs | [feature_production/config/histograms.yaml](feature_production/config/histograms.yaml) |
| Colours, markers and labels | [feature_production/config/plotting.yaml](feature_production/config/plotting.yaml) |

Omit a range to use automatic bounds across the selected events. Production
plots use these configuration files; a histogram range in the plugin schema
does not set the production plot range. Edit settings before creating a new run.

For plotting-only changes, see [reuse calculated features](feature_production/README.md#reuse-calculated-features)
to avoid repeating conversion.

## Full data and normalization

Small samples and testbeam plots use unit-area shapes. Full MC and real-data
yield plots require a validated luminosity audit. Source ROOT files retain
exposure metadata for validation; the small sample is not assigned a physical
yield. See the [normalization guide](feature_dev/NORMALIZATION.md) for the required
information, storage locations and calculation.

Full processing uses your saved HTCondor profile. Set it up once with
`./run_shower_features.sh setup --batch`, then follow the
[bounded batch canary](feature_production/README.md#first-run-a-bounded-batch-canary).
Preview full processing before submitting:

```bash
./run_shower_features.sh --dataset mc-neutrino --plugin silvia \
  --sample full --normalization shape --run full01 --dry-run
```

See [full-dataset execution](feature_production/README.md#batch-setup-and-full-dataset) for batch
requirements. The small-sample file/event caps do not apply to `--sample full`.

## Further reference

- [2024 testbeam catalogue](feature_dev/catalogues/README.md): 100 GeV electrons now use run100916; no separate cutflow is required.

- [SciFi clock-cycle selection](feature_dev/SCIFI_CLOCK_CYCLE.md): timing windows, peak calculation, code locations and commands.

- [Data formats and structure](feature_dev/DATA_FORMAT.md): raw inputs, MC truth,
  hit/cluster fields, event features, Hit3D and metadata.

- [Production reference](feature_production/README.md): complete output layout,
  plot settings, source reuse, batch execution and luminosity metadata.
- [Standalone development guide](feature_dev/README.md): the retained
  `run_feature_dev.sh` interface, including the synthetic example.
- [Previous standalone walkthrough](README_SHOWER_FEATURES_STANDALONE.md):
  preserved detailed examples and historical dataset/output information.
- [Validation record](feature_dev/TEST_RESULTS.md): tests and actual MC runs.
