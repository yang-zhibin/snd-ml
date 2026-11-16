# Develop shower features

For the branch overview, physics datasets, feature definitions, and normalization,
start with [the shower-feature branch guide](../README_SHOWER_FEATURES.md).

New collaborator? Follow the [GitHub clone, update, first-run and contribution instructions](../README_SHOWER_FEATURES.md#git-and-environment-setup) first.

The coworker edits an algorithm and its schema, then runs one command to calculate
features, validate them, make histograms, and open a standalone HTML report.
The existing production clustering is always calculated on the **same selected
hits** as the candidate. Candidate code receives no sample labels or MC truth.

## Run neutrino MC, testbeam, and real data independently

The command now accepts `--dataset mc-neutrino`, `testbeam`, `real-data`,
`synthetic`, and optional `mc-muon`. `--dataset all` runs the first three as
independent Snakemake DAGs and writes an index linking their reports. It excludes
muon MC and the synthetic fixture.

```bash
./run_feature_dev.sh --algorithm gap_clusters \
  --dataset mc-neutrino --sample small --run trial01
./run_feature_dev.sh --algorithm gap_clusters \
  --dataset testbeam --sample small --run trial01
./run_feature_dev.sh --algorithm gap_clusters \
  --dataset real-data --sample small --run trial01

# Run all three, reusing any identical completed dataset runs:
./run_feature_dev.sh --algorithm gap_clusters \
  --dataset all --sample small --run trial01
```

These commands now read the existing source catalogues and automatically prepare
small selected-hit samples from their EOS inputs on first use. You do not need
to supply Zhibin's sample directory. You need read access to the shared catalogue,
digitized ROOT files, geometry and cutflows, plus the established SNDSW build.
Missing inputs or missing physics groups fail explicitly; there is no synthetic
fallback. For an example requiring no physics-data access, use:

```bash
./run_feature_dev.sh --algorithm gap_clusters \
  --dataset synthetic --sample small --run dataset-demo-v1
```

Existing commands without `--dataset` retain their original output layout and
`--sample small` synthetic alias. Framework changes invalidate prior run hashes,
so use a new run name when moving from the earlier implementation.

### Output and sample locations

The default output root is your AFS work campaign's `feature_dev` directory:

```text
feature_dev/
  .samples/<dataset>/<source-signature>/
    plan.json                      selected sources, fingerprints and sampling
    exports/<number>/              selected hits, converter ROOT and parity receipt
    logs/export-<number>.log        converter command and runtime diagnostics
    small.jsonl                    validated prepared sample (+ manifest)
    registry.json                  reusable prepared-sample registry
    complete.json                  checksums required for reuse
  mc-neutrino/trial01/plot/report.html
  testbeam/trial01/plot/report.html
  real-data/trial01/plot/report.html
  all/trial01/index.html
  synthetic/dataset-demo-v1/plot/report.html
```

Each dataset has its own `.cache` and execution lock. Changing testbeam inputs or
its configuration does not invalidate the neutrino or real-data calculations.
The same run name is allowed in different dataset directories. Algorithm changes
still require a new run name within each dataset.

`--output-root` changes your cache and result base. The default is
`/afs/cern.ch/work/<initial>/<username>/snd-ml-outputs/unified_shower_features_v1/feature_dev`
when your AFS work directory exists, otherwise `./feature-dev-output`.
No source directory is written. `--dataset-config` selects another YAML registry;
relative catalogue paths resolve against that registry's directory.

### Automatic small samples from existing EOS sources

The shipped `datasets.yaml` points to the existing campaign metadata under
`/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/metadata`.
Those CSVs supply the authoritative EOS input, geometry and selection paths.

| Dataset | Source strata and default file cap | Event grouping |
| --- | --- | --- |
| `mc-neutrino` | 4 files per flavour subfolder | CC/NC and flavour from each event's existing converter `pdgCode` encoding |
| `testbeam` | 2 files per data/MC family, particle, energy and geometry; includes H4 2024 and H8 2023 | Preserve family, particle, energy and geometry |
| `real-data` | 2 files per saved no-veto/veto-10% selection, preferring distinct runs | Use existing indexed cutflows; no new veto sampling |
| `mc-muon` | 2 files per source family and geometry | Preserve upward-going muon source family |

Files are ordered by a stable hash with seed `feature-dev-v1`. Each selected
file contributes at most its first 100 selected entries. The final sample keeps
up to 100 events per group/geometry (25 for testbeam), sorted by source/event identity hash.
This is a bounded development subset, **not an unbiased sample of the campaign**.
Missing CC/NC coverage is an error; increase the file/event caps deliberately.
The converter may still need to inspect the chosen file's cutflow to resolve its
selection. The input-file cap prevents scanning the entire raw campaign.

Preview without converting or creating result directories:

```bash
./run_feature_dev.sh --dataset all --sample small --algorithm gap_clusters \
  --run trial01 --dry-run
```

For an uncached sample, this previews the preparation DAG; analysis is planned
once the sample exists. `all` preflights every dataset's selected source paths
before starting any preparation. A failure leaves completed earlier datasets
reusable and does not create a new combined index.

Preparation alone, or a larger bounded sample:

```bash
./run_feature_dev.sh --dataset mc-neutrino --sample small --run preparation \
  --prepare-only
./run_feature_dev.sh --dataset mc-neutrino --sample small --run trial02 \
  --files-per-group 6 --events-per-file 200 --algorithm gap_clusters
```

Automatic assembly reads one guarded source at a time and retains only the
lowest identity hashes per group/geometry. Each source and the retained/final
sample must fit within 256 MiB; a larger aggregate of bounded source exports is
allowed. The testbeam retained cap is 25 because the default 45 exports together
exceed the in-memory input guard used by the manual/full preparation path.

Hard development limits are 20 files per stratum and 1000 selected entries per
file. Use Snakemake/HTCondor for long preparation: pass
`--prepare-profile /absolute/path/to/your/htcondor-profile` (an existing profile
compatible with your Snakemake installation). The CLI waits for preparation,
then runs analysis locally. Run the controller in a persistent session. Record
the launch command, scheduler job IDs, status command and cache/log paths in
your task record. No HTCondor plugin or profile is installed by this command.
`launch.json`, `.snakemake/log/` and `logs/export-*.log` live inside the sample cache.

SNDSW export and LCG analysis run in separate processes. Every export must pass
entry-by-entry `(runId, eventId, eventIndex)` and baseline scalar comparison
against its freshly produced converter ROOT tree before sample assembly. Scalar
floats use `rtol=1e-6, atol=1e-7`; integer and identity comparisons are exact.
This verifies adapter parity, not scientific validity of a candidate algorithm
or agreement with an older campaign produced with different software.

The source cache key includes catalogue hashes, selected input path/size/mtime,
geometry and cutflow content hashes, sampling and preparation code. Cache reuse
checks output hashes; interrupted exports are retried atomically. Source digi
files are identified by size/mtime rather than hashing entire large ROOT files.
Shared SNDSW builds are environment dependencies, not bundled immutable images;
retain converter logs when qualifying samples. Algorithm-only edits reuse the
prepared hits. Planning still checks source accessibility and fingerprints.

### Optional: reuse an existing prepared sample

`--sample-root` explicitly bypasses automatic EOS preparation. For a sample
prepared by this workflow, use its printed cache directory and registry:

```bash
./run_feature_dev.sh --dataset mc-neutrino --sample small --run shared-trial \
  --dataset-config /shared/cache/registry.json --sample-root /shared/cache \
  --algorithm gap_clusters
```

The shared files must be readable; outputs still go to the caller's own area.
With the shipped registry, manually prepared samples retain the layout
`<sample-root>/<dataset>/small-v1.jsonl`. No AFS permissions are changed.

### Advanced: manually prepare exported samples

First export selected hits from the correct converter/SNDSW setup as described
below. Keep source provenance `sample_type`, a nonempty `source_uid`, and explicit
group and geometry labels. Sample preparation validates these declarations; it
does not infer particle identity from hit patterns or apply new physics cuts.

- **Neutrino MC:** the default registry requires `CC_nue`, `CC_numu`, `NC_nue`, and
  `NC_numu`. Supply correctly classified exports. For mixed neutrino files use
  `--feature-dev-group __neutrino_truth__` to derive per-event labels. A literal
  group name labels the entire export and must not be used to mislabel mixed inputs.
- **Testbeam:** labels should distinguish data/MC, particle and energy; geometry
  remains a separate grouping field. Configure `required_groups` if you want to
  enforce a particular comparison. No fixed set of energies is required by default.
- **Real data:** the default registry requires `real_data_2024_no_veto` and
  `real_data_2024_veto_10pct`, using the appropriate selected exports.
- **Muon MC:** use `mc-muon` separately; it is never treated as neutrino MC.

For example, prepare a fixed small neutrino sample on lxplus (replace input paths
with your correctly labelled exports):

```bash
DEV_ROOT=/afs/cern.ch/work/${USER:0:1}/$USER/snd-ml-outputs/unified_shower_features_v1/feature_dev

./run_feature_dev.sh prepare-sample --dataset mc-neutrino \
  --input /path/to/CC_nue.jsonl --input /path/to/CC_numu.jsonl \
  --input /path/to/NC_nue.jsonl --input /path/to/NC_numu.jsonl \
  --output "$DEV_ROOT/samples/mc-neutrino/small-v1.jsonl" \
  --version small-v1 --max-events-per-group 100
```

Repeat for testbeam and real data with their dataset name and exports. The cap is
**per `(group, geometry)`**, so a large group cannot crowd out a smaller one.
Selection sorts by a SHA-256 hash of source/event identity, making the chosen set
independent of input-file order. It preserves original hits, identities and labels,
records available/selected counts and input manifests, rejects duplicates, and
refuses to overwrite existing output. A smaller group contributes all its events.

To prepare `full-v1.jsonl`, use the same input list with a new output/version and
`--max-events-per-group 0`. Then run with `--sample full --sample-root "$DEV_ROOT/samples"`. Here **full means all
provided selected-hit exports**, not all raw EOS files or the whole physics
catalog. Both preparation and execution enforce a **256 MiB JSONL input limit**:
the present runner loads events into memory. Larger campaigns need a separate
sharded implementation; increasing the limit is not a substitute for that work.

To use a focused subset of groups, copy the registry and edit `required_groups`,
then pass `--dataset-config /path/to/my-datasets.yaml`. Original converter
provenance types must still match the chosen dataset. Synthetic fixture manifests
are rejected for physics dataset registration, including through merged exports.

### Normalization and testing

All dataset reports currently use **unit-normalized in-range bin fractions**.
There is no automatic luminosity weighting or 10% veto-sampling yield correction.
Group labels and geometry are preserved, with baseline/candidate comparisons
within each panel. Exposure-normalized yields remain a separate production
workflow requiring validated exposure metadata.

Before `--dataset all` starts work, it preflights all selected source paths
(or validates all three samples when `--sample-root` is supplied). A failed child prevents creation of a new index;
already completed child runs remain reusable. `--dry-run` creates no result
folders. A checksum change is rejected when reusing a run name.

Code: `feature_dev/sources.py`, `prepare.py`, `PrepareSnakefile` and
`run_export.sh` (automatic source preparation); `feature_dev/datasets.py` (manual preparation and resolution),
`feature_dev/datasets.yaml` (registry), `feature_dev/cli.py` (routing/index), and
`feature_dev/Snakefile` (each dataset's DAG).

In the LCG environment, run:

```bash
python3 -m unittest discover -s feature_dev/tests -v
python3 feature_dev/tests/dataset_integration.py
python3 feature_dev/tests/integration.py
```

Dataset integration tests create explicitly artificial exported hits with source
metadata to test routing; they do not establish real-data physics acceptance.

## Try the supplied example on lxplus

From this checkout:

```bash
./run_feature_dev.sh --algorithm gap_clusters --sample small --run tutorial-gap1
```

`small` is a versioned **synthetic 24-event software fixture**, not testbeam,
collision, or simulated physics data. The example joins channel runs separated
by one missing channel and adds `charge_fragmentation_v1`:

```
1 - highest_cluster_QDC / total_event_QDC
```

Zero charge gives the schema's missing value, `-999`. Missing channels add no
charge or hits. This is an illustrative hypothesis, not a qualified algorithm.
The example retains the existing cluster summaries; its QDC roughness uses
consecutive observed channels, including across the permitted gap.

Try zero missing channels to reproduce baseline clustering:

```bash
./run_feature_dev.sh --algorithm gap_clusters --sample small \
  --parameters '{"max_missing_channels":0}' --run tutorial-gap0
```

Outputs default to your own AFS work:

```
/afs/cern.ch/work/<initial>/<user>/snd-ml-outputs/unified_shower_features_v1/feature_dev/<run>/
  manifest.json         resolved schemas, parameters, source hashes, input provenance
  plan.json             executable stage plan
  code/                 source snapshots
  calculate/            features.json and features.root (baseline/candidate trees)
  validate/             validation.json
  histogram/            histograms.json
  plot/                 PNGs and standalone report.html
```

The four stage directories link to a shared `.cache` under the output root.
Keep that cache when retaining runs. Copy `plot/report.html` to share one report;
it embeds its images. To archive a whole run, copy it with symlinks dereferenced.
Use `--output-root /your/writable/directory` for another location. Existing EOS
products are not moved or read by this example.

Preview jobs with `--dry-run`; it creates no result directories. Repeating an
identical command reuses results. Use a **new `--run` after editing code,
parameters, or schema**; a conflicting run name fails rather than overwriting.

## The algorithm interface

See `algorithms/baseline.py` and `algorithms/gap_clusters.py`:

```python
def cluster_hits(hits, parameters):
    # Return any clustering structure that calculate_features understands.
    return clusters

def calculate_features(hits, clusters, parameters):
    # Exact schema keys, ordinary Python int/float values, no NaN/inf.
    return {"my_feature": value}

# Optional: raise ValueError when an algorithm-specific invariant fails.
def validate_algorithm(hits, clusters, features):
    ...
```

Every hit contains `station` (1–5), `isVertical` (bool), `layer_channel`
(0–1535), and nonnegative `qdc` (detector QDC). Optional fields are `x_cm`,
`y_cm`, `z_cm`, and `time_ns`. Station/orientation/channel must be unique within
an event. Do not mutate the hits. The functions must be deterministic; put all
adjustable settings in `parameters`. A feature-only change can reuse baseline
`cluster_hits`. A clustering change may return a different structure.

Copy the example algorithm to `feature_dev/algorithms/my_features.py` and its
schema to `feature_dev/schemas/my_features.yaml`, then run:

```bash
./run_feature_dev.sh --algorithm feature_dev/algorithms/my_features.py \
  --schema feature_dev/schemas/my_features.yaml \
  --sample small --run my-features-v1
```

The example schema extends `baseline.yaml`. Add a new entry for every new scalar:

```yaml
version: my-features-v1
extends: baseline.yaml
features:
  my_feature:
    dtype: float64
    description: Explain the physical definition and units here.
    missing_value: -999.0
    valid_range: [0, 1]
    histogram: {bins: 25, range: [0, 1]}
```

Use `int64` with an integer sentinel for integer features. `valid_range` and
`histogram` are optional. Only features with histogram settings are plotted;
all features are calculated and validated. Six existing features are plotted
by default, plus the example's new feature. A new feature has no baseline curve
until it also has a baseline definition. To return only your own features, use
a schema without `extends` or `inherit`.

Imports from the packaged `feature_dev/algorithms` directory work directly.
For external imported helper files, make their modules available in your Python
environment and declare each source with `--dependency /path/to/helper.py` so
changes invalidate the cache. Runtime inputs must be the provided hits and
parameters; undeclared external files or randomness violate this contract.

## Use real selected hits

A one-time export is needed for each chosen testbeam, collision-data, or MC
sample. Use its **existing converter invocation and its matching SNDSW runtime**,
adding these options (replace the path and label):

```text
--max-events 100 --feature-dev-export /afs/.../samples/electron100.jsonl
--feature-dev-group testbeam-electron-100GeV
```

Both `convertData/digi_2_features.py` and the MC-muon wrapper support the options.
Keep a distinct `--source-uid` for each source and the correct
`--source-geometry-id`. Export occurs after the existing timing selection,
using precisely the hit list supplied to clustering. Completion requires its
exported count to equal the written ROOT entry count. Interrupted exports lack
a completion manifest and are rejected. Existing exports cannot be overwritten.

Merge completed exports in the LCG environment:

```bash
set +u
source snakemake/envs/env_lcg.sh
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
python3 feature_dev/selected_hits.py /afs/.../samples/comparison.jsonl \
  /afs/.../samples/testbeam.jsonl /afs/.../samples/collision.jsonl \
  /afs/.../samples/mc.jsonl
./run_feature_dev.sh --algorithm gap_clusters \
  --sample /afs/.../samples/comparison.jsonl --run comparison-v1
```

The merger validates manifests and rejects duplicate identities using
`(source_uid, runId, eventId, eventIndex)`. Exported sample provenance records
converter arguments and checksum. Subsequent algorithm development reads these
fixed exports; it does not rerun detector selection or need SNDSW.

Reports split groups by label **and geometry**, comparing candidate and baseline
within each panel. Histograms use unit-normalized **in-range bin fractions**,
not luminosity-normalized yields or probability density per unit x. Missing
values and under/overflow counts are displayed. The last bin includes its upper
edge, following NumPy. There are no additional physics cuts in this runner:
the exported selected sample defines the population.

This development DAG is separate from the full production campaign. It does
not replace candidate algorithms in legacy fixed-schema converters, rerun
training, or qualify testbeam/MC agreement automatically. Promoting a candidate
requires physics review and an explicit production schema/adapter update.

## Rerun rules and validation

| Change | Work required |
| --- | --- |
| Same inputs/settings/code | Reuse all stages |
| Histogram bins/ranges or descriptions only | Histograms and report |
| Feature schema type/range/sentinel or algorithm/parameters | Calculation onward |
| Selected hits or their manifest | Calculation onward |
| Shared development engine or declared helper source | Calculation onward |

Cache keys include source SHA-256 hashes, input hashes, resolved definitions and
Python/library versions. Completed products have checksum receipts; tampering
is rejected. The runner serializes executions within an output root. Outputs
are intended for modest fixed development samples, loaded into memory.

Validation checks sample completeness/checksum, unique identities/channels,
input types, exact feature schema, finite values, bounds/sentinels, no input
mutation, optional algorithm invariants, and JSON/ROOT identity/value agreement.
It catches software errors; it does not establish physical correctness.

On lxplus, run `./run_feature_dev.sh setup` once. It installs and checks a private
Snakemake environment and saves its path in this checkout's ignored
`.feature-dev-snakemake` file. Future workflow commands load this setting and
LCG automatically; no activation or exports are needed. Full instructions:
[collaborator setup](../README_SHOWER_FEATURES.md#git-and-environment-setup).
Use `setup --check` to verify, or `setup --prefix /your/new/environment/path` to
choose storage. Repeating setup is safe for environments managed by this tool.

Advanced overrides `FEATURE_DEV_PYTHON` and `FEATURE_DEV_SNAKEMAKE` remain
available and take precedence over automatic selection. Normal workflow runs
never install packages; only the explicit `setup` command does. The setup is for
lxplus/LCG 107; other systems need an appropriate analysis Python and Snakemake.

## Environment setup reference

The short setup instructions are in
[the collaborator guide](../README_SHOWER_FEATURES.md#git-and-environment-setup).

| What you need | Command or action |
| --- | --- |
| Check an installed environment | `./run_feature_dev.sh setup --check` |
| Retry an interrupted setup | Run the same setup command again |
| Choose another installation directory | `./run_feature_dev.sh setup --prefix /your/new/environment/path` |
| Choose another output directory | Add `--output-root /your/writable/path` to the workflow command |
| Preview workflow steps | Add `--dry-run` to the workflow command |
| Remove stale manual overrides | `unset FEATURE_DEV_PYTHON FEATURE_DEV_SNAKEMAKE` |

Setup requires lxplus with LCG 107/CVMFS, writable storage, and access to the
Python package index for the initial installation. It installs Snakemake 9.17.2
in `venvs/shower-snakemake-9.17.2-lcg107` under your AFS work area. Analysis
packages come from shared LCG. SNDSW is needed only when exporting raw detector
events; physics dataset commands now prepare and validate those exports automatically.

The path is saved in `.feature-dev-snakemake`, a Git-ignored file in this
checkout. It survives Git updates. A new clone needs its own setup command;
use `--prefix` to reuse an environment managed by this installer. Setup does
not change `.bashrc`. Healthy installations are checked without reinstalling;
unrelated existing environments are refused. Choose a new directory for
`--prefix`, rather than an existing Conda or other virtual environment.

Explicit `FEATURE_DEV_PYTHON` and `FEATURE_DEV_SNAKEMAKE` exports override saved
settings. Remove stale exports from your shell startup file if they return in
new terminals. Installation clears Python path variables to avoid reusing
incompatible packages from LCG. Do not replace it with `pip install --user`.

If setup fails, keep the full command and error message. Permission or quota
errors require fixing storage access/space. A missing CVMFS path requires a
working lxplus/LCG installation. Changing AFS permissions on someone else's
private environment is not part of setup.

## Test plan

Run in the LCG environment above:

```bash
python3 -m unittest discover -s feature_dev/tests -v
python3 -m unittest discover -s convertData -p 'test_scifi_plane_clustering.py' -v
python3 feature_dev/tests/integration.py
```

Unit tests cover known cluster/feature values, empty events, baseline parity at
zero gap, malformed parameters/results, mutation rejection, sample integrity,
export completion, duplicate channels and label independence. Integration checks
exercise the actual Snakemake DAG, ROOT outputs, self-contained HTML, dry-run
isolation, cache reuse, histogram-only changes, changed parameters, conflicting
run names and tamper rejection.

Before physics use, export a small approved sample from each real source with
its normal converter invocation. Check exported identities/counts against ROOT,
compare baseline scalar branches event by event, inspect geometry/group labels,
and review candidate differences. These detector-runtime acceptance checks are
still required; the packaged example is synthetic.

### Current source validation limits (16 November 2026)

Standard small samples passed automatic export, baseline parity, report
generation and unchanged-cache reuse through HTCondor: 633 neutrino events
(including all four classes), 625 testbeam events across 25 groups, and 200
collision-data events across both groups. All 57 source exports passed identity
and 182 baseline-scalar comparisons. The testbeam size issue found by this test
was fixed with bounded assembly and a retained cap of 25 per group/geometry.
Coworker-account access, full campaigns and candidate physics acceptance remain
unqualified; the optional preparation-profile path was not exercised.
The optional `mc-muon` path is implemented, but its current shared catalogue
had no entry with both digi and geometry files present from the validation host;
its command reports the missing input until the catalogue/access is repaired.
See [TEST_RESULTS.md](TEST_RESULTS.md) for exact evidence and limits.
