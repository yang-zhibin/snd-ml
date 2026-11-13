# Develop shower features

The coworker edits an algorithm and its schema, then runs one command to calculate
features, validate them, make histograms, and open a standalone HTML report.
The existing production clustering is always calculated on the **same selected
hits** as the candidate. Candidate code receives no sample labels or MC truth.

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

On CERN the wrapper selects the repository's LCG 107 environment and discovers
your existing `miniconda3/envs/snakemake/bin/snakemake`. Elsewhere set
`FEATURE_DEV_PYTHON` and `FEATURE_DEV_SNAKEMAKE` to suitable existing executables
(Python with NumPy, PyYAML, uproot and matplotlib; separate Snakemake runtime).
No dependencies are installed by the runner.

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
