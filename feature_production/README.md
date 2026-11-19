# Production with coworker features

The production converter calculates the existing features and the chosen plugin
in the same event loop. One `sndData` entry contains both sets of features.
Dataset selection uses the existing EOS catalogues in `feature_dev/datasets.yaml`;
there is no synthetic fallback and no separate copied input dataset.

## Start on your account

From your checkout on lxplus:

```bash
./run_shower_features.sh setup
./run_shower_features.sh --dataset mc-neutrino --plugin silvia \
  --sample small --run trial01 --check-access
./run_shower_features.sh --dataset mc-neutrino --plugin silvia \
  --sample small --run trial01
```

`setup` reuses the private Snakemake installer. Conversion runs through the clean
SNDSW child wrapper, independently of the LCG plotting environment.
`--check-access` checks the selected digi, geometry, cutflow, output parents and
SNDSW imports using **your credentials**. It does not submit a batch job or verify
credential lifetime. Missing or unreadable files fail explicitly.

Choose `--dataset testbeam`, `real-data`, or `mc-muon` for the other catalogues.
The optional muon catalogue has previously contained missing inputs; its current
availability must be checked. Private testbeam cutflows may require access from
the owner. The implementation does not grant permissions.

Small runs default to one file per group/geometry and at most 20 selected events
per file. Change these with `--files-per-group` and `--events-per-file`.
The first selected events are a debugging sample, not an unbiased yield sample.

## Outputs

Default: `/afs/cern.ch/work/<initial>/<user>/snd-ml-outputs/shower-features/`.
Override with `--output-root /your/writable/path`. A run contains:

```text
<output-root>/<dataset>/<run>/
  plan.json, plugin.json
  config/
    histograms.yaml, plotting.yaml     # requested settings
    resolved.yaml, resolved.json       # measured bounds, bins and fallback reasons
    coverage.json, complete.json
  logs/source-<index>.log
  sources/<index>/features.root
  sources/<index>/validation.json, inputs.json, complete.json
  histograms/
    1d/<feature>/<selection>/<geometry>/<class>.root
    2d/<pair>/<selection>/<geometry>/<class>.root
    summary.json, complete.json
  plots/
    1d/<feature>/<selection>/<geometry>/plot.pdf, summary.csv
    2d/<pair>/<selection>/<geometry>/plot.pdf, summary.csv
    coverage.json, normalization.json, complete.json
```

Open the **per-feature PDF** under `plots/1d/` or **per-pair PDF** under
`plots/2d/`. There is no combined `report.pdf` or HTML. Geometry directories are
stable `geo_<hash>` keys; `config/coverage.json` maps them to original geometry
identifiers. Four MC classes appear in each comparison; empty classes are
explicitly shown. Testbeam and real-data groups retain their catalogue labels.

The 1D figures use `evaluation_region_partitions/plot_comparison.py`, including
ROOT markers/error bars, class colours, legends, configured log axes and mean/
standard deviation. 2D uses `plot_group_hist2d.py`: density panels on page 1,
profile comparison on page 2. Profiles are means derived from the binned TH2,
matching the existing EM-shower workflow. ROOT filling is shared with
`make_group_hist.py` and `make_group_hist2d.py`; histogram files retain raw
counts/errors and normalization metadata. Plots apply audited scales once.

Small runs additionally save selected hits and a baseline parity receipt.
Use `--feature-root /your/writable/eos/directory` to put `sources/` under
`<feature-root>/<dataset>/<run>/`; plans, histograms and plots stay under
output-root. Workers must see the checkout and output paths at identical
locations. Inputs and existing production campaign outputs remain read-only.

## Change the plot settings

Edit these files **before creating a new run**:

- [config/histograms.yaml](config/histograms.yaml): features, ROOT expressions,
  cuts, bins, ranges, selections, flow folding and explicitly selected 2D pairs.
- [config/plotting.yaml](config/plotting.yaml): class colours, markers, labels,
  ratio-panel setting. Shape/yield mode remains the CLI `--normalization` option.

Alternatively copy both files and pass `--hist-config /path/histograms.yaml`
and `--plot-config /path/plotting.yaml`. A supplied file replaces that default
configuration. The run snapshots requested and resolved settings under `config/`;
do not edit those published snapshots. Plugin features are included by default,
but old plugin-schema histogram ranges are intentionally not inherited.

Example settings (merge these into your histogram configuration):

```yaml
defaults: {bins: 50, fold_underflow: false, fold_overflow: true}
include_baseline: true
include_plugin: true
selections:
  selected: 'true'                    # already converter-selected events
  # scifi35: 'count_scifi > 35'        # optional original EM-shower selection
features:
  silvia__leading_charge_fraction:
    bins: 40
    range: [0, 1]                    # explicit bounds win
  silvia__scifi_vertical_cluster1_density:
    bins: 50                         # omitted range => automatic
    logy: true
  count_scifi: {dtype: int64, axis_title: Selected SciFi hits}
  qdc_scifi: {axis_title: Total selected SciFi QDC}
features2d:
  total_qdc_vs_hits:
    x: count_scifi
    y: qdc_scifi
    x_bins: 50
    y_bins: 50
    # x_range: [0, 1000]              # omit either axis range for automatic bounds
    # y_range: [0, 10000]
    logz: true
```

Automatic bounds scan finite, non-missing selected values across all classes in
the **same selection and geometry**, with padding so observed maxima stay in
normal ROOT bins. No quantile trimming. Integer features use half-integer bounds;
constant values get padding; empty samples use `[0,1]` (`[1,10]` for log-x),
with the reason recorded in `resolved.yaml`. Explicit bounds preserve ROOT's
upper-edge overflow convention. 2D bounds use jointly valid X/Y pairs and are
resolved independently of 1D ranges. Default 1D overflow is folded into the last
bin, matching the EM-shower setting; 2D retains flow bins and normalizes using
the in-range integral. Missing values never enter the flow bins. Use
`missing_value`, `feature_cut`, `expression`, `axis_title`, `y_max`, `logx` and
`logy` on individual features as needed. Nonpositive explicit log-axis ranges
are rejected. Every selected feature must exist in every applicable source,
including empty files; schema mismatches fail before filling.

For more than four groups the 2D layout adds rows. Only named 2D pairs are
produced; there is no automatic Cartesian product of all features.

## Reuse calculated features

An identical command reuses verified products. Use a **new run name** for changed
settings or Git HEAD; each immutable plan records the checkout commit. Reuse compatible ROOT files explicitly:

```bash
./run_shower_features.sh --dataset mc-neutrino --plugin silvia \
  --sample small --files-per-group 1 --events-per-file 2 --run new-plots \
  --plot-config /path/plotting.yaml \
  --reuse-sources /your/output/mc-neutrino/previous-run
```

Use the same dataset, caps, plugin, converter, normalization and exposure audit
as the source run. Source products and inputs are checked and kept in their
original directory. If histogram settings/code also match a new-backend run,
the range results and histograms are reused through verified references; only
plots rerun. Changing histogram settings reuses ROOT sources but rebuilds ranges
and histograms. Keep referenced runs available. Style changes are recorded in
the new run's own `config/plotting.yaml`. Changed converter/plugin/input settings
require fresh conversion. No source reuse happens implicitly.

Named Snakemake aggregates are `production_sources`, `production_histograms`,
`production_plots`, and default `all`. Dependency order is source validation →
range/coverage resolution → ROOT histogram filling → ROOT PDFs/CSV summaries.
Each stage publishes a completion receipt with recursive product checksums.

## Full dataset

Preview the DAG without submitting or creating campaign products:

```bash
./run_shower_features.sh --dataset mc-neutrino --plugin silvia \
  --sample full --normalization shape --run full01 --dry-run
```

Execute with your qualified Snakemake HTCondor profile:

```bash
./run_shower_features.sh --dataset mc-neutrino --plugin silvia \
  --sample full --normalization shape --run full01 \
  --profile /path/to/your/htcondor-profile
```

The profile must use `executor: htcondor`, with the matching executor installed
in your Snakemake environment and suitable CERN credentials, concurrency and
resource settings. Setup alone does not qualify batch execution. This launcher
creates one conversion job per catalogue source (including empty sources), then
separate range-resolution, histogram and plotting jobs. ROOT event processing
is sequential within each job; all actions are booked together per group to
avoid a separate input scan for every feature. It does not split a large individual input file
into shards. Defaults are 2500 MB and a 3600-minute Snakemake runtime resource;
review profile resource mapping before submission. Full execution is refused
without an HTCondor profile. No full batch campaign has been validated yet.

## Normalization and ROOT metadata

Small and testbeam runs use in-range bin fractions. For full MC/data, `auto`
requires an audited exposure manifest; explicitly choose `shape` for inspection.
Yield invocation adds `--normalization yield --exposure-manifest /path/audit.json`.

Use the [audit format and physics rules](../feature_dev/NORMALIZATION.md), with
these production-specific differences:

- Supply `campaign_sha256` from the plan/coverage report instead of a prepared
  JSONL `sample_sha256`. The campaign identity is independent of the plugin and
  audit, so a shape run can establish coverage for a later yield run.
- Source IDs in audit populations are `family:source_uid` (`audit_source_id` in
  ROOT metadata). Use exact group/geometry names from `coverage.json`.
- Include every parent source, including zero-selected outputs. Establish
  authoritative luminosities and external reference-data coverage separately;
  coverage reports are evidence, not automatically approved exposure audits.

MC weight is target data exposure / parent MC exposure; data weight is 1 at
its effective sampled exposure. The same weights apply to production and plugin
features. Counts, flows, missing values and statistical sumw2 are retained.
Classes are compared within each selection/geometry; geometries stay separate. Energy comes from the same `sndData` entry.
There is no external event join.

Each source ROOT file stores `feature_plugin_json` and its SHA256, plus
`production_exposure_json` and a one-row `source_exposure` TTree. The latter
preserves **catalogue** parent luminosity, sampling probability and validity;
small/unknown exposure has invalid effective luminosity (NaN). This is not
per-event luminosity. A supplied full audit is also embedded in the JSON;
its audited values can differ from unqualified catalogue values. Final group
weights and campaign coverage validation are in `plots/normalization.json`.
Source metadata deliberately says `requires_campaign_validation`: an isolated
source file cannot prove complete campaign exposure. Current real-data
`event_count_only` entries alone cannot establish physical yields.

## Developing a feature

Edit only your plugin folder; see [plugin interface](../feature_plugins/README.md).
The supplied `silvia` plugin exports 13 namespaced scalar branches. The old
`run_feature_dev.sh` remains available for standalone development. The new
launcher always uses the production converter for detector processing.
