# Production with coworker features

New collaborators: start with the [coworker guide](../README_SHOWER_FEATURES.md)
for setup, examples and a complete run command. This page is the detailed reference.

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
Every plan also checks the selected cutflow ROOT headers before submission and
reports all missing/unreadable selections together. Testbeam MC catalogues must
include `selected_event_count`, separately from raw `input_event_count`.

Testbeam MC nominal beam energy uses the converter tolerance (2% relative or
0.5 GeV absolute) during validation. PDG remains exact; existing strict
numerical and event-identity checks are unchanged. Actual MC energy remains
stored in the feature file. After a catalogue or validator
update, use a new run name; existing run plans are immutable.

Choose `--dataset testbeam`, `real-data`, or `mc-muon` for the other catalogues.
The optional muon catalogue has previously contained missing inputs; its current
availability must be checked. Private testbeam cutflows may require access from
the owner. The implementation does not grant permissions.

Small runs default to one file per group/geometry and at most 20 selected events
per file. Change these with `--files-per-group` and `--events-per-file`.
For example, the retained sampling options work with this production launcher:

```bash
./run_shower_features.sh --dataset mc-neutrino --plugin silvia \
  --sample small --files-per-group 6 --events-per-file 200 --run trial02
```

This selects up to six files per dataset group/geometry and up to 200 selected
events per file. Small-run limits are 20 files per group/geometry and 1,000 events
per file. File groups are not quotas for the plotted CC/NC classes. These caps
apply only to `--sample small`; full mode reads all catalogue sources.
The first selected events are a debugging sample, not an unbiased yield sample.

See [data formats and structure](../feature_dev/DATA_FORMAT.md) for the meaning
of input collections, saved event features, hit fields and MC truth.

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
    1d/<feature>/<selection>/<class>.root
    2d/<pair>/<selection>/<class>.root
    summary.json, complete.json
  plots/
    1d/<feature>/<selection>/plot.pdf, summary.csv
    2d/<pair>/<selection>/plot.pdf, summary.csv
    coverage.json, normalization.json, complete.json
```

Open the **per-feature PDF** under `plots/1d/` or **per-pair PDF** under
`plots/2d/`. There is no combined `report.pdf` or HTML. Geometries are combined in each class distribution; there are no `geo_*`
output folders. Original geometry identifiers remain in `config/coverage.json`
and each ROOT histogram's `production_histogram_json` for validation. Four MC classes appear in each comparison; empty classes are
explicitly shown. Testbeam and real-data groups retain their catalogue labels.

The 1D figures use `evaluation_region_partitions/plot_comparison.py`, including
ROOT markers/error bars, class colours, legends, configured log axes and mean/
standard deviation. 2D uses `plot_group_hist2d.py`: density panels on page 1,
profile comparison on page 2. Profiles are means derived from the binned TH2,
matching the existing EM-shower workflow. ROOT filling is shared with
`make_group_hist.py` and `make_group_hist2d.py`; histogram files retain raw
counts/errors and normalization metadata. Yield plots sum components after applying each component's audited scale,
including squared weights in variances. ROOT files keep both raw counts and
`hist_yield`/`hist2d_yield`; summaries retain the component exposures. Because
weights can differ, a combined yield has no single scale factor (CSV field
left empty). Shapes merge raw counts first and normalize once. Profiles remain
raw event means from the combined binned TH2.

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
the **same selection across all geometries**, with padding so observed maxima stay in
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

## Batch setup and full dataset

On your own account, install the batch dependencies and configure your profile:

```bash
./run_shower_features.sh setup --batch
./run_shower_features.sh setup --batch --check
```

This extends the existing private Snakemake environment. It deploys the official
[Snakemake HTCondor profile](https://github.com/Snakemake-Profiles/htcondor) at
pinned revision `a5b7e6c73d9b9305abe4a04a24056050461627d3` using Cookiecutter,
with CERN credential settings. The `cluster-generic` executor submits to
HTCondor through Python bindings; it does not require the native `htcondor`
executor. Package versions are pinned: Snakemake 9.17.2, Cookiecutter 2.7.1,
HTCondor 25.0.14 (the `htcondor2` import), cluster-generic executor 1.0.9.

The generated profile defaults to `<private-environment>/profiles/htcondor`;
logs live under `<private-environment>/htcondor_logs/<job-token>/`. Its location
is saved in ignored `.feature-production-profile` in this checkout. Official
generated files are preserved under the profile's `upstream/`; a CERN adapter
sets portable interpreter paths, shared-filesystem execution, resource units,
status polling and cancellation. `deployment.json` records provenance.
No Zhibin-specific paths are used by the generated profile.

Repeated setup preserves existing profile customizations. `--check` installs or
changes nothing. An unmanaged existing profile is never overwritten. Use
`setup --batch --batch-profile /your/new/profile` to select a new deployment;
`--prefix /your/new/environment` selects a private environment. Initial
concurrency defaults to 20, configurable with `--batch-jobs` on first setup.
For an existing profile, use runtime `--jobs N` or edit its `jobs` setting.

### First run a bounded batch canary

```bash
./run_shower_features.sh --dataset mc-neutrino --plugin silvia \
  --sample small --files-per-group 1 --events-per-file 2 \
  --batch --jobs 2 --run batch-check01 --check-access

./run_shower_features.sh --dataset mc-neutrino --plugin silvia \
  --sample small --files-per-group 1 --events-per-file 2 \
  --batch --jobs 2 --run batch-check01
```

Small runs remain local unless `--batch` or an explicit `--profile` is provided.
Full runs automatically use the saved profile. Explicit `--profile PATH` takes
precedence. Missing setup produces an actionable error; full execution never
silently falls back to local processing. User-supplied profiles may use
`htcondor` or configured `cluster-generic` submit/status commands; external
profiles remain the user's responsibility, including paths/resource mapping.

Before batch execution, the launcher checks the executor import, generated
profile's HTCondor binding/interpreter and log access, Kerberos ticket (`klist
-s`) and scheduler WRITE authentication (`condor_ping`). `--check-access` also
checks selected detector inputs and SNDSW. Setup alone does not establish EOS
permissions or credential lifetime on workers; test the canary on your account.

### Preview and run full processing

```bash
./run_shower_features.sh --dataset mc-neutrino --plugin silvia \
  --sample full --normalization shape --run full01 --dry-run

./run_shower_features.sh --dataset mc-neutrino --plugin silvia \
  --sample full --normalization shape --run full01
```

The full neutrino catalogue currently plans 1,866 source jobs plus range,
histogram and plot stages. `--jobs 20`
limits concurrent jobs, not the number of source files. Keep the controller
running in a persistent terminal session. A dry-run submits nothing.

Each source job uses one CPU, 2500 MB memory and 5000 MB disk. The stage runtime
resource defaults to **60 minutes**, mapped by the generated adapter to
`+MaxRuntime=3600` seconds. To change a rule's runtime, edit the generated profile,
for example `set-resources: {source: {runtime: 120}}`. The adapter supplies 5000 MB
disk for aggregate stages too. Review bounds before a larger campaign.

Monitor with `condor_q "$USER"`; inspect the run's `.snakemake/log/`,
`logs/source-<index>.log` and the profile log directory's `condor.out`,
`condor.err`, `condor.log`, `submission.json` and `job.json`. Snakemake prints the
external job ID `<token>_<cluster>`; `job.json` records the cluster number.
The run's `batch-execution.json` records the chosen profile/config hash.
For cancellation, interrupt the controller normally so its cancel hook can
remove tracked jobs; use `condor_rm <cluster>` for a known remaining job.
Do not remove unrelated jobs. After an interrupted controller, check for live
jobs before rerunning the same unchanged command. Completed products are reused
only after receipt validation; uncompleted jobs may need recovery.

This workflow submits one conversion per source, not shards within a source.
The bounded batch canary and full dry-run are separate from full-campaign
resource/physics qualification. Yield processing still requires an audited
exposure manifest as described below.

## Normalization and ROOT metadata

Small and testbeam runs use in-range bin fractions. For full MC/data, `auto`
requires an audited exposure manifest; explicitly choose `shape` for inspection.
Yield invocation adds `--normalization yield --exposure-manifest /path/audit.json`.

See the [normalization guide](../feature_dev/NORMALIZATION.md) for required
information, storage locations, formulas and reading examples, and the
[audit JSON reference](../feature_dev/NORMALIZATION_AUDIT.md) for the production
`campaign_sha256` format and coverage rules.

MC weight is target data exposure / parent MC exposure; data weight is 1 at
its effective sampled exposure. The same weights apply to production and plugin
features. Counts, flows, missing values and statistical sumw2 are retained.
Classes are compared within each selection with geometries combined. Energy comes from the same `sndData` entry.
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
