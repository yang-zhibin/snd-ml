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
  plan.json, plugin.json        # frozen inputs, code hashes and settings
  logs/source-<index>.log       # converter and validator commands/output
  sources/<index>/features.root
  sources/<index>/validation.json, inputs.json, complete.json
  plots/report.pdf, histograms.json, coverage.json, normalization.json
```

Small runs additionally save selected hits and a baseline parity receipt.
Use `--feature-root /your/writable/eos/directory` to put `sources/` under
`<feature-root>/<dataset>/<run>/`; logs, plans and plots stay under output-root.
Open `plots/report.pdf` for the multipage report: a summary followed by one
plot per feature/group/geometry, with counts and normalization details.
Individual PNG plots and machine-readable JSON remain available.
All workers must see the checkout and both output locations at the same paths.
Existing production campaign files and shared input catalogues are not changed.

Repeat an identical command to reuse verified products. Changed code, Git HEAD,
parameters, exposure audit or inputs require a new run name. Do not edit the
checkout while its campaign is running. Partial source files are published only
after validation; altered cached files are rejected.

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
one streaming distribution job. It does not split a large individual input file
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
Groups and geometries stay separate. Energy comes from the same `sndData` entry.
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
