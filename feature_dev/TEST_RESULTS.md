# Feature development acceptance results

Verified on lxplus922, 2026-11-13, using the repository's LCG 107 environment
(Python 3.11.9) and existing Snakemake environment. No production jobs submitted.

- 17 development unit tests: passed.
- 11 existing SciFi clustering regression tests: passed.
- 13 unified shower feature regression tests: passed.
- 100 deterministic randomized hit collections: full `cluster_feature_payload`
  equals the pre-change implementation loaded from Git HEAD before this change.
- Python syntax checks and `git diff --check`: passed.
- `feature_dev/tests/integration.py`: all 10 acceptance checks passed (dry-run
  isolation; full DAG; same-run reuse; reuse across run names; conflicting run
  rejection; histogram-only cache reuse; parameter invalidation; zero-gap
  baseline equality; self-contained HTML; output tamper rejection).
- Both documented tutorial commands completed the full DAG for 24 synthetic
  events, writing baseline/candidate ROOT trees, JSON, validation receipts,
  seven histograms/PNGs and embedded-image HTML.

Saved examples:

```
/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/feature_dev/tutorial-gap1/plot/report.html
/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/feature_dev/tutorial-gap0/plot/report.html
```

The real detector converter's optional export was syntax checked and its writer
was tested with round trips, counts, coordinates, no-overwrite behavior and
merged source identity checks. The real SNDSW conversion/export path was **not**
executed on detector inputs. Real-source baseline comparison remains the next
acceptance step, as described in `README.md`; synthetic checks establish software
behavior only. Candidate algorithms are not installed into production converters.

## Independent dataset execution (13 November 2026)

Verified on lxplus9104 with LCG 107 / Python 3.11.9 and the existing Snakemake
installation:

- 30 development unit tests passed, including 13 dataset preparation/resolution
  tests: deterministic group sampling, full selection, duplicate rejection,
  source-family checks, synthetic rejection, no overwrite, registry resolution,
  missing files/groups, registration, size limit and normalization policy.
- `tests/dataset_integration.py`: all 10 end-to-end acceptance checks passed:
  routing/isolation, dry run, all/index, independent reuse, unrelated configuration
  changes, immutable runs, full sample selection, synthetic namespace, muon
  exclusion, and missing-input preflight.
- Existing `tests/integration.py`: all 10 acceptance checks passed again,
  including histogram-only reuse, ROOT/JSON validation and cache tamper rejection.
- The documented wrapper command with `--dataset synthetic --algorithm gap_clusters
  --sample small --run dataset-demo-v1` completed all stages. Report:
  `/afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/feature_dev/synthetic/dataset-demo-v1/plot/report.html`.

Dataset integration tests use artificial hit exports explicitly labelled as test
fixtures in their provenance. No real detector conversion or physics campaign
was run. Physics dataset samples must still be prepared and qualified; large
sharded execution and exposure-normalized development reports are not implemented.

## Automatic environment setup — 2026-11-15

- 35 development unit tests passed, including 5 installer tests covering Python
  path isolation, atomic plain-data configuration, preservation of unrelated
  environments, failed-install recovery, and read-only verification.
- Fresh explicit setup installed Snakemake 9.17.2 into an isolated LCG-Python
  virtual environment; `pip check` passed.
- `setup --check` passed. Repeated `setup` reused the healthy environment without
  invoking pip installation.
- From an `env -i` shell with no FEATURE_DEV overrides, the wrapper loaded the
  saved configuration and completed calculation, ROOT/JSON validation,
  histograms and the synthetic HTML report.
- Shell syntax, ignored-local-config check and `git diff --check` passed.
- No AFS access permissions or shell startup files were changed. This is an
  lxplus/CVMFS installer, not a portable laptop setup. Real-data access still
  requires independently provided selected-hit exports and manifests.

## Automatic source preparation — 16 November 2026

Verified on lxplus939 using LCG 107 / Python 3.11.9, the configured personal
Snakemake 9.17.2 installation, and the existing `env_sndsw_nue.sh` plus pinned
June25 converter interpreter used by the unified production rules.

- 51 development unit tests passed (35 existing + 16 source-preparation tests).
- 11 SciFi clustering and 13 unified-feature regressions passed.
- Both existing integration suites passed: 10 dataset checks and 10 workflow
  checks, including immutable runs, cache tampering and dry-run isolation.
- The shipped registry's `--dataset all --dry-run` passed: 8 neutrino, 45
  testbeam and 4 collision-data source files. The output directory was absent
  afterward; planning did not export or alter source files.
- A focused detector canary exported **two selected events per source** from
  six catalogue records: one neutrino MC, one testbeam MC 2024, one H4 data 2024,
  one H8 data 2023, and one each of the no-veto/veto-10% collision-data groups.
  All six passed entry-by-entry identity and **182 baseline scalar** comparisons
  against their fresh converter ROOT trees. Integer checks were exact;
  floating-point checks used `rtol=1e-6, atol=1e-7`.
- The complete baseline/candidate DAG produced three HTML reports and the `all`
  index. Repeating the identical command reused all three validated samples;
  all analysis DAGs reported nothing to do. No converters reran.
- A first canary exposed Snakemake precreating the output parent; this was fixed
  and regression-tested, including failed-export retry and cache preservation.
- Shell syntax and `git diff --check` passed.

Reproduction command for the focused real-source canaries (temporary registry
contains one source per family/group, two events per source; neutrino required
four-class coverage is deliberately disabled for this two-event adapter test):

```bash
./run_feature_dev.sh --dataset all \
  --dataset-config /tmp/zhibin/feature-auto-canary-20261116/all-datasets.json \
  --sample small --run canary-report --algorithm gap_clusters \
  --output-root /tmp/zhibin/feature-auto-canary-20261116/all-output
```

Reports: `/tmp/zhibin/feature-auto-canary-20261116/all-output/` followed by
`{mc-neutrino,testbeam,real-data}/canary-report/plot/report.html`;
combined index: `all/canary-report/index.html`. Source plans, export logs and
parity receipts are under that output root's `.samples/<dataset>/<signature>/`.
Controller logs: `/tmp/zhibin/feature-auto-canary-20261116/all-run.log` and
`reuse.log`. Unit/regression logs: `/tmp/zhibin/feature-{unit,clustering,unified}-20261116.log`.
Integration logs: `/tmp/zhibin/feature-{dataset-integration,integration}-20261116.log`.
Default-registry preview: `/tmp/zhibin/feature-source-preview-20261116.log`.
These are temporary, host-local validation artifacts, not shared production samples.

Limits: the complete default samples were not exported; this does not establish
four-class neutrino coverage, physics acceptance of gap clustering, coworker
account permissions or HTCondor execution. The optional muon source adapter is
unit-tested, but detector validation was blocked: none of the 15 existing muon
catalogue rows had both digi and geometry files present from this host. The
missing-source preflight failed explicitly without substituting another sample.
No production products were changed and no batch jobs were submitted. Automatic
full-campaign preparation remains blocked pending sharded processing.

## Standard small samples and bounded assembly — 16 November 2026

The standard registry was exercised against the existing EOS sources through
HTCondor controllers running the Snakemake-backed launcher. Controllers use the
shared checkout and its configured Snakemake 9.17.2 executable; this tests batch
execution, not the optional `--prepare-profile` integration.

The first complete testbeam export exposed an assembly bug: all 45 source
exports passed parity, but their combined JSONL size (~542 MiB) exceeded the
manual-input aggregate guard. Automatic capped assembly now reads one guarded
source at a time and retains the smallest identity hashes per group/geometry.
It preserves the 256 MiB per-source and final-sample limits. Testbeam retains
25 events per group/geometry; neutrino and collision data retain 100. Source
file selection and the 100-selected-entry cap per file are unchanged.

All 55 development unit tests passed, including bounded/global sampling
equivalence, aggregate/per-file limits, full-mode rejection and oversized
output rejection. The dataset integration suite passed all 10 checks.
Logs: `feature_dev/output/standard-20261116/fix-unit.log` and
`fix-dataset-integration.log` in the same directory.

The 57 detector exports (8 neutrino, 45 testbeam, 4 collision data) passed exact
entry identity and all 182 baseline scalar comparisons. For the assembly-only
fix, those exports were reused after verifying every receipt, identical source
jobs/catalogue fingerprints, and unchanged converter code. The one-off audit
is `feature_dev/output/standard-20261116/assembly-reuse-audits.json`; this is
not an automatic cache-migration feature. Corrected controllers rerun assembly,
analysis/report generation and identical-command reuse.

Validation harness and scheduler evidence are under
`feature_dev/output/standard-20261116/`: `run-test.sh`, `submit.py`, `jobs.json`,
`<dataset>.out`, `<dataset>.err` and `<dataset>.condor.log`. Earlier attempts are
preserved with `attempt1`/`attempt2` names. The first attempt failed before
conversion because LCG activation overwrote a generic harness variable; the
harness now uses `validation_dataset`. The second attempt qualified all source
exports and found the testbeam assembly limit. Corrected controller IDs are
`16585806.0` (neutrino), `16585807.0` (testbeam), `16585808.0` (collision data).

All three corrected controllers exited 0; no validation jobs remain active.

| Dataset | Source files | Retained events | Groups/geometries | Sample bytes |
| --- | ---: | ---: | ---: | ---: |
| mc-neutrino | 8 | 633 | 16 | 92262386 |
| testbeam | 45 | 625 | 25 | 78175626 |
| real-data | 4 | 200 | 2 | 2917300 |

Neutrino class totals: CC_nue 180, CC_numu 298, NC_nue 53, NC_numu 102.
Testbeam retains 25 events in every one of its 25 groups; collision data retains
100 events for each saved no-veto/veto-10% group. Each identical rerun reused
all exported ROOT files with unchanged modification times.

Reproduction (from this checkout, with its configured runtime):

```bash
./run_feature_dev.sh --dataset all --sample small --algorithm gap_clusters \
  --run standard-test-fixed-20261116 \
  --output-root "$PWD/feature_dev/output/standard-20261116/results"
```

Reports: `results/<dataset>/standard-test-fixed-20261116/plot/report.html`;
JSON assertions: `results/<dataset>-test-result.json`; sample-size summary:
`fixed-summary.json`, all relative to the validation directory above.

Remaining limits: coworker-account permissions, full campaigns, optional
`--prepare-profile` execution and physics acceptance are unqualified. Optional
muon sources remain unavailable as described above. No production products or
EOS inputs were modified.

The combined `--dataset all` dry-run and actual rerun also passed, reusing all
three samples with no analysis jobs needed. Index:
`results/all/standard-test-fixed-20261116/index.html`; logs:
`fixed-all-dry-run.log` and `fixed-all-reuse.log`. Shell syntax and
`git diff --check` passed. Verification completed 2026-11-16 (Europe/Zurich).

## Mixed LHCb/SNDSW Python startup — 17 November 2026

The original committed export wrapper retained LHCb entries on PATH. With
`/cvmfs/lhcb.cern.ch/lib/lcg/releases/Python/3.12.11-531c6/x86_64-el9-gcc13-opt/bin`
first on PATH, it reproduced the reported fatal `init_sys_streams` /
`cannot import name 'text_encoding' from 'io'` error. Both SNDSW release setup
scripts capture the incoming interpreter's sys.path, so changing only the
release date does not resolve the contamination.

The isolation change already present in the worktree was preserved and tested:
start a clean child environment before SNDSW activation, retain credential and
scratch variables, use the existing pinned converter Python, and print the
actual interpreter/stdlib. On lxplus9111 it passed ROOT and SndlhcGeo imports
with Python 3.9.25, `/usr/lib64/python3.9/io.py`, ROOT 6.36.04 and no LHCb paths
in sys.path. No source data, geometry or event conversion was opened/run.

Evidence (ignored worktree-local files):
`feature_dev/output/python-env-audit-20261117/` contains `probe.sh`, a snapshot
of the old wrapper with only its repository root adjusted, `probe.log`,
`old-lhcb.log`, `new-lhcb.log`, and `contaminated.log` for the additional
PYTHONHOME/PYTHONPATH/stale-SNDLHC-state probe. Each child was limited to 30 CPU
seconds, 3 GiB virtual memory and 90 seconds wall time; core dumps disabled.
No batch jobs or shared-environment changes. Coworker-account imports and EOS
permissions remain to be verified independently.

## 2026-11-19 — Audited luminosity normalization

Environment: lxplus9120, LCG107 Python 3.11.9, NumPy 1.26.4, uproot 5.3.11;
activated with `source snakemake/envs/env_lcg.sh`, user site and bytecode disabled.
All inputs were tiny artificial fixtures; no EOS conversion or HTCondor job.

Commands (from this checkout):

```bash
python3 -m unittest discover -s feature_dev/tests -p 'test_*.py'
python3 -m unittest discover -s evaluation_region_partitions -p 'test_unified_feature_distributions*.py'
python3 feature_dev/tests/normalization_integration.py feature_dev/output/normalization-20261119/verified-integration
python3 feature_dev/tests/integration.py
```

The normalization integration checks full shape/yield DAGs, exposure-scaled bins
and sumw2 for both algorithms, calculation-cache reuse, standalone report metadata,
identical reruns and rejection of missing audits/small-sample yields. Unit tests
include the audited ABCD reference ratios (ve 14640, vm 13350 fb^-1), shared CC/NC
exposure, empty sources, source deduplication, missing/overlapping shards, sampling
and cap checks. These are arithmetic comparisons, not execution of the ABCD
branch or a physical coverage qualification.

Logs: `feature_dev/output/normalization-20261119/{unit,production-tests,verified-integration,legacy-integration}.log`.
The persistent four-event report is
`feature_dev/output/normalization-20261119/verified-integration/results/mc-neutrino/yield/plot/report.html`.
Its sibling `shape/` report uses the same calculated features; `yield/histogram/histograms.json`
contains counts, normalized bins, sumw2, flow bins and exposure metadata.
The audit and prepared fixture are in `verified-integration/`.

Limits: exposure audit completeness and external reference-data coverage are
supplied by the audit producer; they are not independently verified against EOS.
Full automatic plugin processing, production audit ingestion and campaign-scale
validation remain outside this phase. The bounded prepared-input guard remains.

Final verification 2026-11-19T10:53:39+01:00: 70 development unit tests and
11 production histogram tests passed; the normalization integration passed and
all 10 existing launcher integration checks passed. `git diff --check` passed.
An added ABCD reference fixture initially summed to 14630 rather than 14640;
its nonempty-source test luminosity was corrected to 14635 plus the existing
5 fb^-1 empty-source contribution, then the complete unit suite passed.

## 2026-11-19 — Exposure metadata in published feature ROOT files

Implementation adds a Snakemake export stage that copies validated feature trees
and writes source_exposure/normalization TTrees, the canonical exposure audit and
its SHA-256, and metadata version/unit/sample/calculation hashes. Published file:
`<run>/features.root` (link to `export/features.root`). Internal calculation files
remain reusable and contain no normalization metadata. Export validation compares
every copied branch/entry and every metadata field against the calculation/plan.

Environment: lxplus9120, LCG107 Python3.11.9, uproot5.3.11, NumPy1.26.4;
`source snakemake/envs/env_lcg.sh` in the command shell with user-site/bytecode off.
Commands:

```bash
python3 -m unittest discover -s feature_dev/tests -p 'test_*.py'
python3 feature_dev/tests/normalization_integration.py feature_dev/output/root-exposure-20261119/integration
FEATURE_DEV_SNAKEMAKE=/afs/cern.ch/work/z/zhibin/venvs/shower-snakemake-9.17.2-lcg107/bin/snakemake python3 feature_dev/tests/integration.py
```

New tests cover ROOT round trips, source/normalization values, zero-selected and
external reference sources, data weight1 at effective exposure, explicit NaN/false
shape metadata, embedded audit checksum, corrupted metadata rejection, exposure
changes without recalculation and preservation of previously published files.
The existing launcher suite covers default shape runs and cache/schema behavior.

Evidence: `feature_dev/output/root-exposure-20261119/{unit,integration,legacy-integration}.log`.
Standalone four-event yield fixture:
`feature_dev/output/root-exposure-20261119/integration/results/mc-neutrino/yield/features.root`.
Its `export/validation.json` records metadata and unchanged-feature checks.
The sibling shape fixture contains explicit invalid/unknown exposure metadata.
The changed-exposure fixture is intentionally corrupted at the end of the test
to exercise cache tamper detection; use the original yield fixture for inspection.
No EOS conversion or batch jobs; no physical exposure/coverage qualification.

Final verification 2026-11-19T11:21:51+01:00: 75 development unit tests passed;
extended normalization integration passed, including ROOT publication, exposure
refresh, unchanged calculation timestamp and detection of the deliberately
corrupted exported ROOT file. All 10 existing launcher integration checks passed.
`git diff --check` passed. All local test processes completed.

## 2026-11-19 — Production converter plugins

Evidence directory (ignored): `feature_dev/output/production-plugin-20261119/`.
LCG 107 on lxplus9102: Python 3.11.9, ROOT 6.34.02, NumPy 1.26.4,
uproot 5.3.11. Isolated SNDSW: Python 3.9.25, ROOT 6.36.04.

- `python3 -m unittest discover -s feature_production/tests -p 'test_*.py'`:
  20 passed (`unit.log`), including plugin validation/isolation, known density,
  ROOT round trip, metadata checksum, shard compatibility and exposure copy,
  streaming audited yields/empty sources, capped plans and output isolation.
- Existing `feature_dev/tests`: 75 passed (`dev-unit.log`). Existing converter
  `test_unified_shower_features.py`: 13 passed; shard validator tests: 3 passed
  (`production-regression.log`).
- Actual EOS command: `./run_shower_features.sh --dataset mc-neutrino --plugin
  silvia --sample small --files-per-group 1 --events-per-file 2 --run smoke
  --output-root "$PWD/feature_dev/output/production-plugin-20261119/results"`.
  Both source jobs, validators and report passed (`smoke.log`). Four events;
  all 13 plugin scalars equal coworker commit d8cd5d0 on identical exported
  hits. Separate source-0 no-plugin conversion: all 438 original branches on
  two entries exactly match (`parity.json`, `parity.log`, `baseline-check/`).
- Identical rerun: nothing to do; cache verified (`reuse.log`).
- Full MC shape DAG dry-run: 1866 source jobs plus distribution/aggregate,
  no execution (`full-dry-run.log`). Initial small dry-run exposed a duplicate
  plan keyword; fixed before successful dry-run and real smoke.
- Report: `results/mc-neutrino/smoke/plots/report.html`; ROOT:
  `results/mc-neutrino/smoke/sources/{0,1}/features.root`, each with validation,
  baseline parity, input fingerprints and checksums. Source logs under `logs/`.

No HTCondor submission or full conversion, no coworker-account execution, and
no new real testbeam/data/muon run. Full per-file resource requirements and
physical audit qualification remain unverified. Unit yield fixtures validate
accounting, not the physical provenance of real luminosities. Small sample
exposure remains explicitly invalid for yields. No input datasets modified.

## 2026-11-19 — PDF production reports

Production report renderer now writes a vector multipage `plots/report.pdf`
instead of HTML; launcher output and production guide updated. Individual PNGs
and histogram/coverage/normalization JSON remain available. Re-rendered existing
four-event real MC smoke histograms into the new ignored evidence directory
`feature_dev/output/production-pdf-20261119/` without changing the original run.
`pdfinfo report.pdf` confirms 61 pages (one summary + 60 plot strata), matching
the histogram-derived expected count. Page 2 rasterized with `pdftoppm` and
visually inspected: title, geometry, axes, legend and count/normalization footer
readable without overlap. LCG107 Python3.11.9/matplotlib3.8.3 on lxplus9102.
All 20 production plugin/campaign tests passed (`unit.log`); git diff --check
passed. No new detector conversion, physics recalculation or batch submission.

## 2026-11-19 — Existing ROOT backend and automatic ranges

Replaced the standalone production Matplotlib renderer with shared EM-shower
ROOT filling/drawing: make_group_hist.py, make_group_hist2d.py,
plot_comparison.py and plot_group_hist2d.py. Added explicit plot config files,
range resolution, selection/geometry strata, per-feature/pair PDFs and CSV,
audited 2D yields, and compatible source/histogram reuse.

Evidence: `feature_dev/output/root-backend-20261119/`. Environment lxplus9102,
LCG107/Python3.11.9/ROOT6.34.02/NumPy1.26.4/uproot5.3.11. Detector conversions
used the existing isolated SNDSW wrapper.

- Production plugin/campaign/ROOT tests: `python3 -m unittest discover -s
  feature_production/tests -p 'test_*.py'` (unit.log). Covers range extremes,
  constants, empty samples, integer bounds, invalid/log bounds, missing and
  nonfinite values, paired-axis validity, missing branch rejection, class cuts,
  geometry/testbeam label preservation, parent coverage and yield scales,
  direct calls into both existing plotters and two-page 2D PDFs. Legacy 1D/2D
  histogram entry points match the new workflow's counts and variances exactly
  on the same fixture. Final test count/result recorded in the task checkpoint.
- `python3 -m unittest discover -s evaluation_region_partitions -p
  'test_em_shower*workflow.py'`: 14 passed (legacy.log).
- Existing MC ROOT reuse: `./run_shower_features.sh --dataset mc-neutrino
  --plugin silvia --sample small --files-per-group 1 --events-per-file 2
  --run root-backend-reuse-20261119 --reuse-sources
  /afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/mc-neutrino/mc-small-20261119-143325`:
  four stages passed (reuse.log), no source conversions.
- Fresh actual EOS command: same dataset/plugin/caps, `--run
  root-backend-small-20261119`, without --reuse-sources. Six tasks completed;
  two ROOT files/two entries each, 13 plugin branches, source validators and
  baseline parity passed (small.log). The controller tool reported exit143
  after its final success/path messages; identical rerun returned exit0,
  nothing to do, with all receipts validated (repeat.log).
- Colour-only override (`style.yaml`, CC_numu orange), new run
  root-backend-style-20261119 with --reuse-sources pointing to the fresh run:
  only plots/all scheduled; sources and histogram products reused (style.log).
- Final-code reuse run root-backend-final-20261119 completed four stages
  (final.log); subsequent small fix to paired-cut missing accounting is covered
  by its dedicated unit test, without changing default MC pair selections.
- `--sample full --normalization shape --run root-backend-full-preview
  --dry-run`: 1866 source tasks plus ranges/histograms/plots/all planned;
  no submitted jobs or campaign output directory (full-dry-run.log).
- Independent verifier (`verify_outputs.py`, verification.json) validated all
  receipts, four source events and identities, 192 ROOT histograms, raw sumw2,
  valid-entry conservation, 48 PDFs (46 one-page 1D, two two-page 2D), CSV scales
  and unit integrals, immutable source reuse and histogram reuse. All passed.
- Rasterized 1D and 2D density pages inspected: ROOT colours/markers/legends,
  selection labels and empty-class annotations readable. Preview images one-d.png
  and two-d.png. AFS home/work alias is not used for new output routing.

Default output root:
`/afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/mc-neutrino/`.
Fresh run outputs under `root-backend-small-20261119/`; reused-source runs keep
ROOT files in their source run and record that location in plan.json. Example
PDF paths: `plots/1d/count_scifi/selected/geo_c4519b199b177c78/plot.pdf` and
`plots/2d/total_qdc_vs_hits/selected/geo_c4519b199b177c78/plot.pdf`.

Limits: four events validate software behavior, not physics distributions.
Actual testbeam/data account access, full processing resources/HTCondor, and
physical luminosity audits remain unqualified. Testbeam grouping and yields
are tested with fixtures. No EOS inputs or previous published runs modified.

## 2026-11-19 — Combine geometries in production plots

Removed geometry directories from histogram and PDF paths. Common ranges span
all geometries; raw histograms are added before shape normalization. Yield
histograms add independently exposure-weighted geometry components, preserving
Sumw2 and original geometry/audit metadata. The shared ROOT plotting backend is
unchanged in style; 2D profiles remain binned raw-event means.

Validation on lxplus9102, LCG107 Python3.11.9/ROOT6.34.02:

- `python3 -m unittest discover -s feature_production/tests -p 'test_*.py'`:
  31 passed. New two-geometry fixture verifies raw count4, weighted yield1.2
  and variance0.4 with different exposure scales; existing backend parity,
  testbeam, campaign and plugin checks pass.
- `./run_shower_features.sh --dataset mc-neutrino --plugin silvia --sample small
  --files-per-group 1 --events-per-file 2 --run combined-geometry-20261119
  --reuse-sources /afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/mc-neutrino/root-backend-small-20261119`:
  exit0, four of four tasks complete, existing two source ROOTs/four events reused.
- Independent receipt and uproot checks: 24 PDFs, 96 ROOT histograms, zero
  geo_* directories, count_scifi class counts 1/2/1/0 and matching raw variances.
  All config/histogram/plot receipts passed. `git diff --check` passed.

Evidence: `feature_dev/output/combined-geometry-20261119/{unit.log,run.log,verification.json}`.
Output: `/afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/mc-neutrino/combined-geometry-20261119/`,
including `plots/1d/count_scifi/selected/plot.pdf` and
`plots/2d/total_qdc_vs_hits/selected/plot.pdf`.

Limits: tiny MC sample checks software only; luminosity correctness uses fixtures,
not a new physical audit. No full campaign, HTCondor submission or coworker-account
qualification. Existing outputs are retained.

## 2026-11-19 — Larger MC plotting test

Command: `./run_shower_features.sh --dataset mc-neutrino --plugin silvia
--sample small --files-per-group 2 --events-per-file 500 --run larger-mc-20261119`.
Dry-run and --check-access passed. Actual run exit0, eight tasks complete using
sequential local Snakemake (one core), no HTCondor jobs. Implementation cd13b6c;
launch HEAD49afb76. Existing isolated SNDSW converter and LCG107 analysis used.

Four files contained 394 selected events (70,62,127,135); no file reached the
500-event cap. Class counts: CC_nue97, CC_numu199, NC_nue35, NC_numu63. Source
validation and baseline parity passed, including entrywise selected identity.
Independent checks passed source/stage receipts, all 13 plugin branches,
histogram entry conservation and raw variances, shape normalization and PDF
page counts. Produced96 ROOT histograms and24 PDFs (23one-page1D, one two-page2D),
without geometry directories.

Output: `/afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/mc-neutrino/larger-mc-20261119/`.
Evidence: `feature_dev/output/larger-mc-20261119/` holds dry-run.log, preflight.log,
run.log, verify.py and verification.json. Verification command in activated LCG:
`python3 -m feature_dev.output.larger-mc-20261119.verify`.

This is a four-file shape comparison, not full-dataset yield validation or a
representative-sampling claim. No new code changes, full production, batch
qualification or coworker-account test.

## 2026-11-20 — Official HTCondor profile integration

Implementation: `db10611` (batch setup, adapters, CLI selection and instructions).
No full-campaign submission. Evidence: `feature_dev/output/batch-20261120/`.

- LCG107/Python3.11.9 on lxplus9105; fresh private test environment at
  `feature_dev/output/batch-20261120/venv`, profile at sibling `profile/`.
  Existing environment unchanged; ignored checkout configuration now selects
  the test environment/profile. Package pins: Snakemake9.17.2, Cookiecutter2.7.1,
  htcondor25.0.14, cluster-generic executor1.0.9.
- `./run_shower_features.sh setup --batch --prefix "$PWD/feature_dev/output/batch-20261120/venv" --batch-profile "$PWD/feature_dev/output/batch-20261120/profile" --batch-jobs 2`:
  initial attempt exposed unavailable legacy `htcondor` import. Corrected to
  `htcondor2`; retry passed, including pip check. Setup logs retained.
- `./run_shower_features.sh setup --batch --check` and repeat `setup --batch`:
  passed, existing profile preserved. `check.log`, `repeat.log`.
- `python3 -m unittest discover -s feature_production/tests -p 'test_*.py'`:
  40 passed, including9 batch tests: profile selection/version precedence,
  rejection cases, customization preservation, resources, job IDs/status and
  read-only checks. `unit.log`.
- `python3 -m unittest discover -s feature_dev/tests -p test_setup_env.py`:
  five passed. `setup-unit.log`.
- Full dry-run: `./run_shower_features.sh --dataset mc-neutrino --plugin silvia
  --sample full --normalization shape --run batch-full-preview-20261120 --dry-run`:
  exit0, saved cluster-generic profile loaded,1866 sources plus aggregate stages;
  no campaign products/jobs. `full-dry-run.log`.
- Canary access: `--sample small --files-per-group 1 --events-per-file 2 --batch
  --jobs 2 --run batch-canary-20261120 --check-access`: passed input/SNDSW,
  Kerberos and scheduler WRITE checks. `access.log`.
- Actual canary: same command without `--check-access`, controller `canary.log`.
  Sources16608353/16608354 completed on separate CERN workers with two validated
  events each and13 Silvia branches. Ranges16608356 completed; histogram16608357
  submitted. Source receipts independently verified. Final results follow below.

The official template is pinned at a5b7e6c73d9b9305abe4a04a24056050461627d3;
its generated originals remain in profile/upstream. Adaptations supply shared
filesystem execution, per-account logs, explicit interpreter, cancellation and
minutes-to-seconds runtime mapping. Actual source job ads confirm3600-second
MaxRuntime,1 CPU,5120000KB disk and site-rounded3000MB memory (requested2500MB).
Logs/job IDs/submission specs live in venv/htcondor_logs/<token>/.

Final canary result: exit0, six of six DAG tasks completed (five HTCondor jobs
plus local aggregate). Clusters16608353/16608354 (sources),16608356 (ranges),
16608357 (histograms),16608359 (plots) all completed. Targeted `condor_q` returned
no remaining rows. `python3 -m feature_dev.output.batch-20261120.verify` passed:
all source/config/histogram/plot receipts, four events,13 plugin branches,
96 raw histograms with count/variance conservation,24 PDF page checks and shape
normalization. Output is
`/afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/mc-neutrino/batch-canary-20261120/`.
Independent evidence: `verify.py`, `verification.json`. Identical repeat exit0,
"Nothing to be done", no new submissions (`canary-repeat.log`).

Additional cancellation test uses a mocked scheduler: removes only the cluster
in its recorded receipt, rejects a different cluster without a scheduler call.
Focused final batch tests:10 passed (`unit-batch-final.log`); this adds one new
case to the earlier40 production tests. No live jobs cancelled.

Limits: worker qualification is on Zhibin's account and a four-event MC sample;
coworker credentials/access, full-scale resource bounds and physical yield audits
remain unqualified. No full1866-source campaign launched. The private test
profile/environment are selected only by ignored files in this checkout; the
coworker runs setup to create her own. Runtime/cancellation adapters are maintained
locally on top of the pinned official profile, whose unmodified generated files
are retained for inspection. README local links and git diff --check passed.

## 2026-11-25 — Testbeam MC validation repair

Implementation `89685db3f68a9e1833bec56d1371fd12068c22db`. Shared MC manifest now contains the selected
counts from frozen `master_sources.csv`. All75 raw headers and identity cutflows
were audited before atomic publication;33784 selected events. Original fields
and row order preserved; backup and hashes in
`output/testbeam-fix-20261125/catalogue-repair.json`.

Validation:44 production tests,13 converter tests,4 final focused tests passed.
Focused regressions cover1200 selected versus18000 raw entries, nominal pion
energy, wrong PDG/energy rejection, missing-selection aggregation, and failure
before catalogue publication. Production uses the existing converter beam
tolerance only for nominal MC energy; truth values are not replaced.

Command (13MC groups,20events each,4 concurrent batch jobs):
```bash
./run_shower_features.sh --dataset testbeam --plugin silvia \
  --dataset-config feature_dev/output/testbeam-fix-20261125/testbeam-mc.json \
  --sample small --files-per-group 1 --events-per-file 20 \
  --batch --jobs 4 --run mc-validation-20261125
```

Result: exit0,17/17 DAG steps through16 actual HTCondor jobs; all13 source
validators and parity checks passed,260events. Independent comparison of all260
saved PDGs/energies against raw MC passed (exact PDG, energy rtol1e-9/atol1e-7).
Output: `/afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/testbeam/mc-validation-20261125/`;
299histogram ROOTs and23PDFs, matching every resolved plot. Stage receipts pass.
Initial verifier assumed24PDFs from previous campaign; corrected expectation to
resolved configuration, no missing output.

Evidence/logs: `output/testbeam-fix-20261125/` (canary.log, *tests.log, access.log,
source-verification.json, preview-verification.json, jobs.json). Clusters: 16621100, 16621101, 16621102, 16621103, 16621104, 16621105, 16621108, 16621109, 16621110, 16621111, 16621114, 16621115, 16621116, 16621124, 16621127, 16621130.
All completed; targeted condor_q empty. No full conversion or coworker-account
validation. Use a new run name after catalogue/code updates.

### 2026-11-26 — Configurable fixed SciFi timing windows

Implementation: `a454ef3542fbde1c8c4988c9fe6d22f031f67722`.
Verified 2026-11-26T17:20:56+01:00, lxplus939; LCG107 Python3.11.9 ROOT6.34.02 and isolated
SNDSW Python3.9.25 ROOT6.36.04. Shared installation unchanged.

- Production suite:53 tests passed, including6 timing tests.
- Converter suite:13 tests passed; shard suite:4 passed.
- Actual installed SNDSW synthetic-hit test:1 passed (default parity, wider
  window, corrected data and MC). Total71 tests.
- MC-muon and Hit3D converter `--help` smoke checks passed; original21-rule
  Snakemake file parsed/compiled without running catalogue generation.
- Three real-source HTCondor previews completed7/7 workflow steps each,
  max2concurrent jobs;18jobs total, all exit0, final targeted condor_q empty.
  Each used the same3sources x20events: H8data,2024data and2024MC.
- Data upper edges1.2/1.8/2.0, lower edge-0.5; MC fixed[-0.5,1.2].
  All9 source validation/parity receipts and all stage checksums passed.
- Event identities preserved; wider data windows contain every baseline hit.
  MC arrays identical across runs. Default MC20events x451branches exactly
  equal the pre-change mc-validation-20261125 output (zero tolerance).
- Each run produced26 configured PDFs and78histogram ROOT files. All paths and
  checksums verified; pdfinfo opens one1D and one2D PDF from each run.

| Source (20events each) | Hits at upper1.2 | upper1.8 | upper2.0 |
| --- | ---: | ---: | ---: |
| 2023 H8 data | 1 | 1 | 1 |
| 2024 data | 299 | 300 | 300 |
| 2024 MC | 22884 | 22884 | 22884 |

The added2024data hit has zero QDC; summed QDC/density unchanged in this tiny
sample. H8 is especially sparse. This validates configuration and execution,
not the optimal physics window,7pe threshold, adaptive selection or full
selection efficiency. No real Hit3D or muon-MC conversion campaign was run;
those paths have helper/unit, parser and syntax coverage. Upstream selections
and historical clock periods/extra25ns cuts were preserved. Defaults unchanged.

Reproduce a preview (substitute1.2/1.8/2.0 and a fresh run name):

```bash
./run_shower_features.sh --dataset testbeam --plugin silvia \
  --dataset-config feature_dev/output/timing-selection-20261126/testbeam.json \
  --sample small --files-per-group 1 --events-per-file 20 --batch --jobs 2 \
  --scifi-data-window-cycles -0.5 2.0 --scifi-mc-window-cycles -0.5 1.2 \
  --run timing20-20261126
```

Outputs: `/afs/cern.ch/work/z/zhibin/snd-ml-outputs/shower-features/testbeam/timing{12,18,20}-20261126/`.
Evidence: `feature_dev/output/timing-selection-20261126/` contains copied
catalogues/config, tests, batch logs, default-parity.json, comparison.json,
verify_comparison.py, jobs.json, final-queue.txt and pdfinfo reports. Each job's
cluster/token/log directory is in jobs.json. No new full campaign or push.
