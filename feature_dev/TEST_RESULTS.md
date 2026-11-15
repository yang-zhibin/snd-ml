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
