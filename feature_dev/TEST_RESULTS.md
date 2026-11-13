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
