# Documentation migration map

The former `document/` directory has been replaced by the topic-based `docs/`
hierarchy. Filenames were preserved during the first migration.

| Previous group | Current location |
|---|---|
| Feature and hit-time notes | `docs/data/` |
| Control-region strategy | `docs/analysis/` |
| Region-v2 documents | `docs/analysis/regions/` |
| Current passing-muon references | `docs/analysis/passing_muon/current/` |
| Passing-muon studies | `docs/analysis/passing_muon/studies/` |
| Detector-ID transformation records | `docs/analysis/passing_muon/transforms/` |
| Dataset construction and metadata | `docs/ml/datasets/` |
| Model architecture documents | `docs/ml/models/` |
| Training policy | `docs/ml/training/` |
| Snakemake and evaluation workflows | `docs/workflows/` |
| Containers, overlays, and tests | `docs/operations/` |
| Proposed work | `docs/plans/proposed/` |
| Rejected or superseded designs | `docs/archive/superseded/` |

Do not add new repository-local links to the removed `document/` directory.
