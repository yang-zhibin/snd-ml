# Current project status

## Accepted or authoritative

- The accepted Run-8329 detector-ID transform is V4. V2/V3 are development
  history and V5 is an experimental reconstruction-free comparison.
- The initial seven-class training order and sampling policy are frozen.
- The data-driven five-fold dataset catalogue and validation pass.
- The no-Veto real-data signal region is kept separate from training
  background catalogues.

## Implemented

- Event-level feature and Hit3D/native-SciFi conversion.
- Metadata-driven Snakemake and HTCondor workflows.
- Region-v1 partitioning and passing-muon studies through V8.
- PyG GravNet, sparse ResNet, native-hit, two-view, and clustered-model code.
- Reproducible training-container and LCG-overlay infrastructure.

## In progress or pending production

- Full-scale SciFi preprocessing metadata construction.
- Production GPU qualification and five-fold classifier training.
- Merged evaluation and physics-performance reporting.
- Region-v2 definition and the final multi-region statistical workflow.

## Generated status reports

- [Dataset diagnostics](../../snakemake/metadata/ml_dataset/data_driven_3d_cnn_v1/diagnostics/dataset_report.md)
- [Classifier architecture report](../../snakemake/metadata/ml_dataset/data_driven_3d_cnn_v1/model_report/README.md)
- [Container validation status](../../train_3d/test_outputs/container_validation_eca898623c12ae80/reports/validation_status.md)

Update this file only for milestone-level changes. Feature handoff state belongs
under `docs/tasks/`.
