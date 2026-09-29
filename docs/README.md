# `snd-ml` documentation

This directory is the canonical home for authored project documentation.
Generated reports remain beside the products that created them and are linked
from the relevant topic documents.

## Start here

- [Project overview](project/OVERVIEW.md)
- [Current project status](project/CURRENT_STATUS.md)
- [Documentation conventions](project/DOCUMENTATION_CONVENTIONS.md)
- [Previous-to-current path map](project/DOCUMENT_MIGRATION_MAP.md)

## Data conversion and schemas

- [Feature-converter implementation record](data/FEATURE_FIX_PLAN.md)
- [Hit-time feature implementation](data/HIT_TIME_FEATURE_IMPLEMENTATION_PLAN.md)
- [Event-level feature schema](../convertData/FEATURE_DESCRIPTION.md)
- [Hit3D conversion notes](../convertData/HITS3D_FIX_NOTES.md)
- [Reconstructed-track schema](../convertData/RECO_TRACK_FEATURES.md)

## Physics analysis

- [Control-region and ML strategy](analysis/CONTROL_REGION_ML_STRATEGY.md)
- [Region-v2 inspection inputs](analysis/regions/REGION_V2_INSPECTION_INPUTS_PLAN.md)
- [Region-v2 veto inspection](analysis/regions/REGION_V2_VETO_INSPECTION_PLOTTER.md)
- [Passing-muon documentation index](analysis/passing_muon/README.md)

## Machine learning

- [Data-driven 3D dataset](ml/datasets/DATA_DRIVEN_3D_CNN_DATASET_IMPLEMENTATION.md)
- [Dataset pilot test report](ml/datasets/DATA_DRIVEN_3D_CNN_DATASET_TEST_REPORT.md)
- [PyG GravNet and sparse-CNN implementation](ml/models/GRAVNET_PYG_MIGRATION_AND_SPARSE_CNN_PLAN.md)
- [Native SciFi models](ml/models/NATIVE_SCIFI_GRAVNET_TWO_VIEW_SPARSE_RESNET_IMPLEMENTATION_PLAN.md)
- [Initial seven-class sampling policy](ml/training/INITIAL_7CLASS_TRAINING_SAMPLING_POLICY.md)
- [Maintained training package](../train_3d/README.md)

## Workflows and operations

- [Workflow documentation index](workflows/README.md)
- [Operations and environment index](operations/README.md)
- [Active proposals](plans/proposed/README.md)
- [Superseded documents](archive/superseded/README.md)

Documents should use one of these status labels near the top when their status
is not obvious: `authoritative`, `implemented`, `active`, `proposed`, `paused`,
`superseded`, `archived`, or `generated`.
