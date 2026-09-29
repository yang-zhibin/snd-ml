"""Compatibility imports for the original data-driven module name."""

from train_3d.data.classification_datamodule import (  # noqa: F401
    AdvanceSampledEpoch,
    ClassificationDataModule,
)

DataDrivenClassificationDataModule = ClassificationDataModule
