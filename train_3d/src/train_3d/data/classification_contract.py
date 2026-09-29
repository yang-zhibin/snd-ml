"""Compatibility import for the renamed classification-label contract."""

from train_3d.data.classification_labels import (  # noqa: F401
    EXCLUDED_CLASS_NAMES,
    TRAINING_CLASS_NAMES,
    TRAINING_LABELS,
    make_training_labels,
    training_label,
    validate_model_classification_config,
    validate_scifi_classification_config,
)
