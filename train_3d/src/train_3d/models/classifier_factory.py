"""Construct one of the two frozen model families from resolved config."""

from train_3d.models.pyg_gravnet_classifier import PyGGravNetClassifier
from train_3d.models.clustered_pyg_gravnet_classifier import (
    ClusteredPyGGravNetClassifier,
)
from train_3d.models.sparse_resnet_classifier import SparseResNetClassifier
from train_3d.models.two_view_sparse_resnet_classifier import (
    TwoViewSparseResNetClassifier,
)


def build_classifier(config: dict, num_classes: int):
    family = str(config.get("family", ""))
    kwargs = dict(config)
    kwargs.pop("family", None)
    kwargs["num_classes"] = int(num_classes)
    if family == "pyg_gravnet":
        return PyGGravNetClassifier(**kwargs)
    if family == "clustered_pyg_gravnet":
        return ClusteredPyGGravNetClassifier(**kwargs)
    if family == "sparse_resnet":
        return SparseResNetClassifier(**kwargs)
    if family == "native_scifi_gravnet":
        return PyGGravNetClassifier(**kwargs)
    if family == "two_view_sparse_resnet":
        return TwoViewSparseResNetClassifier(**kwargs)
    raise ValueError(f"Unsupported model family: {family!r}")
