"""Frozen seven-class label contract for data-driven 3D classification."""

from __future__ import annotations


TRAINING_CLASS_NAMES = (
    "CC_numu",
    "CC_nue",
    "NC",
    "neutral_kaon",
    "neutron",
    "passing_muon",
    "muon_DIS",
)
EXCLUDED_CLASS_NAMES = ("CC_nutau",)
TRAINING_LABELS = {
    class_name: label for label, class_name in enumerate(TRAINING_CLASS_NAMES)
}


def make_training_labels(class_names=TRAINING_CLASS_NAMES,
                         excluded_classes=EXCLUDED_CLASS_NAMES) -> dict[str, int]:
    names = tuple(str(name) for name in class_names)
    excluded = {str(name) for name in excluded_classes}
    if not names:
        raise ValueError("At least one training class is required")
    if len(set(names)) != len(names):
        raise ValueError("Training class names must be unique")
    overlap = set(names) & excluded
    if overlap:
        raise ValueError(f"Classes cannot be both trained and excluded: {sorted(overlap)}")
    return {name: index for index, name in enumerate(names)}


def training_label(class_name: str, labels=None,
                   excluded_classes=EXCLUDED_CLASS_NAMES) -> int:
    class_name = str(class_name)
    labels = TRAINING_LABELS if labels is None else labels
    if class_name in labels:
        return int(labels[class_name])
    if class_name in set(excluded_classes):
        raise ValueError(f"Class {class_name!r} is explicitly excluded from training")
    raise ValueError(f"Class {class_name!r} is not in the training class contract")


def validate_model_classification_config(config: dict) -> dict[str, int]:
    if str(config.get("task", "")) != "classification":
        raise ValueError("Training task must be 'classification'")
    names = tuple(config.get("class_names", ()))
    excluded = tuple(config.get("excluded_classes", ()))
    labels = make_training_labels(names, excluded)
    if int(config.get("num_classes", -1)) != len(labels):
        raise ValueError("num_classes must equal the number of class_names")
    if names != TRAINING_CLASS_NAMES:
        raise ValueError(
            f"Class order must be {list(TRAINING_CLASS_NAMES)}, found {list(names)}"
        )
    if excluded != EXCLUDED_CLASS_NAMES:
        raise ValueError(
            f"Excluded classes must be {list(EXCLUDED_CLASS_NAMES)}, "
            f"found {list(excluded)}"
        )
    return labels


def validate_scifi_classification_config(config: dict) -> dict[str, int]:
    labels = validate_model_classification_config(config)
    data = config.get("data", {})
    family = config.get("model", {}).get("family")
    native_families = {"native_scifi_gravnet", "two_view_sparse_resnet"}
    if family in native_families:
        expected_fields = (
            "scifi_native_station", "scifi_native_orientation",
            "scifi_native_channel", "scifi_native_det_id", "scifi_native_x",
            "scifi_native_y", "scifi_native_z", "scifi_native_qdc",
            "scifi_native_time_ns",
        )
    else:
        expected_fields = ("hit_x", "hit_y", "hit_z", "hit_qdc", "hit_detType")
    if family == "clustered_pyg_gravnet":
        expected_fields = (*expected_fields, "hit_station")
    if tuple(data.get("raw_input_fields", ())) != expected_fields:
        raise ValueError(f"raw_input_fields must be exactly {list(expected_fields)}")
    if list(data.get("allowed_detector_types", ())) != [1]:
        raise ValueError("The initial representation must select only SciFi detType 1")
    if data.get("class_probabilities") != "uniform":
        raise ValueError("Initial training requires uniform class sampling")
    if data.get("domain_probabilities") is not None:
        raise ValueError("Initial training requires natural within-class domain sampling")
    if data.get("muondis_density_probabilities") != "uniform":
        raise ValueError("Initial training requires uniform populated Muon-DIS density bins")
    validation_sampling = data.get("validation_sampling")
    if validation_sampling not in {"natural", "stratified_fixed"}:
        raise ValueError(
            "validation_sampling must be 'natural' or 'stratified_fixed'"
        )
    if validation_sampling == "stratified_fixed":
        if int(data.get("validation_size", 0)) <= 0:
            raise ValueError("stratified_fixed validation requires validation_size > 0")
        try:
            validation_seed = int(
                data.get("validation_seed", config.get("runtime", {}).get("seed", 0))
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("validation_seed must be an integer") from exc
        if validation_seed < 0:
            raise ValueError("validation_seed must be non-negative")
    representation = config.get("representation", {})
    for key in ("include_station", "include_time"):
        if representation.get(key) is not False:
            raise ValueError(f"Initial representation requires {key}: false")
    if family == "two_view_sparse_resnet":
        if representation.get("include_occupancy_as_input") is not True:
            raise ValueError("Two-view sparse representation requires occupancy input")
    elif representation.get("include_occupancy_as_input") is not False:
        raise ValueError("Initial representation requires include_occupancy_as_input: false")
    expected_version = {
        "clustered_pyg_gravnet": "scifi_xyz_qdc_dettype_station_partition_v1",
        "native_scifi_gravnet": "scifi_native_hits_v1",
        "two_view_sparse_resnet": "scifi_two_view_detector_grid_v1",
    }.get(family, "scifi_xyz_qdc_dettype_v1")
    if representation.get("version") != expected_version:
        raise ValueError(f"Unexpected representation version for {family}")
    if family == "clustered_pyg_gravnet":
        if representation.get("metadata_version") != "scifi_xyz_qdc_dettype_v1":
            raise ValueError("Clustered representation requires its base metadata version")
        partition = representation.get("partition", {})
        if partition.get("algorithm") != "station_xy_recursive_bisection":
            raise ValueError("Unsupported clustered GravNet partition algorithm")
        if int(partition.get("max_cluster_nodes", 0)) < 1:
            raise ValueError("max_cluster_nodes must be positive")
        if int(config["model"].get("max_cluster_nodes", 0)) != int(
            partition["max_cluster_nodes"]
        ):
            raise ValueError("Model and representation max_cluster_nodes must match")
        if int(config["model"].get("max_nodes_per_forward", 0)) < int(
            partition["max_cluster_nodes"]
        ):
            raise ValueError("max_nodes_per_forward must cover one full cluster")
    expected_channels = {
        "pyg_gravnet": 5, "clustered_pyg_gravnet": 5, "sparse_resnet": 2,
        "native_scifi_gravnet": 7, "two_view_sparse_resnet": 2,
    }
    if family not in expected_channels:
        raise ValueError(f"Unsupported model family: {family!r}")
    if int(config["model"].get("in_channels", -1)) != expected_channels[family]:
        raise ValueError(
            f"{family} requires {expected_channels[family]} initial input channels"
        )
    if family == "native_scifi_gravnet":
        if representation.get("nodes") != "original_fiber_hits":
            raise ValueError("Native GravNet requires original_fiber_hits nodes")
        if representation.get("include_unmeasured_coordinate") is not False:
            raise ValueError("Native GravNet must exclude the unmeasured coordinate")
    if family == "two_view_sparse_resnet":
        if "voxel_size" in representation:
            raise ValueError("Two-view detector-grid input must not define voxel_size")
        if representation.get("coordinates") != ["station_index", "channel"]:
            raise ValueError("Two-view sparse coordinates must use station_index/channel")
        if int(config["model"].get("dimension", -1)) != 2:
            raise ValueError("Two-view sparse ResNet requires dimension=2")
    if config.get("wandb", {}).get("project") != "snd-ml-data-driven-3d":
        raise ValueError("Training must use the new snd-ml-data-driven-3d project")
    objective = config.get("objective", {})
    if objective.get("name") != "cross_entropy":
        raise ValueError("Initial objective must be cross_entropy")
    if objective.get("class_weights") is not None:
        raise ValueError("Initial cross-entropy objective must be unweighted")
    checkpoint = config.get("checkpoint", {})
    if checkpoint.get("monitor") != "val/macro_f1" or checkpoint.get("mode") != "max":
        raise ValueError("Checkpointing must maximize val/macro_f1")
    return labels
