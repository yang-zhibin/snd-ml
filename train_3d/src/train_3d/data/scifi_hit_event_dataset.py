"""Shared ROOT-streamed SciFi event contract for graph and sparse models."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from train_3d.data.classification_labels import (
    EXCLUDED_CLASS_NAMES,
    TRAINING_CLASS_NAMES,
    make_training_labels,
    training_label,
)
from train_3d.data.data_driven_fold_dataset import EventReference
from train_3d.data.native_scifi import (
    NATIVE_SCIFI_BRANCHES,
    NATIVE_SCIFI_DTYPES,
    native_hit_counts,
    validate_native_scifi_hits,
)


SCIFI_DETECTOR_TYPE = 1
RAW_HIT_BRANCHES = {
    "x": "hit_x",
    "y": "hit_y",
    "z": "hit_z",
    "qdc": "hit_qdc",
    "detector_type": "hit_detType",
}
PARTITION_HIT_BRANCHES = {"station": "hit_station"}
IDENTITY_BRANCHES = ("runId", "eventId", "eventIndex", "pdgCode", "energy")


@dataclass(frozen=True)
class SciFiNormalization:
    """Fold-specific constants derived from training folds only."""

    version: str
    dataset_version: str
    cv_fold: int
    position_mean: tuple[float, float, float]
    position_scale: tuple[float, float, float]
    qdc_transform: str
    qdc_mean: float
    qdc_scale: float
    training_events: int
    training_candidates: int

    def __post_init__(self):
        if self.qdc_transform != "log1p":
            raise ValueError("Only qdc_transform='log1p' is supported")
        if len(self.position_mean) != 3 or len(self.position_scale) != 3:
            raise ValueError("Position normalization requires three axes")
        values = np.asarray((*self.position_mean, *self.position_scale,
                             self.qdc_mean, self.qdc_scale), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Normalization constants must be finite")
        if np.any(np.asarray(self.position_scale) <= 0) or self.qdc_scale <= 0:
            raise ValueError("Normalization scales must be positive")

    @classmethod
    def load(cls, path, expected_fold: int | None = None):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["position_mean"] = tuple(payload["position_mean"])
        payload["position_scale"] = tuple(payload["position_scale"])
        result = cls(**payload)
        if expected_fold is not None and result.cv_fold != int(expected_fold):
            raise ValueError(
                f"Normalization fold {result.cv_fold} does not match fold {expected_fold}"
            )
        return result

    def save(self, path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")

    def normalize_position(self, position: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.position_mean, dtype=np.float32)
        scale = np.asarray(self.position_scale, dtype=np.float32)
        return ((position.astype(np.float32, copy=False) - mean) / scale).astype(
            np.float32, copy=False
        )

    def transform_qdc(self, qdc: np.ndarray) -> np.ndarray:
        qdc = np.asarray(qdc, dtype=np.float32)
        if np.any(qdc < 0) or not np.isfinite(qdc).all():
            raise ValueError("SciFi QDC must be finite and non-negative")
        transformed = np.log1p(qdc)
        return ((transformed - self.qdc_mean) / self.qdc_scale).astype(
            np.float32, copy=False
        )


def validate_and_select_scifi(raw_hits: dict, include_station: bool = False) -> dict[str, np.ndarray]:
    """Validate aligned vectors and return only detector-type-one candidates."""
    missing = set(RAW_HIT_BRANCHES) - set(raw_hits)
    if missing:
        raise KeyError(f"Hit event lacks fields: {sorted(missing)}")
    arrays = {
        "x": np.asarray(raw_hits["x"], dtype=np.float32),
        "y": np.asarray(raw_hits["y"], dtype=np.float32),
        "z": np.asarray(raw_hits["z"], dtype=np.float32),
        "qdc": np.asarray(raw_hits["qdc"], dtype=np.float32),
        "detector_type": np.asarray(raw_hits["detector_type"], dtype=np.int16),
    }
    if include_station:
        if "station" not in raw_hits:
            raise KeyError("Hit event lacks fields: ['station']")
        arrays["station"] = np.asarray(raw_hits["station"], dtype=np.int16)
    if any(value.ndim != 1 for value in arrays.values()):
        raise ValueError("Hit fields must be one-dimensional vectors")
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"Hit-vector length mismatch: {sorted(lengths)}")
    for name in ("x", "y", "z", "qdc"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Hit field {name} contains a non-finite value")
    if np.any(arrays["qdc"] < 0):
        raise ValueError("SciFi QDC must be non-negative")
    selected = arrays["detector_type"] == SCIFI_DETECTOR_TYPE
    if include_station and np.any((arrays["station"][selected] < 1) | (arrays["station"][selected] > 5)):
        raise ValueError("Selected SciFi station values must be in [1,5]")
    position = np.column_stack([
        arrays["x"][selected], arrays["y"][selected], arrays["z"][selected]
    ]).astype(np.float32, copy=False)
    result = {
        "position": position,
        "qdc": arrays["qdc"][selected].astype(np.float32, copy=False),
        "detector_type": arrays["detector_type"][selected].astype(np.int16, copy=False),
    }
    if include_station:
        result["station"] = arrays["station"][selected].astype(np.int16, copy=False)
    return result


class RootSciFiEventReader:
    """Read allowed Hit3D fields with a small process-local ROOT-file cache."""

    def __init__(self, tree_name: str = "hit3D", max_open_files: int = 8,
                 include_station: bool = False, input_contract: str = "candidate"):
        if int(max_open_files) < 1:
            raise ValueError("max_open_files must be positive")
        self.tree_name = str(tree_name)
        self.max_open_files = int(max_open_files)
        self.include_station = bool(include_station)
        if input_contract not in {"candidate", "native"}:
            raise ValueError("input_contract must be 'candidate' or 'native'")
        self.input_contract = str(input_contract)
        self._files = OrderedDict()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_files"] = OrderedDict()
        return state

    def close(self) -> None:
        for root_file, _ in self._files.values():
            root_file.Close()
        self._files.clear()

    def _tree(self, path: str):
        import ROOT

        path = str(path)
        cached = self._files.pop(path, None)
        if cached is None:
            root_file = ROOT.TFile.Open(path, "READ")
            if not root_file or root_file.IsZombie():
                raise OSError(f"Cannot open ROOT file {path}")
            tree = root_file.Get(self.tree_name)
            if not tree:
                root_file.Close()
                raise KeyError(f"Tree {self.tree_name!r} is missing from {path}")
            cached = (root_file, tree)
        self._files[path] = cached
        while len(self._files) > self.max_open_files:
            _, (old_file, _) = self._files.popitem(last=False)
            old_file.Close()
        return cached[1]

    def read(self, reference: EventReference) -> dict:
        tree = self._tree(reference.hit3d_path)
        entry = int(reference.source_entry)
        if entry < 0 or entry >= int(tree.GetEntries()):
            raise IndexError(
                f"Entry {entry} is outside {reference.hit3d_path} "
                f"with {tree.GetEntries()} entries"
            )
        available = {branch.GetName() for branch in tree.GetListOfBranches()}
        branches = (
            dict(NATIVE_SCIFI_BRANCHES)
            if self.input_contract == "native" else dict(RAW_HIT_BRANCHES)
        )
        if self.include_station and self.input_contract == "candidate":
            branches.update(PARTITION_HIT_BRANCHES)
        required = set(branches.values())
        missing = required - available
        if missing:
            raise KeyError(f"Hit3D file lacks branches: {sorted(missing)}")
        if tree.GetEntry(entry) < 0:
            raise OSError(f"Failed to read {reference.hit3d_path}:{entry}")
        dtypes = {
            "x": np.float32, "y": np.float32, "z": np.float32,
            "qdc": np.float32, "detector_type": np.int16,
            "station": np.int16,
        }
        if self.input_contract == "native":
            dtypes.update(NATIVE_SCIFI_DTYPES)
        raw = {
            name: np.fromiter(getattr(tree, branch), dtype=dtypes[name])
            for name, branch in branches.items()
        }
        event = (
            validate_native_scifi_hits(raw)
            if self.input_contract == "native" else
            validate_and_select_scifi(raw, include_station=self.include_station)
        )
        for branch in IDENTITY_BRANCHES:
            if branch in available:
                value = getattr(tree, branch)
                event[branch] = value.item() if hasattr(value, "item") else value
        return event


class UprootSciFiEventReader:
    """Equivalent reader that avoids a PyROOT dependency in training images."""

    def __init__(self, tree_name: str = "hit3D", max_open_files: int = 8,
                 include_station: bool = False, input_contract: str = "candidate"):
        if int(max_open_files) < 1:
            raise ValueError("max_open_files must be positive")
        self.tree_name = str(tree_name)
        self.max_open_files = int(max_open_files)
        self.include_station = bool(include_station)
        if input_contract not in {"candidate", "native"}:
            raise ValueError("input_contract must be 'candidate' or 'native'")
        self.input_contract = str(input_contract)
        self._files = OrderedDict()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_files"] = OrderedDict()
        return state

    def close(self):
        for root_file, _ in self._files.values():
            root_file.close()
        self._files.clear()

    def _tree(self, path):
        import uproot

        path = str(path)
        cached = self._files.pop(path, None)
        if cached is None:
            root_file = uproot.open(path)
            if self.tree_name not in root_file:
                root_file.close()
                raise KeyError(f"Tree {self.tree_name!r} is missing from {path}")
            cached = (root_file, root_file[self.tree_name])
        self._files[path] = cached
        while len(self._files) > self.max_open_files:
            _, (old_file, _) = self._files.popitem(last=False)
            old_file.close()
        return cached[1]

    def read(self, reference: EventReference) -> dict:
        tree = self._tree(reference.hit3d_path)
        entry = int(reference.source_entry)
        if entry < 0 or entry >= int(tree.num_entries):
            raise IndexError(
                f"Entry {entry} is outside {reference.hit3d_path} "
                f"with {tree.num_entries} entries"
            )
        available = set(tree.keys())
        branches = (
            dict(NATIVE_SCIFI_BRANCHES)
            if self.input_contract == "native" else dict(RAW_HIT_BRANCHES)
        )
        if self.include_station and self.input_contract == "candidate":
            branches.update(PARTITION_HIT_BRANCHES)
        missing = set(branches.values()) - available
        if missing:
            raise KeyError(f"Hit3D file lacks branches: {sorted(missing)}")
        requested = [*branches.values(), *(
            branch for branch in IDENTITY_BRANCHES if branch in available
        )]
        payload = tree.arrays(
            requested, entry_start=entry, entry_stop=entry + 1,
            library="ak", how=dict,
        )
        raw = {
            name: np.asarray(payload[branch][0])
            for name, branch in branches.items()
        }
        event = (
            validate_native_scifi_hits(raw)
            if self.input_contract == "native" else
            validate_and_select_scifi(raw, include_station=self.include_station)
        )
        for branch in IDENTITY_BRANCHES:
            if branch in payload:
                value = payload[branch][0]
                event[branch] = value.item() if hasattr(value, "item") else value
        return event


def make_scifi_reader(backend="pyroot", tree_name="hit3D", max_open_files=8,
                      include_station=False, input_contract="candidate"):
    if backend == "uproot":
        return UprootSciFiEventReader(
            tree_name, max_open_files, include_station, input_contract
        )
    if backend == "pyroot":
        return RootSciFiEventReader(
            tree_name, max_open_files, include_station, input_contract
        )
    raise ValueError("reader backend must be 'uproot' or 'pyroot'")


class SciFiHitEventDataset(Dataset):
    """Return model-independent, filtered SciFi candidate arrays."""

    def __init__(self, references=None, tree_name="hit3D", max_open_files=8,
                 reader_backend="pyroot",
                 class_names=TRAINING_CLASS_NAMES,
                 excluded_classes=EXCLUDED_CLASS_NAMES, reader=None,
                 include_station=False, input_contract="candidate"):
        self.references = references
        self.training_labels = make_training_labels(class_names, excluded_classes)
        self.excluded_classes = tuple(str(name) for name in excluded_classes)
        self.include_station = bool(include_station)
        if input_contract not in {"candidate", "native"}:
            raise ValueError("input_contract must be 'candidate' or 'native'")
        self.input_contract = str(input_contract)
        self.reader = reader or make_scifi_reader(
            reader_backend, tree_name, max_open_files, self.include_station,
            self.input_contract,
        )

    def __len__(self):
        if self.references is None:
            raise TypeError("Reference-driven dataset length is defined by its sampler")
        return len(self.references)

    def _reference(self, index_or_reference):
        if isinstance(index_or_reference, EventReference):
            return index_or_reference
        if self.references is None:
            raise TypeError("SciFiHitEventDataset expects an EventReference index")
        return self.references[index_or_reference]

    def __getitem__(self, index_or_reference):
        reference = self._reference(index_or_reference)
        event = self.reader.read(reference)
        result = {
            "label": training_label(
                reference.class_name, self.training_labels, self.excluded_classes
            ),
            "source_class_id": int(reference.class_id),
            "class_name": str(reference.class_name),
            "domain": str(reference.domain),
            "density_bin": str(reference.density_bin),
            "dataset_file_uid": str(reference.dataset_file_uid),
            "source_entry": int(reference.source_entry),
            "run_id": int(event.get("runId", -1)),
            "event_id": int(event.get("eventId", -1)),
            "event_index": int(event.get("eventIndex", -1)),
            "pdg_code": int(event.get("pdgCode", -999)),
            # Metadata only; collators deliberately never place energy in x/features.
            "energy": float(event.get("energy", np.nan)),
        }
        if self.input_contract == "native":
            for name in NATIVE_SCIFI_BRANCHES:
                result[name] = event[name]
            result.update(native_hit_counts(event))
            # Retained only as legacy metadata; native collators never use it.
            result["n_candidates"] = -1
        else:
            result.update({
                "position": event["position"],
                "qdc": event["qdc"],
                "detector_type": event["detector_type"],
                "n_candidates": int(len(event["qdc"])),
            })
        if self.include_station:
            if "station" not in event:
                raise KeyError("Clustered GravNet reader did not return station")
            result["station"] = event["station"]
        return result
