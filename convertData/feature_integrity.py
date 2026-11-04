"""ROOT-free verification of validated and frozen feature products."""

import hashlib
import json
import re
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_feature(row, record=None):
    """Verify receipt and bytes; frozen rows additionally bind receipt replacement."""
    feature = Path(str(row["feature_scifi_cluster_v1_path"]))
    marker = Path(str(row["validation_record"]))
    if record is None:
        record = json.loads(marker.read_text())
    if not isinstance(record, dict):
        raise ValueError(f"validation record must be an object: {marker}")
    if record.get("validation_passed") is not True:
        raise ValueError(f"failed validation: {marker}")
    if str(record.get("source_uid", "")) != str(row["source_uid"]):
        raise ValueError(f"source UID mismatch: {marker}")
    if str(record.get("feature_path", "")) != str(feature):
        raise ValueError(f"feature path mismatch: {marker}")
    digest = record.get("feature_sha256", "")
    size = record.get("feature_size_bytes")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"missing or malformed feature SHA-256: {marker}")
    if type(size) is not int or size < 0:
        raise ValueError(f"missing or invalid feature size: {marker}")
    frozen = {"feature_sha256": digest, "feature_size_bytes": size,
              "validation_record_sha256": sha256(marker)}
    if "feature_unit_uid" in row:
        for key, value in frozen.items():
            if str(row.get(key, "")) != str(value):
                raise ValueError(f"frozen {key} mismatch; refresh validated snapshot: {feature}")
    if feature.stat().st_size != size or sha256(feature) != digest:
        raise ValueError(f"feature checksum/size mismatch: {feature}")
    return frozen
