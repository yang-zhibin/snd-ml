#!/usr/bin/env python3
"""Frozen selection policy for the 2024 real-data feature streams."""

from __future__ import annotations

import hashlib


SELECTION_PROFILE_VERSION = "real_data_2024_selection_v1"
NO_VETO_PROFILE = "real_data_2024_no_veto"
VETO_10PCT_PROFILE = "real_data_2024_veto_10pct"
VETO_SAMPLE_DENOMINATOR = 10
VETO_SAMPLE_BUCKET = 0


def event_identity_text(run_id: int, event_id: int, event_index: int) -> str:
    """Return the versioned canonical text hashed by the sampling policy."""
    return f"{int(run_id)}:{int(event_id)}:{int(event_index)}"


def deterministic_sample_bucket(
    run_id: int,
    event_id: int,
    event_index: int,
    denominator: int = VETO_SAMPLE_DENOMINATOR,
) -> int:
    """Map an event identity to a stable sampling bucket using SHA-256."""
    denominator = int(denominator)
    if denominator <= 0:
        raise ValueError("sampling denominator must be positive")
    digest = hashlib.sha256(
        event_identity_text(run_id, event_id, event_index).encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % denominator


def keep_veto_10pct(run_id: int, event_id: int, event_index: int) -> bool:
    """Return whether a veto event belongs to the frozen ten-percent sample."""
    return (
        deterministic_sample_bucket(run_id, event_id, event_index)
        == VETO_SAMPLE_BUCKET
    )


def has_valid_veto_hit(mu_filter_hits) -> bool:
    """Return whether an event has at least one valid Veto-system hit."""
    for hit in mu_filter_hits:
        if hit.isValid() and int(hit.GetSystem()) == 1:
            return True
    return False


def event_header_identity(event_header, event_index: int) -> tuple[int, int, int]:
    """Read the collision-data identity used by feature conversion."""
    return (
        int(event_header.GetRunId()),
        int(event_header.GetEventNumber()),
        int(event_index),
    )
