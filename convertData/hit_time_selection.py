"""Shared hit-time calibration and SciFi plane-selection helpers."""

import math
from collections import Counter, defaultdict


MUON_TIME_CALIBRATED_SAMPLES = frozenset({"mc_muon", "mc_muondis"})


def normalized_sample_type(args_or_type):
    """Return a normalized declared sample type without inspecting file paths."""
    sample_type = getattr(args_or_type, "type", args_or_type)
    return str(sample_type).strip().lower()


def is_muon_time_calibrated_sample(args_or_type):
    """Whether first-arrival calibration is required for this sample."""
    return normalized_sample_type(args_or_type) in MUON_TIME_CALIBRATED_SAMPLES


def _valid_time(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > -900.0


def apply_first_arrival_time0(
    args_or_type,
    hit_collections,
    *,
    time_key,
    invalid_time=-999.0,
    raw_time_key=None,
):
    """Subtract the first detector arrival for declared muon MC samples only."""
    if not is_muon_time_calibrated_sample(args_or_type):
        return float(invalid_time), False

    all_hits = [hit for collection in hit_collections for hit in collection]
    valid_times = [
        float(hit.get(time_key, invalid_time))
        for hit in all_hits
        if _valid_time(hit.get(time_key, invalid_time))
    ]
    if not valid_times:
        return float(invalid_time), False

    time0_ns = min(valid_times)
    raw_key = raw_time_key or f"{time_key}Raw"
    for hit in all_hits:
        time_ns = hit.get(time_key, invalid_time)
        if _valid_time(time_ns):
            hit[raw_key] = float(time_ns)
            hit[time_key] = float(time_ns) - time0_ns

    return time0_ns, True


def select_calibrated_scifi_hits(
    items,
    *,
    hit_getter=None,
    time_key="hitTimeNS",
    bin_width_ns=0.5,
    ns_per_clock=6.25,
    lower_clock_cycles=0.5,
    upper_clock_cycles=1.2,
):
    """Select the main calibrated-time cluster in each SciFi plane.

    A plane is one (station, orientation) pair.  The peak uses sparse 0.5 ns
    bins, matching the bin width of SND ``selectHits`` without its fixed
    0--100 ns histogram range.  Tied peaks choose the lowest-time bin, as
    ROOT's first-maximum behavior does.

    ``items`` may be hit dictionaries or wrappers such as
    ``{"raw": root_hit, "hit": hit_dict}``; ``hit_getter`` selects the
    dictionary while the returned list preserves the original items.
    """
    if bin_width_ns <= 0:
        raise ValueError("bin_width_ns must be positive")

    get_hit = hit_getter or (lambda item: item)
    groups = defaultdict(list)

    for item in items:
        hit = get_hit(item)
        time_ns = hit.get(time_key)
        if not _valid_time(time_ns):
            continue
        key = (int(hit.get("station", 0)), bool(hit.get("isVertical", False)))
        groups[key].append((item, float(time_ns)))

    selected = []
    lower_offset_ns = float(lower_clock_cycles) * float(ns_per_clock)
    upper_offset_ns = float(upper_clock_cycles) * float(ns_per_clock)

    for group in groups.values():
        bin_counts = Counter(
            math.floor(time_ns / bin_width_ns) for _, time_ns in group
        )
        peak_bin = min(
            bin_counts,
            key=lambda bin_index: (-bin_counts[bin_index], bin_index),
        )
        peak_time_ns = (peak_bin + 0.5) * bin_width_ns
        lower_ns = peak_time_ns - lower_offset_ns
        upper_ns = peak_time_ns + upper_offset_ns
        selected.extend(
            item for item, time_ns in group if lower_ns <= time_ns <= upper_ns
        )

    return selected
