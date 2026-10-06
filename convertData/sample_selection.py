"""Shared sample-selection policy for digi converters."""


PASSING_MUON_REAL_DATA = "passing_muon_real_data"
EXTERNAL_CUTFLOW = "external_cutflow"
IDENTITY_CUTFLOW = "identity_cutflow"
PASSING_MUON_DIRECT = "passing_muon_direct"
DIRECT_INPUT = "direct_input"
SCIFI_MINHITS_SELECTION = "SciFiMinHits == 1"
COLLISION_DATA_QUALITY_SELECTION = (
    "StableBeams == 1 && IP1 == 1 && EventDeltat_m1_100 == 1"
)
SELECTION_MODES = (
    EXTERNAL_CUTFLOW,
    IDENTITY_CUTFLOW,
    PASSING_MUON_DIRECT,
    DIRECT_INPUT,
)


def normalized_sample_type(value):
    return str(value or "").strip().lower()


def is_passing_muon_real_data(value):
    return normalized_sample_type(value) == PASSING_MUON_REAL_DATA


def resolve_selection_mode(sample_type, requested_mode=None):
    """Return an explicit mode while preserving the historical type policy."""
    mode = normalized_sample_type(requested_mode)
    if not mode:
        return PASSING_MUON_DIRECT if is_passing_muon_real_data(sample_type) else EXTERNAL_CUTFLOW
    if mode not in SELECTION_MODES:
        raise ValueError(
            f"Unknown selection mode {requested_mode!r}; expected one of {SELECTION_MODES}"
        )
    if mode == PASSING_MUON_DIRECT and not is_passing_muon_real_data(sample_type):
        raise ValueError(
            "passing_muon_direct requires sample type passing_muon_real_data"
        )
    return mode


def uses_external_cutflow(value, requested_mode=None):
    return resolve_selection_mode(value, requested_mode) in {
        EXTERNAL_CUTFLOW,
        IDENTITY_CUTFLOW,
    }


def applies_collision_data_quality(sample_type, requested_mode=None):
    """Return whether collision-only cutflow requirements must be applied."""
    return resolve_selection_mode(sample_type, requested_mode) == EXTERNAL_CUTFLOW


def is_direct_selection(sample_type, requested_mode=None):
    return not uses_external_cutflow(sample_type, requested_mode)


def skips_scifi_minhits_selection(sample_type):
    """Return whether a legacy external cutflow is intentionally inclusive."""
    return normalized_sample_type(sample_type) in {"mc_muon", "mc_muondis"}


def setup_event_deltat_alias(cutflow_tree):
    """Make the collision-data timing flag usable across sanitized ROOT names."""
    alias_name = "EventDeltat_m1_100"
    if cutflow_tree.GetBranch(alias_name) or cutflow_tree.GetLeaf(alias_name):
        return alias_name
    for candidate in ("EventDeltat_1_100", "EventDeltat_-1_100"):
        if cutflow_tree.GetBranch(candidate) or cutflow_tree.GetLeaf(candidate):
            cutflow_tree.SetAlias(alias_name, candidate)
            return candidate
    return None


def cutflow_selection_expression(
    sample_type, requested_mode=None, event_deltat_available=False
):
    """Return the frozen cutflow expression used by feature conversion."""
    mode = resolve_selection_mode(sample_type, requested_mode)
    if mode in {IDENTITY_CUTFLOW, PASSING_MUON_DIRECT, DIRECT_INPUT}:
        return None
    if skips_scifi_minhits_selection(sample_type):
        return None

    clauses = [SCIFI_MINHITS_SELECTION]
    is_mc = normalized_sample_type(sample_type).startswith("mc")
    if applies_collision_data_quality(sample_type, mode) and not is_mc:
        if not event_deltat_available:
            raise RuntimeError("No EventDeltat cut branch found for real-data selection")
        clauses.append(COLLISION_DATA_QUALITY_SELECTION)
    return " && ".join(clauses)
