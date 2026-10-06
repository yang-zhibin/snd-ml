"""Shared sample-selection policy for digi converters."""


PASSING_MUON_REAL_DATA = "passing_muon_real_data"
EXTERNAL_CUTFLOW = "external_cutflow"
IDENTITY_CUTFLOW = "identity_cutflow"
PASSING_MUON_DIRECT = "passing_muon_direct"
DIRECT_INPUT = "direct_input"
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
