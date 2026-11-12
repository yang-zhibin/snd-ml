"""Output location policy shared by task workflows (standard library only)."""
from pathlib import Path
import re

DEFAULT_OUTPUT_BASE = Path("/afs/cern.ch/work/z/zhibin/snd-ml-outputs")

def resolve_output(value, repo_root):
    """Preserve explicit absolute paths/URIs; resolve legacy relative paths."""
    value = str(value)
    if not value or not value.strip():
        raise ValueError("Output path must not be empty")
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        return value
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else Path(repo_root) / path)

def campaign_output(campaign, run, *, configured=None, override=None,
                    base=DEFAULT_OUTPUT_BASE):
    """CLI override > explicit campaign setting > AFS campaign/run default.

    Relative overrides/settings are rejected so invocation cwd cannot redirect
    outputs. Existing EOS settings remain explicit exceptions. Callers must
    propagate the selected path to both Snakemake targets and worker arguments.
    """
    for candidate in (override, configured):
        if candidate is not None:
            candidate = str(candidate)
            if not candidate or not (Path(candidate).is_absolute() or
                    re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", candidate)):
                raise ValueError("Explicit output roots must be absolute paths or URIs")
            return candidate
    for value in (campaign, run):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
            raise ValueError("Campaign and run must be single safe path components")
    if not Path(base).is_absolute():
        raise ValueError("Output base must be absolute")
    return str(Path(base) / campaign / run)
