"""Fixed SciFi timing policy, shared by launchers and both converters.

Offsets are signed clock cycles relative to each plane's peak. The historical
SNDSW and calibrated-muon clock periods are deliberately preserved.
"""
import hashlib
import json
import math
from pathlib import Path

from sample_selection import sample_kind
from hit_time_selection import is_muon_time_calibrated_sample

DEFAULT_WINDOW = [-0.5, 1.2]
METADATA_KEY = 'scifi_timing_json'


def add_arguments(parser):
    for kind in ('data', 'mc'):
        parser.add_argument('--scifi-' + kind + '-window-cycles', type=float, nargs=2,
                            metavar=('LOWER', 'UPPER'), default=list(DEFAULT_WINDOW),
                            help='Signed offsets from each plane time peak (default: -0.5 1.2)')


def window(values):
    if len(values) != 2:
        raise ValueError('SciFi timing window requires two offsets')
    result = list(map(float, values))
    if not all(map(math.isfinite, result)) or result[0] >= result[1]:
        raise ValueError('SciFi timing offsets must be finite and lower < upper')
    return result


def settings(args):
    return {kind: window(getattr(args, 'scifi_' + kind + '_window_cycles', DEFAULT_WINDOW))
            for kind in ('data', 'mc')}


def effective_policy(sample_type, requested):
    kind = 'mc' if sample_kind(sample_type) == 'mc' else 'data'
    lower, upper = window(requested[kind])
    calibrated = is_muon_time_calibrated_sample(sample_type)
    return dict(version=1, mode='fixed', sample_policy=kind,
                lower_cycles=lower, upper_cycles=upper,
                clock_period_ns=6.25 if calibrated else 1e9 / 160.316e6,
                selector='calibrated_plane_peak' if calibrated else 'sndsw_selectHits',
                first_arrival_time0=calibrated)


def cli_arguments(requested):
    result = []
    for kind in ('data', 'mc'):
        result += ['--scifi-' + kind + '-window-cycles', *map(str, window(requested[kind]))]
    return result


def configure(args, late_mc_filter):
    """Configure only this converter process; never modify installed files."""
    from analysis.analyses.snd_analysis_2024_0mu import sciFiTools
    requested = settings(args)
    policy = effective_policy(args.type, requested)
    sciFiTools.min_clock_cycle = policy['lower_cycles']
    sciFiTools.max_clock_cycle = policy['upper_cycles']
    path = Path(sciFiTools.__file__).resolve()
    policy.update(additional_max_time_ns=25.0 if late_mc_filter(args) else None,
                  selector_source=str(path), selector_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if policy['first_arrival_time0']:
        import hit_time_selection
        path = Path(hit_time_selection.__file__).resolve()
        policy.update(selector_source=str(path), selector_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    args._scifi_timing = dict(requested=requested, effective=policy)
    return policy


def calibrated_kwargs(args):
    policy = effective_policy(args.type, settings(args))
    return dict(lower_clock_cycles=-policy['lower_cycles'], upper_clock_cycles=policy['upper_cycles'])


def write_record(root, args):
    root.TObjString(json.dumps(args._scifi_timing, sort_keys=True, allow_nan=False)).Write(METADATA_KEY)


def validate_record(root_file, sample_type, requested=None, required=False):
    obj = root_file.Get(METADATA_KEY)
    if not obj:
        if required:
            raise ValueError('Missing SciFi timing metadata')
        return None
    record = json.loads(str(obj.GetString().Data()))
    actual = record['effective']
    expected = effective_policy(sample_type, requested or record['requested'])
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError('SciFi timing metadata mismatch: ' + key)
    if actual.get('additional_max_time_ns') not in (None, 25.0):
        raise ValueError('Invalid extra MC time cut')
    if not actual.get('selector_source') or len(actual.get('selector_sha256', '')) != 64:
        raise ValueError('Missing SciFi selector provenance')
    return record
