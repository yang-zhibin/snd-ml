"""Self-contained exposure metadata for the published feature ROOT product.

Unknown luminosities/weights use NaN plus explicit validity flags. Reference
sources are included but labelled separately from sources in the feature sample.
"""
from collections import Counter
import hashlib
import json

SOURCE_TYPES = dict(source_uid='string', population='string', role='string',
                    in_sample='bool', exposure_valid='bool', luminosity_fb='float64',
                    sampling_probability='float64', effective_luminosity_fb='float64',
                    selected_entries='int64')
NORMALIZATION_TYPES = dict(group='string', geometry='string', population='string',
                           target_population='string', mode='string', exposure_valid='bool',
                           weight_valid='bool', source_exposure_fb='float64',
                           target_exposure_fb='float64', event_weight='float64')


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def metadata_records(plan, events):
    """Build typed metadata from an already validated run plan."""
    audit = plan.get('exposure_audit')
    sources = []
    if audit is not None:
        for name, population in sorted(audit['populations'].items()):
            for source in sorted(population['sources'], key=lambda s: s['source_uid']):
                lumi, probability = source['luminosity_fb'], source['sampling_probability']
                sources.append(dict(source_uid=source['source_uid'], population=name,
                    role=population['role'], in_sample=population['role'] != 'reference_data',
                    exposure_valid=True, luminosity_fb=lumi, sampling_probability=probability,
                    effective_luminosity_fb=lumi * probability, selected_entries=source['selected_entries']))
    else:
        for uid, count in sorted(Counter(e['source_uid'] for e in events).items()):
            sources.append(dict(source_uid=uid, population='', role='unknown', in_sample=True,
                exposure_valid=False, luminosity_fb=float('nan'), sampling_probability=float('nan'),
                effective_luminosity_fb=float('nan'), selected_entries=count))
    groups = {(g['group'], g['geometry']): g for g in (audit or {}).get('groups', [])}
    normalizations = []
    for record in plan['normalization']:
        valid = record['mode'] == 'yield'
        normalizations.append(dict(group=record['group'], geometry=record['geometry'],
            population=record.get('population', ''),
            target_population=groups.get((record['group'], record['geometry']), {}).get('target_population', record.get('population', '')),
            mode=record['mode'], exposure_valid=valid, weight_valid=valid,
            source_exposure_fb=record.get('source_exposure_fb', float('nan')),
            target_exposure_fb=record.get('target_exposure_fb', float('nan')),
            event_weight=record.get('scale', float('nan'))))
    return sources, normalizations


def metadata_strings(plan, raw_sha):
    audit = canonical_json(plan.get('exposure_audit'))
    return dict(exposure_audit_json=audit,
                exposure_audit_sha256=hashlib.sha256(audit.encode('utf-8')).hexdigest(),
                feature_metadata_json=canonical_json(dict(version=1,
                    audit_version=(plan.get('exposure_audit') or {}).get('version'),
                    exposure_unit='fb^-1', event_weight_unit='dimensionless',
                    sample_sha256=plan['source_hashes'][plan['sample']],
                    calculated_root_sha256=raw_sha,
                    coverage_status='producer_audited' if plan.get('exposure_audit') else 'unavailable')))


def write_metadata(path, plan, events, raw_sha):
    import numpy as np
    import uproot
    sources, normalizations = metadata_records(plan, events)
    with uproot.update(path) as root:
        for name, types, rows in [('source_exposure', SOURCE_TYPES, sources),
                                  ('normalization', NORMALIZATION_TYPES, normalizations)]:
            root.mktree(name, types)
            if rows:
                root[name].extend({key: ([r[key] for r in rows] if dtype == 'string' else
                                        np.asarray([r[key] for r in rows], dtype=dtype))
                                   for key, dtype in types.items()})
        for name, value in metadata_strings(plan, raw_sha).items():
            root[name] = value


def validate_metadata(path, plan, events, raw_sha):
    """Round-trip every metadata field and audit checksum against the run plan."""
    import numpy as np
    import uproot
    sources, normalizations = metadata_records(plan, events)
    with uproot.open(path) as root:
        for name, types, rows in [('source_exposure', SOURCE_TYPES, sources),
                                  ('normalization', NORMALIZATION_TYPES, normalizations)]:
            if root[name].classname != 'TTree' or root[name].num_entries != len(rows):
                raise ValueError(f'Invalid {name} metadata tree')
            arrays = root[name].arrays(library='np')
            for key in types:
                np.testing.assert_array_equal(arrays[key], [r[key] for r in rows])
        for name, value in metadata_strings(plan, raw_sha).items():
            if str(root[name]) != value:
                raise ValueError(f'ROOT metadata mismatch: {name}')
