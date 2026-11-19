"""Validate an explicit parent-source coverage audit before estimating yields.

This verifies consistency of the supplied audit, not the original EOS campaign.
The audit producer must enumerate the complete parent population, including
sources with zero selected events. Luminosity belongs to a parent source, never
to each shard or to a CC/NC subset separately.
"""
from collections import Counter
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation_region_partitions'))
from normalization import exposure_scale


def _positive(value):
    return (type(value) in (float, int) and math.isfinite(value) and value > 0)


def _uncapped(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('max_events_per_group', 'max_events', 'events_per_file') and item not in (0, None):
                raise ValueError('Capped samples cannot be yield-normalized')
            _uncapped(item)
    elif isinstance(value, list):
        for item in value:
            _uncapped(item)


def validate_audit(audit, sample_sha, events, provenance):
    """Return one normalization record per group/geometry from a version-1 audit.

    Source shard ranges cover selected entries [0, selected_entries), without
    gaps or overlap. Empty sources use an empty range list. Reference data
    populations have the same coverage contract, but are external to this sample.
    """
    _uncapped(provenance)
    if audit.get('version') != 1 or audit.get('sample_sha256') != sample_sha:
        raise ValueError('Exposure audit version/sample SHA mismatch')
    populations = audit.get('populations', {})
    if not populations:
        raise ValueError('Exposure audit needs parent populations')
    exposures, source_sets, expected = {}, {}, Counter()
    all_sources = set()
    for name, population in populations.items():
        role = population.get('role')
        if role not in ('mc', 'data', 'reference_data'):
            raise ValueError('Invalid population role')
        if not population.get('coverage_evidence'):
            raise ValueError('Population requires coverage_evidence')
        sources = population.get('sources', [])
        if not sources:
            raise ValueError('Empty parent population')
        total = 0.0
        source_sets[name] = set()
        for source in sources:
            uid = source['source_uid']
            if not isinstance(uid, str) or not uid or uid in all_sources:
                raise ValueError('Duplicate/invalid parent source; do not repeat luminosity per shard')
            all_sources.add(uid)
            source_sets[name].add(uid)
            lumi, probability = source['luminosity_fb'], source['sampling_probability']
            if not _positive(lumi) or not _positive(probability) or probability > 1:
                raise ValueError('Invalid source exposure/sampling probability')
            if role == 'mc' and probability != 1:
                raise ValueError('MC subsampling is not supported for yields')
            if probability != 1 and source.get('sampling_method') != 'uniform_random':
                raise ValueError('Reduced data exposure requires uniform_random sampling')
            count = source['selected_entries']
            if type(count) is not int or count < 0:
                raise ValueError('Invalid selected entry count')
            cursor = 0
            for start, stop in sorted(source['processed_ranges']):
                if type(start) is not int or type(stop) is not int or start != cursor or stop <= start:
                    raise ValueError('Missing or overlapping source shards')
                cursor = stop
            if cursor != count:
                raise ValueError('Incomplete source coverage')
            if role != 'reference_data':
                expected[uid] = count
            total += lumi * probability
        exposures[name] = total
    observed = Counter(event['source_uid'] for event in events)
    if any(uid not in expected for uid in observed) or observed != +expected:
        raise ValueError('Sample event counts do not match audited source coverage')
    records = {}
    actual_groups = {(e['group'], e['geometry']) for e in events}
    used = set()
    for group in audit.get('groups', []):
        key = (group['group'], group['geometry'])
        name = group['population']
        if key in records or name not in populations or populations[name]['role'] == 'reference_data':
            raise ValueError('Duplicate group or invalid population')
        used.add(name)
        if any(e['source_uid'] not in source_sets[name] for e in events
               if (e['group'], e['geometry']) == key):
            raise ValueError('Group contains a source from another parent population')
        role = populations[name]['role']
        source = exposures[name]
        target = source
        if role == 'mc':
            reference = group.get('target_population')
            if reference not in populations or populations[reference]['role'] not in ('data', 'reference_data'):
                raise ValueError('MC requires an audited data target population')
            target = exposures[reference]
        records[key] = dict(mode='yield', population=name, source_exposure_fb=source,
                            target_exposure_fb=target, scale=exposure_scale(target, source))
    if set(records) != actual_groups or used != {n for n, p in populations.items() if p['role'] != 'reference_data'}:
        raise ValueError('Exposure audit groups/populations do not match sample')
    return [dict(group=g, geometry=z, **record) for (g, z), record in sorted(records.items())]


def resolve_normalization(mode, dataset, sample, audit, sample_sha, events, provenance):
    shape_only = dataset in ('synthetic', 'testbeam', None) or sample != 'full'
    if mode == 'yield' and shape_only:
        raise ValueError('Yield requires an uncapped prepared full MC/data sample')
    if mode == 'shape' or (mode == 'auto' and shape_only):
        if audit is not None:
            raise ValueError('Exposure audit supplied for shape-only run')
        return [dict(group=g, geometry=z, mode='shape')
                for g, z in sorted({(e['group'], e['geometry']) for e in events})]
    if audit is None:
        raise ValueError('Full MC/data yields require --exposure-manifest; use --normalization shape for shapes')
    records = validate_audit(audit, sample_sha, events, provenance)
    roles = {p['role'] for p in audit['populations'].values()} - {'reference_data'}
    if roles != ({'data'} if dataset == 'real-data' else {'mc'}):
        raise ValueError('Exposure population role does not match dataset')
    return records


def normalize_counts(counts, record, underflow=0, overflow=0):
    """Preserve raw bins and scale Poisson sumw2, including separate flow bins."""
    total = sum(counts)
    scale = record['scale'] if record['mode'] == 'yield' else (1 / total if total else 0.0)
    return dict(normalization=dict(record, scale=scale), normalized=[n * scale for n in counts],
                sumw2=[n * scale**2 for n in counts],
                normalized_underflow=underflow * scale, normalized_overflow=overflow * scale,
                underflow_sumw2=underflow * scale**2, overflow_sumw2=overflow * scale**2)
