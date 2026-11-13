"""Resolve independent datasets and prepare fixed samples from selected-hit exports."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from engine import ROOT, load_events, event_key, read_json
from selected_hits import SelectedHitWriter

PHYSICS_DATASETS = ('mc-neutrino', 'testbeam', 'real-data')
DATASETS = PHYSICS_DATASETS + ('mc-muon',)
# Both preparation and execution are intentionally limited to development-sized inputs.
MAX_SAMPLE_BYTES = 256 * 1024 * 1024
SAMPLE_TYPES = {
    'mc-neutrino': {'mc_neutrino'},
    'testbeam': {'mc_testbeam', 'real_testbeam'},
    'real-data': {'real_data'},
    'mc-muon': {'mc_muon', 'mc_muondis'},
}


def check_size(paths):
    total = sum(Path(p).stat().st_size for p in paths)
    if total > MAX_SAMPLE_BYTES:
        raise ValueError('Selected-hit inputs exceed the 256 MiB development limit. Prepare smaller exports; large full campaigns require a sharded workflow.')


def source_types(manifest):
    """Inspect original provenance through prepared/merged export manifests."""
    provenance = manifest.get('provenance', {})
    if provenance.get('kind') == 'synthetic':
        raise ValueError('Synthetic hits cannot be registered as a physics dataset; use --dataset synthetic')
    types = set()
    if provenance.get('sample_type'):
        types.add(str(provenance['sample_type']).lower())
    for source in provenance.get('inputs', []):
        types.update(source_types(source['manifest']))
    return types


def group_counts(events):
    counts = Counter((e['group'], e['geometry']) for e in events)
    return [{'group': g, 'geometry': geo, 'events': n} for (g, geo), n in sorted(counts.items())]


def prepare_sample(dataset, inputs, output, version, per_group):
    if dataset not in DATASETS or not version or per_group < 0:
        raise ValueError('Choose a known dataset, a nonempty version, and a nonnegative per-group cap')
    inputs = [Path(p).resolve() for p in inputs]
    check_size(inputs)
    grouped = defaultdict(list); seen = set(); provenance = []
    for source in inputs:
        events, manifest = load_events(source)
        declared = manifest.get('provenance', {}).get('dataset')
        if declared and declared != dataset:
            raise ValueError(f'Export dataset {declared} does not match {dataset}')
        types = source_types(manifest)
        if not types or not types <= SAMPLE_TYPES[dataset]:
            raise ValueError(f'Export sample_type {sorted(types)} does not match {dataset}')
        for event in events:
            key = event_key(event)
            if key in seen:raise ValueError('Duplicate event across dataset exports')
            seen.add(key)
            if not event['source_uid']:raise ValueError('Physics exports require a nonempty source_uid')
            if not event['group'] or event['geometry'] in ('', 'unknown'):
                raise ValueError('Physics exports require explicit group and geometry labels')
            grouped[(event['group'], event['geometry'])].append(event)
        provenance.append({'path': str(source), 'manifest': manifest})
    selected = []
    for group in sorted(grouped):
        # Stable under input-file and event ordering; balanced per group/geometry.
        rows = sorted(grouped[group], key=lambda e: hashlib.sha256(json.dumps(event_key(e)).encode()).hexdigest())
        selected.extend(rows[:per_group] if per_group else rows)
    if not selected:raise ValueError('No selected events')
    writer = SelectedHitWriter(output, {
        'kind': 'prepared dataset', 'dataset': dataset, 'sample_version': version,
        'sampling': 'sha256_identity_per_group_geometry_v1', 'max_events_per_group': per_group,
        'available_groups': group_counts([e for rows in grouped.values() for e in rows]),
        'selected_groups': group_counts(selected), 'inputs': provenance,
    })
    for event in selected:
        writer.stream.write(json.dumps(event, allow_nan=False) + '\n');writer.count += 1
    writer.finish(len(selected))
    return len(selected)


def resolve_dataset(dataset, sample, config_path, sample_root):
    import yaml
    config = yaml.safe_load(Path(config_path).read_text())
    if config.get('version') != 1:raise ValueError('Dataset config version must be 1')
    entry = config.get('datasets', {}).get(dataset)
    if not isinstance(entry, dict):raise ValueError(f'Dataset configuration missing: {dataset}')
    if entry.get('normalization', 'unit_area') != 'unit_area':
        raise ValueError('Dataset development currently supports unit_area shapes only')
    if sample not in ('small', 'full'):raise ValueError('Physics datasets require --sample small or full')
    location = entry.get('samples', {}).get(sample)
    if not isinstance(location, str) or not location:raise ValueError(f'No {sample} sample configured for {dataset}')
    path = Path(location).expanduser()
    if not path.is_absolute():path = Path(sample_root) / path
    path = path.resolve()
    if not path.exists():
        raise ValueError(f'No prepared {dataset}/{sample} sample: {path}. Export real selected hits, then use run_feature_dev.sh prepare-sample (see feature_dev/README.md). No synthetic fallback is used.')
    check_size([path])
    events, manifest = load_events(path)
    provenance = manifest.get('provenance', {})
    if provenance.get('dataset') != dataset or not provenance.get('sample_version'):
        raise ValueError('Sample must declare matching dataset and sample_version; use prepare-sample')
    types = source_types(manifest)
    if not types or not types <= SAMPLE_TYPES[dataset]:raise ValueError('Sample source types do not match dataset')
    expected = entry.get('required_groups', [])
    missing = set(expected) - {e['group'] for e in events}
    if missing:raise ValueError(f'Missing configured groups for {dataset}: {sorted(missing)}')
    identity = {'name': dataset, 'sample': sample, 'version': provenance['sample_version'],
                'normalization': 'unit_area', 'groups': group_counts(events), 'configuration': entry}
    return path, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--max-events-per-group', type=int, default=100,
                        help='Fixed cap per group/geometry; 0 keeps all provided events')
    args = parser.parse_args()
    count = prepare_sample(args.dataset, args.input, args.output, args.version, args.max_events_per_group)
    print(f'Prepared {count} events: {args.output}')

if __name__ == '__main__':
    try:main()
    except (ValueError, OSError) as error:raise SystemExit(str(error))
