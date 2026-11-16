"""Bounded catalogue selection and immutable selected-hit preparation plans."""
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from engine import ROOT, read_json, sha, write_json
from datasets import SAMPLE_TYPES, resolve_dataset


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def fingerprint(path, content=False):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.R_OK):
        raise ValueError(f'Source is missing or unreadable: {path}')
    stat = path.stat()
    result = {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    if content:
        result['sha256'] = sha(path)
    return result


def source_group(dataset, row):
    if dataset == 'mc-neutrino':
        return '__neutrino_truth__'
    if dataset == 'testbeam':
        return f"{row['family']}:pdg{row['source_pdg_code']}:{row['source_energy_gev']}GeV"
    if dataset == 'real-data':
        group = row['selection_profile']
        if group not in ('real_data_2024_no_veto', 'real_data_2024_veto_10pct'):
            raise ValueError(f'Unsupported real-data selection: {group}')
        if row['selection_mode'] != 'indexed_cutflow':
            raise ValueError('Real-data preparation requires frozen indexed cutflows')
        return group
    return row['family']


def stratum(dataset, row):
    if dataset == 'mc-neutrino':
        # CC/NC is assigned per event by the converter; never label a whole file CC.
        return (row['subfolder'],)
    if dataset == 'real-data':
        return (source_group(dataset, row),)
    return (source_group(dataset, row), row['source_geometry_id'] or row['geo_path'])


def build_plan(dataset, config_path, output_root, files_per_group=None, events_per_file=None):
    """Read catalogues, select bounded sources, then preflight only chosen ROOT files."""
    import yaml
    config = yaml.safe_load(Path(config_path).read_text())
    if config.get('version') != 1:
        raise ValueError('Dataset config version must be 1')
    entry = config['datasets'][dataset]
    sources = entry['sources']
    files = files_per_group if files_per_group is not None else sources.get('files_per_group', 2)
    events = events_per_file if events_per_file is not None else sources.get('events_per_file', 100)
    cap = sources.get('events_per_group', 100)
    if any(type(n) is not int or n <= 0 for n in (files, events, cap)):
        raise ValueError('Automatic preparation limits must be positive integers')
    if files > 20 or events > 1000:
        raise ValueError('Development preparation is limited to 20 files per group and 1000 events per file; use a production workflow for larger samples')
    seed = str(sources.get('seed', 'feature-dev-v1'))
    catalogues = []
    grouped = defaultdict(list)
    seen = set()
    for location in sources['catalogues']:
        path = Path(location).expanduser()
        if not path.is_absolute():
            path = Path(config_path).resolve().parent / path
        catalogues.append(fingerprint(path, content=True))
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row['sample_type'].lower() not in SAMPLE_TYPES[dataset]:
                    raise ValueError(f'Wrong source family in {path}: {row["sample_type"]}')
                if row.get('selected_event_count') and int(row['selected_event_count']) == 0:
                    continue
                group = source_group(dataset, row)
                key = (group, row['source_uid'])
                if not row['source_uid'] or key in seen:
                    raise ValueError(f'Duplicate or empty source identity: {key}')
                seen.add(key)
                grouped[stratum(dataset, row)].append(row)
    if not grouped:
        raise ValueError(f'No eligible sources for {dataset}')
    selected = []
    # Prefer different runs for real data, keeping the source order reproducible.
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda r: digest([seed, key, r['source_uid']]))
        if dataset == 'real-data':
            first, rest, runs = [], [], set()
            for row in rows:
                run = str(Path(row['digi_path']).parent)
                (rest if run in runs else first).append(row)
                runs.add(run)
            rows = first + rest
        selected.extend(rows[:files])
    jobs = []
    for row in selected:
        row = dict(row)
        inputs = {name: fingerprint(row[name], content=name != 'digi_path')
                  for name in ('digi_path', 'geo_path', 'cutflow_path') if row.get(name)}
        if row['selection_mode'] != 'direct_input' and 'cutflow_path' not in inputs:
            raise ValueError('Selected source requires a cutflow')
        expected = row.get('selection_index_sha256')
        if expected and inputs['cutflow_path']['sha256'] != expected:
            raise ValueError(f'Selection index checksum mismatch: {row["source_uid"]}')
        if not row.get('source_geometry_id'):
            row['source_geometry_id'] = 'sha256:' + inputs['geo_path']['sha256']
        # Existing geometry checksums can describe a catalogue identity rather than
        # file bytes. Keep that field unchanged and record the file hash separately.
        jobs.append({'record': row, 'group': source_group(dataset, row), 'inputs': inputs,
                     'max_events': events})
    code_paths = [*sorted((ROOT/'convertData').glob('*.py')),
                  ROOT/'feature_dev/sources.py', ROOT/'feature_dev/prepare.py',
                  ROOT/'feature_dev/selected_hits.py', ROOT/'feature_dev/datasets.py',
                  ROOT/'feature_dev/engine.py', ROOT/'feature_dev/algorithms/baseline.py',
                  ROOT/'feature_dev/PrepareSnakefile', ROOT/'feature_dev/run_export.sh',
                  ROOT/'snakemake/envs/env_sndsw_nue.sh', ROOT/'snakemake/envs/env_lcg.sh',
                  ROOT/'run_feature_dev.sh']
    plan = {'version': 1, 'dataset': dataset, 'catalogues': catalogues, 'jobs': jobs,
            'required_groups': entry.get('required_groups', []), 'normalization': 'unit_area',
            'sampling': {'seed': seed, 'files_per_group': files, 'events_per_file': events,
                         'events_per_group': cap, 'event_policy': 'first selected entries, then identity hash per group/geometry'},
            'code': {str(p): sha(p) for p in code_paths}}
    if entry.get('normalization', 'unit_area') != 'unit_area':
        raise ValueError('Automatic preparation supports only unit_area')
    plan['signature'] = digest(plan)
    plan['directory'] = str(Path(output_root).resolve()/'.samples'/dataset/plan['signature'])
    return plan


def converter_arguments(job, directory):
    row = job['record']
    directory = Path(directory)
    converter = 'digi_2_features_mc_muon.py' if row['sample_type'].lower() == 'mc_muon' else 'digi_2_features.py'
    args = [str(ROOT/'convertData'/converter), '-d', row['digi_path'], '-g', row['geo_path'],
            '-t', row['sample_type'], '--tree-name', row['input_tree'],
            '--selection-mode', row['selection_mode'], '--source-domain-id', row['source_domain_id'],
            '--source-uid', row['source_uid'], '--source-geometry-id', row['source_geometry_id'],
            '--max-events', str(job['max_events']), '--feature-dev-group', job['group'],
            '--feature-dev-export', str(directory/'hits.jsonl'), '-o', str(directory/'features.root')]
    if row.get('cutflow_path'):
        args += ['-p', row['cutflow_path']]
    if row.get('source_geometry_checksum'):
        args += ['--source-geometry-checksum', row['source_geometry_checksum']]
    if int(row['source_domain_id']) in (4, 5, 6):
        args += ['--source-pdg-code', str(int(float(row['source_pdg_code']))),
                 '--source-energy-gev', row['source_energy_gev']]
    return args


def verify_inputs(job):
    for name, expected in job['inputs'].items():
        if fingerprint(expected['path'], content=name != 'digi_path') != expected:
            raise ValueError(f'Source changed after planning: {expected["path"]}')


def resolve_prepared(plan):
    directory = Path(plan['directory'])
    receipt = directory/'complete.json'
    if not receipt.exists():
        return None
    for name, expected in read_json(receipt).items():
        if not (directory/name).is_file() or sha(directory/name) != expected:
            raise ValueError(f'Prepared cache changed: {directory/name}')
    return resolve_dataset(plan['dataset'], 'small', directory/'registry.json', directory)


def write_registry(plan, directory):
    write_json(Path(directory)/'registry.json', {'version': 1, 'datasets': {plan['dataset']: {
        'samples': {'small': 'small.jsonl'}, 'required_groups': plan['required_groups'],
        'normalization': plan['normalization']}}})
