"""Snakemake workers: export in SNDSW, qualify in LCG, publish atomically."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from engine import ROOT, load_events, read_json, sha, write_json
from datasets import prepare_sample, check_size, resolve_dataset
from sources import converter_arguments, verify_inputs, write_registry


def verify_receipt(directory):
    directory = Path(directory)
    for name, expected in read_json(directory/'complete.json').items():
        if not (directory/name).is_file() or sha(directory/name) != expected:
            raise ValueError(f'Export cache changed: {directory/name}')


def validate_export(hits, feature_file):
    """Compare every exported identity and baseline scalar with the converter tree."""
    import numpy as np
    import uproot
    from algorithms import baseline
    from scifi_plane_clustering import scalar_feature_dtypes
    check_size([hits])
    rows, _ = load_events(hits)
    names = [name for name, _ in scalar_feature_dtypes()]
    identity = ['runId', 'eventId', 'eventIndex']
    with uproot.open(feature_file) as source:
        tree = source['sndData']
        if tree.num_entries != len(rows):
            raise ValueError('Export/ROOT entry count mismatch')
        arrays = tree.arrays(identity + names, library='np')
    for index, event in enumerate(rows):
        if any(int(arrays[name][index]) != event['event_id'][name] for name in identity):
            raise ValueError(f'Export/ROOT identity mismatch at entry {index}')
        values = baseline.calculate_features(event['hits'], baseline.cluster_hits(event['hits'], {}), {})
        for name, dtype in scalar_feature_dtypes():
            actual = arrays[name][index]
            expected = values[name]
            matches = actual == expected if dtype == 'i' else np.isclose(actual, expected, rtol=1e-6, atol=1e-7)
            if not matches:
                raise ValueError(f'Export/ROOT baseline mismatch at entry {index}: {name}: {actual} != {expected}')
    return {'events': len(rows), 'baseline_scalars': len(names), 'identity': identity,
            'float_rtol': 1e-6, 'float_atol': 1e-7,
            'comparison': 'fresh converter ROOT output, same selected events'}


def export_source(plan, index):
    job = plan['jobs'][index]
    base = Path(plan['directory'])
    destination = base/'exports'/str(index)
    if (destination/'complete.json').exists():
        verify_receipt(destination)
        return
    if destination.exists():
        # Snakemake creates output parents before invoking the worker. Only an
        # empty placeholder is removable; never erase an unrecognized product.
        destination.rmdir()
    verify_inputs(job)
    destination.parent.mkdir(parents=True, exist_ok=True)
    log = base/'logs'/f'export-{index}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{index}-', dir=destination.parent) as temporary:
        temporary = Path(temporary)
        command = ['bash', str(ROOT/'feature_dev/run_export.sh'), *converter_arguments(job, temporary)]
        with log.open('w') as stream:
            stream.write('Command: ' + repr(command) + '\n'); stream.flush()
            try:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as error:
                raise ValueError(f'Converter failed; see {log}') from error
        validation = validate_export(temporary/'hits.jsonl', temporary/'features.root')
        verify_inputs(job)
        write_json(temporary/'validation.json', validation)
        write_json(temporary/'complete.json', {p.name: sha(p) for p in temporary.iterdir()})
        os.replace(temporary, destination)


def assemble(plan):
    base = Path(plan['directory'])
    exports = [base/'exports'/str(i) for i in range(len(plan['jobs']))]
    for directory in exports:
        verify_receipt(directory)
    with tempfile.TemporaryDirectory(prefix='.assemble-', dir=base) as temporary:
        temporary = Path(temporary)
        prepare_sample(plan['dataset'], [p/'hits.jsonl' for p in exports], temporary/'small.jsonl',
                       plan['signature'], plan['sampling']['events_per_group'], bounded=True)
        write_registry(plan, temporary)
        resolve_dataset(plan['dataset'], 'small', temporary/'registry.json', temporary)
        # Include provenance and parity receipts in the final cache integrity check.
        receipt = {}
        for directory in exports:
            for name in ('complete.json', 'validation.json', 'hits.jsonl', 'hits.jsonl.manifest.json', 'features.root'):
                receipt[str((directory/name).relative_to(base))] = sha(directory/name)
        for path in temporary.iterdir():
            receipt[path.name] = sha(path)
            os.replace(path, base/path.name)
        receipt['plan.json'] = sha(base/'plan.json')
        write_json(base/'complete.json', receipt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('plan', type=Path)
    parser.add_argument('stage', choices=['export', 'assemble'])
    parser.add_argument('index', nargs='?', type=int)
    args = parser.parse_args()
    plan = read_json(args.plan)
    for path, expected in plan['code'].items():
        if sha(path) != expected:
            raise ValueError(f'Preparation code changed after planning: {path}; rerun the command')
    if args.stage == 'export':
        if args.index is None or not 0 <= args.index < len(plan['jobs']):
            raise ValueError('Invalid export source index')
        export_source(plan, args.index)
    else:
        assemble(plan)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
