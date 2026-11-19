"""Shared planning/worker helpers for production converter plugins."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'convertData'), str(ROOT/'feature_dev')]
from feature_plugin import canonical, sha, validate_spec
from engine import read_json, write_json
from sources import digest, source_group, stratum, fingerprint, verify_inputs


def plugin_spec(name, parameters):
    import yaml
    path = Path(name)
    if not path.is_dir():
        path = ROOT/'feature_plugins'/name
    path = path.resolve()
    schema = yaml.safe_load((path/'schema.yaml').read_text())
    if schema.get('extends') or schema.get('inherit'):
        raise ValueError('Production plugin schema must explicitly declare exported features, without inheritance')
    sources = list(path.rglob('*.py')) + [path/'schema.yaml', ROOT/'feature_dev/algorithms/baseline.py',
                ROOT/'feature_dev/algorithms/gap_clusters.py', ROOT/'convertData/scifi_plane_clustering.py']
    spec = dict(version=1, name=path.name, algorithm=str(path/'features.py'), parameters=parameters,
                schema_version=schema['version'], features=schema['features'],
                source_hashes={str(p):sha(p) for p in sorted(set(sources))})
    validate_spec(spec)
    return spec


def check_code(plan):
    for path, expected in plan['code'].items():
        if sha(path) != expected:
            raise ValueError('Code changed after planning: ' + path)


def verify_receipt(directory):
    directory = Path(directory)
    receipt = read_json(directory/'complete.json')
    for name, expected in receipt.items():
        if sha(directory/name) != expected:
            raise ValueError('Cached production output changed: ' + str(directory/name))
    return receipt


def source_key(row):
    # Same source can have independent no-veto/veto selected populations.
    return row['family'] + ':' + row['source_uid']


def groups_for_row(dataset, row, pdg):
    if dataset == 'mc-neutrino':
        mapping = {12:'CC_nue',14:'CC_numu',112:'NC_nue',114:'NC_numu'}
        try:
            return [mapping[abs(int(value))] for value in pdg]
        except KeyError as error:
            raise ValueError('Unknown neutrino truth class: ' + str(error)) from error
    return [source_group(dataset,row)] * len(pdg)
