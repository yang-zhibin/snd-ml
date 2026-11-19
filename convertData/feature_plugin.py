"""Production plugin adapter. Standard-library only; safe to import in SNDSW.

Plugins receive copied selected-hit dictionaries, never ROOT objects or truth.
The launcher resolves YAML and records the complete executable source hashes.
"""
import array
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_spec(spec):
    if spec.get('version') != 1 or not re.fullmatch('[A-Za-z][A-Za-z0-9_]*', spec.get('name', '')):
        raise ValueError('Invalid production plugin version/name')
    if not isinstance(spec.get('parameters'), dict) or not spec.get('features'):
        raise ValueError('Plugin requires parameters and feature definitions')
    for name, definition in spec['features'].items():
        if not re.fullmatch('[A-Za-z][A-Za-z0-9_]*', name):
            raise ValueError('Invalid plugin feature name')
        if definition.get('dtype') not in ('int64', 'float64') or not definition.get('description'):
            raise ValueError('Invalid plugin dtype/description: ' + name)
        validate_value(name, definition.get('missing_value'), definition)
        if 'valid_range' in definition:
            limits = definition['valid_range']
            if len(limits) != 2 or any(not math.isfinite(x) for x in limits) or limits[0] > limits[1]:
                raise ValueError('Invalid plugin range: ' + name)
        if 'histogram' in definition:
            h = definition['histogram']
            if type(h.get('bins')) is not int or h['bins'] <= 0 or len(h.get('range', [])) != 2 or any(not math.isfinite(x) for x in h['range']) or h['range'][0] >= h['range'][1]:
                raise ValueError('Invalid histogram: ' + name)
    if spec.get('algorithm') not in spec.get('source_hashes', {}):
        raise ValueError('Plugin algorithm must be fingerprinted')


def validate_value(name, value, definition):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + ': finite Python int/float required')
    if definition['dtype'] == 'int64' and (type(value) is not int or not -(2**63) <= value < 2**63):
        raise ValueError(name + ': int64 required')
    if value != definition.get('missing_value') and 'valid_range' in definition:
        if not definition['valid_range'][0] <= value <= definition['valid_range'][1]:
            raise ValueError(name + ': outside valid range')


def standardized_hits(hits):
    result = []
    for hit in hits:
        row = {name: hit[name] for name in ('station', 'isVertical', 'layer_channel', 'qdc')}
        for internal, public in [('x', 'x_cm'), ('y', 'y_cm'), ('z', 'z_cm'), ('hitTimeNS', 'time_ns')]:
            if public in hit:
                row[public] = hit[public]
            elif internal in hit:
                row[public] = hit[internal]
        result.append(row)
    return result


class Plugin:
    def __init__(self, spec):
        validate_spec(spec)
        self.spec = spec
        for path, expected in spec['source_hashes'].items():
            if sha(path) != expected:
                raise ValueError('Plugin source changed: ' + path)
        repo = Path(__file__).resolve().parents[1]
        # Compatibility for Silvia's existing imports; these helper files are hashed.
        sys.path[:0] = [str(Path(spec['algorithm']).parent), str(repo/'feature_dev/algorithms')]
        module_spec = importlib.util.spec_from_file_location('production_plugin_' + spec['name'], spec['algorithm'])
        self.module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(self.module)
        if not callable(getattr(self.module, 'calculate_features', None)):
            raise ValueError('Plugin needs calculate_features')
        self.buffers = {}

    def branch(self, name):
        return self.spec['name'] + '__' + name

    def create_branches(self, tree):
        existing = {str(b.GetName()) for b in tree.GetListOfBranches()}
        for name, definition in self.spec['features'].items():
            branch = self.branch(name)
            if branch in existing:
                raise ValueError('Plugin output collision: ' + branch)
            integer = definition['dtype'] == 'int64'
            buffer = array.array('q' if integer else 'd', [definition['missing_value']])
            self.buffers[name] = buffer
            tree.Branch(branch, buffer, branch + ('/L' if integer else '/D'))

    def calculate(self, hits):
        selected = standardized_hits(hits)
        before = copy.deepcopy(selected)
        parameters = self.spec['parameters']
        cluster = getattr(self.module, 'cluster_hits', None)
        if cluster is None:
            values = self.module.calculate_features(selected, copy.deepcopy(parameters))
            clusters = None
        else:
            clusters = cluster(selected, copy.deepcopy(parameters))
            values = self.module.calculate_features(selected, clusters, copy.deepcopy(parameters))
        if selected != before:
            raise ValueError('Plugin modified selected hits')
        if not isinstance(values, dict):
            raise ValueError('Plugin must return a feature dictionary')
        from scifi_plane_clustering import scalar_feature_dtypes
        allowed_extra = {name for name, _ in scalar_feature_dtypes()}
        if set(values) - set(self.spec['features']) - allowed_extra:
            raise ValueError('Undeclared plugin outputs')
        for name, definition in self.spec['features'].items():
            if name not in values:
                raise ValueError('Missing plugin output: ' + name)
            validate_value(name, values[name], definition)
        validator = getattr(self.module, 'validate_algorithm', None)
        if validator:
            validator(selected, clusters, values)
        return {name: values[name] for name in self.spec['features']}

    def fill(self, hits):
        for name, buffer in self.buffers.items():
            buffer[0] = self.spec['features'][name]['missing_value']
        values = self.calculate(hits)
        for name, value in values.items():
            self.buffers[name][0] = value

    def write_metadata(self, root):
        text = canonical(self.spec)
        root.TObjString(text).Write('feature_plugin_json')
        root.TObjString(hashlib.sha256(text.encode()).hexdigest()).Write('feature_plugin_sha256')


def validate_root_plugin(source, expected_spec=None):
    """Check embedded provenance and every plugin value with PyROOT."""
    obj = source.Get('feature_plugin_json')
    if not obj:
        if expected_spec is not None:
            raise ValueError('Missing production plugin metadata')
        return None
    text = str(obj.GetString().Data())
    spec = json.loads(text)
    validate_spec(spec)
    digest = hashlib.sha256(text.encode()).hexdigest()
    recorded = source.Get('feature_plugin_sha256')
    if not recorded or str(recorded.GetString().Data()) != digest:
        raise ValueError('Plugin metadata checksum mismatch')
    if expected_spec is not None and canonical(spec) != canonical(expected_spec):
        raise ValueError('Embedded plugin differs from requested plugin')
    tree = source.Get('sndData')
    for name, definition in spec['features'].items():
        branch = spec['name'] + '__' + name
        leaf = tree.GetLeaf(branch)
        if not leaf or leaf.GetTypeName() != ('Long64_t' if definition['dtype'] == 'int64' else 'Double_t'):
            raise ValueError('Wrong/missing plugin ROOT branch: ' + branch)
    for event in tree:
        for name, definition in spec['features'].items():
            validate_value(name, getattr(event, spec['name'] + '__' + name), definition)
    return digest
