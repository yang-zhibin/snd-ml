#!/usr/bin/env python3
"""Audit frozen testbeam MC selections and optionally add missing manifest counts."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
import numpy as np
import uproot


def repair(manifest, upstream, report, apply=False):
    manifest, upstream, report = map(Path, (manifest, upstream, report))
    original = manifest.read_bytes()
    upstream_bytes = upstream.read_bytes()
    reader = csv.DictReader(io.StringIO(original.decode()))
    fields = list(reader.fieldnames)
    rows = list(reader)
    master_rows = list(csv.DictReader(io.StringIO(upstream_bytes.decode())))
    master = {r['source_uid']: r for r in master_rows}
    if len(master) != len(master_rows):
        raise ValueError('Duplicate upstream source_uid')
    if len({r['source_uid'] for r in rows}) != len(rows):
        raise ValueError('Duplicate manifest source_uid')
    results = []
    for row in rows:
        uid = row['source_uid']
        if row['family'] != 'testbeam_mc_2024' or row['selection_mode'] != 'identity_cutflow':
            raise ValueError(f'{uid}: not a testbeam MC identity selection')
        src = master[uid]
        for key in ('digi_path', 'cutflow_path', 'geo_path'):
            if row[key] != src[key]:
                raise ValueError(f'{uid}: upstream {key} differs')
        n, k = int(row['input_event_count']), int(src['n_events'])
        if not 0 <= k <= n:
            raise ValueError(f'{uid}: selected count outside input')
        if row.get('selected_event_count') not in (None, '', str(k)):
            raise ValueError(f'{uid}: existing selected count conflicts')
        with uproot.open(row['digi_path']) as source:
            if source[row['input_tree']].num_entries != n:
                raise ValueError(f'{uid}: raw count differs')
        with uproot.open(row['cutflow_path']) as source:
            tree = source['cutFlowSummary']
            if tree.num_entries != k:
                raise ValueError(f'{uid}: selected count differs')
            if 'entry' in tree:
                offset = 0
                for chunk in tree.iterate(['entry'], step_size=100000, library='np'):
                    values = chunk['entry']
                    if not np.array_equal(values, np.arange(offset, offset + len(values))):
                        raise ValueError(f'{uid}: cutflow is not identity')
                    offset += len(values)
        row['selected_event_count'] = str(k)
        results.append(dict(source_uid=uid, input_event_count=n, selected_event_count=k))
        print(f'{uid}: {k}/{n} verified', flush=True)
    if 'selected_event_count' not in fields:
        fields.append('selected_event_count')
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    corrected = buffer.getvalue().encode()
    backup = manifest.with_name(manifest.name + '.before-selected-counts-' + hashlib.sha256(original).hexdigest()[:12])
    if apply and corrected != original:
        # All source checks must pass before changing the deployed catalogue.
        if manifest.read_bytes() != original or upstream.read_bytes() != upstream_bytes:
            raise ValueError('Catalogue changed during audit')
        if backup.exists():
            if backup.read_bytes() != original:
                raise ValueError('Backup collision')
        else:
            with backup.open('xb') as stream:
                stream.write(original)
        fd, temporary = tempfile.mkstemp(prefix=manifest.name + '.', dir=manifest.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(corrected)
            os.chmod(temporary, manifest.stat().st_mode & 0o777)
            os.replace(temporary, manifest)
        finally:
            Path(temporary).unlink(missing_ok=True)
    payload = dict(verified_at=datetime.now(timezone.utc).isoformat(), manifest=str(manifest),
                   upstream=str(upstream), original_sha256=hashlib.sha256(original).hexdigest(),
                   upstream_sha256=hashlib.sha256(upstream_bytes).hexdigest(),
                   corrected_sha256=hashlib.sha256(corrected).hexdigest(), applied=apply,
                   backup=str(backup), sources=results)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2) + '\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'upstream', 'report'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    repair(args.manifest, args.upstream, args.report, args.apply)
