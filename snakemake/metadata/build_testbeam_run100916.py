"""Build a versioned 2024 catalogue replacing electron100GeV with run100916.

Reads only the named upstream CSV and the 15 ROOT headers; no event scan.
Use the existing LCG environment (uproot). Existing catalogue is never modified.
"""
import argparse
import csv
import hashlib
from pathlib import Path


def build(base, upstream, output):
    import uproot
    with base.open() as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        original = list(reader)
    removed = [r for r in original if int(r['source_pdg_code']) == 11 and float(r['source_energy_gev']) == 100]
    if len(removed) != 1 or removed[0]['partition'] != 'run_100890':
        raise ValueError('Expected exactly the original run100890 electron100GeV row')
    template = removed[0]
    with upstream.open() as stream:
        sources = [(i, r) for i, r in enumerate(csv.DictReader(stream)) if r['partition'] == 'run_100916']
    if len(sources) != 15:
        raise ValueError('Expected 15 run100916 upstream rows')
    names = {Path(r['digi_path']).name for _, r in sources}
    if names != {f'sndsw_raw-{i:04d}.root' for i in range(15)}:
        raise ValueError('Unexpected or duplicate run100916 partitions')
    replacements = []
    for index, source in sources:
        if source['beam_type'] != 'e-' or source['beam_energy'] != '100GeV' or source['geo_path'] != template['geo_path']:
            raise ValueError('Unexpected beam or geometry metadata')
        with uproot.open(source['digi_path']) as root:
            count = root['cbmsim'].num_entries
        if count != int(source['n_event']):
            raise ValueError('ROOT count differs from upstream metadata: '+source['digi_path'])
        row = dict(template)
        # Do not inherit old source products, cutflows, or catalogue-row provenance.
        for key in ('cutflow_path', 'feature_scifi_cluster_v1_path', 'validation_record', 'validation_marker', 'catalog_file', 'catalog_row'):
            row[key] = ''
        row.update(manifest_version='testbeam_data_2024_run100916_v1',
                   source_uid='tb24_run100916_'+hashlib.sha256(source['digi_path'].encode()).hexdigest()[:20],
                   source_metadata_file=str(upstream), source_metadata_row=str(index),
                   partition='run_100916', digi_path=source['digi_path'],
                   selection_mode='direct_input', input_event_count=str(count), selected_event_count=str(count), is_canary='0')
        replacements.append(row)
    result = [r for r in original if r not in removed] + sorted(replacements, key=lambda r:r['digi_path'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(result)
    print(f'{output}: {len(result)} sources, {len(replacements)} replacement files, {sum(int(r["input_event_count"]) for r in replacements)} run100916 events')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.base, args.upstream, args.output)
