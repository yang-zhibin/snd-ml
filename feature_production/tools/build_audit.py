#!/usr/bin/env python3
"""Build a yield exposure audit that scales every MC neutrino flavour to one luminosity.

Input is a completed shape run of the full mc-neutrino campaign (its plan.json and
config/coverage.json). The audit declares one parent MC population per flavour
(nu12 -> 've', nu14 -> 'vm') spanning all geometries, and one nominal target
population of --target-lumi fb^-1. Every class then gets

    scale = target_lumi / (sum of parent-file luminosities of its flavour)

so histograms show absolute expected event counts at the target luminosity.

Group membership, event counts and the final validation use the same functions as
feature_production (groups_for_row, validate_coverage), so an audit written here
passes the production check for the same campaign.

Run in the LCG environment (needs uproot):

    set +u; unset PYTHONHOME; source snakemake/envs/env_lcg.sh >/dev/null
    export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
    python3 feature_production/tools/build_audit.py \
        --run-dir /afs/cern.ch/work/s/svicenti/snd-ml-outputs/shower-features/mc-neutrino/shape-full-v12 \
        --target-lumi 20 --output ~/shower-configs/audit-20fb.json
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'feature_production'))
from common import groups_for_row, sha  # noqa: E402  (common also adds convertData/ and feature_dev/)
from exposure import validate_coverage  # noqa: E402

FLAVOUR_POPULATION = {'12': 've', '14': 'vm'}
GROUP_POPULATION = {'CC_nue': 've', 'NC_nue': 've', 'CC_numu': 'vm', 'NC_numu': 'vm'}
TARGET = 'target_nominal'


def fail(message):
    raise SystemExit('ERROR: ' + message)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='Completed shape run directory containing plan.json and config/coverage.json')
    parser.add_argument('--target-lumi', type=float, default=20.0, help='Target luminosity in fb^-1 (default 20)')
    parser.add_argument('--output', type=Path, required=True, help='Audit JSON to write')
    parser.add_argument('--allow-capped', action='store_true',
                        help='Testing only: accept a small/capped run. Production rejects yield for such runs.')
    args = parser.parse_args()

    run = args.run_dir.resolve()
    plan = json.loads((run / 'plan.json').read_text())
    coverage_path = run / 'config' / 'coverage.json'
    if not coverage_path.exists():
        fail(f'{coverage_path} missing: the shape run has not finished its ranges stage')
    coverage = json.loads(coverage_path.read_text())

    campaign = plan['campaign_sha256']
    if plan['dataset'] != 'mc-neutrino':
        fail('only the mc-neutrino dataset is supported, got ' + plan['dataset'])
    if coverage['campaign_sha256'] != campaign:
        fail('coverage.json and plan.json belong to different campaigns')
    if plan['sample'] != 'full' and not args.allow_capped:
        fail(f"sample is '{plan['sample']}'; yield normalization needs the full campaign")
    if not args.target_lumi > 0:
        fail('--target-lumi must be positive')

    covered = {s['source_uid']: s for s in coverage['sources']}
    if len(covered) != len(plan['jobs']) or len(coverage['sources']) != len(plan['jobs']):
        fail(f"coverage lists {len(coverage['sources'])} sources, plan has {len(plan['jobs'])}")

    import uproot

    sources = defaultdict(list)
    source_population = {}
    counts = Counter()
    class_events = Counter()
    memberships = defaultdict(set)
    for index, job in enumerate(plan['jobs']):
        key, row = job['source_key'], job['record']
        entry = covered.get(key)
        if entry is None:
            fail('source missing from coverage.json: ' + key)
        match = re.search(r'/Genie/\d+/nu(\d+)/', row['digi_path'])
        if not match or match.group(1) not in FLAVOUR_POPULATION:
            fail(f'cannot determine flavour of {key} from {row["digi_path"]}')
        population = FLAVOUR_POPULATION[match.group(1)]

        exposure = entry['exposure']
        lumi = float(exposure['parent_luminosity_fb'])
        if exposure['audit_source_id'] != key or exposure['campaign_sha256'] != campaign:
            fail('exposure identity mismatch for ' + key)
        if lumi != float(row['lumi_per_file']):
            fail(f'{key}: ROOT luminosity {lumi} differs from catalogue {row["lumi_per_file"]}')
        if float(exposure['sampling_probability']) != 1.0:
            fail(f'{key}: MC sampling probability must be 1')

        path = Path(plan['source_directory']) / str(index) / 'features.root'
        if sha(path) != entry['root_sha256']:
            fail(f'{path} changed since coverage.json was written')
        observed = set()
        counts[key] = 0
        with uproot.open(path) as handle:
            for arrays in handle['sndData'].iterate(['pdgCode'], step_size='32 MB', library='np'):
                groups = groups_for_row(plan['dataset'], row, arrays['pdgCode'])
                counts[key] += len(groups)
                observed.update(groups)
                class_events.update(groups)
        if counts[key] != entry['selected_entries']:
            fail(f'{key}: {counts[key]} events in ROOT, coverage.json says {entry["selected_entries"]}')
        for group in observed:
            if GROUP_POPULATION[group] != population:
                fail(f'{key} ({population}) contains a {group} event')
            memberships[(group, row['source_geometry_id'])].add(key)

        source_population[key] = population
        sources[population].append(dict(
            source_uid=key, luminosity_fb=lumi, sampling_probability=1,
            selected_entries=entry['selected_entries'], processed_ranges=entry['processed_ranges']))
        if (index + 1) % 200 == 0:
            print(f'  scanned {index + 1}/{len(plan["jobs"])} sources', flush=True)

    populations = {name: dict(role='mc', coverage_evidence=str(coverage_path), sources=items)
                   for name, items in sorted(sources.items())}
    populations[TARGET] = dict(
        role='reference_data',
        coverage_evidence=f'Nominal target luminosity {args.target_lumi:g} fb^-1 for MC-only yields; '
                          'no data events are processed',
        sources=[dict(source_uid=f'nominal:target-{args.target_lumi:g}fb', luminosity_fb=args.target_lumi,
                      sampling_probability=1, selected_entries=0, processed_ranges=[])])
    groups = [dict(group=group, geometry=geometry, population=GROUP_POPULATION[group], target_population=TARGET)
              for group, geometry in sorted(memberships)]
    audit = dict(version=1, campaign_sha256=campaign, populations=populations, groups=groups)

    # Same validator and call as feature_production/distributions.py:coverage.
    records = validate_coverage(dict(audit, sample_sha256=campaign), campaign, counts, memberships, {})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=1) + '\n')

    print(f'Audit written: {args.output}')
    print(f'campaign_sha256: {campaign}')
    print(f'target luminosity: {args.target_lumi:g} fb^-1')
    for name in sorted(sources):
        total = sum(s['luminosity_fb'] for s in sources[name])
        print(f'  {name}: {len(sources[name])} files, {total:g} fb^-1, scale {args.target_lumi / total:.6g}')
    scales = {record['group']: record['scale'] for record in records}
    print('Selected-stream events (before plot selections) -> expected at target luminosity:')
    for group in sorted(class_events):
        print(f'  {group}: {class_events[group]} MC events x {scales[group]:.6g} = '
              f'{class_events[group] * scales[group]:.1f}')
    print(f'{len(groups)} populated (group, geometry) pairs; production validation PASSED')


if __name__ == '__main__':
    main()
