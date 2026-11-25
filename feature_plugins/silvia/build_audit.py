import json
import glob

# Parametri
CAMPAIGN_SHA256 = 'a727bf1954463604c635f90e36f652e26cec5df4c4fdc5ae606751327344d7cc'
LUMI_PER_FILE = 15.0
TARGET_LUMI = 20.0
SOURCES_DIR = '/afs/cern.ch/work/s/svicenti/snd-ml-outputs/shower-features/mc-neutrino/trial01/sources'

# Leggi tutte le sorgenti
source_dirs = sorted(glob.glob(f'{SOURCES_DIR}/*/'))

ve_sources = []  # nu_e (pdg 12)
vm_sources = []  # nu_mu (pdg 14)

for sdir in source_dirs:
    val = json.load(open(f'{sdir}/validation.json'))
    uid = val['source_uid']
    selected = val['validated_event_count']
    ranges = [[0, selected]] if selected > 0 else []

    source = {
        'source_uid': f'unified:{uid}',
        'luminosity_fb': LUMI_PER_FILE,
        'sampling_probability': 1,
        'selected_entries': selected,
        'processed_ranges': ranges
    }

    # usa il path del file digi per capire se è nue o numu
    digi_path = val.get('input_digi_path', '')
    if 'nu14' in digi_path:
        vm_sources.append(source)
    else:
        ve_sources.append(source)

# Leggi le geometrie dal coverage
coverage = json.load(open(f'{SOURCES_DIR}/../config/coverage.json'))
groups = []
seen = set()
for norm in coverage['normalizations']:
    key = (norm['group'], norm['geometry'])
    if key in seen:
        continue
    seen.add(key)
    pdg_group = norm['group']
    pop = 'vm' if 'numu' in pdg_group else 've'
    groups.append({
        'group': norm['group'],
        'geometry': norm['geometry'],
        'population': pop,
        'target_population': 'target_data'
    })

audit = {
    'version': 1,
    'campaign_sha256': CAMPAIGN_SHA256,
    'populations': {
        've': {
            'role': 'mc',
            'coverage_evidence': SOURCES_DIR,
            'sources': ve_sources
        },
        'vm': {
            'role': 'mc',
            'coverage_evidence': SOURCES_DIR,
            'sources': vm_sources
        },
        'target_data': {
            'role': 'reference_data',
            'coverage_evidence': 'target luminosity 20 fb-1',
            'sources': [
                {
                    'source_uid': 'external:target-20fb',
                    'luminosity_fb': TARGET_LUMI,
                    'sampling_probability': 1,
                    'selected_entries': 0,
                    'processed_ranges': []
                }
            ]
        }
    },
    'groups': groups
}

output_path = '/afs/cern.ch/user/s/svicenti/snd-ml-shower-features/audit.json'
with open(output_path, 'w') as f:
    json.dump(audit, f, indent=2)

print(f'Audit salvato in {output_path}')
print(f've sources: {len(ve_sources)}')
print(f'vm sources: {len(vm_sources)}')
print(f'groups: {len(groups)}')