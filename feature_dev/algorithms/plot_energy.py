import uproot
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Leggi tutti i file features.root del campione
sample_dir = '/afs/cern.ch/work/s/svicenti/snd-ml-outputs/unified_shower_features_v1/feature_dev/.samples/mc-neutrino/c03692c1d64b72a87f1eadac81b70113ca294fb3559e2c921f4deace3afacc79'

# Leggi il features.json per sapere il gruppo di ogni evento
features_json = json.load(open('/afs/cern.ch/work/s/svicenti/snd-ml-outputs/unified_shower_features_v1/feature_dev/mc-neutrino/density-mc-v19/calculate/features.json'))

# Crea dizionario runId+eventIndex -> group
event_group = {
    (r['event_id']['runId'], r['event_id']['eventIndex']): r['group']
    for r in features_json
}

# Leggi energia da tutti i file ROOT
energies = {'CC_nue': [], 'CC_numu': [], 'NC_nue': [], 'NC_numu': []}

files = glob.glob(f'{sample_dir}/exports/*/features.root')
for fpath in files:
    with uproot.open(fpath) as f:
        tree = f['sndData']
        run_ids = tree['runId'].array()
        event_indices = tree['eventIndex'].array()
        mc_energies = tree['mc_incoming_energy'].array()

        for run, idx, energy in zip(run_ids, event_indices, mc_energies):
            key = (int(run), int(idx))
            if key in event_group:
                group = event_group[key]
                if group in energies:
                    energies[group].append(float(energy))

# Plot
fig, ax = plt.subplots(figsize=(9, 5))
colors = {'CC_nue': 'blue', 'CC_numu': 'red', 'NC_nue': 'cyan', 'NC_numu': 'orange'}
for group, vals in energies.items():
    if vals:
        ax.hist(vals, bins=30, range=(0, 500), histtype='step',
                label=f'{group} (N={len(vals)})', color=colors[group], linewidth=2, density=True)

ax.set_xlabel('MC incoming energy [GeV]')
ax.set_ylabel('Fraction per bin')
ax.set_title('Incoming neutrino energy by interaction type')
ax.legend()
fig.tight_layout()
fig.savefig('energy_plot.png', dpi=110)
print("Salvato in energy_plot.png")