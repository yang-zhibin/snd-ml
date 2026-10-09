# Testbeam Distribution Source Selection Plan

Status: proposed

Last updated: 2026-10-07

## Objective

Define geometry-controlled source samples for five SciFi shower-feature plot
groups: MC pion and electron energy scans, measured-data pion and electron
energy scans, and separate MC/data pion-versus-electron comparisons.

Every curve within one plot must use the same absorber material and wall
setup. Particle charge must also remain fixed within an energy scan.

## Authoritative catalogues

Use the 2024 W2 catalogue:

```text
/eos/user/z/zhibin/TestBeam/binary_full_v1/catalog/master_sources.csv
```

Its 90 sources use validated tungsten geometry `W_2walls_28plates`.

Use these 2023 H8 catalogues:

```text
/afs/cern.ch/work/z/zhibin/snd-ml/testbeam/metadata/updated/real_data_testbeam_June2023_H8_metadata.csv
/afs/cern.ch/work/z/zhibin/snd-ml/testbeam/metadata/updated/MC_data_testbeam2023_metadata.csv
```

The measured catalogue contains distinct `1wall`, `2walls`, and `3walls`
iron configurations. Their geometry checksums differ. The 2023 MC geometry
checksum matches the measured `3walls` geometry exactly.

The current geometry auditor cannot accept these legacy files as-is: their
`ShipGeo` Fe-target flags disagree with the active iron placements in
`FAIRGeom`. Assign versioned canonical 2023 setup IDs from both placements and
checksums before unified production; never classify them as 2024 W2.

## Coverage limitation

No one geometry supports every requested plot:

- 2024 W2 has complete MC electron and positive-pion energy coverage and a
  six-energy measured-electron sample, but measured pions only at 180 GeV.
- 2023 iron-wall data has measured-pion energy scans but no electrons.
- 2023 MC has positive pions at 100 and 180 GeV and negative pions at 300 GeV.

Therefore, never mix 2023 pions with 2024 electrons in one plot, never treat
plots made in different geometries as a direct MC/data comparison, and keep
the 2023 one-, two-, and three-wall samples separate.

## Recommended plot sources

### 1. MC pion distributions across energy

Use 2024 W2 MC:

```text
domain == mc_2024
geometry_setup_id == W_2walls_28plates
pdg_code == 211
beam_energy_gev in [50, 100, 150, 180, 200, 250, 300]
```

Use at most 2,520 events per energy, the limiting complete cell. Sample
deterministically and preserve complete source/event identity.

### 2. MC electron distributions across energy

Use 2024 W2 MC:

```text
domain == mc_2024
geometry_setup_id == W_2walls_28plates
abs(pdg_code) == 11
beam_energy_gev in [50, 100, 150, 200, 250, 300]
```

Use at most 2,880 events per energy, the limiting electron cell.

### 3. Measured-data pion distributions across energy

Use the 2023 measured `3walls` iron sample because it has the largest
statistics and is the only 2023 setup with geometry-matched MC.

Positive-pion scan:

```text
geo_path ends with geofile_sndlhc_H8_2023_3walls.root
beam_type == pi+
beam_energy in [100GeV, 140GeV, 180GeV]
```

| Energy | Files | Raw events |
| --- | ---: | ---: |
| 100 GeV | 56 | 52,245,752 |
| 140 GeV | 54 | 51,325,305 |
| 180 GeV | 89 | 84,780,197 |

Select 50,000 events per energy deterministically and distribute them across
all available runs. Do not take all selected events from one run.

Make a separate negative-pion figure if desired:

```text
geo_path ends with geofile_sndlhc_H8_2023_3walls.root
beam_type == pi-
beam_energy in [240GeV, 300GeV]
```

Do not combine positive- and negative-pion scans because charge and energy
would change simultaneously. The `1wall` and `2walls` data may support
separate geometry-systematics plots but must not enter the primary scan.

### 4. Measured-data electron distributions across energy

Use 2024 W2 measured data:

```text
domain == data_2024
geometry_setup_id == W_2walls_28plates
abs(pdg_code) == 11
beam_energy_gev in [50, 100, 150, 200, 250, 300]
```

Use 50,000 selected events per energy. The 150, 250, and 300 GeV unified
feature files are absent from the current validated snapshot and must be
produced before the complete plot is made.

### 5. Pion versus electron

For exact-energy MC plots, use 2024 W2 MC:

```text
domain == mc_2024
geometry_setup_id == W_2walls_28plates
pdg_code in [11, 211]
beam_energy_gev in [50, 100, 150, 200, 250, 300]
```

Use 2,520 events per particle/energy cell.

No exact-energy measured pion/electron pair exists. The nearest pair in one
geometry is 2024 W2 positive pion at 180 GeV versus electron at 200 GeV. Both
currently provide 65,120 selected events. Use all 65,120 per class or the same
deterministic 50,000-event cap. Label the mismatch explicitly.

For interpretation, make three separate figures:

1. MC 200 GeV positive pion versus 200 GeV electron: exact-energy control;
2. MC 180 GeV positive pion versus 200 GeV electron: reproduce the data
   energy mismatch;
3. measured-data 180 GeV positive pion versus 200 GeV electron.

If exact same energy is mandatory for measured data, the requested comparison
cannot be made from the current catalogues.

## Plot contract

Use these five unified shower features:

```text
density_sndsw_scifi
scifi_cluster_qdc_per_hit
scifi_qdc_per_hit_direct
scifi_vertical_cluster1_qdc_std
scifi_horizontal_cluster1_qdc_std
```

Apply `count_scifi > 35` to every curve. Apply the matching
`cluster1_exists == 1` cut to each leading-cluster QDC-standard-deviation
feature. Use unit-area probability-density normalization for primary shape
plots.

Every CSV summary must record raw and selected entries, flow counts, source
count, geometry ID/checksum, particle charge, energy, and sampling policy.

## Preparation required for 2023

1. Extend geometry auditing for legacy 2023 files using active placements and
   the three observed checksums; fail on an unknown or mixed checksum.
2. Build immutable 2023 source manifests with stable source UIDs, particle
   charge, energy, geometry ID, raw event count, and geometry checksum.
3. Build deterministic selection indexes for the balanced 50,000-event cells,
   distributed across runs.
4. Produce new `unified_shower_features_v1` files. Do not reuse legacy feature
   files: only 573/865 measured files and 2/30 MC files exist, and they lack
   the unified validation contract.
5. Validate schema, provenance, event identity, selection membership, and
   geometry before histogram production.
6. Add 2023 families and groups under a new versioned distribution namespace;
   do not modify accepted 2024 outputs.

## Acceptance checks

- Every plot contains one geometry checksum and absorber material.
- Every energy scan contains one particle charge.
- MC electron/pion comparisons use equal event counts per cell.
- Measured-data comparisons use equal event counts, or record why the full
  counts are equal.
- The 2024 data comparison is labelled with its 180/200 GeV mismatch.
- The primary 2023 pion scan uses `3walls` only.
- Legacy and unified feature products are never mixed.
- Dry-runs resolve only explicitly requested geometry, particle, energy, and
  feature targets before production submission.

## Approved cross-geometry diagnostic extension

The same-geometry plots above remain the primary particle-response results.
In addition, version `unified_testbeam_distributions_v3` explicitly permits
the following measured-data diagnostic overlays:

| Pion | Electron | Geometry policy |
| --- | --- | --- |
| 2023 Fe 3-wall pi+ 100 GeV | 2024 W 2-wall e- 100 GeV | exact energy |
| 2023 Fe 3-wall pi+ 140 GeV | 2024 W 2-wall e- 150 GeV | labelled mismatch |
| 2023 Fe 3-wall pi+ 180 GeV | 2024 W 2-wall e- 200 GeV | labelled mismatch |
| 2023 Fe 3-wall pi- 300 GeV | 2024 W 2-wall e- 300 GeV | exact energy |

These plots are unit-area-only and must use the explicit
`cross_geometry_diagnostic` policy. Their labels and CSV summaries must retain
both energies, particle charges, geometry IDs, and geometry checksums. They
must not be presented as geometry-controlled measurements of particle type:
particle, absorber material, wall layout, year, and detector conditions all
change together. The 240/250 GeV comparison remains excluded while the
250 GeV electron source is held.
