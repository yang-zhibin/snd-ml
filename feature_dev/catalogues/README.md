# Versioned testbeam catalogue

The default `feature_dev/datasets.yaml` uses
`testbeam_data_2024_run100916.csv` for 2024 testbeam data.

For 100 GeV electrons this replaces run **100890** with all **15 files of
run 100916** (`sndsw_raw-0000.root` through `sndsw_raw-0014.root`), each containing
100,000 events: **1,500,000 events total**. The other 14 data sources are unchanged.
The MC and 2023 H8 catalogues are unchanged.

The replacement rows use the existing `direct_input` mode: all input entries
are eligible, so no cutflow file is needed. No SciFi occupancy or DS veto is
applied by this catalogue. Standard converter hit validity/timing rules still
apply; analysis cuts can be applied in the histogram configuration. This is an
inclusive replacement, not the old 50,000-entry subset of run100890.

## Use it

After updating the branch, use the normal launcher with a fresh run name:

```bash
./run_shower_features.sh --dataset testbeam --plugin silvia \
  --sample small --run testbeam-run100916
```

The small default picks one file per group/geometry and reads 20 events per file;
it does not process all 1.5 million events. Use `--files-per-group` and
`--events-per-file` to increase a preview. `--sample full` uses all catalogued
files/events and requires your configured HTCondor profile.

No custom `--dataset-config` is required. An old run or custom dataset config
continues to refer to its original catalogue. Source reuse from run100890 is
incompatible with this replacement: use a fresh conversion.

## Provenance and regeneration

Verified on 2026-11-28 against all 15 ROOT tree headers and the existing upstream
metadata. Geometry is `W_2walls_28plates`, using
`geofile_sndlhc_H4_2024_W_2walls.root`. The October2024 run-list workbook labels
this run as 100 GeV electrons on tungsten, with the beam moved up 20 mm.
The existing upstream metadata specifies the same geometry as run100890;
this check does not establish detailed alignment or physics data quality.

Rebuild in the project's LCG environment:

```bash
python3 snakemake/metadata/build_testbeam_run100916.py \
  --base /afs/cern.ch/work/z/zhibin/snd-ml-outputs/unified_shower_features_v1/metadata/manifests/testbeam_data_2024.csv \
  --upstream /afs/cern.ch/work/z/zhibin/snd-ml/testbeam/metadata/updated/real_data_testbeam_24_metadata.csv \
  --output feature_dev/catalogues/testbeam_data_2024_run100916.csv
```

The builder preserves the other sources, verifies beam/geometry and ROOT entry
counts, assigns new source IDs, and clears inherited cutflow/output references.
It does not modify the shared original catalogue or upstream metadata. Per-row
`source_metadata_file`/`source_metadata_row` record the upstream origin. The new
CSV is tracked with the code so collaborators receive it when updating Git.
