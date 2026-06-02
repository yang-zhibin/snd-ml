# Region/Particle Partition Workflow Plan

## Goal

Build region-aware feature and hit3D partition files from existing production
metadata, without creating a separate event catalog for every source file.

The current implemented workflow is:

```text
metadata CSVs
  + region_v1.yaml
  + partition_v1.yaml
  + feature.root / hit3d.root products
        |
        v
build_region_partitions.py
        |
        +--> feature partition ROOT files on EOS
        +--> hit3D partition ROOT files on EOS
        +--> one Snakemake row CSV per region/particle/partition
        |
        v
combine_region_partition_metadata
        |
        v
partition_v1_region_v1.csv
```

The key invariant is:

```text
feature_partition.root:sndData entry i == hit3d_partition.root:hit3D entry i
```

Downstream code can therefore read feature and hit3D partition files by the
same local entry number.

## Current Design Choice

The original plan considered an event catalog ROOT file and event-level
pre-scanning. The implemented workflow intentionally does not do that.

Instead:

- partition boundaries are chosen using `lumi_per_file` from metadata;
- selected source files are kept whole;
- if a file falls partly inside a lumi interval, the whole file is used;
- region cuts and particle cuts are applied with ROOT RDataFrame;
- output files contain only events passing the requested region and particle
  selection;
- missing source feature/hit3D file pairs are skipped when configured.

This avoids scanning all real-data files during DAG construction and keeps
Snakemake jobs independent.

## Inputs

The builder consumes metadata CSVs from:

```text
snakemake/metadata/updated
```

Each metadata row must provide:

```text
data_type
subfolder
partition
output_base_path
feature_path
hit3d_path
```

Optional but important columns are:

```text
lumi_per_file
energy_range
source_metadata_file
```

The ROOT trees are configured in `partition_v1.yaml`:

```yaml
input:
  feature_tree: sndData
  hit3d_tree: hit3D
```

## Region Config

Region definitions live in:

```text
snakemake/metadata/configs/regions/region_v1.yaml
```

The current region version is:

```text
region_v1
```

Regions are defined from feature branches, mainly:

```text
count_scifi
count_veto
count_us
avg_scifi*_x
avg_scifi*_y
```

The current regions are eight orthogonal combinations of:

```text
veto: no_veto / has_veto
US: no_us / has_us
SciFi fiducial: inside / outside
```

The main signal-like region is:

```text
signal_no_veto_has_us_fiducial_inside
```

The other regions are sidebands. Events with `count_scifi <= 35` are outside
this region version and are skipped.

## Partition Config

Partition behavior lives in:

```text
snakemake/metadata/configs/partitions/partition_v1.yaml
```

The current partition version is:

```text
partition_v1
```

The basic processing unit is:

```text
region + particle_group + part_index
```

Examples:

```text
signal_no_veto_has_us_fiducial_inside + CC_nue + part001
sideband_has_veto_has_us_fiducial_inside + real_data + part054
signal_no_veto_has_us_fiducial_inside + kaon_20_30GeV + part001
```

Particle groups are derived from metadata:

- `real_data`
- `CC_nue`, `CC_numu`, `NC_nue`, `NC_numu`
- `kaon_<emin>_<emax>GeV`
- `neutron_<emin>_<emax>GeV`
- `muon`
- `muonDIS`

For neutrinos, the particle group also has an RDF event filter on `pdgCode`.
For kaons and neutrons, the particle group is derived from the metadata
`energy_range` column.

## Meaning Of `fraction`, `n_partitions`, And `max_events`

In the implemented workflow, these are applied at the metadata/lumi level,
not by scanning every event first.

For each `region + particle_group`:

1. collect matching metadata rows;
2. keep rows in source order;
3. compute total lumi from `lumi_per_file`;
4. keep the first `fraction * total_lumi` worth of whole files;
5. split those whole files into `n_partitions` contiguous lumi chunks;
6. run RDF selection on the files belonging to the requested partition.

`max_events` is retained in metadata/config for future use, but the current
main partitioning strategy is lumi-driven.

This means partition sizes are approximate. A file is never split merely
because it crosses a partition boundary; the small lumi difference is accepted.

## Builder Logic

The implemented script is:

```text
convertData/build_region_partitions.py
```

For one Snakemake job, the script receives exactly one:

```text
region
particle_group
part_index
```

The main logic is:

1. load region and partition YAML;
2. select metadata rows for the requested particle group;
3. select the requested lumi partition from those rows;
4. resolve feature and hit3D paths;
5. optionally skip missing feature/hit3D pairs;
6. build one feature `TChain` and one hit3D `TChain`;
7. add the feature chain as a friend of the hit3D chain;
8. build the RDF selection expression from the region cut and particle cut;
9. count selected events;
10. snapshot `sndData` to the feature partition ROOT;
11. snapshot `hit3D` to the hit3D partition ROOT;
12. write the metadata row CSV.

The script uses ROOT RDataFrame as the only active engine.

## Output ROOT Files

ROOT outputs are written to temporary local storage first, then copied to EOS
by the Snakemake rule.

The configured EOS root is:

```text
/eos/experiment/sndlhc/users/zhibin/snd-ml/region_partitions
```

The layout is:

```text
/eos/experiment/sndlhc/users/zhibin/snd-ml/region_partitions/
  partition_v1/
    region_v1/
      <region>/
        <particle_group>/
          feature_<region>__<particle_group>__part001.root
          hit3d_<region>__<particle_group>__part001.root
```

Feature snapshots currently keep all feature branches plus added identity
branches.

hit3D snapshots keep the configured slim branch list:

```text
runId
eventId
eventIndex
pdgCode
isMC
label
energy
hit_x
hit_y
hit_z
hit_qdc
hit_station
hit_detType
event_uid
original_entry
source_file_index
particle_group
particle_id
```

Compression level is currently `1`.

## Metadata Row Files

Each Snakemake job produces one row CSV under:

```text
snakemake/metadata/region_partitions/partition_v1_region_v1/rows
```

The row filename is:

```text
<region>__<particle_group>__partNNN.csv
```

The row records:

- partition and region version;
- region and particle group;
- feature and hit3D EOS paths;
- selected event count;
- lumi assigned to this whole-file partition;
- requested/written partition counts;
- RDF selection expression;
- particle filter expression;
- source metadata CSV;
- source feature and hit3D paths;
- source metadata row indices;
- event identity range and seed/split method.

The combined metadata output is:

```text
snakemake/metadata/region_partitions/partition_v1_region_v1.csv
```

This combined CSV is the catalog for downstream stages.

## Snakemake Integration

The active rule file is:

```text
snakemake/rules/region_partitions.smk
```

The important rules are:

```text
build_region_partition
combine_region_partition_metadata
region_partitions
```

`build_region_partition` is intentionally split into many jobs:

```text
one job per region + particle_group + part_index
```

This gives better Condor scheduling and avoids one huge monolithic partition
job.

The rule output tracked by Snakemake is the row CSV, not the EOS ROOT files.
The ROOT files are side outputs recorded in the row CSV.

The shell logic is:

1. source the LCG environment;
2. create a temporary local directory;
3. run `build_region_partitions.py` with temporary output paths;
4. copy feature ROOT to EOS if it was produced;
5. copy hit3D ROOT to EOS if it was produced;
6. copy the metadata row CSV last.

Copying the row CSV last makes the row file act as the completion marker.

## Validation And Failure Handling

Current validation behavior:

- missing selected feature/hit3D pairs can be skipped with
  `validation.skip_missing_files: true`;
- optional per-file entry-count checks can be enabled with
  `validation.check_entry_counts: true`;
- output entry counts can be checked after production by opening the produced
  ROOT files and comparing `sndData`/`hit3D` entries with CSV `n_events`;
- hit3D branch slimming can be checked by comparing branches against the
  configured branch list.

Known real-data behavior:

- some metadata rows point to missing output chunks;
- some ROOT source files may exist but contain zero entries;
- both cases should not stop the whole campaign unless strict validation is
  explicitly enabled.

## Current Status

Implemented:

1. `region_v1.yaml`
2. `partition_v1.yaml`
3. `build_region_partitions.py`
4. Snakemake rule for one job per region/particle/partition
5. local temporary output then EOS copy
6. RDF-only selection/snapshot engine
7. ROOT implicit multithreading option
8. hit3D branch slimming
9. metadata row CSVs and combined metadata CSV
10. missing source-file skipping

Still to do:

1. finish a full `region_partitions` production run;
2. validate the combined metadata and ROOT outputs;
3. implement split metadata generation;
4. update the ML dataset builder to consume split partition metadata;
5. implement prediction output and merge-back workflow.

## Next Step

After `region_partitions` finishes cleanly, create:

```text
convertData/split_region_partitions.py
snakemake/metadata/configs/splits/split_v1.yaml
```

The split stage should read:

```text
snakemake/metadata/region_partitions/partition_v1_region_v1.csv
```

and write a split metadata CSV assigning whole partition files to:

```text
train
val
test
inference
```
