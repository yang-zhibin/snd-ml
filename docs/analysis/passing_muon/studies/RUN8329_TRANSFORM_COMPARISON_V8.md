# Run 8329 matched transform comparison

## Scope

The parent population is every event in the existing Run 8329 `RealDataDQ`
feature production.  No Veto requirement is applied while producing either
transformed dataset.

Two compact, source-aligned datasets are produced:

- `reco_scifi`: events with a usable frozen reconstructed SciFi track, using
  track-type priority 11, 1, 15 and the accepted v4 quality/axis/edge policy;
- `simple`: events with nonempty horizontal and vertical SciFi activity in all
  five stations.  An equal-weight line is fitted to the five station averages
  in each projection, with assumed positive-z travel.

Both methods remap SciFi and MuFilter detector IDs.  Digi payload fields are
preserved.  An event that cannot be transformed is audited and is never copied
unchanged into a transformed dataset.

## Event matching and selections

Original and transformed feature rows are matched by `(runId,eventId)`.  ROOT
entry numbers are not compared because transformed files are compacted.

Selections are frozen from the original RealDataDQ row:

1. `scifi_track`: `reco_scifi_eligible`, original versus reco-SciFi transform;
2. `no_veto`: original `count_veto == 0`, original versus simple transform;
3. `loose_v7`: original v7 `coverage_loose`, original versus simple transform.

Only successfully transformed partners enter a feature histogram.  A feature
is filled only when its configured validity cut passes in both rows.  Summary
counters retain the original population, matched-success count, and missing
transform count.

## Layout

Production files are written below:

```text
/eos/experiment/sndlhc/users/zhibin/snd-ml/evaluation_region_partitions/
passing_muon_run8329_transform_comparison_v8/
├── manifests/sourceNNN.{csv.gz,json}
├── transformed_digi/{simple,reco_scifi}/run008329_sourceNNN.root
├── audits/{simple,reco_scifi}/sourceNNN.{csv.gz,json}
├── features/{simple,reco_scifi}/transformedFeatures_run008329_sourceNNN.root
├── validation/{digi,features}/{simple,reco_scifi}/sourceNNN.ok
└── histograms/
```

Comparison PDFs are written below:

```text
evaluation_region_partitions/passing_muon_run8329_transform_comparison_v8/
├── scifi_track.pdf
├── no_veto.pdf
└── loose_v7.pdf
```

## Running

The workflow is executor-independent.  Use one of the following launch modes;
both produce the same files and can resume output produced by the other mode.

### HTCondor production (recommended for the complete run)

From the Snakemake directory:

```bash
cd /afs/cern.ch/work/z/zhibin/snd-ml/snakemake

/afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake \
  --profile .config/snakemake/htcondor --jobs 100 \
  passing_muon_run8329_transform_comparison_v8
```

### Local execution

For a complete local run, limit both cores and aggregate memory.  Each
transformation job can use roughly 6 GB, so two simultaneous jobs are a safe
default on a typical LXPLUS node:

```bash
cd /afs/cern.ch/work/z/zhibin/snd-ml/snakemake

/afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake \
  --cores 2 --resources mem_mb=12000 disk_mb=24000 \
  --rerun-incomplete --latency-wait 120 \
  passing_muon_run8329_transform_comparison_v8
```

For a single-source local smoke test, request the source histogram explicitly:

```bash
/afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake \
  --cores 1 --resources mem_mb=6000 disk_mb=12000 \
  --rerun-incomplete --latency-wait 120 \
  /eos/experiment/sndlhc/users/zhibin/snd-ml/evaluation_region_partitions/\
passing_muon_run8329_transform_comparison_v8/histograms/shards/source002.root
```

Do not combine `--profile .config/snakemake/htcondor` with `--cores`; the
former selects batch submission and the latter selects the local executor.
Temporary output is staged below Snakemake's `tmpdir` resource in either mode;
the workflow creates that directory if it is absent.

Intermediate targets allow staged production:

```bash
# Eligibility and event-key manifests only
snakemake --profile .config/snakemake/htcondor --jobs 100 \
  passing_muon_run8329_transform_manifests_v8

# Transformed and validated digi shards
snakemake --profile .config/snakemake/htcondor --jobs 100 \
  passing_muon_run8329_transformed_digis_v8

# Detector features and feature validation
snakemake --profile .config/snakemake/htcondor --jobs 100 \
  passing_muon_run8329_transformed_features_v8
```

The transform outputs are synthetic detector-ID remappings: channel detector
IDs change while the original DAQ payload identifiers remain in the copied
digi payload.  They are intended for the documented background-shape study,
not as newly acquired detector data.
