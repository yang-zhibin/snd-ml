# Project overview

`snd-ml` is the SND@LHC detector-data and machine-learning analysis repository.
It combines ROOT/SNDSW conversion, metadata-driven Snakemake production,
passing-muon and control-region studies, data-driven dataset construction, and
multi-model event classification.

## Main data flow

```text
digi ROOT + geometry + cut-flow products
                  |
                  v
         metadata CSV catalogues
                  |
                  v
        Snakemake / HTCondor rules
                  |
       +----------+-----------+
       |                      |
       v                      v
feature ROOT (`sndData`)   hit ROOT (`hit3D`)
       |                      |
       +----------+-----------+
                  |
       event identity and row alignment
                  |
       +----------+-----------+
       |                      |
       v                      v
physics-region studies    five-fold ML datasets
       |                      |
       v                      v
ROOT/PDF products         GravNet/sparse ResNet
```

The maintained implementation areas are `convertData/`, `snakemake/`,
`evaluation_region_partitions/`, `train_3d/`, and `docs/`.

The `src/`, `training/`, `law/`, `data_quality_check/`, `on_going_work/`, and
`testbeam/` areas contain older, specialized, or exploratory workflows. They
may still be useful, but they are not the default starting point for new core
development.

## Core invariant

Feature and Hit3D products are joined by `(runId, eventId, eventIndex)`. Any
workflow that materializes aligned feature and hit files must preserve entry
identity or validate the join explicitly.
