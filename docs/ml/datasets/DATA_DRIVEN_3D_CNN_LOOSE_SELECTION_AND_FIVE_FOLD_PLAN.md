# Data-Driven 3D-Hit CNN: Loose Selection and Five-Fold Plan

## Existing partition workflow

The existing partition workflow:

1. Selects complete source-file rows from the metadata.
2. Groups consecutive source files into output partitions.
3. Builds matching feature and Hit3D chains.
4. Adds the feature chain as an entrywise friend of the Hit3D chain.
5. Applies the same feature-based selection to both trees.
6. Saves aligned feature and Hit3D partitions.
7. Writes metadata for each output partition.

Relevant implementation:

- `snakemake/metadata/configs/partitions/partition_v1.yaml`
- `convertData/build_region_partitions.py`
- `snakemake/metadata/configs/splits/split_v1.yaml`
- `convertData/split_region_partitions.py`

The filtering and friend-tree approach can be reused. The current configuration and splitter should not be reused unchanged because Hit3D production is disabled, entry-count checking is disabled, normal partitions do not retain useful source-file identity, and the splitter supports only two folds assigned independently within each class.

## 1. Build the original source-file table

Create one row per original source file before event selection:

```text
source_uid
source_metadata_csv
source_metadata_row
data_type
run
production
feature_path
hit3d_path
lumi_per_file
energy_bin
```

`source_uid` identifies the original source, not a derived physics class. If one neutrino file later contributes both CC and NC events, both derived samples retain the same `source_uid`.

## 2. Validate feature-Hit3D alignment

For every feature/Hit3D file pair:

- Require equal `sndData` and `hit3D` entry counts.
- Check `runId`, `eventId`, and `eventIndex` at corresponding entries.
- Reject or record files that fail alignment.
- Preserve the original source and entry identifiers.

This is a file-level alignment check; detailed hit-vector integrity checks are not required here.

## 3. Apply the V7 loose selection

Use the exact `coverage_loose` definition from:

- `document/PASSING_MUON_RUN8329_DETECTOR_SELECTION_V7.md`
- `snakemake/metadata/configs/region_evaluation/passing_muon_binary_region_study_v7.yaml`
- `evaluation_region_partitions/make_passing_muon_binary_region_hist_v5.py`

The loose selection requires:

- Activity in all three Veto stations.
- X and Y activity in all five SciFi stations.
- Activity in all five upstream MuFilter stations.
- At least four active downstream stations.
- Fewer than 20 hits per SciFi station.
- At most two hits per Veto station and at most four Veto hits in total.
- At most two hits per upstream station and at most six upstream hits in total.

Define the frozen Boolean expression as `L = coverage_loose`. For both 2024 data and MC muon:

```text
passing_muon = L
muon_DIS     = !L
```

The data muon-DIS class remains recorded as a V7 loose-failure proxy rather than a particle-truth label.

Evaluate `L` in the feature tree and apply it to the aligned Hit3D entry through the friend relationship:

```text
hit3D entry i
     <-> aligned feature entry i
                     |
              coverage_loose
                 /       \
               pass      fail
```

For every source file, require the closure:

```text
N_input = N_pass + N_fail
```

Retain these fields on every selected event:

```text
source_uid
original_entry
class_id
class_name
domain
selection_version
selection_pass
density_sndsw_scifi
```

## 4. Organize the selected muon samples

Use separate logical outputs:

```text
prepared/
|-- passing_muon/
|   |-- data/
|   `-- mc/
`-- muon_DIS/
    |-- data/
    `-- mc/
```

Prefer one filtered output per original source file. Passing and failing outputs derived from the same input must retain the same `source_uid`.

If merging is needed, merge only after fold assignment and never combine sources assigned to different folds.

## 5. Include the prepared physics samples

The eight CNN classes are prepared as follows:

| Class | Source |
|---|---|
| CC muon neutrino | Existing CC numu Hit3D |
| CC electron neutrino | Existing CC nue Hit3D |
| CC tau neutrino | CC nutau Hit3D when available |
| NC neutrino | Existing NC nue and NC numu with one CNN label |
| Neutral kaon | Existing kaon Hit3D, retaining energy bin |
| Neutron | Existing neutron Hit3D, retaining energy bin |
| Passing muon | V7 loose pass from data and MC muon |
| Muon-DIS | V7 loose fail from data and MC muon |

Keep original neutrino flavor as a subclass for the combined NC class.

Before fold assignment, every event must have:

```text
source_uid
class
subclass
domain
density_bin
energy_bin
hit3d_path
entry
```

## 6. Assign folds by original source

Assign each original `source_uid` to exactly one fold, then propagate that fold to every event derived from the source:

```text
2024 data source A -> fold 2
|-- loose-pass events -> passing_muon, fold 2
`-- loose-fail events -> muon_DIS, fold 2
```

Likewise, CC and NC events derived from one neutrino source stay in the same fold.

This avoids placing detector conditions or simulated events from one original source in both training and validation.

## 7. Balance complete source groups across five folds

Summarize each source group by:

```text
events per class
data/MC counts
muon-DIS counts per density bin
kaon/neutron counts per energy bin
lumi_per_file
```

Use a deterministic group-balancing assignment to make the five folds reasonably similar in:

- Eight-class event counts.
- Data/MC representation.
- Muon-DIS SciFi-density coverage.
- Kaon and neutron energy coverage.
- Data luminosity.
- MC productions.

Source independence has higher priority than identical event counts.

Prefer grouping complete data runs when enough independent 2024 runs exist. If complete-run grouping produces unusably uneven folds, group complete source files within runs and retain the run identifier for later stability tests.

## 8. Save one natural five-fold manifest

Store every event exactly once in its natural distribution:

```text
event_uid
source_uid
hit3d_path
entry
class_id
class_name
subclass
domain
run
production
density_sndsw_scifi
density_bin
energy_bin
fold
selection_version
physics_weight
```

The permanent dataset is:

```text
all prepared natural events
|-- fold 0
|-- fold 1
|-- fold 2
|-- fold 3
`-- fold 4
```

## 9. Apply uniform density sampling only during training

For model `k`:

```text
fold k       -> natural held-out evaluation
other folds  -> training pool
```

Inside the four-fold training pool, sample muon-DIS events as follows:

```text
choose muon-DIS
-> choose domain according to the chosen data/MC balance
-> choose a populated density bin uniformly
-> choose a source group
-> choose an event
```

Do not write duplicated, uniformly sampled ROOT datasets. Repetition happens only inside the training loader. The held-out fold contains every natural event exactly once.

## Reuse and replacement summary

Reuse:

- Metadata-driven discovery.
- ROOT RDataFrame filtering.
- Entrywise feature-Hit3D friend selection.
- ROOT snapshot creation.
- Per-partition metadata.
- Physics-class metadata branches.

Replace or extend:

- `partition_v1.yaml`, because the current regions are unrelated to V7 pass/fail and Hit3D is disabled.
- `split_v1.yaml`, because it describes five older classes and two folds.
- `split_region_partitions.py`, because folds must be assigned jointly by original source across derived classes.
- Normal source identity handling, because the current default does not retain a usable source-file index.

## Final workflow

```text
Original feature and Hit3D files
             |
strict entrywise alignment check
             |
apply V7 loose/not-loose to data and MC muon
             |
combine with prepared neutrino, kaon, and neutron samples
             |
preserve original source_uid
             |
assign complete source groups to five balanced folds
             |
write one natural event manifest
             |
uniform density sampling inside each training loader
```
