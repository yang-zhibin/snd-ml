# Region-v2 Pre-Cut Inspection Input Plan

## Goal

Build a deterministic set of consolidated feature ROOT files for inspecting
real-data and EventBuilder signal distributions before defining `region_v2`.

This stage does not define region cuts or histograms. It produces reusable
event trees that can later support arbitrary one- or two-dimensional ROOT
RDataFrame expressions.

```text
source metadata CSVs
    -> deterministic source selection
    -> sample-specific minimal selection
    -> one consolidated sndData file per sample
       + constant luminosity-normalization branch
       + embedded exposure and provenance
    -> combined manifest and validation report
    -> later, independent 1D/2D plotting
```

This workflow is separate from region partitioning, `region_v2.yaml`, ML
splitting, and the final statistical fit.

## Implementation Status

The isolated workflow described here is implemented in the configured files
listed below. It is opt-in: the only modification to the existing workflow is
one `include` statement in `snakemake/snakefile`; no existing production,
region-v1, ML, training, or evaluation rule was changed by this implementation.

Run a dependency-only check with:

```bash
snakemake -s snakemake/snakefile region_v2_inspection_inputs --cores 1 -n
```

Run all six builders and validation with:

```bash
snakemake -s snakemake/snakefile region_v2_inspection_inputs --cores 4
```

The configured first-10% vm selection intentionally remains blocked until its
row-38/partition-44 feature file exists. The workflow will report the missing
fixed input and will not replace it with another row.

## Analysis Contract

1. Apply no veto, SciFi fiducial, US, density, ML, or other region cut.
2. Apply only the agreed data-quality selection to inclusive real data.
3. Apply no additional selection to passing-muon data.
4. Apply only the truth-particle filter to EventBuilder MC.
5. Never select source files randomly.
6. Fail on a missing selected input; do not replace it with a later file.
7. Record actual source exposure and target luminosity inside every output.
8. Add a constant event weight that normalizes each sample to the selected
   inclusive-real-data luminosity.
9. Preserve event-level feature branches so histogram choices remain open.
10. Record exact selected metadata rows and resolved paths for reproducibility.

## Planned Version and Files

Use:

```text
inspection_version: region_v2_inspection_v1
```

Add:

```text
snakemake/metadata/configs/region_v2_inspection/inspection_inputs_v1.yaml
convertData/build_region_v2_inspection_input.py
convertData/validate_region_v2_inspection_inputs.py
snakemake/rules/region_v2_inspection_inputs.smk
```

Write metadata products under:

```text
snakemake/metadata/region_v2_inspection/
  region_v2_inspection_v1_manifest.csv
  region_v2_inspection_v1_source_rows.csv
  region_v2_inspection_v1_validation.csv
```

Write ROOT outputs under:

```text
/eos/experiment/sndlhc/users/zhibin/snd-ml/
  region_v2_inspection_inputs/region_v2_inspection_v1/
```

## Input Metadata

Use:

```text
snakemake/metadata/updated/real_data_2024_skim_runs_metadata.csv
snakemake/metadata/updated/passing_muon_real_data_2024_metadata.csv
snakemake/metadata/updated/MC_neutrino_2024_ve_EventBuilder_metadata.csv
snakemake/metadata/updated/MC_neutrino_2024_vm_EventBuilder_metadata.csv
```

Every input requires:

```text
output_base_path
feature_path
lumi_per_file
```

The builder must attach the original zero-based CSV row index before applying
any metadata selection.

## Deterministic Dataset Definitions

### Inclusive real data

Select the complete run:

```text
partition: run_009715
subfolder: run_2410
metadata rows: 337
input luminosity: 0.6102 fb^-1
```

This is the closest available complete run to the requested approximately
`0.5 fb^-1`. Do not truncate the run or randomly select files.

The run-9715 feature files are preselected during digi-to-feature production
with:

```cpp
StableBeams == 1 && IP1 == 1 && EventDeltat_m1_100 == 1
```

Here `EventDeltat_m1_100` is the converter alias for the available
`EventDeltat_1_100`/`EventDeltat_-1_100` branch. The inspection consolidation
must therefore use `selection_mode: preselected_data_direct`, must not attach
`cutFlowSummary`, and must apply `event_selection: "true"`. Repeating the same
data-quality filter adds no information and creates unnecessary friend-tree
I/O and alignment requirements.

Do not apply `count_scifi > 35`, `NoVetoHits`, `AvgSFChan`, `USPlanesHit`,
`USQDC`, `SciFiContinuity`, or any V/F/U condition.

The upstream data-quality selection changes event count, not exposure. The
inspection output luminosity remains `0.6102 fb^-1`.

Output:

```text
real_data_run009715.root
```

### Passing-muon real data

Select the complete run:

```text
canonical run: 8329
metadata rows: 2,983
effective luminosity: 0.5323492746290897 fb^-1
```

Apply no additional selection. Do not request inclusive-data quality flags;
its honest pass-through cutflow contains only `entry` and `PassingMuonReco`.

Embed the following existing selection provenance:

```text
upstream PassingMuonReco selection
feature-production exclusion: count_veto > 0 && count_us > 0
```

This file is a muon-enriched reference, not an additive background sample to
stack on top of inclusive data.

Output:

```text
passing_muon_data_run008329.root
```

### Electron-neutrino EventBuilder MC

Read the ve metadata without reordering it. Select:

```text
first ceil(0.10 * 976) = 98 rows
original row indices: 0 through 97
source luminosity: 1,470 fb^-1
```

Use the same source rows for both filters:

```cpp
CC_nue: abs(pdgCode) == 12
NC_nue: abs(pdgCode) == 112
```

Outputs:

```text
CC_nue_EventBuilder_first10pct.root
NC_nue_EventBuilder_first10pct.root
```

### Muon-neutrino EventBuilder MC

Read the vm metadata without reordering it. Select:

```text
first ceil(0.10 * 890) = 89 rows
original row indices: 0 through 88
source luminosity: 1,335 fb^-1
```

Use the same source rows for:

```cpp
CC_numu: abs(pdgCode) == 14
NC_numu: abs(pdgCode) == 114
```

Outputs:

```text
CC_numu_EventBuilder_first10pct.root
NC_numu_EventBuilder_first10pct.root
```

At plan-writing time, selected partition 44 is missing:

```text
/eos/experiment/sndlhc/users/zhibin/MC_neutrino_EventBuilder/2024_vm/44/
  feature_EventBuilder_MC_neutrino_2024_vm_44.root
```

Produce it before materialization. Do not substitute metadata row 89 or a
later partition because that changes the versioned first-10% definition.

## Luminosity and Weights

The normalization target is the actual selected inclusive-data exposure:

```text
target_luminosity_fb = sum selected run-9715 lumi_per_file = 0.6102
```

Calculate it from metadata rather than using an independent hard-coded value.
The configuration should contain `0.6102` only as a validation expectation.

Add this constant branch to every output event:

```text
inspection_event_weight
```

For sample `s`:

```text
inspection_event_weight = target_luminosity_fb / source_luminosity_fb
```

Expected values are:

```text
inclusive real data: 0.6102 / 0.6102 = 1
passing-muon data:   0.6102 / 0.5323492746290897 ~= 1.1462
ve EventBuilder MC:  0.6102 / 1470
vm EventBuilder MC:  0.6102 / 1335
```

CC and NC outputs from one production have the same weight because they use
the same source exposure. Do not apply passing-muon `scale_factor` again; it is
already encoded in `lumi_per_file`.

The EventBuilder interpretation assumes `lumi_per_file` is its correct
equivalent luminosity. Cross-check generated-event and interaction
normalization before treating MC as an absolute expected yield.

Later histograms use:

```cpp
Histo1D(model, expression, "inspection_event_weight")
Histo2D(model, x_expression, y_expression, "inspection_event_weight")
```

## Consolidated ROOT Format

Each file contains:

```text
sndData
embedded inspection metadata objects
```

### Event tree

Preserve all source `sndData` branches and add:

```text
inspection_event_weight  double
inspection_source_row    integer
```

`inspection_source_row` maps each event to the original metadata CSV row. The
complete row-to-path mapping lives in the source-row audit CSV. This avoids
storing a long source path string for every event.

Fail if either added branch name already exists.

### Embedded metadata

Use `TParameter` for numbers and `TNamed` for strings, matching existing
region-evaluation conventions.

Required numeric objects:

```text
inspection_source_luminosity_fb
inspection_target_luminosity_fb
inspection_normalization_weight
inspection_selected_file_count
inspection_input_entry_count
inspection_output_entry_count
inspection_mc_fraction
```

Required string objects:

```text
inspection_version
inspection_dataset_name
inspection_data_type
inspection_source_metadata_csv
inspection_source_row_definition
inspection_event_selection
inspection_particle_filter
inspection_selected_runs
inspection_normalization_note
inspection_production_timestamp_utc
```

Do not write predefined histograms. Future code must be able to evaluate
arbitrary scalar or derived 1D/2D expressions.

## Manifest and Source Audit

Write one manifest row per consolidated output with:

```text
inspection_version
dataset_name
data_type
output_path
tree_name
source_metadata_csv
source_row_definition
selected_file_count
missing_file_count
selected_runs
event_selection
particle_filter
source_luminosity_fb
target_luminosity_fb
normalization_weight
input_entry_count
output_entry_count
mc_fraction
production_timestamp_utc
```

Write one source-audit row per selected input with:

```text
dataset_name
source_metadata_row
resolved_feature_path
partition
run
lumi_per_file
file_exists
file_opened
sndData_entries
cutFlowSummary_entries
```

The embedded ROOT metadata, manifest, and source audit must agree.

## Builder Design

Implement:

```text
convertData/build_region_v2_inspection_input.py
```

Use configuration-driven behavior. Do not infer selections from output
filenames.

The builder should:

1. Load metadata without reordering it.
2. attach the original row index;
3. apply the configured complete-run or first-fraction selection;
4. resolve every feature path;
5. require every selected file and tree to exist;
6. sum `lumi_per_file` over exactly the selected rows;
7. calculate the target-luminosity weight;
8. build the required ROOT chains;
9. apply only the configured event or truth filter;
10. add weight and source-row identity;
11. snapshot all feature branches;
12. embed metadata;
13. write manifest and audit rows;
14. validate locally before publishing to EOS.

### Inclusive real data

Build `sndData` and `cutFlowSummary` chains from the same paths and order.
Require equal entries, attach cutflow with alias `cut`, and apply only the
three data-quality flags. Validate `runId` and `eventId` alignment on sampled
entries, with a full check if affordable.

### Passing-muon data

Build only `sndData`, apply `true`, and preserve all events already retained
by feature production. Do not attach the minimal cutflow.

### EventBuilder MC

Build `sndData` from the deterministic source rows and apply only the
configured `abs(pdgCode)` filter.

### Correct source-row identity

A global `rdfentry_` is not a source-row index. Either process one metadata row
at a time and append, or map the current TChain tree number to its original
metadata row. Never fill the same source-row value for every event.

## Configuration Structure

Create `inspection_inputs_v1.yaml` with shared output and target-luminosity
settings plus these dataset records:

```yaml
version: region_v2_inspection_v1

output:
  root_dir: /eos/experiment/sndlhc/users/zhibin/snd-ml/region_v2_inspection_inputs
  tree_name: sndData
  compression_level: 1

target_luminosity:
  source_dataset: real_data_run009715
  expected_fb: 0.6102
  tolerance_fb: 1.0e-6

datasets:
  real_data_run009715:
    metadata_csv: snakemake/metadata/updated/real_data_2024_skim_runs_metadata.csv
    selection_mode: inclusive_data_quality
    metadata_filter: {partition: run_009715}
    event_selection: >
      cut.StableBeams == 1 && cut.IP1 == 1 &&
      cut.EventDeltat_1_100 == 1
    output_file: real_data_run009715.root

  passing_muon_data_run008329:
    metadata_csv: snakemake/metadata/updated/passing_muon_real_data_2024_metadata.csv
    selection_mode: passing_muon_direct
    metadata_filter: {run: 8329}
    event_selection: "true"
    output_file: passing_muon_data_run008329.root

  CC_nue_EventBuilder_first10pct:
    metadata_csv: snakemake/metadata/updated/MC_neutrino_2024_ve_EventBuilder_metadata.csv
    selection_mode: eventbuilder_truth_filter
    row_selection: {method: first_fraction, fraction: 0.10, rounding: ceil}
    particle_filter: "abs(pdgCode) == 12"
    output_file: CC_nue_EventBuilder_first10pct.root

  NC_nue_EventBuilder_first10pct:
    metadata_csv: snakemake/metadata/updated/MC_neutrino_2024_ve_EventBuilder_metadata.csv
    selection_mode: eventbuilder_truth_filter
    row_selection: {method: first_fraction, fraction: 0.10, rounding: ceil}
    particle_filter: "abs(pdgCode) == 112"
    output_file: NC_nue_EventBuilder_first10pct.root

  CC_numu_EventBuilder_first10pct:
    metadata_csv: snakemake/metadata/updated/MC_neutrino_2024_vm_EventBuilder_metadata.csv
    selection_mode: eventbuilder_truth_filter
    row_selection: {method: first_fraction, fraction: 0.10, rounding: ceil}
    particle_filter: "abs(pdgCode) == 14"
    output_file: CC_numu_EventBuilder_first10pct.root

  NC_numu_EventBuilder_first10pct:
    metadata_csv: snakemake/metadata/updated/MC_neutrino_2024_vm_EventBuilder_metadata.csv
    selection_mode: eventbuilder_truth_filter
    row_selection: {method: first_fraction, fraction: 0.10, rounding: ceil}
    particle_filter: "abs(pdgCode) == 114"
    output_file: NC_numu_EventBuilder_first10pct.root
```

The implementation may factor the duplicated ve/vm source selection into
shared source groups, but the final resolved rows must remain identical for CC
and NC outputs from the same production.

## Snakemake Integration

Add and include `snakemake/rules/region_v2_inspection_inputs.smk`.

Provide:

```text
rule region_v2_inspection_input
rule region_v2_inspection_inputs
rule validate_region_v2_inspection_inputs
```

Each consolidated dataset is a separate job. The aggregate target requires:

```text
six ROOT outputs
combined manifest
source-row audit
validation CSV
```

Write each ROOT file to local temporary storage first, validate it, then copy
atomically to EOS. Do not expose incomplete files at their final paths.

Initial resource estimates:

```text
real run 9715:     source features about 8.8 GB, request >=25 GB temp disk
passing run 8329:  source features about 13.6 GB, request >=35 GB temp disk
first 10% ve MC:   source features about 0.04 GB
first 10% vm MC:   source features about 0.05 GB
```

Tune runtime and memory after the integration test.

## Validation

Implement `validate_region_v2_inspection_inputs.py` and fail on any required
check.

### Inputs

Check:

```text
exact configured metadata rows selected
all selected paths exist and open
sndData exists everywhere
cutFlowSummary and all three DQ branches exist for inclusive data
selected luminosities equal metadata sums
source tree entry counts are recorded
```

### Outputs

Check:

```text
ROOT file opens and sndData exists
all source feature branches are preserved
inspection_event_weight exists, is finite, and is constant
inspection_source_row maps to the source audit
embedded luminosities and weight match the manifest
no region expression was applied
```

Independently count the inclusive-data DQ expression and each MC PDG filter;
require exact agreement with output entries. Passing-muon output entries must
equal its input entries because this stage adds no filter.

Expected exposure checks:

```text
inclusive data: 0.6102 fb^-1
passing muon:   0.5323492746290897 fb^-1
ve first 10%:  1470 fb^-1
vm first 10%:  1335 fb^-1
```

Any metadata change that alters these values requires a new review or version;
it must not silently change `region_v2_inspection_v1`.

## Tests

### Unit tests

Test metadata filtering, first-fraction ceil rounding, path resolution,
luminosity sums, weight calculation, PDG filters, and manifest serialization.

### Small integration test

Use one real-data file, one passing-muon file, two ve files, and two vm files.
Verify friend-tree DQ filtering, truth filtering, snapshot branches, source-row
identity, embedded metadata, and validation.

### Full dry run

```bash
snakemake -s snakemake/snakefile \
  region_v2_inspection_inputs \
  --cores 1 \
  -n
```

It should schedule only consolidation, manifest, and validation jobs—not
region partitioning or histograms. Source feature paths are deliberately
validated by the builder instead of being attached to the generic Snakemake
`process_features` wildcard: that rule uses the workflow's currently enabled
metadata lists and could otherwise resolve a missing fixed input against the
wrong production context. A missing selected feature therefore produces a
precise hard failure and must be produced separately; it is never substituted.

## Production Order

1. Produce the missing vm EventBuilder partition-44 feature file.
2. Confirm all 337 run-9715 inclusive-data feature files.
3. Confirm all 2,983 run-8329 passing-muon feature files.
4. Run the small local integration test.
5. Materialize and validate the four small MC outputs.
6. Materialize and validate inclusive real-data run 9715.
7. Materialize and validate passing-muon run 8329.
8. Combine manifest and source-audit rows.
9. Run the aggregate validation target.

## Deferred Distribution Workflow

Histogram choices remain deliberately deferred. A later plotter will read the
manifest and accept:

```text
arbitrary 1D expression
arbitrary 2D x/y expressions
binning and axis scale
feature-specific validity cut
weighted-yield or unit-area normalization
```

Possible later expressions include `count_veto`, `count_us`, station-average
positions, signed SciFi distance, Veto-versus-US correlations, timing
differences, or visible-energy proxies. Rebuilding the consolidated inputs
must not be necessary when these choices change.

## Completion Criteria

The stage is complete when:

```text
all six consolidated ROOT files exist
selected inputs are deterministic and recorded
no region cut was applied
inclusive data used only the three DQ flags
passing-muon data used no new filter
MC used exactly the first 10% in metadata order
all outputs embed luminosity and normalization weight
manifest and source audit are complete
all validation checks pass
arbitrary later 1D/2D RDataFrame expressions are supported
```
