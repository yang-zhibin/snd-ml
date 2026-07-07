# Region Partition SNDSW Event Display Plan

## Goal

Build a new event plotting workflow for the region partition evaluation that uses
the official SNDSW `shipLHC/scripts/2dEventDisplay.py` backend, similar in spirit
to `evaluation/plot_saved_events.py`, but driven by region partition metadata and
a YAML config.

The first testing configuration should be intentionally small: draw only 2 or 3
events per region/particle group so the workflow can be debugged quickly.

## Main Requirements

1. Use SNDSW `2dEventDisplay.py`, not the hit3D/simple fallback.
2. Use a config file to define:
   - event selection criteria, for example `density_sndsw_scifi > 20000`
   - max number of plotted events per region/particle group
   - feature values to print on the event display
   - output directory and display mode
3. Add text to each plotted event page:
   - region
   - particle group
   - partition ID
   - event identity, such as run/event/eventIndex where available
   - selected feature values
   - selection expression or a short selection label

## Proposed Files

Add:

```text
evaluation_region_partitions/plot_saved_events_sndsw.py
snakemake/metadata/configs/region_evaluation/event_display_v1.yaml
```

Optionally add a Snakemake rule later:

```text
rules/region_evaluation.smk: plot_region_partition_event_displays
```

## Example Config

```yaml
version: event_display_v1

input:
  metadata_csv: snakemake/metadata/region_partitions/partition_v1_region_v1.csv
  feature_tree: sndData
  event_tree_candidates: [cbmsim, rawConv]

selection:
  label: "High SciFi density"
  expression: "density_sndsw_scifi > 20000"

limits:
  default_max_events_per_region_particle: 2
  per_particle:
    real_data: 3
    muonDIS: 2
    CC_nue: 2
    CC_numu: 2
    NC_nue: 2
    NC_numu: 2

display:
  backend: sndsw
  output_mode: vector_pdf
  disable_logo_for_vector_pdf: true
  canvas_width: 1600
  canvas_height: 1000

annotations:
  features:
    - density_sndsw_scifi
    - density_sndsw_scifi_second
    - count_scifi
    - count_us
    - count_veto
    - avg_scifi_x
    - avg_scifi_y
  max_lines: 8

outputs:
  outdir: evaluation_region_partitions/event_display
```

## Important Design Point

SNDSW `2dEventDisplay.py` needs a displayable FairRoot-style event tree, normally
`cbmsim` or `rawConv`, plus geometry and FairRoot bookkeeping objects. The region
partition feature files contain `sndData` rows, not full event-display data.

Therefore the new script should not try to pass the region partition
`feature_partition_path` directly to SNDSW. Instead, it should:

1. Use the region partition feature rows to decide which events should be drawn.
2. Map those selected rows back to their original event source file and event
   entry.
3. Copy only those selected original events into a temporary ROOT file.
4. Run SNDSW `2dEventDisplay.py` on that temporary ROOT file.

This is the main difference from the current `evaluation/plot_saved_events.py`.
The current script starts from `eff_*.root` files that already contain copied
`cbmsim` or `rawConv` trees. The region-partition version starts from selected
`sndData` feature entries, so it must reconstruct a temporary display input.

## Event Mapping Strategy

The safest implementation is to replay the source metadata instead of relying
only on the final partition file.

Each region partition metadata row contains:

```text
region
particle_group
partition_id
feature_partition_path
source_metadata_csv
source_feature_paths
selection_expression
particle_filter_expression
```

The feature trees produced by `convertData/digi_2_features.py` contain:

```text
runId
eventId
eventIndex
```

`eventIndex` is the original entry index in the raw `cbmsim` digi tree. This is
the key needed for event display.

Recommended extraction logic:

1. Read the region partition metadata CSV.
2. Filter rows by config-selected regions/particles if those options are added.
3. For each `(region, particle_group)` group:
   - iterate its partition rows
   - open each `feature_partition_path`
   - apply the config selection expression, for example
     `density_sndsw_scifi > 20000`
   - collect selected feature entries until the max limit is reached
4. For each selected feature entry:
   - read `eventIndex`
   - read annotation features
   - record region, particle group, partition ID, local feature entry, runId,
     eventId, and eventIndex
5. Resolve the original display source file.

The current partition metadata may not preserve a per-entry source file path in
the final snapshot. If a partition is built from multiple source feature files,
`eventIndex` alone is not enough to know which original ROOT file to reopen.

For a robust first implementation, use this approach:

1. For each partition row, read `source_metadata_csv`.
2. Reconstruct the source metadata rows used by that partition in the same way
   as `convertData/build_region_partitions.py`.
3. Process source feature files one by one rather than only the merged partition
   file.
4. For each source feature file:
   - apply the region expression
   - apply the particle filter expression
   - apply the event-display config expression
   - selected entries now have a known source metadata row
   - use that metadata row to find the original event-display file

This avoids ambiguity when multiple source files contribute to one region
partition.

## Source Event File Choice

Preferred source for SNDSW display:

1. `nueAnalysisFilter_path`, if it contains a displayable `cbmsim` or `rawConv`
   tree with `EventHeader`
2. otherwise `digi_path`, if available and displayable

The source path should be resolved with:

```text
output_base_path + relative path column
```

as existing scripts do.

## Temporary Display ROOT File

For each group or batch, create a temporary ROOT file containing only selected
events:

```text
display_input_<region>__<particle>.root
```

Implementation detail:

1. Open the original event file.
2. Find `cbmsim` or `rawConv`.
3. Create an output file.
4. Copy FairRoot bookkeeping:
   - `BranchList`
   - `TimeBasedBranchList`
   - `FileHeader`
   - `FileHeaderHeader`
5. Clone the tree structure with `CloneTree(0)`.
6. For each selected `eventIndex`, call `input_tree.GetEntry(eventIndex)` and
   `output_tree.Fill()`.
7. Write the output tree.

This is similar to `prepare_sndsw_display_file()` in
`evaluation/plot_saved_events.py`, but it must copy arbitrary selected entries,
not just the first `N` entries.

## SNDSW Runner

Reuse the current runner pattern from `evaluation/plot_saved_events.py`:

1. Locate `2dEventDisplay.py` from:
   - explicit config/CLI path
   - `$SNDSW_ROOT/shipLHC/scripts/2dEventDisplay.py`
   - local fallback if needed
2. Generate a temporary Python runner with `runpy.run_path`.
3. Configure:
   - input file
   - geometry file
   - batch mode
   - output prefix
   - output extension
4. Call `loopEvents(start=[0, 1, ...], auto=True, save=False, ...)`.

Because the temporary display file contains only selected events, the SNDSW
event indices are simply `0..N-1`.

## Text Annotation Strategy

The cleanest way to add region/particle/feature text is to patch the SNDSW
`drawInfo` function inside the generated runner, as the current script already
does for vector PDF mode.

The runner should receive a list like:

```python
event_annotations = [
    {
        "region": "...",
        "particle_group": "...",
        "partition_id": "...",
        "source_event_index": 12345,
        "runId": 999,
        "eventId": 123,
        "features": {
            "density_sndsw_scifi": 24310.5,
            "count_scifi": 412,
        },
        "selection_label": "High SciFi density",
    },
]
```

Then the custom `drawInfo` can draw the standard run/event text plus the
region-partition annotation. Keep the text compact because the event display is
already visually busy.

Suggested annotation lines:

```text
Region: signal_no_veto_has_us_fiducial_inside
Particle: muonDIS
Partition: signal_no_veto_has_us_fiducial_inside__muonDIS__part001
Selection: High SciFi density
density_sndsw_scifi=24310.5, count_scifi=412, count_us=3
```

## Output

One PDF per region/particle group:

```text
evaluation_region_partitions/event_display/
  events__signal_no_veto_has_us_fiducial_inside__muonDIS.pdf
  events__signal_no_veto_has_us_fiducial_inside__real_data.pdf
  events__sideband_no_veto_no_us_fiducial_inside__muonDIS.pdf
```

Also write a small summary CSV:

```text
evaluation_region_partitions/event_display/summary.csv
```

with columns:

```text
region
particle_group
partition_id
source_file
source_event_index
runId
eventId
feature_entry
output_pdf
selected_feature_values
```

This makes it easy to trace any plotted page back to the feature/event source.

## Snakemake Rule Sketch

Add a target in `rules/region_evaluation.smk` after region partition metadata is
available:

```python
REGION_EVENT_DISPLAY_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/"
    "region_evaluation/event_display_v1.yaml"
)

REGION_EVENT_DISPLAY_TARGET = (
    f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
    "event_display/selected_event_displays.done"
)

rule plot_region_partition_event_displays:
    input:
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/plot_saved_events_sndsw.py",
        config=REGION_EVENT_DISPLAY_CONFIG,
        metadata=REGION_EVAL_METADATA_CSV,
    output:
        done=REGION_EVENT_DISPLAY_TARGET,
    params:
        outdir=lambda wc, output: os.path.dirname(output.done),
    threads: 1
    resources:
        runtime=60 * 60,
        mem_mb=4000,
        disk_mb=4000,
        nvidia_gpu=0,
    shell:
        r'''
        echo "plotting region partition SNDSW event displays"
        export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u

        source {env_script_sndsw_nue}
        export EOSSHIP=root://eosuser.cern.ch/

        mkdir -p "{params.outdir}"

        python {input.script} \
            --config "{input.config}" \
            --metadata-csv "{input.metadata}" \
            --outdir "{params.outdir}"

        touch "{output.done}"
        '''
```

## Testing Plan

Start with a very small config:

```yaml
limits:
  default_max_events_per_region_particle: 2
  per_particle:
    real_data: 3
```

Suggested first test scope:

```yaml
filters:
  regions:
    - signal_no_veto_has_us_fiducial_inside
  particles:
    - muonDIS
```

Run one group first and inspect:

1. Does the script find selected feature events?
2. Does it resolve the original event source file correctly?
3. Does the temporary display ROOT file contain `cbmsim` or `rawConv`?
4. Does `2dEventDisplay.py` produce pages?
5. Are region, particle, feature values, and event identity visible on the PDF?

Only after that should the rule be expanded to all configured
region/particle groups.

## Risks And Mitigations

### Risk: source file ambiguity

If selected events are taken from a merged region partition file, `eventIndex`
does not identify which original source file to reopen.

Mitigation: replay source metadata and process source feature files one by one,
so every selected feature entry has a known original metadata row.

### Risk: missing `EventHeader`

SNDSW display requires `EventHeader` or `EventHeader.`.

Mitigation: check the event tree before adding it to the temporary display file.
Skip with a clear warning if the branch is missing.

### Risk: vector PDF logo/image issues

The current script patches `drawInfo` for `vector_pdf` mode to avoid image-logo
problems.

Mitigation: reuse the same patching approach and extend it with region
annotations.

### Risk: too much text on event display

Feature lists can become long.

Mitigation: keep `annotations.features` configurable and cap the number of text
lines with `annotations.max_lines`.

## Implementation Order

1. Add `event_display_v1.yaml` with test limits of 2/3 events.
2. Create `plot_saved_events_sndsw.py` by reusing utility pieces from
   `evaluation/plot_saved_events.py`:
   - `find_sndsw_2d_display_script`
   - FairRoot bookkeeping writer
   - temporary runner generation
   - page merging helpers
3. Add new logic for:
   - reading region partition metadata
   - applying feature selections
   - resolving original event files
   - copying arbitrary selected event entries
   - building per-event annotations
4. Test one region/particle pair manually.
5. Add Snakemake rule.
6. Enable broader region/particle plotting after the small test succeeds.
