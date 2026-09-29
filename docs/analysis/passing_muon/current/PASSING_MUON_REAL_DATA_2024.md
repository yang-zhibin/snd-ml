# Passing-muon real data 2024

## Purpose

`passing_muon_real_data` is a control sample of 2024 proton-collision data in
which a muon track was reconstructed by the SND@LHC track-reconstruction
workflow. It is intended for studying real reconstructed muons and for use as a
muon-background reference sample.

This is not an inclusive copy of the original run data. Most runs were randomly
subsampled before track reconstruction, and the saved reconstructed trees
contain only events retained by that reconstruction workflow. Consequently:

- saved output entries measure the selected event yield;
- attempted source entries measure the sampled exposure;
- `lumi_per_file` is an effective sampled luminosity;
- efficiencies measured with this sample are conditional on the upstream
  passing-muon reconstruction.

## Dataset provenance

The reconstructed files were provided by Ivaylo Dionisov from the earlier muon
flux analysis. The principal reconstructed-data location is:

```text
/eos/user/i/idioniso/1_Data/Tracks/run_XXXXXX/
```

The convenience directory below contains symlinks to 2024 proton data, but is
not used as the authoritative metadata input because normalization and scale
factors differ by run:

```text
/eos/user/i/idioniso/1_Data/Tracks/protons2024/
```

Run 8329 was reconstructed separately with full event coverage by Simona and
is stored at:

```text
/eos/experiment/sndlhc/users/sii/2024/8329/
```

The run luminosities and random-sampling scales come from:

```text
/eos/user/i/idioniso/1_Data/Tracks/pp24.csv
```

The detector geometry used for all six runs is:

```text
/eos/experiment/sndlhc/convertedData/physics/2024/geofile_sndlhc_TI18_V12_2024.root
```

## Runs, locations and scales

Only 2024 proton runs listed in `pp24.csv` are included.

| Canonical run | `pp24.csv` run | Fill | Scale | Reconstruction | Event tree | Source directory | Reconstructed directory |
|---:|---:|---:|---:|---|---|---|---|
| 8007 | 8006 | 9523 | 200 | Whole source file | `cbmsim` | `/eos/experiment/sndlhc/convertedData/physics/2024/run_241/run_008007` | `/eos/user/i/idioniso/1_Data/Tracks/run_008007` |
| 8329 | 8329 | 9609 | 1 | 100,000-event chunks | `rawConv` | `/eos/experiment/sndlhc/convertedData/physics/2024/run_242/run_008329` | `/eos/experiment/sndlhc/users/sii/2024/8329` |
| 8638 | 8637 | 9691 | 400 | Whole source file | `cbmsim` | `/eos/experiment/sndlhc/convertedData/physics/2024/run_243/run_008638` | `/eos/user/i/idioniso/1_Data/Tracks/run_008638` |
| 9288 | 9288 | 9931 | 350 | Whole source file | `cbmsim` | `/eos/experiment/sndlhc/convertedData/physics/2024/run_246/run_009288` | `/eos/user/i/idioniso/1_Data/Tracks/run_009288` |
| 9446 | 9445 | 10001 | 350 | Whole source file | `cbmsim` | `/eos/experiment/sndlhc/convertedData/physics/2024/run_247/run_009446` | `/eos/user/i/idioniso/1_Data/Tracks/run_009446` |
| 9936 | 9934 | 10219 | 300 | Whole source file | `cbmsim` | `/eos/experiment/sndlhc/convertedData/physics/2024/run_2411/run_009936` | `/eos/user/i/idioniso/1_Data/Tracks/run_009936` |

For a scale factor `s`, approximately one event in every `s` source events was
sent through the track-reconstruction selection. Run 8329 has scale 1 and was
processed without random subsampling.

## Run-number mismatch in `pp24.csv`

Several reconstructed directories contain an `f0` file whose filename and
internal run number belong to a nearby run:

- run 8007 contains one run-8006 file;
- run 8638 contains one run-8637 file;
- run 9446 contains one run-9445 file;
- run 9936 contains one run-9934 and one run-9935 file.

The mismatched files are real foreign-run files, not merely incorrectly named
files. The `Run` values in `pp24.csv` match these foreign files for four of the
six rows, which strongly indicates that the CSV run label was derived from the
first file found in each reconstructed directory.

Metadata therefore uses the reconstructed directory and the majority/internal
run as the canonical run. The original `pp24.csv` run is retained in the
`pp24_run` column for luminosity provenance. Foreign-run files are excluded and
listed in the audit report.

## How the reconstructed sample was produced

For the randomly subsampled runs, track reconstruction selected events across
the entire run using the reconstruction program's `--scale`/`-sc` argument.
For example, scale 200 means that approximately one in every 200 input events
was attempted. The resulting ROOT files contain events saved by the passing
track-reconstruction selection rather than every attempted source event.

This distinction is essential:

```text
source-tree entries attempted     -> sampled exposure
reconstructed-tree entries saved -> selected passing-muon yield
```

It is incorrect to calculate exposure from the reconstructed output entry
count. Doing so would make the assigned luminosity depend on the muon-selection
efficiency.

For whole-file productions, one reconstructed file maps to one original file:

```text
muonReco_RUN-<run>_f<index>.0.0.root
    -> sndsw_raw-<index, four digits>.root
```

Run 8329 uses:

```text
sndsw_raw-<chunk>_<source-index>_8329_muonReco.root
```

with a nominal chunk size of 100,000 events. The final chunk is clipped to the
actual number of entries in its source file. Metadata generation checks that
the chunks have no gaps, overlaps or duplicates.

## Exact upstream `PassingMuonReco` criterion

The event-saving criterion is implemented upstream in SNDSW, not in this
repository. The relevant code is:

```text
/afs/cern.ch/user/z/zhibin/work/public/SndBuild/sndsw/
  shipLHC/scripts/run_TrackSelections.py
  shipLHC/scripts/Monitor.py
  python/SndlhcMuonReco.py
  python/TrackingParams.xml
```

For every event selected by the random sampling stage,
`Monitor.TrackSelector.ExecuteEvent` runs the enabled track-reconstruction
tasks and combines their output in `fittedTracks`. For real data the output
tree is filled only when that collection is non-empty:

```python
self.ExecuteEvent(self.eventTree)
if not self.MonteCarlo and self.fittedTracks.GetEntries() == 0:
    continue
self.fSink.Fill()
```

The exact event-level definition is therefore:

```text
PassingMuonReco == 1
    <=> at least one enabled tracking task produced a fitted track
    <=> fittedTracks.GetEntries() >= 1
```

This final event-saving decision does not add an explicit veto, US, timing,
fiducial, density, momentum, or chi2/NDF cut. Such detector information can
affect whether reconstruction succeeds, but it is not a separate final cut in
`TrackSelector.Execute`.

The enabled tasks can contribute simple or Hough-transform SciFi and
downstream-MuFilter tracks. The reconstructed run-8329 products contain the
following supported track types:

| Type | Meaning |
|---:|---|
| 1 | Simple SciFi track |
| 3 | Simple downstream track |
| 11 | Hough SciFi track |
| 13 | Hough downstream track |
| 15 | Hough combined-detector track |

For Hough reconstruction, the configured passing-muon cases require at least
three hit planes in each projection before fitting. A Hough track is stored
only when its Kalman fit converges. The standard `passing_mu_Sf` case uses
SciFi hits and a 0.1 cm association tolerance; `passing_mu_DS` uses downstream
MuFilter hits and a 1 cm tolerance. These are track-construction conditions;
the final event condition remains the union of all successfully fitted tracks
from the tasks enabled in the production command.

The exact historical command used to make the provided run-8329 directory is
not stored in `snd-ml`. Consequently, the event-saving union above is known
exactly, while the complete set of enabled tasks should be recorded from the
production provenance if it becomes available.

In `snd-ml`, `PassingMuonReco` is not recalculated. Direct feature conversion
sets it to `1` for every reconstructed input entry because upstream
`TrackSelector` has already removed events with no fitted track.

### Run-8329 efficiency

Run 8329 has sampling scale 1. The random sampling stage therefore attempts
every source event; it does not guarantee that every event is written. The
subsequent track requirement reduces the sample:

```text
source events attempted:                     298,285,666
events saved with at least one fitted track:  82,145,120
events without a saved fitted track:          216,140,546
```

The inclusive raw passing efficiency is:

```text
epsilon(PassingMuonReco)
    = 82,145,120 / 298,285,666
    = 0.275391...
    ~= 27.54 percent
```

This is the efficiency of the complete upstream configuration on all run-8329
source events. Efficiencies after an analysis selection must instead use the
same selection in both numerator and denominator.

## Metadata generation

The authoritative configuration is:

```text
snakemake/metadata/configs/passing_muon_real_data_2024.yaml
```

The dedicated builder is:

```text
snakemake/metadata/generate_passing_muon_metadata.py
```

The generic real-data metadata updater is deliberately not used. It allocates
luminosity using output `n_event`, which would be wrong for this selected
sample.

The generated files are:

```text
snakemake/metadata/updated/passing_muon_real_data_2024_metadata.csv
snakemake/metadata/updated/passing_muon_real_data_2024_excluded_files.csv
```

They can be generated with:

```bash
snakemake -s snakemake/snakefile \
  generate_passing_muon_real_data_2024_metadata \
  --cores 1
```

The builder performs the following operations for every configured run:

1. Discover reconstructed ROOT files.
2. Parse their canonical run, source index and optional chunk index.
3. Exclude files whose filename run does not equal the canonical run.
4. Map every reconstructed file or chunk to its original source ROOT file.
5. Read the source and reconstructed tree entry counts.
6. Calculate the exact attempted source range.
7. Validate duplicate ranges, missing sources, chunk gaps and chunk overlaps.
8. Join fill, scale and luminosity from the configured `pp24.csv` row.
9. Allocate full and effective luminosity from attempted source events.
10. Validate run-level luminosity closure before atomically writing the CSV.

Important metadata columns include:

| Column | Meaning |
|---|---|
| `run` | Canonical run number |
| `pp24_run` | Run label used to join the original `pp24.csv` row |
| `scale_factor` | Random sampling scale |
| `source_path` | Original converted-data ROOT file |
| `reco_path`, `digi_path` | Passing-muon reconstructed ROOT file |
| `source_file_index`, `chunk_index` | Mapping to the original source range |
| `attempt_start`, `attempt_stop` | Attempted source-tree interval, with an exclusive stop |
| `source_n_event` | Total source-tree entries |
| `n_attempted_source_events` | `attempt_stop - attempt_start` |
| `n_saved_event`, `n_event` | Saved reconstructed entries; this is a yield, not exposure |
| `input_tree` | `cbmsim` or `rawConv` |
| `run_lumi_fb` | Full run luminosity |
| `effective_run_lumi_fb` | `run_lumi_fb / scale_factor` |
| `full_lumi_per_file` | Full-exposure share assigned to the output row |
| `lumi_per_file` | Effective sampled luminosity used downstream |
| `selection_mode` | `passing_muon_direct` for newly generated metadata |
| `uses_nue_analysis_filter` | `False` for newly generated metadata |

## Luminosity normalization

For reconstructed output row `j` in run `r`, define:

```text
N_attempted(j) = number of source events attempted by that output job
N_attempted(r) = sum of N_attempted(j) over represented outputs in the run
L_run(r)       = full run luminosity in fb^-1
s(r)           = random-sampling scale factor
```

The exposure fraction and luminosities are:

```text
fraction(j) = N_attempted(j) / N_attempted(r)

full_lumi_per_file(j) = L_run(r) * fraction(j)

lumi_per_file(j) = full_lumi_per_file(j) / s(r)
```

The builder requires both closure equations:

```text
sum_j full_lumi_per_file(j) = L_run(r)

sum_j lumi_per_file(j) * s(r) = L_run(r)
```

Equivalently:

```text
sum_j lumi_per_file(j) = L_run(r) / s(r)
```

Downstream histograms use unweighted saved events and normalize a sample with:

```text
target_luminosity / sum(lumi_per_file)
```

The sampling scale is therefore already encoded exactly once in
`lumi_per_file`. Do not multiply event weights by `scale_factor` again.

## Current metadata totals

| Run | Metadata rows | Attempted events | Saved events | Full lumi (fb^-1) | Effective lumi (fb^-1) |
|---:|---:|---:|---:|---:|---:|
| 8007 | 224 | 223,811,690 | 1,117,717 | 0.327414269 | 0.001637071 |
| 8329 | 2,983 | 298,285,666 | 82,145,120 | 0.532349275 | 0.532349275 |
| 8638 | 589 | 589,000,000 | 1,448,940 | 1.068939354 | 0.002672348 |
| 9288 | 511 | 510,329,473 | 1,444,726 | 0.912102140 | 0.002606006 |
| 9446 | 486 | 485,644,348 | 1,374,846 | 0.863536274 | 0.002467246 |
| 9936 | 495 | 494,249,128 | 1,647,803 | 0.896671174 | 0.002988904 |
| **Total** | **5,288** | **2,601,320,305** | **89,179,152** | **4.601012485** | **0.544720851** |

All six runs satisfy the luminosity closure equations to floating-point
precision.

## Excluded and absent files

The audit report contains 44 records:

- 5 foreign-run files excluded because of a filename/internal run mismatch;
- 39 source files for which no reconstructed output exists.

The missing-output counts are:

| Run | Source files without reconstructed output |
|---:|---:|
| 8007 | 8 |
| 8638 | 16 |
| 9288 | 4 |
| 9446 | 11 |
| 9936 | 0 |

For the subsampled whole-file productions, absent outputs are allowed only
because an output may not have been stored when no event passed reconstruction
or when a source interval was not processed. They are always reported. Run
8329 is a full chunked production and is required to have complete, gap-free
coverage.

There is no authoritative luminosity per original source file. The run
luminosity is therefore allocated over represented attempted events and is
exact at run/sample level. File-level luminosity should be regarded as an
exposure allocation for normalization and partitioning, not an independently
measured per-file luminosity.

## Feature and hit3D production

Passing-muon reconstructed data must not run through `sndEventFilter` or the
neutrino `nueAnalysisFilter` pipeline. These reconstructed files are already a
selected passing-muon sample, and their ROOT structure is not universally
compatible with `sndEventFilter`.

The passing-muon production path is:

```text
reconstructed cbmsim/rawConv
    -> direct feature conversion
    -> direct hit3D conversion
```

For `data_type == passing_muon_real_data`, both converters:

1. Open the event tree named by metadata `input_tree`.
2. Iterate directly over every reconstructed-tree entry.
3. Apply the same optional real-data veto+US exclusion.
4. Save the original reconstructed-tree entry as `eventIndex`.
5. Do not require or open `nueAnalysisFilter_path`.

The current Snakemake configuration supplies
`--drop-real-has-veto-has-us`, so events containing both a valid veto hit and a
valid upstream MuFilter hit are removed consistently from feature and hit3D
outputs.

Feature files contain a minimal, honest pass-through `cutFlowSummary` with:

```text
entry
PassingMuonReco
```

It has one row per retained `sndData` entry, and:

```text
sndData.eventIndex[i] == cutFlowSummary.entry[i]
```

The workflow does not fabricate `StableBeams`, `IP1`, `EventDeltat`,
`SciFiMinHits`, or neutrino cutflow results. Analyses requiring those branches
must not silently interpret their absence as passing the cut.

Output provenance records:

```text
selection_mode = passing_muon_direct
input_tree = cbmsim or rawConv
external_cutflow_used = 0
```

Existing MC and inclusive real-data samples retain their external-cutflow
workflow and are unaffected by this direct mode.

## Run-8329 data-quality intersection used by comparison v2

Comparison v2 does not use direct-inclusive real-data production. Its
denominator is the existing run-8329 real-data feature sample after the
external data-quality cutflow. The existing passing-muon track features are
filtered, one original source file at a time, by exact membership in that
denominator:

```text
(source_file_index, runId, eventId)_passing
    must occur in
(source_file_index, runId, eventId)_real_data_DQ
```

No cutflow is rerun. `eventIndex` is not used for cross-product matching
because it can be local to a reconstructed chunk. Duplicate keys, wrong-run
events, incomplete source mappings, and unmatched retained events are fatal
validation errors. The workflow reports both:

```text
PassingMuonReco efficiency after DQ = N(PassingMuonReco intersect DQ) / N(DQ)
passing-muon DQ acceptance          = N(PassingMuonReco intersect DQ) / N(PassingMuonReco)
```

For run 8329, the validated event-key accounting is:

| | DQ yes | DQ no | Total |
|---|---:|---:|---:|
| `PassingMuonReco` yes | 11,025,227 | 71,119,893 | 82,145,120 |
| `PassingMuonReco` no | 90,755 | 216,049,791 | 216,140,546 |
| Total | 11,115,982 | 287,169,684 | 298,285,666 |

This gives:

```text
raw pre-DQ passing efficiency       = 82,145,120 / 298,285,666 = 27.539077%
passing efficiency within DQ        = 11,025,227 / 11,115,982  = 99.183563%
DQ events without a passing track   =     90,755 / 11,115,982  =  0.816437%
DQ acceptance within passing sample = 11,025,227 / 82,145,120  = 13.421646%
```

The 90,755 missing events pass the external DQ requirement but were not saved
in the reconstructed passing-muon output. The verified SNDSW writer therefore
had no fitted track from any enabled tracking task for those events. DQ hit
requirements alone do not guarantee successful track fitting. The available
comparison validation contains the combined DQ result, not sequential counts
after each individual DQ flag.

The full plotting-selection count table and output interpretation are recorded
in `document/PASSING_MUON_RUN8329_COMPARISON_V2_PLAN.md`.

## Production commands

Inspect one production batch:

```bash
snakemake -s snakemake/snakefile \
  production \
  --batch production=1/5289 \
  -n
```

Run it locally with one core:

```bash
snakemake -s snakemake/snakefile \
  production \
  --batch production=1/5289 \
  --cores 1
```

For a passing-muon feature or hit3D target, the DAG should contain only the
corresponding converter job. It should not schedule `process_nueAnalysis`.

## Validated output behavior

A completed run-8007 feature file and a run-8329 hit3D file were inspected in
detail.

Feature validation confirmed:

- output count equals input count minus the veto+US exclusions;
- feature and pass-through cutflow entry counts agree;
- `eventIndex` and cutflow `entry` agree exactly;
- no non-finite scalar values;
- detector totals equal station sums;
- timing-vector lengths agree;
- real-data truth fields use their documented `-999` sentinel.

Hit3D validation confirmed:

- output count equals input count minus the veto+US exclusions;
- event indices are unique and strictly increasing;
- all 16 hit vectors have equal length in every event;
- no empty retained events or non-finite vector values;
- detector, index-validity and index-type codes are valid;
- unavailable hit times use the documented `-999` sentinel.

## Rules for correct use

1. Normalize with `lumi_per_file`, not `n_saved_event`.
2. Do not apply `scale_factor` again when using `lumi_per_file`.
3. Treat `n_saved_event` as a passing-muon yield only.
4. Treat efficiencies as conditional on the upstream track reconstruction.
5. Use canonical `run`, not `pp24_run`, for run identity.
6. Keep foreign-run files excluded.
7. Use metadata `input_tree`; run 8329 differs from the other runs.
8. Do not require neutrino cutflow branches from direct passing-muon outputs.
9. When splitting the sample, balance partitions using `lumi_per_file`, not
   row count.
10. For event-by-event feature/hit3D comparisons, use outputs generated from
    the same metadata row and compare their `eventIndex` sequences.
