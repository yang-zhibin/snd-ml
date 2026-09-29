# Passing-muon run 8329 detector-only selection study v5

## Purpose and current status

V5 partitions the complete run-8329 `RealDataDQ` sample using detector-hit
topology only. It no longer constructs populations from `PassingMuonReco` or
its missing 0.82% complement, and it never reads a reconstructed-track branch.

The study has two stages:

1. run topology selections and data-only multiplicity scans now;
2. enable calibrated clean-muon working points after v3 provides a
   through-going MC sample with usable SciFi activity.

The two population names depend on the stage:

```text
Topology-only point:
    ThroughGoingTopologyCandidate
    TopologyFailureComplement

Final topology + multiplicity point:
    CleanThroughGoingMuonCandidate
    ShowerEnrichedComplement
```

The complement is called shower enriched, not a pure shower sample. It can
also contain detector inefficiency, geometrical losses, stopping muons, noise,
and other events that fail the clean topology.

## Input population

The histogram builder reads only `real_data_dq_run8329` from the validated v2
partition metadata. The production expectation is:

```text
N(RealDataDQ) = 11,115,982
```

The previous v5 inputs `passing_muon_dq_run8329` and
`no_passing_muon_reco_run8329` are not used. This makes the selection
independent of track reconstruction by construction.

V5 owns an independent upward-muon MC input contract and no longer reuses the
single-file V3 reference. It selects all 15 upward-muon metadata rows and
keeps the four source productions separate:

| Production | Files | Declared events |
|---|---:|---:|
| `scoring_1.8` | 3 | 697,641 |
| `scoring_2` | 10 | 912,922 |
| `scoring_2.5` | 1 | 260,258 |
| `scoring_at449m` | 1 | 1,228,583 |

The baseline is `abs(pdgCode) == 13` with activity in at least one detector
system. Every working point is applied independently to every production.
The ROOT output records per-production candidate and complement counts,
efficiencies, closure, luminosity, provenance, and feature histograms.

No combined-production histogram is produced because no physics weighting
prescription has been defined. Aggregate MC counts remain as compatibility
metadata only. The four feature files that existed when this workflow was
implemented all had zero SciFi activity; empty selected populations are kept
as valid diagnostics rather than hidden or treated as job failures.

## Detector coverage

Station-level activity, rather than a total hit count, establishes whether an
event crosses a detector system.

Veto coverage is

```cpp
count_veto1 > 0 && count_veto2 > 0 && count_veto3 > 0
```

US coverage is

```cpp
count_us1 > 0 && count_us2 > 0 && count_us3 > 0 &&
count_us4 > 0 && count_us5 > 0
```

The active DS-station count is

```cpp
int(count_ds1 > 0) + int(count_ds2 > 0) +
int(count_ds3 > 0) + int(count_ds4 > 0)
```

SciFi coverage uses the existing `scifi_n_active_stations_xy` branch, which
requires activity in both views of each counted station.

Two topology working points are enabled initially:

| Working point | Veto | SciFi X/Y stations | US | DS stations |
|---|---:|---:|---:|---:|
| `coverage_loose` | all 3 | at least 4 | all 5 | at least 3 |
| `coverage_tight` | all 3, exactly 1 hit/station | all 5, fewer than 10 combined X+Y hits/station | all 5, exactly 1 hit/station | all 4 |

The tight point's multiplicity conditions are strict inequalities. Full Veto
and US coverage combined with fewer than two hits per station therefore means
exactly one hit in every station. The feature schema stores `count_scifiN` as
the combined X+Y count for station `N`; requiring all five active X/Y stations
still guarantees both views are present, and the new limit requires
`count_scifiN < 10` for every station.

An event with three veto hits concentrated in one station does not pass. An
event with five US hits but no activity in one US station also does not pass.

## Exact candidate/complement closure

For each configured point with expression `S`, the implementation constructs

```cpp
candidate  = S
complement = !(S)
```

and requires

```text
N(candidate) + N(complement) = N(RealDataDQ).
```

The job stops on a closure failure, an unexpected RealDataDQ count, or a
negative detector count. Every expression, component cut, event count,
per-feature valid count, and closure flag is written to ROOT metadata.

## Data-only multiplicity scans

Four independent upper-limit scans run after `coverage_loose`:

```text
count_veto  <= VETO_MAX
count_scifi <= SCIFI_MAX
count_us    <= US_MAX
count_ds    <= DS_MAX
```

For every threshold, v5 stores:

```text
selected event count
selected fraction of complete RealDataDQ
selected fraction of the base topology population
```

These are diagnostic scan points, not calibrated cuts. In particular,
`count_scifi = 100` is one scan point and is not treated as a final boundary.

The diagnostics PDF contains:

- one study-definition page;
- one cumulative cutflow per enabled working point;
- one page per multiplicity scan.

## Final working points and N-minus-one plots

The configuration contains a disabled `nominal` final working point. A final
point selects one coverage definition and may set any of:

```yaml
veto_max: null
scifi_max: null
us_max: null
ds_max: null
```

`null` disables that particular maximum. At least one maximum is required
when a final point is enabled.

For every enabled maximum, the builder automatically produces an N-minus-one
distribution: all other cuts are applied, while the displayed upper-limit
cut is split into pass and fail populations. This prevents the threshold
variable from being hidden by its own final selection. The selection PDF adds
one N-minus-one page for every enabled maximum.

## Features and plotting

V5 uses every common feature from
`passing_muon_run8329_hist_v2.yaml` and adds:

```text
scifi_n_active_stations_xy
ds_n_active_stations
count_scifi versus count_us
count_scifi versus count_ds
```

The common list already supplies `count_veto versus count_scifi`,
`count_veto versus count_us`, `count_veto versus count_ds`, and
`count_us versus count_ds`, as well as both SciFi X-versus-Y maps.

Each enabled working point has one data-only PDF:

- 1D pages overlay normalized candidate and complement shapes;
- the lower panel is normalized complement divided by normalized candidate;
- legends show valid event counts in scientific notation;
- 2D pages show candidate and complement side by side with a shared z scale;
- no efficiency subplot is used for 2D plots.

`mc_muon_up_summary.pdf` reports the independent production baselines and
selection efficiencies. For every working point and MC production, an
additional `<selection>__mc__<production>.pdf` compares the normalized
real-data candidate with that production's normalized MC candidate. This
avoids obscuring provenance in one six-curve plot.

## Updating the selection after the MC result

After all productions have been generated and validated:

1. apply exactly the same station-coverage definitions to MC;
2. overlay the MC multiplicity distributions with coverage-conditioned data;
3. compute MC efficiency versus each threshold and combined working point;
4. choose loose, nominal, and tight thresholds from explicit efficiency and
   background-enrichment targets;
5. edit only `selection_working_points` in the YAML and set `enabled: true`;
6. rerun v5 without modifying the builder or plotter.

Distributions used directly by the selection must be interpreted through the
N-minus-one pages. Candidate/complement differences in those variables are
otherwise expected cut sculpting rather than independent validation.

## Outputs

The ROOT file is written to:

```text
/eos/experiment/sndlhc/users/zhibin/snd-ml/evaluation_region_partitions/
passing_muon_run8329_detector_selection_study_v5/histograms/detector_selection.root
```

The current PDFs are written under:

```text
evaluation_region_partitions/passing_muon_run8329_detector_selection_study_v5/
```

with the initial files:

```text
diagnostics.pdf
mc_muon_up_summary.pdf
coverage_loose.pdf
coverage_tight.pdf
coverage_loose__mc__scoring_1p8.pdf
coverage_loose__mc__scoring_2.pdf
coverage_loose__mc__scoring_2p5.pdf
coverage_loose__mc__scoring_at449m.pdf
```

Additional enabled working points automatically add one PDF using the working
point name.

## Files

- Configuration: `snakemake/metadata/configs/region_evaluation/passing_muon_binary_region_study_v5.yaml`
- MC metadata preparation: `snakemake/metadata/prepare_mc_muon_up_reference_v5.py`
- Inclusive MC cutflow writer: `convertData/make_identity_cutflow.py`
- Histogram builder: `evaluation_region_partitions/make_passing_muon_binary_region_hist_v5.py`
- Plotter: `evaluation_region_partitions/plot_passing_muon_binary_region_study_v5.py`
- Synthetic test: `evaluation_region_partitions/test_passing_muon_binary_region_study_v5.py`
- Workflow: `snakemake/rules/passing_muon_binary_region_study_v5.smk`

The V5 metadata-preparation rule explicitly depends on all 15 feature paths.
Snakemake therefore applies its normal missing-file and input-freshness checks
to every source. Every resulting file is checked for a readable `sndData`
tree, exact metadata/tree event-count agreement, and all V5 branches. The
validation CSV also flags the existing `scoring_2` metadata whose geometry
path contains `scoring_2.5`.

At implementation time, 11 feature outputs and 11 corresponding filter
outputs were missing. The corrected dry-run schedules all 15 identity
cutflows and all 15 feature jobs, deliberately giving the final set a
consistent selection mapping and feature schema.

The historical neutrino `sndEventFilter` pipeline aborts on the older
`scoring_2` MC files. For `data_type == MC_muon` only, `process_nueAnalysis`
therefore writes an inclusive one-to-one `cutFlowSummary` with one `entry` row
per `cbmsim` event. The feature converter continues to use its established
external-cutflow interface, but no physics selection is applied. Neutrino,
real-data, muon-DIS, and hit3D processing retain their existing behavior.

The old `scoring_2` and `scoring_2.5` digi files also persist a legacy
`FairEventHeader`, while the current detector `InitEvent` API accepts an
`SNDLHCEventHeader`. MC-muon feature jobs use the compatibility wrapper
`convertData/digi_2_features_mc_muon.py`: it skips per-event detector
initialization only when it detects `FairEventHeader`, while retaining normal
initialization for current MC-muon files. Static geometry is still loaded and
used for hit positions. Other data types continue to call the shared converter
directly.

For `coverage_tight`, V5 also writes an explicit `all` MC comparison. Each bin
is the unweighted sum of the corresponding bins from all four configured
MC-muon-up productions (all 15 source files). The output is diagnostic: it is
an event-count mixture, not a luminosity-weighted prediction. The individual
production histograms and PDFs remain available unchanged.

The synthetic test includes two independent MC productions, multiple-group
closure, one deliberately SciFi-empty MC production, metadata validation,
station-coverage boundaries, threshold scans, N-minus-one output, and all 1D
and 2D plot modes.

## Commands

Run the focused synthetic test:

```bash
cd /afs/cern.ch/work/z/zhibin/snd-ml
snakemake -s snakemake/snakefile \
  passing_muon_run8329_binary_region_study_test_v5 \
  --cores 1
```

Build the production histogram and all enabled PDFs:

```bash
snakemake -s snakemake/snakefile \
  passing_muon_run8329_binary_region_study_v5 \
  --cores 4
```

Histogram only:

```bash
snakemake -s snakemake/snakefile \
  passing_muon_run8329_binary_region_study_hist_v5 \
  --cores 4
```

Plots only, after the ROOT file exists:

```bash
snakemake -s snakemake/snakefile \
  passing_muon_run8329_binary_region_study_plots_v5 \
  --cores 1
```
