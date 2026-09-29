# Run-8329 Passing-Muon Track-Type and Reconstruction-Failure Study

## Purpose

This study will compare detector and event features according to which type of
muon track was reconstructed. It will also compare reconstructed events with
the 0.82% of real-data DQ events that were not saved by the upstream
`PassingMuonReco` production.

The study has three stages:

1. For each target track type, compare events with that track, other passing
   events without that track, and events with no `PassingMuonReco` result.
2. Compare the different track types within the passing-muon sample.
3. Repeat both comparisons after applying increasingly strict track
   chi-square/NDF requirements.

This is an asymmetric track study and must be implemented separately from the
existing symmetric comparison-v2 workflow. Track branches exist only in the
passing-muon features, so the same track expression cannot be applied to the
real-data denominator.

## Implementation status and commands

The isolated v3 workflow described here is implemented. It reuses the validated
v2 real-DQ and passing-DQ partitions and does not change any v2 output or
selection semantics.

Implemented files:

```text
snakemake/metadata/configs/region_evaluation/passing_muon_track_study_v3.yaml
convertData/build_no_passing_muon_partition.py
convertData/validate_passing_muon_track_study_v3.py
evaluation_region_partitions/make_passing_muon_track_type_hist.py
evaluation_region_partitions/plot_passing_muon_track_type.py
evaluation_region_partitions/test_passing_muon_track_study_v3.py
snakemake/rules/passing_muon_track_study_v3.smk
```

Run the synthetic end-to-end test in the LCG environment:

```bash
cd /afs/cern.ch/user/z/zhibin/work/snd-ml
source snakemake/envs/env_lcg.sh
python evaluation_region_partitions/test_passing_muon_track_study_v3.py
```

Inspect the complete production DAG:

```bash
snakemake -s snakemake/snakefile \
  passing_muon_run8329_track_study_v3 \
  --cores 1 --rerun-triggers mtime -n
```

Run the complete production:

```bash
snakemake -s snakemake/snakefile \
  passing_muon_run8329_track_study_v3 \
  --cores 4 --rerun-triggers mtime
```

The `--rerun-triggers mtime` option avoids an unrelated provenance-triggered
attempt to overwrite the protected passing-muon metadata file in the current
workspace. It does not weaken the explicit v2/v3 validation dependencies in
this workflow.

## Input population and terminology

The baseline population is the existing run-8329 real-data sample after its
combined external data-quality selection:

```text
RealDataDQ = 11,115,982 events
```

Exact event-key matching partitions this population into:

```text
PassingMuonReco = 11,025,227 events
NoPassingMuonReco = 90,755 events

PassingMuonReco + NoPassingMuonReco = RealDataDQ
```

`NoPassingMuonReco` is the official name for the missing 0.82% population. It
means that the event passed the real-data DQ selection but the enabled upstream
tracking tasks supplied no fitted track that caused the event to be saved. It
does not claim that the event could never be reconstructed by another tracking
configuration.

In this document, “real data with no selection” means no additional plotting
selection beyond the existing DQ requirement. It does not mean all 298,285,666
raw events in run 8329. A comparison against the complete raw run would require
a separate inclusive feature production.

## Supported track types and analysis categories

The current passing-muon feature schema contains five supported track types:

| Study key | Track type | Feature prefix | SNDSW type |
|---|---|---|---:|
| `simple_scifi` | Simple SciFi | `reco_scifi_simple` | 1 |
| `simple_ds` | Simple downstream | `reco_ds_simple` | 3 |
| `hough_scifi` | Hough SciFi | `reco_scifi_hough` | 11 |
| `hough_ds` | Hough downstream | `reco_ds_hough` | 13 |
| `hough_combined` | Hough combined-detector | `reco_combined_hough` | 15 |

The implemented study adds two event-level intersection categories:

| Study key | Required component tracks |
|---|---|
| `simple_scifi_and_ds` | Converged Simple SciFi AND converged Simple DS |
| `hough_scifi_and_ds` | Converged Hough SciFi AND converged Hough DS |

These intersections are not new SNDSW track types. In particular,
`hough_scifi_and_ds` requires separate type-11 and type-13 tracks, whereas
`hough_combined` requires one type-15 combined-detector track.

There is no US-only reconstructed-track type in the current schema. `US`
refers to upstream MuFilter event activity, such as `count_us`. The available
MuFilter track types are downstream (`DS`). US activity can be studied later
as a detector-feature selection, but it must not be labelled as a US track.

For every prefix, the important track-presence and quality branches are:

```text
<prefix>_has_any
<prefix>_has_converged
<prefix>_has_best
<prefix>_n_all
<prefix>_n_converged
<prefix>_best_chi2
<prefix>_best_ndf
<prefix>_best_chi2ndf
<prefix>_best_npoints
```

The baseline track-presence definition uses `<prefix>_has_converged == 1`.
`has_any` is not sufficient because it may include a track object whose fit is
not marked as converged.

## Study 1: three populations for each target track type

For each target track type `T`, construct three disjoint populations.

### A. `HasTargetTrack`

The event is in `PassingMuonReco` and has at least one converged target track:

```text
HasTargetTrack(T) =
    PassingMuonReco AND <T_prefix>_has_converged == 1
```

### B. `OtherPassingReco`

The event is in `PassingMuonReco` but does not have a converged target track:

```text
OtherPassingReco(T) =
    PassingMuonReco AND <T_prefix>_has_converged == 0
```

This population may contain other track types. For example,
`OtherPassingReco(simple_scifi)` can contain a Hough SciFi, simple DS, Hough
DS, or Hough combined track. It must not be labelled “no SciFi track,” because
it may still contain a different SciFi track. The precise label is “passing
without a converged Simple SciFi track.”

### C. `NoPassingMuonReco`

The event is in `RealDataDQ` but absent from `PassingMuonReco`:

```text
NoPassingMuonReco = RealDataDQ minus PassingMuonReco
```

Construct this population by an exact anti-join on:

```text
(source_file_index, runId, eventId)
```

Do not use `eventIndex` across the real and reconstructed products because it
can be local to a reconstructed chunk.

### Required count closure

For every target type, require:

```text
N(HasTargetTrack) + N(OtherPassingReco) = 11,025,227

N(HasTargetTrack)
  + N(OtherPassingReco)
  + N(NoPassingMuonReco)
  = 11,115,982

N(NoPassingMuonReco) = 90,755
```

The three populations must be mutually exclusive. The closure checks must be
stored in the output metadata and treated as fatal validation requirements.

### Features to compare

Study 1 can compare only branches common to all three populations. Initial
priority features are:

```text
count_veto and qdc_veto
earliest valid veto time
count_scifi and qdc_scifi
SciFi counts by station when available
density_sndsw_scifi
average and QDC-weighted SciFi x/y positions
count_us and qdc_us
count_ds and qdc_ds
Region-v2 veto/SciFi two-dimensional features
source-file or event-time stability variables when available
```

Track-only quantities such as chi-square, momentum, direction, start position,
and track points are undefined for `NoPassingMuonReco` and must not be filled
with a sentinel and compared as ordinary values.

The common-feature set contains all six unordered 2D pairs among veto hit
count, inclusive earliest veto-hit time, US hit count, and DS hit count:

```text
earliest veto time vs veto count
veto count vs US count
veto count vs DS count
earliest veto time vs US count
earliest veto time vs DS count
US count vs DS count
```

For the inclusive time coordinate, events with zero veto hits are retained at
`-1 ns`. Events with one or more veto hits but an absent/non-finite invalid
time are excluded from timing-dependent plots and their per-feature selected
counts are recorded. The separate “earliest valid veto-hit time” 1D feature
continues to require a real veto hit and a valid time.

The SciFi-QDC plotting range is `0 <= QDC < 30000`; values at or above 30000
are folded into the final bin so the event count is preserved.

### Plot format

Produce one multi-page PDF per target track type.

For every 1D common feature:

- overlay the three unit-normalized shapes;
- show the absolute event count for each population in the legend;
- use `HasTargetTrack` as the reference in the lower pad;
- draw `OtherPassingReco / HasTargetTrack` and
  `NoPassingMuonReco / HasTargetTrack` as normalized-shape ratios;
- omit ratio points where the reference bin is empty;
- label the lower pad “normalized shape ratio,” not “efficiency.”

Unit normalization is required because the 90,755-event
`NoPassingMuonReco` population is much smaller than the passing population.
Absolute population yields must still be reported on the first summary page.

For every 2D common feature:

- draw the three normalized densities side by side;
- use identical x/y binning;
- use a shared z range across the three panels;
- show the population name and event count on each panel;
- do not make a 2D efficiency or ratio panel in this first implementation.

The seven baseline target-category PDFs are:

```text
simple_scifi.pdf
simple_ds.pdf
simple_scifi_and_ds.pdf
hough_scifi.pdf
hough_ds.pdf
hough_scifi_and_ds.pdf
hough_combined.pdf
```

The `NoPassingMuonReco` distribution is identical in all seven PDFs. Repeating
it is intentional because it provides the same reconstruction-failure
reference for each target type.

## Study 2: compare track types within PassingMuonReco

Use only the 11,025,227 `PassingMuonReco` events after DQ intersection. For
each of the five physical track types and two intersection categories, apply
the configured one- or two-component convergence expression. For one physical
type this reduces to:

```text
<prefix>_has_converged == 1
```

Compare the normalized common-feature distributions of all seven categories
in one multi-page PDF:

```text
all_track_types.pdf
```

The primary comparison should use the same detector and event features as
Study 1. Homologous track properties such as `npoints`, start position, and
direction may be placed in a separate track-properties section, but
chi-square/NDF comparisons belong to Study 3.

### Track-type overlap

The seven samples are inclusive and may overlap because one event can contain
multiple track types. Therefore:

- do not add the seven event counts together;
- do not describe their fractions as an exclusive composition;
- add a summary page with each type's event count and fraction of passing
  events;
- add a 7-by-7 row-conditional overlap matrix showing
  `100 * N(row and column) / N(row)`; every non-empty diagonal is therefore
  100%, and the matrix is generally asymmetric;
- report the fraction with `reco_track_n_types_present > 1`.

This retains the physical information without imposing an arbitrary rule for
assigning a multi-track event to a single type.

### Four SciFi/DS states within each algorithm

In addition to the seven inclusive categories, build a disjoint comparison
for Simple tracking and another for Hough tracking. Let `S` mean that the
algorithm's separate SciFi component passes the active working point and `D`
mean that its separate DS component passes it. Partition every
`PassingMuonReco` event into exactly one state:

```text
SciFi=0, DS=0: NOT S AND NOT D
SciFi=1, DS=0: S AND NOT D
SciFi=0, DS=1: NOT S AND D
SciFi=1, DS=1: S AND D
```

Compare those four passing populations with `NoPassingMuonReco`, producing:

```text
simple_track_combinations.pdf
hough_track_combinations.pdf
```

At every working point, require the four passing counts to sum exactly to
`N(PassingMuonReco)`, and require those four plus `NoPassingMuonReco` to close
to `N(RealDataDQ)`. For the Hough comparison, the bits refer only to the
separate type-11 SciFi and type-13 DS tracks. A type-15 combined-detector Hough
track does not set either bit and therefore falls in the `SciFi=0, DS=0`
state unless separate type-11 or type-13 tracks are also present.

The algorithm PDFs contain the same common 1D and 2D feature comparisons. The
1D ratio reference is the `SciFi=1, DS=1` shape. Track-quality 2D pages are not
added to these PDFs because no single score is defined across all four states,
especially `SciFi=0, DS=0`.

## Study 3: repeat with chi-square/NDF quality requirements

Repeat Studies 1 and 2 using increasingly strict chi-square/NDF requirements.
Lower chi-square/NDF is treated as better track-fit quality.

For a target category `T` containing one or more component prefixes and a
threshold `q`, define:

```text
GoodTargetTrack(T, q) =
    PassingMuonReco
    AND every component has_best == 1
    AND every component best_chi2ndf < q

OtherPassingReco(T, q) =
    PassingMuonReco AND NOT GoodTargetTrack(T, q)

NoPassingMuonReco =
    RealDataDQ minus PassingMuonReco
```

`has_best` is required for the quality selection. The existing feature
extractor defines the best track only among tracks that are converged, have
positive NDF, and have finite non-negative chi-square/NDF. If more than one
valid target-type track exists, the one with the smallest chi-square/NDF is
used, with larger `npoints` and then input order as tie-breakers.

The three populations remain disjoint and exhaustive at every threshold:

```text
N(GoodTargetTrack(q)) + N(OtherPassingReco(q)) = 11,025,227

N(GoodTargetTrack(q))
  + N(OtherPassingReco(q))
  + 90,755
  = 11,115,982
```

`OtherPassingReco(T, q)` intentionally contains:

- passing events with no target-type track;
- passing events with a target track that is not a valid best candidate;
- passing events whose best target track fails the chi-square/NDF threshold;
- passing events reconstructed only by other track types.

Report separately how many events have `<prefix>_has_converged == 1` but
`<prefix>_has_best == 0`. This identifies target tracks for which a valid
chi-square/NDF could not be formed and prevents silent event loss.

### Initial working points

Use configurable working points rather than hard-coding them in plotting code:

```yaml
chi2ndf_working_points:
  no_quality_cut: null
  loose: 50
  medium: 20
  tight: 10
```

The numerical thresholds above are initial exploratory values. Before freezing
a final analysis, inspect the type-specific chi-square/NDF distributions and
document why the retained thresholds are appropriate. The same numerical cut
does not necessarily have identical performance across track types because
the fits use different detector information.

### Track-quality 2D features

For target-category events with valid best-track scores, each PDF also shows:

```text
Veto hit count versus best chi-square/NDF
US hit count versus best chi-square/NDF
DS hit count versus best chi-square/NDF
SciFi hit count versus best chi-square/NDF
SciFi density versus best chi-square/NDF
```

The score is on the x-axis and detector activity is on the y-axis. The score
range is fixed to 0--300 across working points, with overflow folded into the
last bin. Detector-activity overflow is also folded. The maps are normalized
by their valid target-category entries, and all-category pages use a shared z
scale.

For the two intersection categories, define the displayed joint score as:

```text
joint score = max(best SciFi chi-square/NDF, best DS chi-square/NDF)
```

This is the worse component score and is consistent with requiring both
component scores below the working-point threshold. `OtherPassingReco` and
`NoPassingMuonReco` do not enter these maps because the relevant target score
is not defined for them.

For each non-null threshold:

1. Repeat the seven Study-1 target PDFs using `GoodTargetTrack`.
2. Repeat `all_track_types.pdf` using the seven good-track selections.
3. Add a summary page with the retained yield and efficiency versus threshold
   for every track type.
4. Record the exact `<` expression and threshold in the ROOT and PDF metadata.

## Output organization

Use a new versioned output area and do not change the meaning or output
contract of comparison v2:

```text
evaluation_region_partitions/passing_muon_run8329_track_study_v3/
  no_quality_cut/
    simple_scifi.pdf
    simple_ds.pdf
    simple_scifi_and_ds.pdf
    hough_scifi.pdf
    hough_ds.pdf
    hough_scifi_and_ds.pdf
    hough_combined.pdf
    all_track_types.pdf
    simple_track_combinations.pdf
    hough_track_combinations.pdf

  chi2ndf_lt_50/
    simple_scifi.pdf
    simple_ds.pdf
    simple_scifi_and_ds.pdf
    hough_scifi.pdf
    hough_ds.pdf
    hough_scifi_and_ds.pdf
    hough_combined.pdf
    all_track_types.pdf
    simple_track_combinations.pdf
    hough_track_combinations.pdf

  chi2ndf_lt_20/
    ...

  chi2ndf_lt_10/
    ...
```

Store one ROOT histogram file per working point and reuse it for that working
point's seven target PDFs, `all_track_types.pdf`, and two algorithm-combination
PDFs. There are 27 common-feature pages (15 one-dimensional and 12
two-dimensional). Each target/all-types PDF contains one summary page, 27
common-feature pages, and five target-quality 2D pages, for 33 pages. Each
algorithm-combination PDF contains one summary plus the 27 common-feature
pages, for 28 pages. The full production therefore contains four ROOT files,
40 track-study PDFs, and one MC-reference PDF. No per-feature CSV files are
needed. Store counts, expressions,
target categories, working point, feature order, event-key provenance, and
closure results in each ROOT file.

## Implementation structure

Keep this study separate from
`make_passing_muon_efficiency_hist.py`, whose current contract applies one
symmetric selection to real and passing samples.

The implemented workflow contains:

1. A source-aligned anti-join step that builds and validates the
   `NoPassingMuonReco` feature population from real DQ minus passing DQ.
2. A track-study YAML configuration containing seven one- or two-component
   categories, two algorithm-bit comparisons, their labels, common features,
   five track-quality 2D features, and chi-square/NDF working points.
3. A histogram builder that reads real DQ, passing DQ, and
   `NoPassingMuonReco`; constructs `HasTargetTrack`/`GoodTargetTrack` and
   `OtherPassingReco`; and books all populations in a small number of
   `RDataFrame` graph passes.
4. A plotter that creates one multi-page PDF per target/working-point pair,
   one all-track-types PDF, and two algorithm-combination PDFs per working
   point.
5. Synthetic tests containing overlapping track types, missing events,
   invalid NDF, non-finite chi-square, threshold boundary values, empty bins,
   and histogram flow entries.
6. Snakemake targets for the missing-event construction, histogram production,
   plots, tests, and the complete study.

## Upward-muon MC detector reference

V3 additionally uses one explicitly configured `MC_muon/up` production:

```text
subfolder: up
partition: scoring_at449m
feature: feature_MC_muon_up_scoring_at449m_14.root
```

The input is validated and standardized by
`prepare_mc_muon_up_reference_v3.py`. The validator requires a readable,
nonempty `sndData` tree, exact agreement between the metadata and tree event
counts, and every detector branch referenced by the shared histogram
configuration. Reconstructed-track branches are neither present nor required.

The plotted detector-response reference is

```cpp
std::abs(pdgCode) == 13 &&
(count_veto + count_scifi + count_us + count_ds) > 0
```

It is named `MCMuonUpDetectorActivity`. It is not described as a
truth-trajectory selection because the available feature file does not contain
a usable truth intersection with each detector plane.

Production inspection found 1,228,583 inclusive muons and 93,742 events with
some detector activity in the configured file. Critically, the file contains
zero events with `count_scifi > 0`. Its SciFi distributions and both SciFi x-y
MC panels will therefore be empty by construction. This fact is retained in
the output rather than hidden by changing the requested sample. MC-muon-up is
useful for diagnosing veto, US and DS response, but it must not be used to tune
the v5 SciFi threshold. A compatible through-going or downward MC production
is required for that later calibration.

No station-coverage or upper-multiplicity requirement is applied: those
quantities remain inputs to the subsequent v5 cut study where the MC schema is
applicable.

The dedicated output is written once, under the no-quality-cut directory:

```text
evaluation_region_partitions/passing_muon_run8329_track_study_v3/
  no_quality_cut/mc_muon_up_reference.pdf
```

Its summary is followed by all 27 common feature pages. One-dimensional pages
compare unit-normalized MC, RealDataDQ, PassingMuonReco and
NoPassingMuonReco shapes, with data/MC shape ratios. Two-dimensional pages use
a 2-by-2 layout with a common normalized z scale. They include both the
unweighted `scifi_x_vs_y` map and `qdc_weighted_scifi_x_vs_y`.

All four one-dimensional SciFi position features reject non-finite values and
the `-999` missing-position sentinel. Their detector-coordinate underflow and
overflow are not folded into visible edge bins. Consequently an MC sample with
no SciFi hits is reported with `valid N=0`, not as a false peak at the edge of
the detector.

MC is never added to the reconstructed-track overlap, Simple/Hough component,
or chi-square/NDF studies. It is a reference shape and is not part of the
real-data closure equation.

## Required validation

Before treating the plots as analysis results, require:

```text
all events have runId == 8329
event keys are unique within each population
PassingMuonReco is an exact subset of RealDataDQ
NoPassingMuonReco is the exact anti-join
PassingMuonReco and NoPassingMuonReco are disjoint
NoPassingMuonReco contains exactly 90,755 events
each target's Has/Other populations are disjoint
each target's Has/Other counts close to 11,025,227
each complete three-way comparison closes to 11,115,982
each algorithm's four SciFi/DS states are disjoint and close to 11,025,227
each algorithm's four states plus NoPassingMuonReco close to 11,115,982
common feature schemas and binning are identical
quality selections require has_best and finite valid chi-square/NDF
all output files store selection and provenance metadata
```

## Final analysis sequence

The intended progression is:

```text
Baseline track presence
    HasTargetTrack vs OtherPassingReco vs NoPassingMuonReco

Track-method comparison
    seven inclusive physical/intersection categories within PassingMuonReco

Track-quality study
    repeat both comparisons at stricter chi-square/NDF working points
```

This organization answers three distinct questions without mixing their
denominators:

1. How do events containing a particular track type differ from other passing
   events and from reconstruction failures?
2. How do the detector topologies associated with the five physical track
   types and two SciFi-and-DS intersection categories differ?
3. How do those differences change when progressively better fitted tracks are
   required?
