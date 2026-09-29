# Passing-muon background-composition implementation plan

## Implementation status (2026-09-02)

The supported V7 scope is complete.

| Item | Status |
|---|---|
| Raw-count detector PDFs | Implemented and tested |
| Detector-family N-minus-one populations | Implemented and tested |
| PassingMuonReco flat-v2 track diagnostics | Implemented and produced |
| Three-class observable composition | Implemented for `passing_muon_reco_only` |
| Raw one-dimensional class stacks and fractions | Implemented |
| Two-dimensional track/activity correlations | Implemented |
| ROOT metadata and total/class closure | Implemented |
| Isolated Snakemake targets | Implemented |
| Synthetic end-to-end test | Implemented and passing |
| Small real-data integration test | Passed on source 298 (4,850 events) |
| Inclusive RealDataDQ track decomposition | Not possible with current inputs |
| Analytic Gaussian/Poisson mixture fit | Intentionally deferred |

The 299 inclusive Run-8329 converted-data files do not provide
`Reco_MuonTracks`. Producing inclusive flat-v2 tracks would require a new
reconstruction production over the inclusive data and is outside the current
V7 input scope. The code therefore rejects an inclusive scope rather than
silently interpreting absence from PassingMuonReco as no track.

The implemented track tag requires valid SciFi-Hough and downstream-Hough best
tracks. Combined Hough was rejected after checking the completed Run-8329
track histograms: its valid-best count is zero, whereas both subsystem Hough
categories are populated. The initial composition uses existing loose/tight
coverage as the low-activity definition and introduces no unvalidated numeric
track-quality thresholds.

## 1. Objective

Investigate whether the Run-8329 real-data background is consistent with at
least two observable components:

1. a clean, approximately straight through-going muon;
2. additional or scattering activity, ranging from a small secondary deposit
   to a large shower, either with or without a reconstructed incoming track.

The first goal is descriptive: expose raw detector-count distributions without
selection-induced artifacts. The second goal is to determine whether
track-related observables support a useful data-driven decomposition. This
study must not call a category "passing muon" or "scattering" solely from a
detector cut; those names require validation.

## 2. Existing V7 baseline

V7 already provides two complementary products.

### 2.1 Shape-normalized detector reports

- `coverage_loose.pdf`
- `coverage_tight.pdf`

These compare unit-area distributions for Run-8329 data and pooled
muon-up MC. They are useful for shape comparisons but do not preserve the
number of events in each bin.

### 2.2 Raw-count detector reports

- `coverage_loose_counts.pdf`
- `coverage_tight_counts.pdf`

These use raw data event counts and detector-family N-minus-one populations.
For the detector family being plotted, that family's requirements are removed
while all other coverage requirements remain applied. Each parent population
is split into pass and fail categories, with the closure

```text
pass + fail = inclusive parent
```

checked both in total and bin by bin.

The current reports include detector totals, per-station counts, active-plane
counts, and Veto excess measurements. They are the correct starting point for
the composition study and should be retained unchanged.

### 2.3 Reconstructed-track reports

The flat-v2 PassingMuonReco sample provides event-level track multiplicities
and best-track positions, associated measurements, measured planes, slopes,
and fit quality. These distributions are conditional on a valid best track.

The present track sample covers PassingMuonReco events only. It cannot by
itself measure the full "no incoming track" component of the inclusive
RealDataDQ population.

## 3. Interpretation safeguards

The implementation must enforce the following distinctions.

### 3.1 Detector counts are discrete observables

A clean through-going muon can produce a narrow Veto peak near one measurement
per active station. This does not imply that the peak is Gaussian. Detector
inefficiency, multiple bars, cross-talk, reconstruction details, and secondary
activity can change the shape.

Back-scattering or shower activity is not assumed to follow a simple Poisson
distribution. Correlated particles and event-to-event energy variation can
produce a compound-Poisson or over-dispersed distribution. No analytic fit
should be introduced until the raw N-minus-one distributions are inspected.

### 3.2 SciFi count scale

SciFi station counts include X and Y activity. A clean track can therefore
produce approximately two measurements per fully active station before
clustering and detector effects. The expected total is not assumed to be five.

### 3.3 Selection sculpting

Full-selection histograms answer "what remains after all requirements?" They
must not be used alone to infer an underlying multiplicity distribution. The
N-minus-one histograms answer "what does this detector family look like before
its own requirement is imposed?" and are the primary composition inputs.

### 3.4 Observable classes are not truth labels

Labels in code and ROOT output should use neutral names such as
`track_like_low_activity`, `track_plus_activity`, and `no_valid_track`.
Physics interpretations can be shown in explanatory text only after validation.

## 4. Analysis populations

### 4.1 Coverage definitions

Run the complete analysis independently for:

- `coverage_loose`;
- `coverage_tight`.

Do not merge the two. Tight is nested inside loose and represents a different
selection, not an independent sample.

### 4.2 Detector-family N-minus-one parents

For each coverage definition, construct four parents:

- Veto N-minus-one: keep SciFi, US, and DS requirements;
- SciFi N-minus-one: keep Veto, US, and DS requirements;
- US N-minus-one: keep Veto, SciFi, and DS requirements;
- DS N-minus-one: keep Veto, SciFi, and US requirements.

Within each parent, store:

- `inclusive`;
- `pass` for all removed family requirements;
- `fail` for at least one removed family requirement.

The current V7 count diagnostics already implement these populations.

### 4.3 Proposed track/activity classes

The next stage should add mutually exclusive observable classes. Exact numeric
thresholds must be chosen from distributions or a documented control sample,
not inserted as unvalidated assumptions.

1. `no_valid_track`
   - no chosen incoming-track definition has `has_best == 1`;
2. `track_like_low_activity`
   - a valid chosen track exists;
   - it has sufficient associated measurements and measured planes;
   - fit-quality and endpoint requirements pass;
   - detector excess activity is below a configurable threshold;
3. `track_plus_activity`
   - a valid chosen track exists;
   - the low-activity requirement fails.

Required closure:

```text
no_valid_track + track_like_low_activity + track_plus_activity = parent
```

Before production, compare candidate incoming-track definitions based on:

- Combined Hough;
- Hough SciFi plus Hough downstream;
- Simple SciFi plus Simple downstream.

Do not silently combine them. The selected definition and fallback order must
be written into the configuration and ROOT metadata.

## 5. Required input decision

The inclusive composition study requires track information for every event in
the chosen RealDataDQ parent, including events rejected by PassingMuonReco.
Implement one of the following only after checking the source data.

### Preferred option: inclusive flat-v2 conversion

Run the same flat-v2 track extraction on the inclusive Run-8329 reconstructed
data, apply the existing real-data DQ event-key intersection, and validate the
result with exact `(runId, eventId)` keys. This provides explicit zero-track
events and avoids inferring `no_valid_track` from absence in a selected sample.

### Restricted option: PassingMuonReco-only study

If inclusive reconstructed inputs are unavailable, restrict the track/activity
decomposition to PassingMuonReco and label every output accordingly. In this
mode, do not interpret the missing part of RealDataDQ as `no_valid_track`.

The implementation must stop with a clear error if the configured analysis
scope and manifest population do not agree.

## 6. Feature definitions

### 6.1 Core detector counts

Store and plot:

- `count_veto1`, `count_veto2`, `count_veto3`, `count_veto`;
- `count_scifi1` through `count_scifi5`, `count_scifi`;
- `count_us1` through `count_us5`, `count_us`;
- `count_ds1` through `count_ds4`, `count_ds`;
- SciFi and DS active-station counts.

### 6.2 Derived activity counts

Generalize the existing Veto definitions to every detector family where the
semantics are valid.

```text
n_active_<family> = sum(count_<family><station> > 0)
```

```text
excess_<family> = count_<family> - expected_minimum_from_active_units
```

The expected minimum must be detector-specific. For Veto and MuFilter, one per
active plane is a useful diagnostic baseline. For SciFi, X/Y views and cluster
semantics require an explicitly documented definition; do not reuse the Veto
formula automatically.

Recommended combined activity observables:

- total non-track detector measurements;
- Veto excess;
- SciFi excess or per-station excess after its definition is validated;
- US excess;
- DS excess;
- maximum station multiplicity;
- number of stations above one configurable multiplicity threshold.

### 6.3 Track observables

Use flat-v2 scalar best-track features:

- `has_any`, `has_converged`, `has_best`;
- start and stop coordinates;
- endpoint `delta z` and separation;
- associated measurements;
- unique measured planes;
- `chi2/NDF` and NDF;
- `dx/dz` and `dy/dz`.

Track-property histograms must require `has_best == 1`. Missing tracks must
never be represented by zero-valued track properties.

### 6.4 Correlations

Add the following two-dimensional histograms for each selected track definition:

- Veto excess versus SciFi total count;
- Veto excess versus track-associated measurements;
- SciFi total count versus track-associated measurements;
- total detector count versus unique measured planes;
- `chi2/NDF` versus Veto excess;
- `chi2/NDF` versus SciFi total count;
- track start `z` versus Veto excess;
- endpoint separation versus total detector activity.

Use explicit underflow/overflow folding and record the chosen ranges in YAML.

## 7. PDF organization

Keep count and composition plots in separate PDFs so raw yields are never
confused with unit-area shapes.

### 7.1 Existing detector-count PDFs

Retain the current 30-page structure for each coverage selection:

1. summary and closure;
2. full-selection detector totals;
3. four detector-family N-minus-one overview pages;
4. one detailed raw-count page per detector feature.

Each detailed page contains:

- upper panel: stacked pass/fail raw event counts plus inclusive black points;
- lower panel: `pass / inclusive` with binomial uncertainty;
- a clear label stating whether overflow is folded.

### 7.2 New composition PDF

Create one PDF per coverage selection and track definition, for example:

```text
coverage_loose_composition_combined_hough.pdf
coverage_tight_composition_combined_hough.pdf
```

Recommended order:

1. scope, input population, track definition, thresholds, and closure summary;
2. raw event counts for the three observable classes;
3. detector-total overview, with classes stacked;
4. Veto detail: per-plane, total, active planes, and excess;
5. SciFi detail: per-station, total, active stations, and validated excess;
6. US and DS detail;
7. track availability and quality;
8. track/activity correlation pages;
9. loose-versus-tight efficiency summary for each observable class.

For every one-dimensional observable, use two panels:

- upper: raw stacked event counts with the inclusive closure overlay;
- lower: class fraction in each bin.

Normalized class shapes may be added on a separate page, but must never replace
the raw-count page.

## 8. Configuration changes

Add a new block to the V7 YAML only after the inclusive-input decision is made:

```yaml
background_composition:
  enabled: true
  scope: inclusive_real_data_dq  # or passing_muon_reco_only
  track_definition: combined_hough
  class_order:
    [no_valid_track, track_like_low_activity, track_plus_activity]
  track_quality:
    min_measurements: null
    min_unique_planes: null
    max_chi2ndf: null
    min_endpoint_distance_cm: null
  low_activity:
    max_veto_excess: null
    max_scifi_excess: null
    max_us_excess: null
    max_ds_excess: null
  correlations: [...]
```

`null` thresholds are intentionally invalid for enabled production. A config
validator must require finite, physically allowed values before booking any
composition histogram. Threshold selection and justification should be kept in
a separate study note or audit YAML.

## 9. Code changes

### 9.1 Shared configuration helper

Add a helper module or extend `passing_muon_v7_reco.py` to:

- validate the composition block;
- build class expressions;
- verify mutual exclusivity and complete coverage;
- expose required input branches;
- format stable ROOT directory and metadata names.

Do not duplicate selection-expression construction in the plotter.

### 9.2 Histogram builder

Add a separate builder, for example:

```text
evaluation_region_partitions/
  make_passing_muon_background_composition_v7.py
```

Responsibilities:

1. load and validate the event manifest;
2. check run, tree, schema, input population, source uniqueness, and event counts;
3. define detector activity and track-quality columns once;
4. construct loose/tight and N-minus-one parents using existing helpers;
5. construct mutually exclusive observable classes;
6. book all counts and histograms in one `RDataFrame` graph;
7. check class and histogram closure;
8. write atomically to a separate ROOT file.

Suggested ROOT hierarchy:

```text
selections/<coverage>/
  parents/<parent>/
    features/<feature>/<class>
    correlations/<feature>/<class>
```

Required metadata:

- run and study version;
- input scope and schema;
- source and event counts;
- selection, parent, class, feature, and correlation order;
- complete class expressions and thresholds;
- class totals and fractions;
- closure flags;
- overflow-folding policy.

### 9.3 Plotter

Add a dedicated plotter, for example:

```text
evaluation_region_partitions/
  plot_passing_muon_background_composition_v7.py
```

The plotter must consume stored metadata rather than reconstructing class
definitions. It should fail if the requested coverage point, track definition,
or feature order differs from the ROOT metadata.

### 9.4 Snakemake

Add separate targets so the composition study does not force regeneration of
the established detector PDFs:

```text
passing_muon_run8329_background_composition_hist_v7
passing_muon_run8329_background_composition_plots_v7
passing_muon_run8329_background_composition_v7
```

Keep outputs under the v7 directory but use unique names:

```text
histograms/background_composition_flat_v2.root
coverage_<selection>_composition_<track_definition>.pdf
```

Existing protected DQ products must remain external reference parameters, as in
the corrected v7 track workflow. New wildcard outputs must not be placed under
`snakemake/metadata/updated/`, where they would collide with `update_metadata`.

## 10. Validation and testing

### 10.1 Configuration tests

Reject:

- unknown track definitions;
- missing or non-finite thresholds;
- overlapping or incomplete class definitions;
- unknown features or detector families;
- unsupported schema or input scope.

### 10.2 Synthetic end-to-end test

Create a small ROOT tree containing at least:

- a clean three-plane Veto track with zero excess;
- a valid track plus small extra activity;
- a valid track plus large shower activity;
- an event with no valid track;
- an event with a converged but ineligible track;
- values in underflow and overflow regions.

Assert:

- exact class totals;
- class closure for loose and tight selections;
- N-minus-one parent closure;
- bin-by-bin stacked-histogram closure;
- valid-track feature counts exclude missing tracks;
- folded overflow remains in the total integral;
- expected ROOT paths and metadata exist;
- expected PDF page count is stable.

### 10.3 Small real-data integration test

Before full production, run on a documented subset of Run 8329 and record:

- manifest event count;
- duplicate and wrong-run key counts;
- class totals and fractions;
- loose/tight nesting;
- empty or unexpectedly saturated histograms;
- fraction of events with each track algorithm available.

Do not proceed to full production if class closure fails or if the chosen track
definition leaves a large unexplained reconstruction-dependent population.

### 10.4 Full-production checks

After production:

- confirm all source files are represented exactly once;
- confirm tree entries equal the manifest total;
- confirm every PDF reads successfully;
- archive the resolved YAML and ROOT metadata with the result;
- compare loose and tight class fractions without treating them as independent
  statistical samples.

## 11. Model-fitting stage

Do not include a Gaussian-plus-Poisson fit in the initial implementation.
After inspecting the raw distributions:

1. choose a control observable, preferably Veto excess rather than total Veto
   count, because it removes the active-plane baseline;
2. test dispersion of the high-activity component against a Poisson model;
3. compare Poisson, negative-binomial, and data-template descriptions;
4. treat the clean component with a discrete detector-response template rather
   than assuming a Gaussian;
5. validate fit stability across loose/tight and track definitions;
6. report goodness-of-fit and parameter covariance, not only component yields.

A template-based decomposition from validated observable classes is preferred
to imposing analytic component shapes without detector-response evidence.

## 12. Implementation sequence

1. Freeze and review the existing raw-count/N-minus-one PDFs.
2. Decide whether inclusive Run-8329 flat-v2 track input can be produced.
3. Produce and validate the inclusive track manifest, or explicitly select the
   restricted PassingMuonReco-only scope.
4. Study track algorithms and choose a configurable incoming-track definition.
5. Define and document track-quality and low-activity thresholds.
6. Implement the composition config validator and class expressions.
7. Implement the histogram builder with closure metadata.
8. Implement the raw-count composition plotter.
9. Add synthetic and small-real-data integration tests.
10. Add isolated Snakemake targets and verify the complete dry-run graph.
11. Run full production.
12. Only then evaluate whether a mixture fit is justified.

## 13. Acceptance criteria

The composition implementation is complete only when:

- raw event counts are available for both loose and tight coverage;
- detector-family N-minus-one distributions remain available and unchanged;
- the analysis scope is explicit and matches the manifest population;
- all observable classes are mutually exclusive and exhaustive;
- missing tracks are never encoded as zero-valued track properties;
- class and parent closure pass in totals and histogram bins;
- thresholds and track definitions are stored in configuration and ROOT
  metadata;
- synthetic and small-real-data tests pass;
- Snakemake dry runs without ambiguity or protected-product rebuilds;
- no existing V6 or V7 detector/track product is overwritten.
