# Detector-only clean through-going muon selection plan

> Implementation status (2026-08-12): the detector-only v5 framework is now
> implemented. It currently enables loose and tight topology points plus
> data-only multiplicity scans; the final nominal multiplicity point remains
> disabled until a v3 MC sample with usable SciFi response is available. See
> `PASSING_MUON_RUN8329_BINARY_REGION_STUDY_V5.md` for the implemented
> interfaces, outputs, tests, and commands.

## 1. Purpose

This study will replace the current v5 interpretation based on
`PassingMuonReco` with a detector-only definition of a clean through-going
muon candidate.

The scientific goal is to partition the complete run-8329 RealDataDQ sample
into two exact, disjoint populations:

```text
RealDataDQ
├── CleanThroughGoingMuonCandidate
└── ShowerEnrichedComplement
```

The selection must not use any reconstructed-track information. In
particular, it must not use branches with names such as:

```text
reco_*_has_converged
reco_*_has_best
reco_*_best_chi2ndf
reco_track_n_types_present
```

The detector-only selection should instead describe the physical topology of
a clean muon:

1. it enters through all three veto stations;
2. it crosses the SciFi system with activity in both views;
3. it crosses all five US stations;
4. it reaches the DS system;
5. it does not produce excessive activity consistent with a large shower.

The complementary sample is expected to be shower enriched. It must initially
be called `ShowerEnrichedComplement`, rather than simply `Shower`, because an
event can fail the clean-muon selection for other reasons, including detector
inefficiency, geometrical acceptance, noise, or a stopping muon. Its shower
enrichment has to be demonstrated from the resulting distributions.

## 2. Why total hit counts are not station-coverage requirements

The following expression is not a reliable statement that a muon crossed all
three veto stations:

```cpp
count_veto == 3
```

It can accept three hits in one station while the other stations have no hit.
It can also reject a clean muon if one station records two adjacent channels.

The physical coverage condition must use the station-level counts:

```cpp
veto_coverage =
    count_veto1 > 0 &&
    count_veto2 > 0 &&
    count_veto3 > 0
```

The same principle applies to the US and DS systems. Total counts should be
used to measure excess activity only after station coverage has been
established.

The existing feature tree contains the required station-level branches:

```text
count_veto1 ... count_veto3
count_scifi1 ... count_scifi5
count_us1 ... count_us5
count_ds1 ... count_ds4
```

It also contains `scifi_n_active_stations_xy`, which counts the number of
SciFi stations containing at least one selected hit in both the x and y views.

## 3. Detector-coverage definitions

### 3.1 Veto coverage

The clean topology requires activity in every veto station:

```cpp
veto_coverage =
    count_veto1 > 0 &&
    count_veto2 > 0 &&
    count_veto3 > 0
```

This replaces the current v5 condition `count_veto == 3` as the physical
entrance requirement.

### 3.2 SciFi coverage

The nominal complete SciFi condition should be:

```cpp
scifi_coverage_tight = scifi_n_active_stations_xy == 5
```

This requires all five stations to contain both x-view and y-view activity.

A looser condition should also be studied:

```cpp
scifi_coverage_loose = scifi_n_active_stations_xy >= 4
```

The loose condition protects against losing otherwise clean muons because of
one inefficient station or view. The final choice must be based on the MC
efficiency and the data distributions.

### 3.3 US coverage

The clean topology requires activity in all five US stations:

```cpp
us_coverage =
    count_us1 > 0 &&
    count_us2 > 0 &&
    count_us3 > 0 &&
    count_us4 > 0 &&
    count_us5 > 0
```

This must not be replaced by `count_us == 5`, because a station can contain
more than one valid hit.

### 3.4 DS coverage

Two DS definitions should be compared.

Loose DS coverage:

```cpp
ds_coverage_loose =
    ((count_ds1 > 0) +
     (count_ds2 > 0) +
     (count_ds3 > 0) +
     (count_ds4 > 0)) >= 3
```

Tight DS coverage:

```cpp
ds_coverage_tight =
    count_ds1 > 0 &&
    count_ds2 > 0 &&
    count_ds3 > 0 &&
    count_ds4 > 0
```

The three-of-four definition is likely to be more robust against one
inefficient station. The four-of-four definition should be tested as a higher
purity alternative.

## 4. Multiplicity requirements

Station coverage establishes penetration through the detector. Separate total
hit-count upper bounds reject shower-like activity:

```cpp
low_veto_activity  = count_veto  <= VETO_MAX
low_scifi_activity = count_scifi <= SCIFI_MAX
low_us_activity    = count_us    <= US_MAX
low_ds_activity    = count_ds    <= DS_MAX
```

Their purposes are:

- `VETO_MAX`: reject excessive entrance activity;
- `SCIFI_MAX`: reject large showers in the SciFi target region;
- `US_MAX`: reject shower development in the target and US system;
- `DS_MAX`: reject events leaving the detector with many particles.

The initial clean-muon expression should have the configurable form:

```cpp
clean_throughgoing_muon =
    veto_coverage &&
    scifi_coverage &&
    us_coverage &&
    ds_coverage &&
    count_veto  <= VETO_MAX &&
    count_scifi <= SCIFI_MAX &&
    count_us    <= US_MAX &&
    count_ds    <= DS_MAX
```

The exact complement is:

```cpp
shower_enriched_complement = !(clean_throughgoing_muon)
```

The current threshold `count_scifi < 100` is useful as one scan point, but it
must not yet be treated as the final physical threshold.

## 5. Limits of a hit-count-only definition

Coverage and multiplicity cuts provide a simple, auditable first selection,
but counts alone cannot prove that an event contains a clean track. A compact
shower can occasionally satisfy the count requirements, while an inefficient
clean muon can fail them.

After the first result, detector-only refinements can be investigated without
using `RecoTrack`:

- total and per-station QDC in SciFi, US, veto and DS;
- SciFi transverse spreads such as `scifi_std_x` and `scifi_std_y`;
- SciFi peak fractions and station-to-station balance;
- detector hit-time ranges and time consistency;
- consistency of per-station average SciFi x/y positions with a straight
  trajectory.

The last item is a detector-centroid consistency observable, not an imported
`RecoTrack` decision. If introduced, its definition must be explicit and its
effect must be studied separately.

## 6. Add an MC-muon detector reference to v3

### 6.1 Purpose of MC in v3

MC muons should provide a reference for the expected detector response of a
true through-going muon. They should be used to determine reasonable
multiplicity ranges and to measure the efficiency of candidate detector-only
cuts.

MC must not be treated as another reconstructed-track type. It should not be
inserted into:

- the track-type overlap matrix;
- track-quality versus chi2/NDF pages;
- Simple/Hough component categories.

Those quantities require the reconstructed-track branches used by the v3
passing-muon partitions, while the existing MC feature files contain the
common detector features but not those v3 track-feature branches.

### 6.2 Recommended v3 comparison

Add one dedicated v3 comparison:

```text
mc_muon_reference.pdf
```

For every common one-dimensional feature, compare normalized shapes for:

```text
MCThroughGoingMuon
RealDataDQ
PassingMuonReco
NoPassingMuonReco
```

The MC distribution is a reference shape. It is not part of the real-data
closure equation.

For a common two-dimensional feature, show the same populations in a 2-by-2
layout with a common normalized z scale. This includes:

```text
scifi_x_vs_y
qdc_weighted_scifi_x_vs_y
```

These two spatial features are already defined in the shared v2 histogram
configuration used by v3:

```text
snakemake/metadata/configs/region_evaluation/
  passing_muon_run8329_hist_v2.yaml
```

Therefore v3 will receive both SciFi x-y pages when its histograms and PDFs are
regenerated. The remaining work is to add the MC population and the dedicated
comparison output.

### 6.3 MC input preparation

The existing general partition configuration declares an `MC_muon` group, but
the currently materialized file

```text
snakemake/metadata/region_partitions/partition_v1_region_v1.csv
```

contains no `muon` rows.

Existing metadata list 56 MC-muon inputs:

```text
35 downward
15 upward
 6 horizontal
```

At the time of inspection, detector-feature files were present for only:

```text
5 / 35 downward inputs
4 / 15 upward inputs
0 / 6  horizontal inputs
```

A dedicated, validated MC metadata file is therefore required. It should:

1. resolve `feature_path` relative to `output_base_path`;
2. retain only files that exist and contain a readable `sndData` tree;
3. validate the required detector branches;
4. record the production direction and configuration;
5. record event counts and normalization metadata;
6. clearly distinguish nominal samples from systematic alternatives.

Different generator, physics-list or detector-production variants must not be
blindly concatenated as independent nominal events. One consistent production
set should define the nominal template. Alternative configurations should be
used to estimate modelling uncertainty.

### 6.4 Do not use inclusive generated MC as the clean-muon template

A diagnostic scan of nine currently available upward/downward MC feature files
contained approximately 1.93 million events. Only about 1.25% satisfied:

```text
all three veto stations
+ all five US stations
+ at least one DS hit
```

Most inclusive MC events contained no detector activity. The inclusive MC
sample is therefore not itself a through-going detector-muon reference.

The originally proposed v3 reference name was:

```text
MCThroughGoingMuon
```

For the implemented upward-only MC study this is not achievable: the selected
nominal `MC_muon/up/scoring_at449m` feature file has zero events with SciFi
hits. V3 therefore uses the explicitly diagnostic name
`MCMuonUpDetectorActivity` and must not use this sample to calibrate a SciFi
threshold. A compatible through-going or downward production is still needed
before the MC-efficiency procedure below can be applied to SciFi.

It should preferably be selected using simulation truth:

1. the relevant primary particle is a muon;
2. its truth trajectory intersects the active detector acceptance;
3. it traverses the detector depth required by the study;
4. no `RecoTrack` result is used;
5. the detector multiplicity bounds being calibrated in v5 are not used to
   define the truth label.

Using truth geometry avoids circularly defining the MC reference with the same
hit-count cuts that are being optimized.

If the available feature files do not contain enough truth trajectory
information, detector station coverage may be used as an interim MC
preselection. This limitation and the resulting circularity must be documented,
and a truth-based selection should remain the preferred final method.

### 6.5 MC normalization

For cut-efficiency calculations within a consistent MC sample, use:

```text
efficiency = selected weighted MC events / truth-reference weighted MC events
```

For normalized shape comparisons, every population can be normalized to unit
integral after applying its validity requirement.

When combining MC files with different generated exposures, use their correct
per-event weights. Do not let a file contribute in proportion to its stored
event count when its simulated exposure is different. The chosen weighting and
the effective MC sample size must be written to the ROOT output and summary
page.

## 7. Refactor v5 around RealDataDQ

### 7.1 Current limitation

The current v5 implementation is hard-coded to two binary axes:

```text
veto_three
scifi_100
```

It reads `PassingMuonReco` and `NoPassingMuonReco` separately and reconstructs
RealDataDQ by addition. That organization is no longer appropriate for the new
question.

### 7.2 New input and populations

The revised v5 builder should read only the complete group:

```text
real_data_dq_run8329
```

It should define the two populations from the same `RDataFrame`:

```text
clean     = RealDataDQ.Filter(clean_expression)
complement = RealDataDQ.Filter(!(clean_expression))
```

The original `PassingMuonReco` membership can be joined or inspected later as
a validation label, but it must not be referenced by the detector-only
selection expression.

### 7.3 Configurable working points

The v5 configuration should contain named cut components and several working
points. A conceptual structure is:

```yaml
definitions:
  veto_coverage: >-
    count_veto1 > 0 && count_veto2 > 0 && count_veto3 > 0
  scifi_coverage_tight: "scifi_n_active_stations_xy == 5"
  scifi_coverage_loose: "scifi_n_active_stations_xy >= 4"
  us_coverage: >-
    count_us1 > 0 && count_us2 > 0 && count_us3 > 0 &&
    count_us4 > 0 && count_us5 > 0
  ds_coverage_loose: >-
    ((count_ds1 > 0) + (count_ds2 > 0) +
     (count_ds3 > 0) + (count_ds4 > 0)) >= 3
  ds_coverage_tight: >-
    count_ds1 > 0 && count_ds2 > 0 &&
    count_ds3 > 0 && count_ds4 > 0

working_points:
  loose:
    scifi_coverage: scifi_coverage_loose
    ds_coverage: ds_coverage_loose
    veto_max: VETO_MAX_LOOSE
    scifi_max: SCIFI_MAX_LOOSE
    us_max: US_MAX_LOOSE
    ds_max: DS_MAX_LOOSE

  nominal:
    scifi_coverage: scifi_coverage_tight
    ds_coverage: ds_coverage_loose
    veto_max: VETO_MAX_NOMINAL
    scifi_max: SCIFI_MAX_NOMINAL
    us_max: US_MAX_NOMINAL
    ds_max: DS_MAX_NOMINAL

  tight:
    scifi_coverage: scifi_coverage_tight
    ds_coverage: ds_coverage_tight
    veto_max: VETO_MAX_TIGHT
    scifi_max: SCIFI_MAX_TIGHT
    us_max: US_MAX_TIGHT
    ds_max: DS_MAX_TIGHT
```

The placeholder numerical bounds must be replaced after examining the updated
v3 MC reference. The expressions assembled by the code must be saved verbatim
in the ROOT file.

## 8. Required v5 studies and plots

### 8.1 Cumulative cutflow

For RealDataDQ and MCThroughGoingMuon, report the number and efficiency after:

```text
all events
veto coverage
SciFi coverage
US coverage
DS coverage
veto multiplicity upper bound
SciFi multiplicity upper bound
US multiplicity upper bound
DS multiplicity upper bound
```

Both incremental and cumulative efficiencies are needed:

```text
incremental efficiency = N(after this cut) / N(before this cut)
cumulative efficiency  = N(after this cut) / N(initial sample)
```

### 8.2 N-minus-one distributions

For every numerical upper bound, apply all other nominal requirements and omit
only the bound being plotted. Produce N-minus-one distributions for:

```text
count_veto
count_scifi
count_us
count_ds
```

Show the candidate loose, nominal and tight thresholds as vertical lines. This
is more informative than inspecting fully inclusive distributions because it
isolates the effect of the cut under study.

### 8.3 Coverage studies

Plot these discrete observables before applying the corresponding coverage
requirements:

```text
number of active veto stations
scifi_n_active_stations
scifi_n_active_stations_xy
number of active US stations
number of active DS stations
```

The active-station counts for veto, US and DS can be defined inside
`RDataFrame` as sums of Boolean station tests.

### 8.4 Correlation plots

At minimum, add normalized two-dimensional comparisons for:

```text
count_scifi versus count_us
count_scifi versus count_ds
count_us versus count_ds
count_veto versus count_scifi
scifi_x_vs_y
qdc_weighted_scifi_x_vs_y
```

The final clean and complement populations should be shown side by side with a
shared z scale.

### 8.5 Failure-reason overlap

The complement can fail more than one component. Store and plot an overlap
matrix for failure reasons:

```text
fails veto coverage
fails SciFi coverage
fails US coverage
fails DS coverage
excess veto activity
excess SciFi activity
excess US activity
excess DS activity
```

This matrix should use row-conditional percentages:

```text
M(i,j) = 100 * N(fails i and fails j) / N(fails i)
```

The diagonal must therefore be 100% for every nonempty failure category.

This will establish whether the complement is dominated by one shower
observable or by several correlated failures.

### 8.6 Final comparison pages

For each common one-dimensional feature:

- overlay the unit-normalized clean and complement shapes;
- show `complement / clean` in the ratio pad;
- show valid event counts in scientific notation.

For each common two-dimensional feature:

- show clean and complement side by side;
- normalize each population to unit integral;
- use a common z scale;
- do not create an efficiency or ratio panel.

## 9. Choosing the numerical thresholds after v3

### 9.1 Condition on a true through-going reference

Multiplicity quantiles must be calculated after selecting the truth-based
MCThroughGoingMuon population. Quantiles from inclusive generated MC are not
appropriate because most generated muons do not cross the complete detector.

### 9.2 Produce efficiency scans

For each possible upper threshold `t`, calculate:

```text
epsilon_veto(t)  = N_MC(count_veto  <= t) / N_MC(reference)
epsilon_scifi(t) = N_MC(count_scifi <= t) / N_MC(reference)
epsilon_us(t)    = N_MC(count_us    <= t) / N_MC(reference)
epsilon_ds(t)    = N_MC(count_ds    <= t) / N_MC(reference)
```

These one-dimensional curves are diagnostic. The final selection must also be
evaluated jointly because the observables are correlated.

### 9.3 Target combined working-point efficiencies

A useful initial target is:

```text
loose working point:   approximately 99% combined MC efficiency
nominal working point: approximately 95% combined MC efficiency
tight working point:   approximately 90% combined MC efficiency
```

These percentages refer to the complete combined detector selection, not to
each individual cut. Four independently chosen 95% cuts can have a combined
efficiency substantially below 95%.

The thresholds should be chosen together by scanning candidate tuples:

```text
(VETO_MAX, SCIFI_MAX, US_MAX, DS_MAX)
```

For every tuple, record:

- combined MC efficiency;
- selected fraction of RealDataDQ;
- retention of the existing PassingMuonReco label;
- retention of NoPassingMuonReco;
- population of each failure reason;
- stability across nominal and alternative MC productions.

The reconstructed labels are used here only to interpret and validate a cut
that was defined without them.

### 9.4 Avoid optimization on the validation sample

Use independent MC subsets for threshold choice and validation. A practical
split is by complete input file or production partition, rather than random
events from the same file. This tests stability against file-level and
production-level differences.

The final reported efficiency should come from the held-out sample. Variations
between compatible MC production configurations should contribute to a
modelling systematic uncertainty.

### 9.5 Data checks after choosing the thresholds

After fixing the nominal thresholds from MC:

1. apply them to RealDataDQ without retuning;
2. verify the exact clean/complement closure;
3. inspect the N-minus-one distributions near every threshold;
4. inspect detector-coordinate maps for edge or acceptance effects;
5. check time and QDC distributions for pileup or detector noise;
6. compare the detector-selected clean population with PassingMuonReco;
7. inspect disagreements event by event on a small sample.

The disagreement populations are particularly useful:

```text
detector clean AND PassingMuonReco
detector clean AND NOT PassingMuonReco
detector non-clean AND PassingMuonReco
detector non-clean AND NOT PassingMuonReco
```

They reveal whether the new definition is removing showers, losing clean
muons, or exposing reconstruction-specific inefficiency.

## 10. Validation requirements

The revised v5 workflow should fail if any of the following conditions is not
met:

```text
N_clean + N_complement == N_RealDataDQ
N_clean_intersection_complement == 0
every event is assigned exactly once
all detector counts used by the selection are non-negative
all configured expressions are nonempty
all requested input branches exist
```

The ROOT output should store:

- the complete clean-selection expression;
- every component expression;
- working-point values;
- population counts;
- cutflow counts and efficiencies;
- valid-entry counts for every feature;
- closure flags;
- MC file list, production identity, weights and total effective sample size.

The synthetic v5 test should deliberately use an input tree containing no
reconstructed-track branches. Successful execution will then provide a direct
test that the selection is independent of `RecoTrack`.

Boundary events must explicitly test:

- exactly the configured upper bounds;
- one count above every upper bound;
- four versus five active SciFi xy stations;
- three versus four active DS stations;
- a station with multiple hits;
- total counts that look correct while station coverage is incomplete.

## 11. Recommended implementation order

### Phase A: MC preparation

1. Define the nominal MC-muon production set.
2. Build a validated metadata CSV for existing feature files.
3. Define the truth-based through-going muon requirement.
4. Validate detector branches, event counts and weights.

### Phase B: v3 extension

1. Add an MC-muon metadata input to the v3 histogram rule.
2. Book each common feature once for MCThroughGoingMuon.
3. Store MC counts, valid-feature counts and normalization metadata.
4. Add `mc_muon_reference.pdf`.
5. Regenerate v3 so both SciFi x-y maps are included.
6. Add synthetic tests for 1D and 2D MC reference pages.

### Phase C: threshold study

1. Produce coverage distributions.
2. Produce multiplicity distributions after coverage.
3. Produce individual and combined efficiency scans.
4. Select provisional loose, nominal and tight working points.
5. Validate them on independent MC files.

### Phase D: v5 refactor

1. Replace the two hard-coded axes with configurable cut components.
2. Read RealDataDQ directly as the only data input.
3. Build clean and exact-complement frames.
4. Add cutflow, N-minus-one, failure-overlap and final comparison outputs.
5. Store selection provenance and closure metadata.
6. Replace the current synthetic test with a detector-only boundary test.

### Phase E: research validation

1. Apply the fixed thresholds to run-8329 RealDataDQ.
2. Compare against the previous PassingMuonReco membership only after the cut
   has been fixed.
3. Inspect the four agreement/disagreement populations.
4. Evaluate MC-production and threshold variations.
5. Document the final working point, efficiency and uncertainty.

## 12. Final recommended interpretation

The detector-selected population should be reported as:

```text
CleanThroughGoingMuonCandidate
```

It should not be called a truth-level real passing muon in collision data.
Its measured performance should be expressed through:

```text
MC through-going efficiency
data selected fraction
agreement with independent detector observables
stability under working-point and MC-production variations
```

The complement should be reported as:

```text
ShowerEnrichedComplement
```

until the updated v3 and v5 distributions establish its shower purity. This
naming keeps the partition mathematically exact while preserving a careful
physical interpretation.
