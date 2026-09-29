# Data-Driven Control-Region and ML Analysis Strategy

## Purpose

This document proposes a physics strategy for defining signal, control,
training, and validation regions in the `snd-ml` analysis when a machine
learning model is trained with signal MC and background-dominated real data,
then applied in the signal region.

The current region definition uses three detector-level decisions:

```text
veto activity:       count_veto == 0 or count_veto > 0
SciFi containment:   fiducial inside or outside
US activity:         count_us > 0 or count_us == 0
```

These decisions produce one signal-like region and seven orthogonal
sidebands. That partition is useful for studying the data. It is not yet, by
itself, a complete background-estimation or ML-training strategy.

The main problem is:

> How can background data from a signal-depleted region be used to train a
> classifier that will later be applied in the signal region without the
> classifier learning only the difference between the two regions?

The answer requires separate definitions for region bookkeeping, training,
validation, and the final statistical analysis.

## Executive Summary

The recommended design is:

1. Keep the current eight regions as diagnostic categories.
2. Replace each binary region axis with a continuous or ordinal variable.
3. Define nested loose and tight working points for veto, SciFi containment,
   and US activity.
4. Freeze and blind one final signal region.
5. Use one-inversion regions to identify and constrain individual background
   components, not automatically as one combined training sample.
6. Use regions immediately adjacent to the signal region as validation
   regions.
7. Choose the background training sample by measured transportability to the
   signal region, not only by event count.
8. Prevent the ML model from reconstructing the region definition through its
   inputs.
9. Validate the classifier score, efficiency, and background normalization in
   independent validation regions.
10. Use transfer factors and systematic uncertainties when control- and
    signal-region efficiencies are not equal.

The central principle is:

> A control region does not have to contain a background distribution that is
> exactly identical to the signal-region background. It must provide a
> background prediction that can be transported to the signal region with a
> correction that is independently validated and assigned an uncertainty.

## Chosen Baseline: Data-Driven Background Analysis

The observed background MC yield is lower than the real-data yield. Therefore,
the baseline analysis must not use the absolute background MC prediction as
the final SR background estimate.

The selected division of responsibilities is:

```text
Signal model and efficiency:
    signal MC, corrected and assigned data/MC uncertainties

Background examples for ML training:
    signal-depleted real data from control/training regions

Background yield and classifier-score shape:
    real data constrained by control regions

CR-to-SR extrapolation:
    transfer factors constrained and validated with data

Background MC:
    process interpretation, transfer guidance, closure tests,
    correlations, and systematic variations; not absolute normalization
```

The baseline final inference should be a simultaneous binned likelihood fit
to multiple control regions and the signal region. The observable can be the
ML classifier score, possibly also divided into broad bins of an energy proxy.
An ABCD/pass-fail estimate in score bins should be developed as a transparent
cross-check and may replace the fit only if its background-independence and
closure tests are demonstrably stronger.

### Analysis flow

```text
real data
  |
  +--> CR-V, CR-F, CR-U -----------------------------+
  |        |                                          |
  |        +--> background training data              |
  |        +--> background normalization constraints  |
  |        +--> background score templates            |
  |                                                   |
  +--> near-SR VR-V, VR-F, VR-U                       |
  |        |                                          |
  |        +--> test transfer and score closure        |
  |        +--> define nonclosure uncertainties        |
  |                                                   |
  +--> blinded SR <------ validated transfer model ---+
                           + signal MC model
```

### What the MC deficit can mean

Before treating the discrepancy as a physics background, rule out technical
causes:

- incorrect luminosity, cross section, generator, or filter weights;
- missing or duplicated files and incorrect generated-event counts;
- feature/cut-flow friend-tree misalignment;
- inconsistent event selections between data and MC;
- production-profile or EventBuilder differences;
- timing-window, noise, dead-channel, or detector-efficiency mismodeling.

After those checks, distinguish three cases by studying `data / MC` and
`data - MC` in all control regions:

1. **Normalization deficit:** shapes agree and one approximately constant
   factor describes the relevant regions.
2. **Composition deficit:** known MC processes exist but their relative yields
   are wrong.
3. **Missing or mismodeled background:** the excess has shapes not described
   by the available MC or changes strongly across regions.

The data-driven framework covers all three cases more safely than scaling the
total MC once. The complexity of the transfer model should increase only when
the closure studies require it.

### Core equations

For region `r` and variable `x`, first measure:

```text
R_r(x) = N_data,r(x) / N_MC,r(x)
E_r(x) = N_data,r(x) - N_MC,r(x)
```

The excess `E_r` should be compared among CR-V, CR-F, CR-U, and double-
inversion regions. A common shape suggests one missing normalization or
background population. Different shapes indicate multiple components or
region-dependent detector effects.

For one CR and classifier threshold `s > s0`, a conceptual data-driven
prediction is:

```text
N_B,pred(SR, s > s0)
    = [N_data(CR) - N_non-target(CR)]
    * T_yield(CR -> SR)
    * epsilon_B(CR, s0)
    * T_epsilon(CR -> SR, s0)
```

where:

```text
T_yield   = N_B(SR) / N_B(CR)
epsilon_B = N_B(CR, s > s0) / N_B(CR)
T_epsilon = epsilon_B(SR, s0) / epsilon_B(CR, s0)
```

In the preferred simultaneous-fit implementation, these factors become
constrained nuisance parameters instead of a sequence of fixed corrections.
The CR observations constrain background normalizations, while VR nonclosure
and alternative models constrain the extrapolation uncertainties.

### Preferred multi-region likelihood

For region `r` and classifier-score bin `i`, write the expected yield as:

```text
lambda_ri = mu * S_ri(theta) + sum_k beta_k * B_k,ri(theta)
```

where:

- `mu` is the signal-strength parameter;
- `S_ri` is the signal MC expectation;
- `k` labels background components or data-driven templates;
- `beta_k` are background normalization parameters constrained mainly by CRs;
- `theta` contains transfer, shape, detector, signal-efficiency, and
  nonclosure nuisance parameters.

The likelihood is conceptually:

```text
L(mu, beta, theta)
  = product over regions and score bins Poisson(N_ri | lambda_ri)
    * constraint terms for theta
```

During development, include CRs and VRs while keeping the SR observation
blinded. After all choices are frozen, include the SR and extract `mu`, a
signal yield, significance, or limit.

### Transparent ABCD/pass-fail cross-check

Choose two region axes that are least correlated for background, for example
veto and SciFi containment at fixed US category:

```text
A: veto pass, fiducial pass
B: veto fail, fiducial pass
C: veto pass, fiducial fail
D: veto fail, fiducial fail
```

For each classifier-score bin `i`:

```text
N_A,pred(i) = N_B(i) * N_C(i) / N_D(i)
```

Measure the closure factor:

```text
kappa(i) = N_A(i) * N_D(i) / [N_B(i) * N_C(i)]
```

The method is reliable only if `kappa` is compatible with a stable,
well-constrained correction in independent signal-depleted regions. With
three axes, repeat the ABCD closure test in different US categories. This uses
the third axis as a powerful validation rather than assuming three-way
factorization.

### Operational veto-US ABCD background estimate

For the proposed SciFi-only ML analysis, use veto activity and US activity as
the two background-estimation axes. Hold the SciFi fiducial requirement fixed
at its tight value when defining the nominal ABCD plane:

| Region | Veto | US | SciFi fiducial | Role |
|---|---|---|---|---|
| A | pass | pass | tight inside | blinded signal region |
| B | fail | pass | tight inside | veto control region |
| C | pass | fail | tight inside | US control region |
| D | fail | fail | tight inside | double-inverted control region |

For example, the first count-based implementation can start from:

```text
veto pass: count_veto == 0
veto fail: count_veto > 0

US pass:   count_us >= U_tight
US fail:   count_us < U_loose
```

The interval:

```text
U_loose <= count_us < U_tight
```

must remain separate from the nominal ABCD plane so it can be used as a
near-SR validation category. The final `U_loose` and `U_tight` values must be
chosen from signal efficiency, control-region contamination, background
composition, available statistics, and closure studies, then frozen before SR
unblinding.

If veto and US pass/fail decisions factorize for background, the nominal
prediction is:

```text
N_A,pred = N_B * N_C / N_D
```

Equivalently, `N_C / N_D` is a veto pass/fail transfer factor measured in the
US-fail population and applied to the US-pass control region B:

```text
T_veto = N_C / N_D
N_A,pred = N_B * T_veto
```

Before forming the estimate, subtract or model contributions that are not part
of the target background:

```text
N_tilde_R = N_data,R - N_non-target,R - N_signal,R
```

The corrected data-driven prediction is then:

```text
N_A,pred = kappa * N_tilde_B * N_tilde_C / N_tilde_D
```

where the background correlation correction is:

```text
kappa = N_A * N_D / (N_B * N_C)
```

`kappa = 1` corresponds to exact factorization. A stable non-unit value is
acceptable as a correction. A value that changes strongly with SciFi
fiducial distance, ML score, run period, veto threshold, or US threshold means
that the simple ABCD model is not transportable and must be extended or
rejected.

In the final inference, signal contamination and non-target backgrounds should
preferably be included as explicit likelihood components rather than manually
subtracted. Manual subtraction is useful for diagnostic plots but can produce
awkward uncertainties or negative corrected yields in low-statistics bins.

### SciFi fiducial validation of the ABCD estimate

Use a signed event-level SciFi fiducial distance `F` to define categories that
are orthogonal to the nominal tight-inside selection:

```text
F-tight:     F > F_tight
F-boundary:  F_loose < F <= F_tight
F-edge:      0 < F <= F_loose
F-outside:   F <= 0
```

Construct a complete veto-US ABCD plane independently in each SciFi category.
The `F-boundary` A-like region is signal depleted and can therefore be
observed during development. Its prediction is:

```text
N_A_boundary,pred
    = kappa_boundary
    * N_B_boundary * N_C_boundary / N_D_boundary
```

Compare this prediction with the observed `N_A_boundary`. Repeat the test in
the edge and outside categories and, when statistics allow, in multiple narrow
bands approaching `F-tight`. The method is acceptable only if the observed
nonclosure is understood, stable toward the SR boundary, and covered by the
assigned uncertainty.

The SciFi validation samples must not be used to train the final classifier or
derive the correction that they are intended to test. Model-development,
transfer-calibration, and final-closure samples should be separated by run or
whole source partition.

### Background estimate in ML-score bins

Assign every event an out-of-fold SciFi-only classifier score. Do not score an
event with a model that trained on that event or its source partition. Repeat
the ABCD estimate separately in each classifier-score bin `i`:

```text
N_A,pred(i)
    = kappa(i) * N_B(i) * N_C(i) / N_D(i)
```

This predicts both the background normalization and the classifier-score
shape in the SR. Validate `kappa(i)` and the prediction in every usable SciFi
fiducial validation band. In particular, check the most signal-like score
bins separately because they normally have the smallest CR statistics and the
largest sensitivity to domain shift.

The classifier must not receive veto, US, fiducial-distance, or direct
fiducial-position variables. A SciFi-only fixed detector grid can nevertheless
reconstruct the fiducial decision from absolute hit indices and can infer US
or veto behavior through correlated shower topology. Therefore, compare at
least:

```text
topology model:          relative or recentered SciFi coordinates
absolute-position model: raw detector-index grid
```

Use CR-to-VR closure and score stability, rather than nominal classification
AUC alone, to choose between them.

### Statistical uncertainty and likelihood implementation

For sufficiently populated independent Poisson counts, an approximate
relative statistical uncertainty is:

```text
(sigma_A / N_A)^2
    ~= 1 / N_B + 1 / N_C + 1 / N_D
       + (sigma_kappa / kappa)^2
```

This approximation is not reliable for small or zero counts. The preferred
implementation is a simultaneous Poisson likelihood. For score bin `i`, one
possible parameterization is:

```text
lambda_D,i = b_i
lambda_B,i = r_U,i * b_i
lambda_C,i = r_V,i * b_i
lambda_A,i = mu * S_i
             + kappa_i * r_U,i * r_V,i * b_i
```

where:

```text
b_i       = background normalization in D
r_U,i     = B/D transfer
r_V,i     = C/D transfer
kappa_i   = residual veto-US correlation correction
S_i       = signal MC expectation
mu        = signal-strength parameter
```

The likelihood should contain the observed A/B/C/D counts in all score bins
and constraint terms for `kappa`, nonclosure, signal efficiency, detector
effects, and non-target backgrounds. During development, exclude the blinded
nominal A observation while including the corresponding observable A-like
regions in the SciFi validation categories.

Required uncertainties include:

- finite B, C, and D data statistics;
- finite statistics used to constrain `kappa`;
- residual SciFi-validation nonclosure;
- variation across veto and US working points;
- run-period and detector-condition dependence;
- signal and non-target contamination in the CRs;
- background-composition dependence;
- any MC-derived correction or transfer uncertainty.

The final SR estimate is therefore not the number of events in the ML training
sample. The training CR supplies background examples, while B, C, D and the
validated transfer model supply the SR background normalization and score
shape.

### Immediate implementation priority

The first data-driven result should not be a final neural network. It should be
a region report containing:

1. data and MC yields in every current region;
2. `data / MC` and `data - MC` for all important features;
3. signal contamination in every candidate CR and VR;
4. MC background composition and its uncertainty;
5. run-period and production-profile dependence;
6. EventBuilder versus non-EventBuilder comparisons;
7. candidate pass/fail transfer factors and ABCD closure;
8. a conclusion about whether the deficit is normalization, composition, or
   shape dependent.

Only after this report should the final training-region mixture and classifier
feature list be frozen.

## 1. Terminology

The terms below should be used consistently in code, configuration, plots,
and documentation.

### 1.1 Signal region (SR)

The signal region is the final analysis region in which the signal yield is
measured or a search is performed. A possible conceptual definition is:

```text
no in-time veto activity
SciFi activity inside a tight fiducial volume
sufficient US activity
common event-quality and SciFi preselection
```

Real data in the SR should normally remain blinded while region definitions,
model architecture, score threshold, corrections, and uncertainties are
chosen.

### 1.2 Control region (CR)

A control region is a signal-depleted data region used to constrain a
background normalization, composition, or shape. A good CR has:

- negligible or acceptably small signal contamination;
- enough data for the intended measurement;
- a known relationship to the SR;
- preferably the same dominant background processes as the SR;
- a transfer factor that can be validated.

Different backgrounds may require different CRs. A veto-inverted muon CR and
an outside-fiducial neutral-hadron CR should not automatically be treated as
samples of the same background distribution.

### 1.3 Training region (TR)

A training region supplies events used to optimize the ML parameters. It may
be a CR, a subset of a CR, or a weighted mixture of multiple CRs.

Calling it a training region makes clear that its events cannot also provide
an unbiased final validation of the method.

### 1.4 Validation region (VR)

A validation region is signal-depleted, statistically independent of the
training data, and not used to derive the final correction being tested. It is
used to test the full prediction procedure:

```text
training region -> correction/transfer -> validation region
```

The important result is closure: the prediction must agree with observed data
within the assigned uncertainties.

### 1.5 Sideband

“Sideband” should be reserved for a region adjacent to the SR along a
continuous variable. For example, a narrow band immediately outside a
fiducial boundary is a natural sideband. A region obtained by fully inverting
the veto selection may be physically far from the SR and is more naturally
called a control region.

## 2. Current Eight-Region Definition

The current `region_v1` design uses three binary axes after requiring:

```text
count_scifi > 35
```

The axes are:

```text
V: count_veto == 0 versus count_veto > 0
F: separate-plane SciFi fiducial inside versus outside
U: count_us > 0 versus count_us == 0
```

This creates the following regions:

| Role in current naming | Veto | Fiducial | US |
|---|---:|---:|---:|
| Signal-like | no veto | inside | has US |
| Sideband 1 | no veto | inside | no US |
| Sideband 2 | no veto | outside | has US |
| Sideband 3 | no veto | outside | no US |
| Sideband 4 | has veto | inside | has US |
| Sideband 5 | has veto | inside | no US |
| Sideband 6 | has veto | outside | has US |
| Sideband 7 | has veto | outside | no US |

This is an excellent diagnostic partition because the regions are exclusive
and interpretable. However, the seven non-SR regions can have very different
background compositions:

- a veto inversion can strongly enrich incoming muons;
- a fiducial inversion can enrich edge-crossing, geometry, or acceptance
  effects;
- a US inversion can enrich neutral, low-energy, inefficient, or unusual
  shower topologies;
- double and triple inversions can be far from the SR in several correlated
  detector variables.

Therefore, the seven regions should not initially be merged into one
background class for training.

## 3. Required Conditions for a Useful Control Region

### 3.1 Low signal contamination

For every candidate CR or VR, calculate from signal MC:

```text
N_signal_region
signal efficiency relative to the baseline selection
signal fraction after adding expected background normalization
```

Low signal contamination must be assessed relative to the signal enrichment
in the SR, not only as a small absolute number. Signal contamination in CRs
can bias a data-driven prediction or teach a classifier to reject signal-like
events.

### 3.2 Relevant background composition

For each candidate region, evaluate the expected fractions of:

```text
muon DIS
ordinary muons
kaons
neutrons
neutrino interactions
EventBuilder variants
other available background categories
```

Compare both yields and fractions with the expected SR background. An
inclusive distribution can look similar even when compensating changes in
different components hide a composition mismatch.

### 3.3 Transportable feature distributions

Let `x` denote the variables used by the classifier and `z` the variables used
to define regions:

```text
z = {veto activity/timing, SciFi fiducial coordinate, US activity}
x = remaining ML features
```

The simplest direct sideband training assumes approximately:

```text
p_background(x | TR) ~= p_background(x | SR)
```

Exact equality is not normally achievable. The practical requirement is that
remaining differences are smooth, correctable, validated, and smaller than
the assigned uncertainty.

### 3.4 Stable classifier efficiency

For score threshold `s > s0`, define:

```text
epsilon_B(R, s0) = N_B(R, s > s0) / N_B(R)
```

The background efficiency does not have to be numerically identical in CR and
SR. If it differs, define an efficiency transfer factor:

```text
T_epsilon(s0) = epsilon_B(SR, s0) / epsilon_B(CR, s0)
```

The transfer factor must be derived without using unblinded SR data, tested in
VRs, and assigned statistical and modeling uncertainties.

## 4. Continuous Region Variables

The current binary cuts should be retained for compatibility, but new region
versions should be based on continuous or ordinal quantities.

### 4.1 Veto activity and timing

The binary `count_veto` decision does not distinguish prompt veto activity
from early noise, late hits, or activity unrelated to the reconstructed event.

A more physical primary variable is:

```text
N_veto_in_time(t_low, t_high)
```

where veto hits are counted relative to a defined event time reference. Useful
supporting quantities include:

```text
earliest veto hit time
latest veto hit time
minimum veto-SciFi time difference
number of veto hits inside the timing window
veto QDC inside the timing window
veto plane pattern
```

The absolute earliest time alone may not be robust: one unrelated early hit
could control the category. The analysis should first establish:

- time units and offsets in data and every MC production;
- the event time reference;
- timing resolution;
- detector and run-period dependence;
- behavior of empty-hit and invalid-time events;
- effects of the MC first-arrival shift and 25 ns filtering.

Suggested nested categories are:

```text
V-tight-pass: no veto hit in the nominal in-time window
V-loose-only: passes a relaxed definition but fails tight
V-fail:       clear in-time veto activity
```

The exact numeric boundaries should be selected from timing studies and frozen
before looking at the final SR.

### 4.2 SciFi signed fiducial distance

Represent containment using a signed distance to the closest allowed edge:

```text
F > 0: inside
F = 0: on the boundary
F < 0: outside
```

For rectangular bounds in one station:

```text
F_station = min(x - x_min, x_max - x, y - y_min, y_max - y)
```

An event-level value could be:

```text
F_event = minimum F_station over stations to which the position cut applies
```

This preserves the conservative “all relevant stations must be contained”
logic. The current exception for stations with at most two hits should either
be preserved explicitly or replaced only after an efficiency study.

Suggested nested categories are:

```text
F-tight-pass: F_event > m_tight
F-boundary:   m_loose < F_event <= m_tight
F-edge:       0 < F_event <= m_loose
F-outside:    F_event <= 0
```

A narrow boundary band is usually more informative for transport tests than
combining every event outside the fiducial area.

### 4.3 US activity

The current `count_us > 0` decision should be generalized with:

```text
count_us
number of US planes hit
qdc_us
station-by-station US counts and QDC
```

Raw hit multiplicity may depend strongly on energy and noise. The number of
planes hit or a count/QDC combination may be more stable.

Suggested nested categories are:

```text
U-tight-pass: count_us >= N_tight and optional plane/QDC conditions
U-loose-only: passes N_loose but fails tight
U-fail:       count_us < N_loose
```

The thresholds should be scanned in MC and signal-depleted data, but the final
choice must be frozen before unblinding.

## 5. Proposed Region Hierarchy

Using three levels instead of two gives enough freedom to separate training
from validation.

### 5.1 Final signal region

```text
SR:
    baseline event quality
    count_scifi > 35, or a frozen revised threshold
    V-tight-pass
    F-tight-pass
    U-tight-pass
```

### 5.2 Near-SR validation regions

Invert or relax exactly one requirement by one adjacent step:

```text
VR-V:
    V-loose-only
    F-tight-pass
    U-tight-pass

VR-F:
    V-tight-pass
    F-boundary
    U-tight-pass

VR-U:
    V-tight-pass
    F-tight-pass
    U-loose-only
```

These are the most important regions for demonstrating that the method is
stable close to the SR boundary.

### 5.3 Background control regions

More strongly inverted regions can constrain particular processes:

```text
CR-V:
    V-fail
    F-tight-pass
    U-tight-pass

CR-F:
    V-tight-pass
    F-outside
    U-tight-pass

CR-U:
    V-tight-pass
    F-tight-pass
    U-fail
```

Possible interpretations, to be verified rather than assumed, are:

```text
CR-V -> incoming-muon-enriched
CR-F -> detector-edge or neutral-hadron-enriched
CR-U -> low-US or neutral-topology-enriched
```

Double-inversion regions are valuable for closure tests and for constraining
correlations between axes. Triple-inversion regions are usually farthest from
the SR and should not be the first choice for classifier training.

## 6. The ML Domain-Shift Problem

Suppose the supervised training sample is:

```text
positive class: signal MC satisfying the SR selection
negative class: real data from a control region
```

The classifier can learn both:

```text
signal versus background
SR domain versus CR domain
```

Only the first distinction is desired.

For example, if the negative data satisfy `count_us == 0` while signal MC
satisfies `count_us > 0`, then `count_us` makes the classification trivial. US
QDC, station-level counts, occupancy, and shower size may allow the network to
reconstruct the same distinction even if `count_us` itself is removed.

When applied to the SR, all background events already satisfy the US
requirement. A classifier trained primarily on the region difference can have
excellent test metrics in the training domain and no valid interpretation in
the SR.

## 7. ML Input Policy

### 7.1 Explicitly identify region variables

Maintain a configuration list of all direct region variables and their close
proxies. For example:

```text
Veto family:
    count_veto*
    qdc_veto*
    veto timing variables

Fiducial family:
    avg_scifi*_x/y
    qdcAvg_scifi*_x/y
    signed fiducial distance

US family:
    count_us*
    qdc_us*
    avg_us*
    number of US planes
```

### 7.2 Use staged feature sets

Train at least three diagnostic models:

```text
Model A: all features
Model B: remove direct region variables
Model C: remove direct variables and strongly correlated proxies
```

Compare signal performance, CR-to-VR closure, and domain distinguishability.
The highest nominal AUC is not necessarily the best physics model.

### 7.3 Measure residual correlations

For background MC and data control samples, measure:

- linear and rank correlations;
- mutual-information estimates where practical;
- feature shapes across bins of `V`, `F`, and `U`;
- classifier score versus each region variable;
- classifier efficiency versus each region variable.

A flat linear correlation is not sufficient evidence of independence. Shape
comparisons and closure tests are required.

### 7.4 Reweight when necessary

Before training, background CR data may be reweighted to match the expected SR
distribution in nuisance variables that are not intended to provide signal
discrimination, such as:

```text
run period
beam condition
energy proxy
overall occupancy
source partition
geometry or detector condition
```

The reweighting variables must not introduce signal sculpting. Weight
distributions should be bounded or regularized so a few events do not control
the training.

### 7.5 Decorrelate if removal is insufficient

Possible methods include:

- adversarial training against `V`, `F`, and `U`;
- distance-correlation penalties;
- flat-efficiency training in bins of the region variables;
- parameterized models conditioned on the region variables;
- conditional background-density modeling.

These methods add complexity and should be introduced only after a transparent
baseline with feature removal and reweighting has been evaluated.

## 8. Domain-Classifier Test

Before training signal versus background, directly test whether a candidate CR
represents the SR background.

For each background component, train:

```text
class 0: background MC in candidate CR
class 1: the same background MC in SR
```

Do not include the region-defining variables. Use source-file-separated
training and test samples.

Interpret the held-out AUC:

```text
AUC near 0.5: domains are difficult to distinguish
large AUC:     CR and SR contain transport-relevant differences
```

There is no universal acceptable AUC boundary. Inspect the variables causing
separation and test the effect on the final classifier efficiency.

Repeat the test:

- for every major background separately;
- for the expected background mixture;
- before and after reweighting;
- between adjacent real-data CR/VR regions;
- between run periods and production versions.

This diagnostic does not prove SR agreement in real data, but it can reject
clearly unsuitable control regions.

## 9. Multiple Background Components

Let the expected SR background be:

```text
B_SR(x) = sum_k f_k^SR B_k^SR(x)
```

where `k` labels muon DIS, muons, kaons, neutrons, and other components. A
control region instead contains:

```text
B_CR(x) = sum_k f_k^CR B_k^CR(x)
```

Both the component fractions and the component shapes can change.

If no single CR represents the SR mixture, use several CRs in a simultaneous
likelihood fit. Conceptually:

```text
CR-V constrains muon-like normalization
CR-F constrains edge/neutral background
CR-U constrains low-US background
VRs test the extrapolation
SR receives all components through transfer factors
```

This is generally safer than forcing one heterogeneous data sample to act as
the complete background class.

## 10. Background Prediction and Transfer Factors

For one background component, a simple pre-ML yield transfer factor is:

```text
T_yield = N_B(SR) / N_B(CR)
```

After applying a classifier threshold:

```text
epsilon_B(CR, s0) = N_B(CR, s > s0) / N_B(CR)
epsilon_B(SR, s0) = N_B(SR, s > s0) / N_B(SR)
T_epsilon(s0) = epsilon_B(SR, s0) / epsilon_B(CR, s0)
```

A conceptual data-driven prediction is then:

```text
N_B_predicted(SR, s > s0)
    = N_data(CR)
    * T_yield
    * epsilon_B(CR, s0)
    * T_epsilon(s0)
```

In a real analysis, non-target backgrounds and signal contamination must be
subtracted or included as components in a simultaneous fit.

Transfer factors may need to be binned in:

```text
energy proxy
run period
classifier score
fiducial distance
US multiplicity
timing category
```

Use the lowest-dimensional parameterization supported by closure tests and
available statistics.

## 11. Closure Tests

A closure test applies the complete CR-to-target prediction to a region where
the answer can be checked.

### 11.1 MC closure

Treat one MC region as pseudo-data:

1. derive the correction from another MC region;
2. predict the target yield and score shape;
3. compare prediction with truth;
4. repeat for each background component and mixture;
5. repeat under generator and detector variations when available.

### 11.2 Data closure

Use signal-depleted real-data VRs:

1. derive the model/correction in a TR or CR;
2. apply it to the VR;
3. compare predicted and observed feature distributions;
4. compare classifier score shape;
5. compare efficiencies for all candidate score thresholds;
6. repeat for near and far VRs.

### 11.3 Quantities to report

For every closure test, report:

```text
observed and predicted yields
ratio and uncertainty
chi-square or likelihood goodness of fit
shape comparisons for ML inputs
classifier score comparison
efficiency versus score threshold
efficiency versus energy proxy
efficiency versus V, F, and U
background composition
```

### 11.4 Nonclosure uncertainty

If the prediction differs from the observed VR data, investigate the cause.
After correcting a demonstrated effect, the remaining credible discrepancy
must be included as a systematic uncertainty. The uncertainty should cover
the SR extrapolation, not only reproduce the statistical fluctuation of one
VR.

## 12. Training and Evaluation Splitting

Random event-level splitting is unsafe when events from the same ROOT file,
run, or production partition share detector and simulation conditions.

Split using groups such as:

```text
real data: run or run block
MC: source ROOT file or production partition
```

Recommended logical subsets are:

```text
training fold:      optimize model parameters
selection fold:     choose architecture and score working point
calibration fold:   derive reweighting or transfer correction
closure fold:       final unbiased VR test
```

Cross-fitting can use statistics efficiently: train several models, each of
which scores only events excluded from its own training fold. No event should
be evaluated by a model that trained on that event or a highly correlated copy.

## 13. Blinding and Analysis Freeze

Before unblinding SR data, freeze:

- the baseline selection;
- definitions and numeric boundaries of SR, CRs, and VRs;
- feature list and preprocessing;
- model architecture and hyperparameters;
- data/MC and CR/SR reweighting;
- classifier score threshold or fit strategy;
- transfer-factor parameterization;
- systematic uncertainty procedure;
- statistical model.

If the SR must be inspected for technical validation, use a prescaled sample,
a masked score range, or scrambled identifiers under an agreed blinding
procedure.

## 14. Concrete Study Sequence for `snd-ml`

### Phase A: build continuous variables

1. Confirm veto timing conventions for data and every MC type.
2. Add or validate `N_veto_in_time` and veto-relative-time variables.
3. Add a signed per-event SciFi fiducial distance.
4. Preserve `count_us`, US-plane count, and US QDC as separate quantities.
5. Produce QA plots and sentinel-value counts.

### Phase B: characterize the existing regions

For each of the current eight regions, produce:

- real-data yield and luminosity-normalized rate;
- signal MC efficiency and expected contamination;
- background yields and fractions;
- feature distributions per background component;
- timing, fiducial-distance, and US-multiplicity distributions;
- EventBuilder/non-EventBuilder comparisons.

### Phase C: scan nested boundaries

Scan one axis at a time while fixing the other two near their SR definition:

```text
veto time window and allowed hit count
fiducial margin
US hit/plane/QDC threshold
```

For each boundary calculate:

```text
signal efficiency
background rejection
signal contamination in proposed CR/VR
background composition change
domain-classifier AUC
available data statistics
```

Do not optimize only `S/sqrt(B)`. Transportability and systematic uncertainty
are also optimization objectives.

### Phase D: choose TR, CRs, and VRs

1. Select a blinded tight SR.
2. Select three near-SR one-step VRs.
3. Assign far one-inversion CRs to the backgrounds they actually constrain.
4. Reserve double-inversion regions for correlation and closure studies.
5. Document signal contamination and expected composition for every region.

### Phase E: ML baseline

1. Train signal MC versus the selected data TR.
2. Exclude all direct region variables.
3. Use grouped splitting and out-of-fold scoring.
4. Train the domain classifiers.
5. Remove or decorrelate the most problematic proxies.
6. Repeat after justified reweighting.

### Phase F: validate the complete method

1. Freeze the model and score threshold.
2. Derive CR-to-VR transfer factors without using the closure fold.
3. Test score shape and efficiency in every VR.
4. Test stability by run period, background model, and production version.
5. Assign nonclosure and modeling uncertainties.
6. Build a simultaneous likelihood with CR and SR bins if practical.
7. Unblind only after review of all closure criteria.

## 15. Suggested Configuration Structure

Do not overwrite `region_v1.yaml` during development. Create a versioned region
configuration, for example:

```text
snakemake/metadata/configs/regions/region_v2.yaml
```

A possible conceptual structure is:

```yaml
region_version: region_v2

axes:
  veto:
    variable: n_veto_in_time
    tight_pass: "n_veto_in_time == 0"
    loose_only: "n_veto_narrow == 0 && n_veto_in_time > 0"
    fail: "n_veto_narrow > 0"

  scifi_fiducial:
    variable: scifi_fiducial_distance
    tight_pass: "scifi_fiducial_distance > F_TIGHT"
    boundary: "scifi_fiducial_distance > F_LOOSE && scifi_fiducial_distance <= F_TIGHT"
    edge: "scifi_fiducial_distance > 0 && scifi_fiducial_distance <= F_LOOSE"
    outside: "scifi_fiducial_distance <= 0"

  us:
    variable: count_us
    tight_pass: "count_us >= U_TIGHT"
    loose_only: "count_us >= U_LOOSE && count_us < U_TIGHT"
    fail: "count_us < U_LOOSE"

analysis_regions:
  SR: [veto.tight_pass, scifi_fiducial.tight_pass, us.tight_pass]
  VR_V: [veto.loose_only, scifi_fiducial.tight_pass, us.tight_pass]
  VR_F: [veto.tight_pass, scifi_fiducial.boundary, us.tight_pass]
  VR_U: [veto.tight_pass, scifi_fiducial.tight_pass, us.loose_only]
  CR_V: [veto.fail, scifi_fiducial.tight_pass, us.tight_pass]
  CR_F: [veto.tight_pass, scifi_fiducial.outside, us.tight_pass]
  CR_U: [veto.tight_pass, scifi_fiducial.tight_pass, us.fail]
```

`F_TIGHT`, `F_LOOSE`, `U_TIGHT`, `U_LOOSE`, and timing boundaries are
placeholders. They must be determined by the studies above, not copied as
arbitrary values.

Store region purpose as metadata:

```yaml
purpose: signal | training | control | validation | diagnostic
blinded: true | false
target_backgrounds: [muonDIS, muon, kaon, neutron]
allowed_for_training: true | false
allowed_for_closure: true | false
```

This prevents an event sample from silently being reused for incompatible
roles.

## 16. Decision Criteria

A candidate training/control strategy should be accepted only if all of the
following are documented:

- signal contamination is sufficiently low;
- the dominant background components are identified;
- the CR-to-SR relationship is physically motivated;
- domain-classifier separation is understood;
- important ML inputs have acceptable stability or corrections;
- score efficiency closes in independent VRs;
- corrections are stable against alternative background simulations;
- run-period and detector-condition dependence is covered;
- statistical uncertainties from finite CR samples are propagated;
- residual nonclosure is assigned as a systematic uncertainty;
- the SR was not used to choose the method.

## 17. Common Failure Modes

### Combining all sidebands

This maximizes statistics but can create an artificial background mixture far
from the SR. The classifier may identify sideband type rather than physics
background.

### Removing only the three exact cut variables

The model can reconstruct them from station-level counts, QDC, positions, and
timing. Proxy variables must also be studied.

### Selecting the region with the best validation AUC

High AUC can indicate a large CR/SR domain difference. Physics sensitivity
must be considered together with closure and systematic uncertainty.

### Testing on the training control region

Agreement there does not validate extrapolation. An independent VR is needed.

### Assuming MC composition is exact

MC can guide transfer factors and composition, but data CRs and VRs must test
the model. Generator disagreement should contribute to uncertainties.

### Optimizing boundaries after viewing SR data

This introduces selection bias and invalidates a clean statistical
interpretation. Boundaries and working points must be frozen first.

### Demanding exactly equal CR and SR efficiency

Exact equality is unnecessarily restrictive and often unrealistic. A
non-unit efficiency transfer is acceptable if it is robustly predicted,
validated, and assigned an uncertainty.

## 18. Relation to Established HEP Methods

The proposed strategy follows standard control-region logic:

- **ABCD/2D sidebands:** use approximately independent background axes and low
  signal contamination to infer the SR from control regions. With three axes,
  the extra regions provide correlation tests, but independence must still be
  demonstrated. See [ABCDisCo: Automating the ABCD Method with Machine
  Learning](https://arxiv.org/abs/2007.14400).

- **Transfer factors:** use simulation or auxiliary data to connect a CR
  background yield or efficiency to the SR, validate the transfer in VRs, and
  propagate correction and nonclosure uncertainties.

- **Sideband-trained ML:** direct substitution of sideband background works
  only when classifier features are sufficiently independent of the sideband
  variable. Otherwise the classifier sculpts or learns the region boundary.

- **Conditional/reweighted background modeling:** when the background changes
  with the region variable, learn a correction or conditional distribution in
  sidebands and interpolate it into the SR. See [Simulation Assisted
  Likelihood-free Anomaly Detection](https://arxiv.org/abs/2001.05001).

- **Modern weakly supervised sideband searches:** compare direct sideband
  models with conditional density models and use sidebands on both sides of a
  target region when possible. See the [CMS machine-learning anomaly-detection
  study](https://cds.cern.ch/record/2938054/files/MLG-23-002-pas.pdf).

These methods do not remove the need for analysis-specific detector studies.
In this analysis, veto timing, SciFi geometry, US response, EventBuilder
effects, and background composition must determine which method is credible.

## 19. Choice of Fit Observable Without a Reconstructed Mass

### 19.1 A reconstructed mass is not required

A final likelihood fit does not require a reconstructed invariant mass. Mass
is frequently used in collider analyses because it can be a well-modeled,
highly discriminating observable, especially for a narrow resonance. Its
statistical role is nevertheless the same as that of any other observable:
it separates signal from background and helps the fit identify the relevant
physics components.

When no useful reconstructed mass exists, a standard physics approach is to
fit one or a small number of observables chosen for:

```text
signal/background separation
background-component identification
CR-to-SR transportability
robust detector and simulation modeling
adequate control-region statistics
small total uncertainty after profiling systematics
```

The observable with the best raw separation or largest classifier AUC is not
automatically the best fit observable. The preferred observable is the one
that gives the best expected inference performance after CR constraints,
transfer uncertainties, detector effects, and closure uncertainties are
included.

For this analysis, the recommended baseline is:

```text
primary SR observable:
    cross-fitted signal-classifier score

secondary categorization or fit dimension:
    a small number of broad visible-energy-proxy bins
```

The fit should be developed progressively, starting with event counts before
adding shape information.

### 19.2 The three jobs performed by fit observables

The analysis needs to distinguish three related tasks:

1. separate the target signal from the total background;
2. distinguish important background components from one another;
3. constrain the transfer of each background from its CR to the SR.

One observable does not need to perform all three tasks. A simultaneous fit
may use different observables in different statistically disjoint channels,
while sharing normalization and nuisance parameters among those channels.

For example:

```text
SR:
    signal-classifier score, optionally binned in visible energy

CR-V:
    veto timing/activity or a muon-background score

CR-F:
    signed SciFi fiducial distance or an edge-background score

CR-U:
    US activity, shower development, or a low-US-background score

near-SR VRs:
    preferably the same score and energy observables used in the SR
```

This is supported naturally by HistFactory-style statistical models, which
combine multiple regions, samples, normalization parameters, coherent shape
variations, bin-by-bin statistical uncertainties, and shared nuisance
parameters. See the [HistFactory
formalism](https://cds.cern.ch/record/1456844/files/CERN-OPEN-2012-016.pdf).

### 19.3 Recommended primary observable: a cross-fitted classifier score

If the target signal is, for example, `CC_nue`, define a final signal score
from the model outputs. For a multiclass classifier, one possible compression
is:

```text
s_CCnue = p(CC_nue) /
          [p(CC_nue) + sum of probabilities assigned to fitted backgrounds]
```

The exact denominator depends on the physics definition. Before defining the
score, the analysis must state explicitly:

```text
parameter of interest
target signal process or processes
other neutrino processes treated as backgrounds
detector and beam-induced background components
```

The full classifier-score distribution should normally be fitted instead of
applying only one hard threshold:

```text
low-score bins:       background dominated
intermediate bins:    mixed signal and background
high-score bins:      signal enriched
```

Classifier-output distributions are routinely used as final binned fit
observables when no resonance mass provides the best discrimination. An
example is the ATLAS s-channel single-top analysis, which used a binned
maximum-likelihood fit to a classifier output. See the [ATLAS classifier-fit
example](https://arxiv.org/abs/1410.0647).

The fitted score must be produced with strict cross-fitting:

```text
event in fold 0 -> score from a model trained without fold 0
event in fold 1 -> score from a model trained without fold 1
combined out-of-fold scores -> templates used in the likelihood
```

This applies especially to CR data used both as ML background examples and as
the source of a fitted background template. A model must not create the final
score template from events on which it trained.

Before it is accepted as the final observable, the score must satisfy:

```text
stable out-of-fold performance
no use of blinded SR data for optimization
understood correlations with V, F, and U
CR-to-VR score-shape closure
CR-to-VR score-efficiency closure
stability by energy, run period, and production profile
systematic templates for credible detector and modeling variations
```

### 19.4 Recommended second dimension: a visible-energy proxy

Signal and background composition, classifier efficiency, and CR-to-SR
transfer factors can vary strongly with shower energy. A fully inclusive
score fit can hide these dependencies. The recommended second observable is a
calibrated visible-energy proxy based on the SciFi and US detector response:

```text
E_vis = k * QDC_SciFi + alpha * QDC_US
```

SND@LHC test-beam and detector studies motivate a combined SciFi and US QDC
response for shower-energy estimation. See [Results and perspectives from the
first two years of neutrino physics at
SND@LHC](https://cds.cern.ch/record/2890347/files/Publication.pdf) and the
[SND@LHC detector and energy-response
roadmap](https://cds.cern.ch/record/2924332/files/document.pdf).

Until the calibration and response are validated for the analysis samples,
this quantity should be called a `visible-energy proxy`, not a fully
reconstructed neutrino energy. It must be checked for:

```text
data/MC response differences
shower-starting-position dependence
electromagnetic versus hadronic response
QDC saturation and nonlinear response
dead and noisy channels
run-period dependence
EventBuilder and production-version dependence
selection and region dependence
```

Alternative or supporting energy proxies already present in the feature
workflow include:

```text
total SciFi QDC
total US QDC
density_sndsw_scifi
SciFi hit multiplicity
station-by-station SciFi and US QDC
longitudinal shower profile
```

Simple detector observables are valid fit variables when they are
discriminating and robustly modeled. For example, the SND@LHC collider-neutrino
observation reported the SciFi-hit distribution for selected candidates. See
the [SND@LHC collider-muon-neutrino
observation](https://cds.cern.ch/record/2858828/plots?ln=bg).

The first two-dimensional model should use coarse binning, for example:

```text
3 to 5 broad E_vis bins
    x
4 to 8 classifier-score bins
```

The two dimensions may be represented as a two-dimensional histogram or as a
flattened list of bins. Fine binning should be avoided until CR statistics,
simulation statistics, and systematic variations support it.

### 19.5 Suggested multi-region likelihood

Let `r` denote the analysis region, `e` the visible-energy bin, `i` the
classifier-score bin, and `k` a background component. A suitable model is:

```text
lambda_r,e,i
    = mu * S_r,e,i(theta)
    + sum_k beta_k * B_k,r,e,i(theta)
```

For example, the components could be organized initially as:

```text
B_mu:       incoming muons and/or muon DIS
B_neutral:  neutral-hadron backgrounds
B_lowUS:    low-US, inefficient, or unusual shower backgrounds
B_nu:       other neutrino interactions
```

This grouping is illustrative. Components should be merged when their
templates and transfers cannot be separately identified and split only when
the data and auxiliary information can constrain them.

The same normalization parameter `beta_k` is shared between the CR and SR.
The CR-to-SR relationship is supplied through the predicted templates or an
explicit transfer factor:

```text
T_k,e,i(CR -> SR) = B_k,SR,e,i / B_k,CR,e,i
```

A conceptual assignment is:

```text
CR-V data -> primarily constrain beta_mu
CR-F data -> primarily constrain beta_neutral or edge background
CR-U data -> primarily constrain beta_lowUS
auxiliary neutrino samples and MC -> constrain beta_nu
SR score shape -> constrain the signal strength mu
VR observations -> test transfer and define nonclosure uncertainties
```

Every CR will generally contain several processes. Non-target backgrounds and
signal contamination must therefore either be included as separate likelihood
components or subtracted with propagated uncertainties; they must not simply
be ignored.

The use of shared parameters across control, validation, and signal regions is
standard in HistFitter/HistFactory-style analyses. See the [HistFitter control
and validation-region
workflow](https://cds.cern.ch/record/2194575/files/10.1088_1742-6596_608_1_012049.pdf)
and the [pyhf HistFactory
implementation](https://cds.cern.ch/record/2751465).

### 19.6 Same observable in every region versus channel-specific observables

Two fit designs should be compared.

#### Design A: the same score and energy observable in all regions

Use:

```text
classifier score x E_vis
```

in the SR, CRs, and VRs.

Advantages:

```text
direct measurement of background score shapes in data
direct test of score transport in VRs
simple channel interpretation
same final observable throughout the likelihood
```

Risks:

```text
score may be strongly correlated with region definitions
CR and SR shapes may be very different
some CRs may not distinguish their target backgrounds
score-dependent transfer factors may need many nuisance parameters
```

This should be the preferred baseline if score-shape and efficiency closure
are satisfactory.

#### Design B: channel-specific control-region observables

Use a signal classifier in the SR but background-specific observables in the
CRs:

```text
SR:    signal score x E_vis
CR-V:  in-time veto multiplicity, veto QDC, or veto timing
CR-F:  signed SciFi fiducial distance
CR-U:  US plane count, US multiplicity, or US QDC
```

Advantages:

```text
each CR may constrain its target background more strongly
background components may be more identifiable
detector-level effects can be constrained directly
```

Risks:

```text
greater reliance on the transfer model
CR data do not directly determine the SR score shape
more systematic variations and validation tests are required
```

The analysis should begin with Design A and move to Design B only when the
common score fit cannot identify the backgrounds or does not close.

### 19.7 Use of region-defining variables in the fit

Veto timing, signed fiducial distance, and US activity may legitimately be
fitted in control regions. The danger is not fitting those variables; the
danger is defining a hard CR/SR boundary in a variable and then assuming an
unvalidated extrapolation across the boundary.

For example, a continuous signed fiducial-distance distribution can connect:

```text
far outside -> near boundary -> loose inside -> tight inside
```

but only if geometry, detector gaps, reconstruction failures, and acceptance
effects are accurately described. Near-boundary VRs are required to test this
transport.

Region variables are therefore:

```text
useful for constraining background components in CRs
useful for measuring distance from the SR
risky as the only SR signal discriminator
dependent on validated interpolation or transfer across cut boundaries
```

An ABCD fit in classifier-score bins requires particular care. If the score
uses SciFi geometry and the ABCD axis is SciFi containment, the background
score can be correlated with that axis. The correction

```text
kappa(i) = N_A(i) * N_D(i) / [N_B(i) * N_C(i)]
```

must then be measured or constrained as a function of score and validated in
the third axis. ABCD independence must never be inferred merely from the fact
that the direct cut variable was removed from the ML input.

### 19.8 Multiclass background categorization

If the fitted backgrounds have distinct detector signatures, a multiclass
model can be more useful than one binary signal/background classifier. A
physics-oriented class structure could be:

```text
target signal, for example CC_nue
other neutrino interactions
muon-like background
neutral-hadron-like background
low-US or inefficient-shower background
```

Possible likelihood inputs include:

```text
one signal score derived from all class probabilities
separate categories defined by the dominant background probability
a signal score fitted inside each background-enriched category
two compressed axes: signal-likeness and background-component-likeness
```

For example:

```text
muon-like category:
    fit signal score x E_vis

neutral-like category:
    fit signal score x E_vis

other-background category:
    fit signal score x E_vis
```

This can improve component identifiability, but only if classes correspond to
physically meaningful processes or carefully defined proxy samples. The
current `split_v1` structure, consisting of one no-US real-data proxy class
and four neutrino truth classes, does not yet provide the detector-background
components needed by the final likelihood.

### 19.9 Quantitative criteria for selecting the observable

Every candidate fit design should be evaluated using blinded simulations,
Asimov data, or pseudo-experiments. The following tests are required.

#### Signal separation

Determine whether the shape distinguishes the target signal from the total
background after all selections.

#### Background identifiability

Determine whether the fit can separately constrain the proposed background
components. If two components have nearly identical shapes and transfers,
their independent normalization parameters will be highly correlated or
unconstrained. They should then be combined or constrained by additional
auxiliary information.

#### CR-to-VR closure

After deriving the CR model, compare the prediction with every VR for:

```text
total yield
fit-observable shape
classifier-score efficiency
visible-energy dependence
V, F, and U dependence
run-period dependence
```

#### Systematic stability

Repeat the templates and transfers under credible variations of:

```text
QDC and energy calibration
veto timing and time window
SciFi fiducial boundaries
US threshold and plane requirements
dead and noisy channels
generator and particle-production model
EventBuilder and reconstruction version
run period and detector conditions
CR background mixture and reweighting
```

#### Statistical robustness

Require sufficient data and simulation in every fitted bin. Merge bins with
poor CR statistics or unstable transfer factors. Propagate finite-template
statistics with bin-by-bin nuisance parameters when appropriate.

#### Expected inference performance

Compare fit designs using:

```text
expected uncertainty on signal strength
expected discovery significance or expected limit
fit bias in pseudo-experiments
nuisance-parameter pulls and constraints
background-normalization correlations
coverage and goodness of fit
stability under alternative truth models
```

The final binning and observable must be frozen before unblinding. Binning
must not be tuned to make observed SR data agree with a preferred model.

### 19.10 Required staged fit development

The likelihood should be built in increasing complexity.

#### Stage 1: count-only fit

Use one event-count bin for each CR and a blinded or Asimov SR count.

Purpose:

```text
validate luminosity and normalization conventions
test whether background normalizations are identifiable
test inclusive transfer factors
understand nuisance correlations
identify missing background components
```

If the count-only fit is unstable or biased, adding classifier shapes will not
solve the underlying normalization or component problem.

#### Stage 2: visible-energy shape fit

Add a small number of broad `E_vis` bins in the CRs, VRs, and blinded SR.
Validate energy-dependent transfer factors and detector-response
uncertainties.

#### Stage 3: classifier-score shape fit

Fit the out-of-fold classifier score, initially inclusive in visible energy.
Require score-shape and efficiency closure in all VRs.

#### Stage 4: score and visible energy

After both one-dimensional models are stable, fit:

```text
classifier score x coarse E_vis
```

This is the recommended baseline final observable for the current analysis.

#### Stage 5: background-enriched categories

Only if important backgrounds remain degenerate, introduce multiclass-derived
categories or channel-specific CR observables. The increased complexity must
provide a demonstrable reduction in the total expected uncertainty.

### 19.11 Concrete recommendation for `snd-ml`

The first serious shape-fit configuration should be:

```text
SR observable:
    cross-fitted target-signal classifier score

second dimension:
    3 to 5 bins of calibrated
    k * QDC_SciFi + alpha * QDC_US

CR/VR baseline observable:
    initially the same score and energy bins

additional or alternative CR constraints:
    CR-V -> in-time veto multiplicity, timing, and QDC
    CR-F -> signed SciFi fiducial distance
    CR-U -> US plane count, multiplicity, and QDC
```

Before implementing this final fit, the following questions must have explicit
answers:

```text
What is the parameter of interest?
Which neutrino interaction is the target signal?
Which independently normalized background components can the data constrain?
Which CR primarily constrains each component?
How is each component transferred to the SR?
Which folds define training, calibration, and final closure?
Which score and energy systematic templates are required?
What quantitative closure criterion must be passed before unblinding?
```

The central recommendation is:

> In the absence of a reconstructed mass, use the most transportable and
> systematically robust approximation to the signal/background likelihood
> ratio. For this analysis that is likely an out-of-fold classifier score,
> supplemented by a coarse calibrated visible-energy proxy. Use separate CR
> observables only when needed to identify individual background components,
> and accept the final design based on profiled expected uncertainty and VR
> closure rather than nominal AUC.

## 20. Final Recommendation

The selected baseline is a data-driven multi-region analysis. Background MC
must not set the absolute SR background normalization because it is observed
to underpredict real data. Real CR data should train the background side of
the classifier and constrain background normalizations and score shapes.
Background MC remains necessary for process interpretation, CR-to-SR transfer
guidance, signal-contamination estimates, and systematic variations.

The current eight-region partition should remain as the first layer of the
analysis because it is orthogonal, interpretable, and useful for diagnostics.
The next region version should add a second layer based on nested working
points:

```text
one blinded tight SR
three adjacent one-step VRs
three background-specific one-inversion CRs
double-inversion diagnostic/closure regions
```

For ML training, do not use the unweighted union of all non-SR regions. Select
the TR, or mixture of TRs, that best models the expected SR background after
validated corrections. Exclude region-defining information from the ML model,
measure residual domain differences, and require score-efficiency closure in
independent VRs.

The preferred final workflow is:

```text
1. diagnose and characterize the data-MC deficit;
2. define continuous V, F, and U variables with nested working points;
3. freeze and blind the tight SR;
4. train signal MC versus cross-fitted CR data;
5. validate score transport in near-SR VRs;
6. constrain backgrounds with a simultaneous CR/SR likelihood fit;
7. use ABCD/pass-fail closure in score bins as an independent cross-check;
8. propagate VR nonclosure as a background systematic uncertainty;
9. unblind only after the model, transfers, and uncertainties are frozen.
```

The classifier and background estimate must remain conceptually separate. A
classifier trained on CR data helps represent backgrounds missing from MC, but
does not by itself predict their SR yield. The simultaneous fit or validated
data transfer supplies that prediction.

The final choice should answer this quantitative question:

> Which control-region data can be transported to the signal region with the
> smallest correction and the smallest validated total uncertainty while
> retaining enough statistics to train and constrain the background?

That criterion is more robust than choosing the sideband with the largest
event count or the model with the largest nominal classification AUC.

## 21. Reconstructed-track inspection implementation

The passing-muon reconstructed tracks are now handled by a separate,
versioned production. The existing feature files and the existing region-v2
inspection inputs are not overwritten.

### 21.1 Why the ROOT objects are converted

The input branch is a `TClonesArray` of `sndRecoTrack` objects named
`Reco_MuonTracks`. Copying that branch into the feature tree would couple all
downstream analysis to the exact SNDSW class dictionary and make vectorized
analysis unnecessarily difficult. The converter therefore writes primitive
vectors plus stable scalar summaries.

The supported track-type mapping is:

| Type | Prefix | Meaning |
|---:|---|---|
| 1 | `reco_scifi_simple` | Simple SciFi track |
| 3 | `reco_ds_simple` | Simple downstream track |
| 11 | `reco_scifi_hough` | Hough SciFi track |
| 13 | `reco_ds_hough` | Hough downstream track |
| 15 | `reco_combined_hough` | Hough combined-detector track |

Unknown type codes are retained in `reco_track_type` and counted in
`reco_track_n_unknown_type`; they are not silently discarded.

### 21.2 Saved branches and absence convention

All reconstructed tracks are stored in aligned vectors:

```text
reco_track_type, reco_track_flag
reco_track_chi2, reco_track_ndf, reco_track_chi2ndf
reco_track_npoints
reco_track_start_{x,y,z}, reco_track_stop_{x,y,z}
reco_track_mom_{x,y,z}
reco_track_slope_{xz,yz}, reco_track_angle_{xz,yz}
```

The event-level branches are:

```text
reco_track_branch_available
reco_track_n_all
reco_track_n_converged
reco_track_n_types_present
reco_track_n_unknown_type
```

For every prefix in the table, the converter saves:

```text
<prefix>_n_all
<prefix>_n_converged
<prefix>_has_any
<prefix>_has_converged
<prefix>_has_best
<prefix>_best_index
<prefix>_best_{flag,chi2,ndf,chi2ndf,npoints}
<prefix>_best_start_{x,y,z}
<prefix>_best_stop_{x,y,z}
<prefix>_best_mom_{x,y,z}
<prefix>_best_slope_{xz,yz}
<prefix>_best_angle_{xz,yz}
```

Absence is explicit: counts and `has_*` flags are zero, `best_index` is `-1`,
and unavailable floating-point best-track values are `NaN`. Consequently,
“this event has no track of this type” is distinct from “a track object exists
but its fit did not converge.”

The best track must have `flag == 1`, `ndf > 0`, and a finite non-negative
`chi2/ndf`. It is selected by minimum `chi2/ndf`, then maximum number of track
points, then original collection index. `best_index` always refers to the
global aligned track vectors.

The start and stop positions and fitted direction components come from
`sndRecoTrack`. Because this reconstruction uses a zero-field fit, the
momentum magnitude must not be interpreted as a calibrated particle energy.
The slopes and angles are the safer quantities for topology inspection.
Unsafe legacy direction/velocity accessors are not called.

### 21.3 Versioned production and validation

The implementation is in:

```text
convertData/reco_track_features.py
convertData/digi_2_features.py --extract-reco-tracks
convertData/validate_reco_track_feature_file.py
snakemake/rules/passing_muon_track_features.smk
```

Outputs are written below:

```text
/eos/experiment/sndlhc/users/zhibin/
  passing_muon_track_features_v1/2024/run_XXXXXX/
```

This production deliberately does **not** use
`--drop-real-has-veto-has-us`. It retains the complete reconstructed
passing-muon sample and therefore avoids inheriting the category removal in
the older passing-muon feature production. Each temporary ROOT file is checked
for the complete `flat_v1` schema, aligned vector lengths, per-type count
closure, and valid best-track sentinels before publication to EOS.

Run the metadata and feature production with:

```bash
source snakemake/envs/env_lcg.sh

snakemake -s snakemake/snakefile \
  passing_muon_track_metadata --cores 1 -p

snakemake -s snakemake/snakefile \
  passing_muon_track_features \
  --executor slurm --jobs 100 --rerun-incomplete -p
```

For a dry-run or one-file test, request the exact versioned output path:

```bash
snakemake -s snakemake/snakefile \
  /eos/experiment/sndlhc/users/zhibin/passing_muon_track_features_v1/2024/run_008329/featureTracks_sndsw_raw-9_0_8329_muonReco.root \
  --cores 1 -n -p
```

### 21.4 Dedicated inspection and plots

The first inspection is deliberately limited to passing-muon run 8329. Only
passing-muon data contain this reconstruction branch, so a six-sample
data/MC overlay would imply that the absent branches in the other five samples
were physical zero-track observations. They are not.

The `passing_muon_track_inspection` target:

1. produces all run-8329 track-feature files;
2. consolidates them without an additional event selection;
3. fills track multiplicity and best-track `chi2/ndf` histograms;
4. fills veto-count versus track multiplicity and best-track `chi2/ndf`;
5. fills earliest valid veto time versus track multiplicity and best-track
   `chi2/ndf`;
6. fills track-start x/y occupancy maps and track-start x/y versus best-track
   `chi2/ndf` in three dimensions; and
7. writes exactly one plot per PDF, with no PNG copies.

ROOT underflow and overflow bins are not drawn by default. The histogram step
therefore folds values outside each configured display range into the
corresponding first or last visible bin, including the propagated statistical
error. This is important for the long high-`chi2/ndf` tail and prevents the
inspection plots from silently losing poor-quality tracks.

It produces eight PDFs for each of the five track types:

```text
<prefix>_n_all.pdf
<prefix>_best_chi2ndf.pdf
<prefix>_veto_count_vs_n_all.pdf
<prefix>_veto_time_vs_n_all.pdf
<prefix>_veto_count_vs_chi2ndf.pdf
<prefix>_veto_time_vs_chi2ndf.pdf
<prefix>_start_xy_2d.pdf
<prefix>_start_xy_chi2ndf_3d.pdf
```

It also produces four SciFi-versus-DS correlation PDFs:

```text
reco_simple_ds_n_all_vs_scifi_n_all.pdf
reco_simple_ds_best_chi2ndf_vs_scifi_best_chi2ndf.pdf
reco_hough_ds_n_all_vs_scifi_n_all.pdf
reco_hough_ds_best_chi2ndf_vs_scifi_best_chi2ndf.pdf
```

Run it with:

```bash
source snakemake/envs/env_lcg.sh
snakemake -s snakemake/snakefile \
  passing_muon_track_inspection \
  --executor slurm --jobs 100 --rerun-incomplete -p
```

The plot outputs are under:

```text
evaluation_region_partitions/region_v2_inspection/
  passing_muon_tracks_v1/plots/
```

These plots are diagnostic inputs to region-v2 design. Track `chi2/ndf` should
not automatically become a region-defining cut: first check its stability
across runs, its correlation with veto activity, the failed-fit population,
and the relative behavior of the five reconstruction algorithms.

### 21.5 Full-schema partitioned inspection workflow

The preferred scalable inspection input is now twenty full-schema partitions
rather than one 58 GB consolidated ROOT file. It reuses the existing
`build_region_partitions.py` whole-file, luminosity-partition logic.

The dedicated configuration applies:

```text
source dataset: passing_muon_real_data, run 8329
region:         all_events, expression true
particle group: passing_muon, event filter true
fraction:       1.0
partitions:     20
shuffle:        false
```

No `feature_branches` list is configured. The existing partition builder
therefore preserves every input `sndData` branch. Five additional provenance
branches are written:

```text
source_file_index
original_entry
event_uid
particle_group
particle_id
```

`source_file_index` is the zero-based row in the run-8329 source metadata.
The ordered `source_file_indices` and `source_feature_paths` fields provide
the exact mapping in each partition. `original_entry` is copied from
`eventIndex`, and `event_uid` is constructed from those two globally stable
values.

The twenty outputs are written below:

```text
/eos/experiment/sndlhc/users/zhibin/snd-ml/
  passing_muon_track_partitions/
  passing_muon_track_partition_v1/
  passing_muon_track_inspection_region_v1/
  all_events/passing_muon/
```

Before histogram production, a strict validator requires:

```text
20 readable, non-recovered ROOT partition files
2,983 unique source feature files with no gap or duplication
82,145,120 total sndData entries
0.5323492746290897 fb^-1 luminosity closure
identical branch schema/order in every partition
every original sndData branch retained
valid source_file_index/original_entry/event_uid samples
```

The histogram program supports both input modes:

```text
--input <legacy consolidated ROOT file>
--partition-metadata <validated combined partition CSV>
```

Both paths produce the same forty-four histogram names, so the PDF plotting code
is unchanged. The partitioned outputs are stored separately under
`passing_muon_track_partitions_v1`; the legacy consolidated target remains
available for comparison.

Run a dry-run with:

```bash
source snakemake/envs/env_lcg.sh
snakemake -s snakemake/snakefile \
  passing_muon_track_partition_inspection \
  --cores 20 -n -p \
  --ignore-incomplete --max-inventory-time 0 --rerun-triggers mtime
```

Run the production on Slurm with:

```bash
snakemake -s snakemake/snakefile \
  passing_muon_track_partition_inspection \
  --executor slurm --jobs 20 \
  --rerun-incomplete -p
```

The partitions reduce wall time and failure scope through parallelism, but
because every branch is retained they do not reduce total output bytes
substantially relative to the consolidated file.
