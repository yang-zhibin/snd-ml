# Region-v1 ABCD Background-Estimation Baseline Plan

## Purpose

Establish transparent baseline background estimates with the existing
`partition_v1` / `region_v1` real-data regions before developing `region_v2`.
Following the cited ABCD guide, each baseline must define a nominal plane,
test the relationship in observable control regions, quantify nonclosure, and
avoid presenting an unvalidated correction as a final signal-region result.

The plan covers inclusive counts and a SciFi-density-binned shape estimate. It
does not authorize production jobs or additional unblinding.

## Existing regions

Every region requires `count_scifi > 35`.

| Region | Veto | US | SciFi fiducial |
|---|---|---|---|
| Signal region | no veto | has US | inside |
| SB1 | no veto | no US | inside |
| SB2 | no veto | has US | outside |
| SB3 | no veto | no US | outside |
| SB4 | has veto | has US | inside |
| SB5 | has veto | no US | inside |
| SB6 | has veto | has US | outside |
| SB7 | has veto | no US | outside |

## Shared analysis requirements

1. Use raw counts and explicit source luminosities in the likelihood. Use
   luminosity-normalized rates for diagnostic arithmetic.
2. Verify that paired regions cover identical source exposure. Current
   no-veto partitions use all input exposure, while veto-active partitions use
   the first 10%.
3. Treat available signal-region data as development data only. Do not use the
   same events to tune a correction and claim independent closure.
4. Evaluate signal-MC leakage in every region and density bin. Background
   normalization remains data-driven.
5. Account explicitly for histogram underflow and overflow. Plot-range
   integrals are not inclusive event counters.
6. Report a transparent arithmetic estimate and a simultaneous Poisson
   likelihood result.
7. Do not merge the seven sidebands into one background template.

## Baseline 1: veto x US, fiducial-inside

This is the standard axis-aligned `region_v1` ABCD plane.

### Nominal plane

| ABCD role | Region | Veto | US | Fiducial |
|---|---|---|---|---|
| A | Signal region | no veto | has US | inside |
| B | SB4 | has veto | has US | inside |
| C | SB1 | no veto | no US | inside |
| D | SB5 | has veto | no US | inside |

The inclusive and density-binned estimates are:

```text
N_A_pred = N_SB4 * N_SB1 / N_SB5

T_veto_inside(i) = N_SB1(i) / N_SB5(i)
N_A_pred(i)      = N_SB4(i) * T_veto_inside(i)
```

### Outside-fiducial validation plane

| Role | Region | Veto | US | Fiducial |
|---|---|---|---|---|
| A' | SB2 | no veto | has US | outside |
| B' | SB6 | has veto | has US | outside |
| C' | SB3 | no veto | no US | outside |
| D' | SB7 | has veto | no US | outside |

```text
N_SB2_pred(i) = N_SB6(i) * N_SB3(i) / N_SB7(i)

kappa_outside(i)
  = N_SB2_obs(i) * N_SB7(i) / (N_SB6(i) * N_SB3(i))
```

Report the uncorrected result (`kappa = 1`) before considering a nonclosure
correction. A correction measured in this validation plane is not thereby
validated for the fiducial-inside signal region.

Principal risks are the different veto/no-veto density shapes, strong veto-US
correlation, sparse SB1/SB5 tails, and the lack of a second near-boundary
validation category.

## Baseline 2: fiducial transfer with composite veto-US categories

This nominal estimate avoids SB1 and SB4. It defines two disjoint categories:

```text
Q_signal_like = no veto and has US
Q_control     = has veto and no US
```

The mixed categories are omitted from the nominal plane.

### Nominal plane

| ABCD role | Region | Composite category | Fiducial |
|---|---|---|---|
| A | Signal region | no veto, has US | inside |
| B | SB2 | no veto, has US | outside |
| C | SB5 | has veto, no US | inside |
| D | SB7 | has veto, no US | outside |

```text
N_A_pred = N_SB2 * N_SB5 / N_SB7

T_fid_control(i) = N_SB5(i) / N_SB7(i)
N_A_pred(i)      = N_SB2(i) * T_fid_control(i)
```

This assumes that the fiducial inside/outside transfer measured in the
`has veto, no US` population transports to the `no veto, has US` population.

### Mixed-category validation plane

| Role | Region | Composite category | Fiducial |
|---|---|---|---|
| A' | SB1 | no veto, no US | inside |
| B' | SB3 | no veto, no US | outside |
| C' | SB4 | has veto, has US | inside |
| D' | SB6 | has veto, has US | outside |

```text
N_SB1_pred(i) = N_SB3(i) * N_SB4(i) / N_SB6(i)

kappa_mixed(i)
  = N_SB1_obs(i) * N_SB6(i) / (N_SB3(i) * N_SB4(i))
```

SB1 and SB4 do not enter the nominal Baseline-2 SR estimate, but they provide
the only available independent `region_v1` diagnostic validation. If they must
also be excluded from validation, Baseline 2 is unvalidated.

Principal risks are the different physical mixtures in the two composite
categories, their non-complementary definition, and the intrinsic correlation
between SciFi density and fiducial containment.

## Signal-contamination treatment

For an assumed signal strength:

```text
N_tilde_R(i) = N_data_R(i) - mu * N_signal_R(i)
```

For each baseline report the result with no signal subtraction, with nominal
signal subtraction, and their difference. Existing plots indicate very small
integrated contamination in the inspected control regions, but the fraction
must be checked in sparse high-density bins. If signal strength is unknown,
float it in the likelihood instead of manually subtracting it.

Muon-DIS MC may guide shape interpretation, but it is not a complete
background model and must not normalize the total data background.

## Density binning

Use `density_sndsw_scifi`, but freeze a versioned ABCD binning before detailed
nominal-SR closure inspection. The existing 500-unit plotting bins may be too
fine in the tail. The binning must be common to all regions, merge sparse tail
bins, and document underflow/overflow handling. Record raw counts, weighted
yields, sum of squared weights, and effective counts.

### Frozen sideband-only merged binning

The versioned policy `density_sndsw_scifi_abcd_merged_v1` was frozen using
only raw regular-bin counts from SB1--SB7. The signal-region distribution,
ABCD predictions, and closure results were not used. Starting from the 80
500-unit bins, the policy preserves consecutive bins from zero while every
sideband has at least 50 events, then merges the remaining tail from high to
low until the same criterion holds in every common bin. The frozen regular-bin
edges are:

```text
[0, 500, 1000, 1500, 2000, 3000, 40000]
```

Thus the six regular bins are `[0,500)`, `[500,1000)`, `[1000,1500)`,
`[1500,2000)`, `[2000,3000)`, and `[3000,40000)`. The final bin includes
the already-folded overflow. Underflow remains separate and is not subject to
the minimum-count criterion. Every sideband has at least 50 raw events in
each regular bin. This bounds the conservative four-count statistical design
proxy at `sqrt(4/50) = 0.2829`, below the frozen 0.30 target; it is not a
closure calculation or the uncertainty model for the final likelihood. These
edges must not be retuned after inspecting closure.

## Required outputs

For each baseline produce:

1. inclusive A/B/C/D counts and exposures;
2. inclusive arithmetic prediction and uncertainty;
3. per-density-bin A/B/C/D table;
4. transfer factor versus density;
5. predicted and observed validation distributions;
6. validation ratio and `kappa(i)`;
7. pull or likelihood goodness of fit;
8. inclusive and per-bin signal contamination;
9. stability by complete run or run block;
10. comparison of the two baseline predictions.

For populated unweighted bins, show the arithmetic approximation:

```text
(sigma_A / N_A_pred)^2 ~= 1/N_B + 1/N_C + 1/N_D
```

Use a Poisson likelihood for sparse bins and include exposure, signal leakage,
and any nonclosure correction as explicit parameters or nuisances.

## Interpretation rules

1. Agreement between baselines is informative but does not prove correctness.
2. A non-unit `kappa` is usable only if stable in density and independent data.
3. A derived nonclosure correction requires secondary validation. Use a run
   split as an additional test and document the remaining limitation of the
   binary fiducial categories.
4. The available SR development sample can document diagnostic closure, but
   cannot tune the method and then serve as the final unbiased measurement.
5. If neither method closes in normalization and shape, record that result; do
   not force a correction. It becomes the quantitative motivation for
   continuous `region_v2` categories.

## Proposed implementation sequence

1. Validate catalogue exposure and event-count closure for all eight regions.
2. Freeze density binning and underflow/overflow handling.
3. Materialize inclusive and binned raw-count tables.
4. Add nominal signal-contamination tables.
5. Calculate Baseline 1 and its outside-fiducial validation.
6. Calculate Baseline 2 and its mixed-category validation.
7. Repeat validation quantities by complete run or run block.
8. Implement the simultaneous likelihood and compare it with the arithmetic
   calculations.
9. Classify each method as closed, usable only with a validated correction, or
   rejected.

No implementation or production run should start until this plan and the role
of SB1/SB4 in Baseline 2 validation are approved.
