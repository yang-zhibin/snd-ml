# Region V1 ABCD Statistical Likelihood Guide

## Purpose

This document explains how to construct a statistical likelihood from first
principles and how to apply it to the `partition_v1` / `region_v1` ABCD
background-estimation baselines. It also defines the appropriate role of
`cabinetry` and `pyhf` in the workflow.

The main conclusion is that a likelihood provides the correct propagation of
control-region statistics, exposure differences, nuisance parameters, and
signal contamination, but it cannot repair a failed ABCD factorization
assumption. Closure must be established in independent validation data before
an ABCD model is used for a final signal-region inference.

This guide complements `REGION_V1_ABCD_BASELINE_PLAN.md`. The numerical values
below are diagnostic values from the existing inclusive `selected_entries`
counts. They are not a final background estimate.

## Start with a counting experiment

If a region contains an observed raw count `n` and the model predicts an
expected yield `nu`, the probability of that observation is Poisson:

```text
P(n | nu) = nu^n exp(-nu) / n!
```

For independent regions `R` and analysis bins `i`, the joint likelihood is:

```text
L = product over R,i of Pois(n_Ri | nu_Ri)
```

The likelihood must use raw event counts. Luminosity-scaled or otherwise
weighted histogram contents are not Poisson observations.

## Define signal and background expectations

For every region and bin, write the expected yield as:

```text
nu_Ri(mu, theta)
  = mu * s_Ri(theta)
  + b_Ri(theta)
  + b_other_Ri(theta)
```

where:

- `s_Ri` is the expected signal yield for nominal signal strength;
- `mu` is the common signal-strength parameter;
- `b_Ri` is the background described by the ABCD model;
- `b_other_Ri` is any background estimated separately;
- `theta` denotes nuisance parameters.

The background-only hypothesis is `mu = 0`. The nominal signal hypothesis is
normally `mu = 1`. If signal can enter a control region, the same `mu` must
scale that contribution. Control-region signal leakage must not be omitted
merely because the region is called a sideband.

## Parameterize an ABCD background

Use the conventional roles:

| Role | First selection | Second selection |
| --- | --- | --- |
| A | pass | pass |
| B | fail | pass |
| C | pass | fail |
| D | fail | fail |

Define:

- `rho`: reference background rate;
- `u`: transfer factor along one ABCD axis;
- `v`: transfer factor along the other ABCD axis;
- `L_R`: source exposure for region `R`;
- `kappa`: residual ABCD nonclosure.

The expected background yields are:

```text
b_D = L_D * rho
b_B = L_B * rho * u
b_C = L_C * rho * v
b_A = L_A * rho * u * v * kappa
```

The ideal factorized ABCD model has `kappa = 1`. Its exposure-corrected
arithmetic prediction is:

```text
b_A_pred = (n_B * n_C / n_D) * (L_A * L_D) / (L_B * L_C)
```

When `L_A = L_C` and `L_B = L_D`, as in each current Region V1 baseline,
the exposure correction cancels and the familiar expression remains:

```text
b_A_pred = n_B * n_C / n_D
```

The cancellation must be demonstrated from the metadata. It must not be
assumed simply because the regions were produced by the same workflow.

## Derive arithmetic ABCD from the likelihood

For a blinded control-region fit using only B, C, and D:

```text
L_CR(rho, u, v)
  = Pois(n_B | L_B * rho * u)
  * Pois(n_C | L_C * rho * v)
  * Pois(n_D | L_D * rho)
```

The maximum-likelihood estimates are:

```text
rho_hat = n_D / L_D

u_hat = (n_B / L_B) / (n_D / L_D)
v_hat = (n_C / L_C) / (n_D / L_D)
```

Substitution into `b_A = L_A * rho * u * v` gives:

```text
b_A_hat = (n_B * n_C / n_D) * (L_A * L_D) / (L_B * L_C)
```

The usual arithmetic ABCD prediction is therefore the maximum-likelihood
prediction of the elementary three-parameter control-region model.

For sufficiently populated, independent control regions, the approximate
statistical uncertainty is:

```text
(sigma_bA / b_A_hat)^2 ~= 1/n_B + 1/n_C + 1/n_D
```

This approximation is not reliable for sparse or zero-count bins. The Poisson
likelihood should be used directly in that case.

## Add nuisance-parameter constraints

Systematic uncertainties introduce nuisance parameters with constraint terms:

```text
L_total(mu, theta)
  = L_counts(mu, theta)
  * product over j of pi_j(theta_j)
```

Possible nuisance parameters include:

- signal efficiency and acceptance;
- integrated luminosity for simulated signal or simulated backgrounds;
- finite signal-MC statistics;
- detector calibration and reconstruction effects;
- run-dependent selection efficiency;
- independently validated ABCD nonclosure;
- separately estimated background normalization and shape.

Control-region data statistics should remain Poisson observations. They should
not be replaced by Gaussian constraints unless there is a specific validated
reason to make that approximation.

## Measure and validate nonclosure

For an orthogonal validation plane:

```text
b_Aprime_pred
  = (n_Bprime * n_Cprime / n_Dprime)
  * (L_Aprime * L_Dprime) / (L_Bprime * L_Cprime)
```

Define the validation closure factor:

```text
kappa_validation
  = n_Aprime / b_Aprime_pred
```

Equivalently:

```text
kappa_validation
  = (n_Aprime * n_Dprime) / (n_Bprime * n_Cprime)
  * (L_Bprime * L_Cprime) / (L_Aprime * L_Dprime)
```

For populated independent regions, its approximate statistical uncertainty is:

```text
(sigma_kappa / kappa)^2
  ~= 1/n_Aprime + 1/n_Bprime + 1/n_Cprime + 1/n_Dprime
```

A non-unit value can arise from correlation between the ABCD axes, changing
background composition, detector effects, or mismatched exposure. Its cause
should be investigated before assigning a correction.

An unconstrained `kappa` must not be added to the signal region because it can
absorb a genuine excess and become degenerate with the signal strength. A
signal-region nonclosure parameter must be fixed, constrained by independent
evidence, or determined through a model that contains adequate independent
validation information.

## Fit and profile the likelihood

Ignoring constants independent of the parameters, the Poisson deviance is:

```text
-2 log L
  = 2 * sum over R,i of
      [nu_Ri - n_Ri + n_Ri * log(n_Ri / nu_Ri)]
    + constraint terms
```

The best-fit parameters minimize this expression. For a chosen signal strength
`mu`, profile all nuisance parameters and form:

```text
lambda(mu)
  = L(mu, theta_hat_hat(mu)) / L(mu_hat, theta_hat)

q_mu = -2 log lambda(mu)
```

Typical uses are:

- `q_0` for the background-only p-value and discovery significance;
- `q_mu` for a confidence interval or upper limit on signal strength;
- likelihood scans;
- nuisance-parameter pulls, constraints, and impacts;
- goodness-of-fit tests.

Asymptotic formulae should be checked with pseudoexperiments when counts are
sparse, parameters lie near physical boundaries, or the likelihood is strongly
non-Gaussian.

## Apply the model to Baseline 1

Baseline 1 uses veto activity and US activity within the fiducial-inside
population:

| Role | Region | Count | Exposure |
| --- | --- | ---: | ---: |
| A | Signal region | 158707 | 19.7806809279 |
| B | SB4 | 20348106 | 1.96031925224 |
| C | SB1 | 40300 | 19.7806809279 |
| D | SB5 | 26510 | 1.96031925224 |

Because `L_A = L_C` and `L_B = L_D`, the exposure factor cancels:

```text
b_A_pred = 20348106 * 40300 / 26510
         = 30932805
```

The approximate relative statistical uncertainty is:

```text
sqrt(1/20348106 + 1/40300 + 1/26510) = 0.00791
```

Therefore:

```text
b_A_pred ~= (30.93 +/- 0.24) million events, statistical only
```

The diagnostic signal-region closure factor is:

```text
kappa_inside = 158707 / 30932805 = 0.00513
```

This is an extreme failure of the veto-US factorization assumption, not a
statistical fluctuation.

### Baseline 1 validation plane

The outside-fiducial validation mapping is:

| Role | Region | Count | Exposure |
| --- | --- | ---: | ---: |
| A prime | SB2 | 447152 | 19.7806809279 |
| B prime | SB6 | 18928315 | 1.96031925224 |
| C prime | SB3 | 73186 | 19.7806809279 |
| D prime | SB7 | 29142 | 1.96031925224 |

The prediction and closure factor are:

```text
b_Aprime_pred = 18928315 * 73186 / 29142
              = 47535779

kappa_outside = 447152 / 47535779
              = 0.00941
```

Baseline 1 should be implemented to document its nonclosure quantitatively,
but it must not be used for a final signal-region prediction in its current
form.

## Apply the model to Baseline 2

Baseline 2 uses the fiducial transfer between the two composite veto-US
categories:

| Role | Region | Count | Exposure |
| --- | --- | ---: | ---: |
| A | Signal region | 158707 | 19.7806809279 |
| B | SB2 | 447152 | 19.7806809279 |
| C | SB5 | 26510 | 1.96031925224 |
| D | SB7 | 29142 | 1.96031925224 |

The prediction and diagnostic closure factor are:

```text
b_A_pred = 447152 * 26510 / 29142
         = 406767

kappa_nominal = 158707 / 406767
              = 0.390
```

### Baseline 2 validation plane

The mixed-category validation mapping is:

| Role | Region | Count | Exposure |
| --- | --- | ---: | ---: |
| A prime | SB1 | 40300 | 19.7806809279 |
| B prime | SB3 | 73186 | 19.7806809279 |
| C prime | SB4 | 20348106 | 1.96031925224 |
| D prime | SB6 | 18928315 | 1.96031925224 |

The prediction and closure factor are:

```text
b_Aprime_pred = 73186 * 20348106 / 18928315
              = 78676

kappa_validation = 40300 / 78676
                 = 0.512
```

Baseline 2 is less pathological than Baseline 1, but it still has about 49%
validation nonclosure. It is not yet suitable for a final discovery or limit
calculation.

## Role of cabinetry and pyhf

The workflow boundary should be:

```text
Snakemake event processing
  -> raw region counts and signal templates
  -> arithmetic closure checks
  -> frozen likelihood and nuisance model
  -> cabinetry HistFactory workspace
  -> pyhf fit, likelihood scans, significance, and limits
```

Do not use `cabinetry` to choose ABCD variables, optimize cuts on signal-region
data, test physical independence, produce the region partitions, or repair
nonclosure automatically.

Use `cabinetry` and `pyhf` to:

- encode Poisson channels and template bins;
- share transfer-factor parameters between regions;
- include signal contamination in every region;
- encode correlated systematic uncertainties;
- perform maximum-likelihood fits;
- inspect pulls and parameter correlations;
- calculate profile-likelihood scans, significances, and limits;
- serialize a reproducible HistFactory workspace.

## Stable HistFactory parameterization

Directly fitting rates of millions of events or transfer factors far from one
can produce poor numerical conditioning. Choose nominal reference values and
fit dimensionless scale factors near one:

```text
b_D = b_D0 * beta
b_B = b_B0 * beta * alpha_u
b_C = b_C0 * beta * alpha_v
b_A = b_A0 * beta * alpha_u * alpha_v * kappa
```

The nominal templates must satisfy:

```text
b_A0
  = (b_B0 * b_C0 / b_D0)
  * (L_A * L_D) / (L_B * L_C)
```

The background sample then receives these shared modifiers:

| Region | Modifiers |
| --- | --- |
| D | `beta` |
| B | `beta`, `alpha_u` |
| C | `beta`, `alpha_v` |
| A | `beta`, `alpha_u`, `alpha_v`, and optionally constrained `kappa` |

The signal sample in every region receives the same `mu` normalization factor.

For the first inclusive one-bin prototype, a small direct `pyhf` workspace may
be clearer than the complete `cabinetry` template-building configuration.
After the parameterization is verified, `cabinetry` can build, fit, inspect,
and visualize the full binned model.

## Recommended construction sequence

1. Reproduce both arithmetic estimates from authoritative raw metadata counts.
2. Build a four-channel, one-bin validation workspace with `kappa = 1`.
3. Confirm that the control-only fit reproduces the arithmetic prediction.
4. Fit all four validation regions with `kappa` free to measure nonclosure.
5. Repeat closure measurements by complete run or run block.
6. Freeze common `density_sndsw_scifi` binning and overflow handling.
7. Extend the model to density bins without silently merging sparse tails.
8. Add signal templates and control-region signal contamination.
9. Validate the fit with Asimov datasets and pseudoexperiments.
10. Calculate a significance or limit only for a model that passes the agreed
    closure and secondary-validation criteria.

## References

- [Region V1 ABCD baseline plan](REGION_V1_ABCD_BASELINE_PLAN.md)
- [Background Estimation with the ABCD Method](https://indico.cern.ch/event/1122790/contributions/4713580/attachments/2381493/4270369/ABCDGuide_draft18Oct18.pdf)
- [cabinetry documentation](https://cabinetry.readthedocs.io/)
- [cabinetry configuration schema](https://cabinetry.readthedocs.io/en/latest/config.html)
- [pyhf likelihood specification](https://scikit-hep.org/pyhf/likelihood.html)
- [Asymptotic formulae for likelihood-based tests of new physics](https://arxiv.org/abs/1007.1727)
