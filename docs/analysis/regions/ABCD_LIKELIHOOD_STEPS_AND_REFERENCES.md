# ABCD Likelihood: Analysis Steps and References

## Purpose

This note summarizes how to build a Poisson likelihood for a data-driven ABCD
background estimate and identifies the main statistical references. It is a
short companion to `REGION_V1_ABCD_STATISTICAL_LIKELIHOOD_GUIDE.md`; it is not
a final Region V1 result or a substitute for an independent closure test.

## 1. Start from observed event counts

For bin `i`, let:

- `n_i` be the integer number of events observed in data;
- `nu_i` be the mean number of events predicted by the model.

The probability of observing `n_i` given `nu_i` is

```text
Pois(n_i | nu_i) = nu_i^n_i * exp(-nu_i) / n_i!
```
The vertical bar means "given." This probability is one likelihood term; it is
not itself a p-value or a significance.

## 2. Decide whether the analysis is unbinned or binned

### Unbinned likelihood

For event-level measurements `x_j`, a simple extended unbinned likelihood is

```text
L(mu, b) = exp[-(mu*s + b)] / N!
           * product over events j of
             [mu*s*f_s(x_j) + b*f_b(x_j)]
```

Here `s` and `b` are nominal signal and background yields, `mu` is the signal
strength, and `f_s` and `f_b` are normalized continuous probability-density
functions. This approach preserves all event-level information but requires
well-validated continuous signal and background densities. ABCD relations
constrain yields; they do not automatically provide these densities.

### Binned likelihood

Divide the discriminating variable into bins and count events. For independent
regions `R` and bins `i`, use

```text
L(mu, theta) = product over R,i of Pois(n_Ri | nu_Ri(mu, theta))
               * product over j of pi_j(theta_j)
```

The `pi_j` terms constrain systematic nuisance parameters `theta_j`. Binned
likelihoods are the standard starting point for an ABCD analysis and are
directly supported by HistFactory and `pyhf`.

## 3. Define the predicted yield

For each region and bin, write

```text
nu_Ri = mu*s_Ri + sum over k of gamma_k*q_kRi + b_Ri
```

where:

- `mu` is the common signal-strength parameter;
- `s_Ri` is the nominal signal yield, normally calculated from weighted signal
  MC after the region and bin selections;
- `q_kRi` is the yield of separately modelled background component `k`;
- `gamma_k` controls the normalization of component `k`;
- `b_Ri` is the background described by the ABCD model.

Signal contamination in B, C, or D must use the same `mu` as signal in A. A
background included in `b_Ri` must not also be counted in `q_kRi`.

## 4. Parameterize the ABCD background

For each bin `i`, define:

- `L_R`: exposure associated with region `R`;
- `rho_i`: reference background rate in D;
- `u_i`: transfer factor along the first ABCD axis;
- `v_i`: transfer factor along the second ABCD axis;
- `kappa_i`: independently validated residual nonclosure correction.

Then

```text
b_Di = L_D*rho_i
b_Bi = L_B*rho_i*u_i
b_Ci = L_C*rho_i*v_i
b_Ai = L_A*rho_i*u_i*v_i*kappa_i
```

In the simplest control-only fit, with negligible signal and other-background
contamination,

```text
rho_hat_i = n_Di / L_D
u_hat_i   = (n_Bi/L_B) / (n_Di/L_D)
v_hat_i   = (n_Ci/L_C) / (n_Di/L_D)
```

For `kappa_i = 1`, the corresponding signal-region prediction is

```text
b_Ai_hat = (n_Bi*n_Ci/n_Di) * (L_A*L_D)/(L_B*L_C)
```

If signal leakage or other backgrounds are non-negligible, do not manually
subtract them and treat the remainder as exact. Fit all four regions
simultaneously with their Poisson terms and nuisance parameters.

## 5. Validate the ABCD assumption

Use an orthogonal validation plane or independent data to calculate

```text
kappa_validation
  = (n_A*n_D)/(n_B*n_C) * (L_B*L_C)/(L_A*L_D)
```

Check closure inclusively and versus important variables such as run period,
density, energy, detector position, and distance from the ABCD boundaries.
Investigate correlations and changing background composition. Do not introduce
an unconstrained signal-region `kappa`: it can absorb a genuine signal and
become degenerate with `mu`.

## 6. Fit and perform statistical inference

Maximize the likelihood to obtain `mu_hat` and the fitted nuisance parameters.
For a tested value of `mu`, profile the nuisance parameters and form

```text
lambda(mu) = L(mu, theta_hat_hat(mu)) / L(mu_hat, theta_hat)
q_mu       = -2*log(lambda(mu))
```

Use the fitted model to report:

- the signal-strength estimate and confidence interval;
- the fitted signal and background yields;
- the background-only p-value and significance when appropriate;
- an upper limit if there is no significant excess;
- nuisance-parameter pulls, constraints, and impacts;
- goodness-of-fit and post-fit closure diagnostics.

Validate asymptotic results with pseudo-experiments when event counts are low,
parameters are near physical boundaries, or the likelihood is non-Gaussian.

## 7. Recommended order for this analysis

1. Define the CC electron-neutrino signal and all background components.
2. Audit data counts, exposures, event identity, MC weights, and MC statistical
   uncertainties.
3. Choose two ABCD variables that are weakly correlated for the dominant
   data-driven background and define A, B, C, and D before examining the final
   signal result.
4. Test the inclusive arithmetic ABCD estimate as a diagnostic.
5. Demonstrate closure in independent validation data, including differential
   closure checks.
6. Build a one-bin, four-region Poisson likelihood.
7. Add signal leakage, separately modelled backgrounds, and systematic
   constraints.
8. Extend to multiple bins only when every bin has adequate control-region
   statistics and acceptable closure.
9. Validate the fit with Asimov data and pseudo-experiments.
10. Freeze the model, unblind the reserved data, and report the measurement or
    limit.

For the current Region V1 study, a likelihood cannot repair failed ABCD
factorization. A defensible signal result requires a region definition with
independently demonstrated closure.

## Core references

1. **Particle Data Group, "Statistics," Review of Particle Physics (2024).**
   Introductory reference for probability, likelihoods, estimation, confidence
   intervals, hypothesis tests, p-values, and significance.
   <https://pdg.lbl.gov/2024/reviews/rpp2024-rev-statistics.pdf>

2. **K. Cranmer, "Practical Statistics for the LHC."**
   Pedagogical explanation of Poisson models, maximum-likelihood estimation,
   nuisance parameters, and profile likelihoods.
   <https://cds.cern.ch/record/2004587/files/267-308%20Cranmer.pdf>

3. **G. Cowan, K. Cranmer, E. Gross, and O. Vitells, "Asymptotic formulae for
   likelihood-based tests of new physics," Eur. Phys. J. C 71 (2011) 1554.**
   Standard reference for profile-likelihood tests, discovery significance,
   confidence intervals, nuisance parameters, and Asimov datasets.
   <https://arxiv.org/abs/1007.1727>

4. **K. Cranmer et al., "HistFactory: A tool for creating statistical models
   for use with RooFit and RooStats," CERN-OPEN-2012-016.**
   Defines the multi-channel binned Poisson likelihood with signal/background
   samples and systematic constraint terms.
   <https://cds.cern.ch/record/1456844/files/CERN-OPEN-2012-016.pdf>

5. **L. Heinrich et al., "pyhf: pure-Python implementation of HistFactory
   statistical models," JOSS 6 (2021) 2823.**
   Practical implementation reference for constructing and fitting binned
   HistFactory likelihoods in Python.
   <https://doi.org/10.21105/joss.02823>

6. **G. Kasieczka et al., "ABCDisCo: Automating the ABCD Method with Machine
   Learning," Phys. Rev. D 103 (2021) 035021.**
   Discusses ABCD background estimation, independence of the ABCD variables,
   and methods for reducing background correlations.
   <https://arxiv.org/abs/2007.14400>

7. **S. Choi and H. Oh, "Improved Extrapolation Methods of Data-driven
   Background Estimation in High-Energy Physics."**
   Discusses residual correlations and improvements to conventional ABCD
   extrapolation.
   <https://arxiv.org/abs/1906.10831>

## Suggested reading order

```text
PDG Statistics
  -> Practical Statistics for the LHC
  -> HistFactory
  -> Cowan et al.
  -> ABCDisCo and improved ABCD extrapolation
  -> pyhf implementation
```
