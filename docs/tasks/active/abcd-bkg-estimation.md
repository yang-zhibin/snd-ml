# Task: abcd-bkg-estimation

Status: active
Last updated: 2026-10-01 13:28 Europe/Zurich

## Goal

I would like to do is to apply the ABCD method to estimate the background in signal regions

## Work location

- Worktree: `/afs/cern.ch/user/z/zhibin/worktrees/snd-ml-abcd-bkg-estimation`
- Branch: `task/abcd-bkg-estimation`
- Starting commit: `5793ce82cfc4a1f0bd11c551e2032266ac53c504`
- Chat/session ID: `not recorded`

## Discovered scope

- Region definitions are in
  `snakemake/metadata/configs/regions/region_v1.yaml`. All regions require
  `count_scifi > 35` and form the eight combinations of veto/no-veto,
  US/no-US, and separate-plane SciFi fiducial inside/outside.
- Partition policy and per-region real-data fractions are in
  `snakemake/metadata/configs/partitions/partition_v1.yaml`. The four no-veto
  regions use the full selected exposure; the four veto-active regions use
  the first 10% in source order.
- Partition construction and metadata aggregation are implemented in
  `snakemake/rules/region_partitions.smk`. Histogram and plot orchestration is
  in `snakemake/rules/region_evaluation.smk`, using
  `evaluation_region_partitions/make_group_hist.py` and
  `evaluation_region_partitions/plot_comparison.py`.
- Existing outputs under
  `/afs/cern.ch/user/z/zhibin/work/snd-ml/evaluation_region_partitions/plots/partition_v1_region_v1`
  contain 119 feature summaries and plots. The authoritative partition
  catalogue is
  `/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/region_partitions/partition_v1_region_v1.csv`.
- The current evaluation workflow compares regions and particles but does not
  implement an ABCD arithmetic estimator, closure test, simultaneous
  likelihood, discovery significance, or limit calculation.

## Task-specific constraints

- Use raw event counts in Poisson likelihoods. Luminosity-scaled histogram
  integrals are diagnostic rates, not Poisson observations.
- Model the source exposure explicitly. Current no-veto regions have
  `19.780680927948` exposure units, while veto-active regions have
  `1.960319252235`.
- `hist_features_v1.yaml` has `fold_underflow: false` and
  `fold_overflow: false`; plot-range integrals are not inclusive event counts.
  Use metadata `n_events` / evaluation `selected_entries` for inclusive ABCD
  counts and state the underflow/overflow policy for binned estimates.
- Treat the already inspected signal-region count as development data. Do not
  tune a correction on it and then claim an independent closure measurement.
- Account for signal leakage in every region. Use a shared signal-strength
  parameter in the eventual simultaneous likelihood rather than assuming all
  sidebands are signal-free.
- An unconstrained signal-region nonclosure parameter would be degenerate with
  signal. Transfer a correction only after independent validation.
- Snakemake remains the orchestration layer. Use the documented LCG runtime
  for analysis Python, with preflight and `PYTHONNOUSERSITE=1`.

## Completed checkpoints

- Inspected and verified the Region V1 partition, evaluation workflow,
  existing outputs, authoritative metadata, cited ABCD guide, Git state, and
  scheduler state.
- Wrote and approved the baseline analysis plan in
  `docs/analysis/regions/REGION_V1_ABCD_BASELINE_PLAN.md`.
- Recorded the likelihood derivation, current arithmetic calculations, and
  the intended role of `cabinetry` / `pyhf` in
  `docs/analysis/regions/REGION_V1_ABCD_STATISTICAL_LIKELIHOOD_GUIDE.md`.
- Implemented baseline-plan step 1 as the focused Snakemake rule
  `region_v1_abcd_input_audit`, with a frozen input contract, atomic CSV and
  Markdown outputs, and a synthetic pass/fail test. The realized audit passes
  for all eight Region V1 real-data regions.
- Added the short companion note
  `docs/analysis/regions/ABCD_LIKELIHOOD_STEPS_AND_REFERENCES.md`, covering the
  unbinned and binned likelihood sequence, ABCD parameterization, inference,
  and core references.
- Verified the realized catalogue and physical EOS feature files for real
  data and all standard and EventBuilder CC/NC electron- and muon-neutrino MC
  groups in every Region V1 region.
- Froze the initial Region V1 ABCD density histogram definition as
  `density_sndsw_scifi_v1`: 80 uniform bins over `[0, 40000)`, bin width
  500, underflow preserved, overflow folded into the final bin, and unit event
  weights for raw Poisson data counts.
- Implemented and ran the eight-region real-data density histogram workflow.
  All eight ROOT histograms and CSV summaries pass catalogue count and
  exposure closure.
- Audited the standard-neutrino normalization source and implemented the
  eight-region by four-sample MC density workflow. All 32 ROOT histograms and
  CSV summaries pass raw-count, finite-value, source-luminosity, weighted-yield,
  and sum-of-squared-weights closure.
- Implemented and saved the versioned Step 3 density analysis table. It
  materializes raw data, exposure, and all four standard-neutrino MC
  raw/weighted/sumw2/effective-entry quantities without calculating ABCD
  closure or modifying the frozen 500-unit binning.
- Rendered the Step 3 table as a versioned 21-page LaTeX/PDF report with an
  inclusive region summary, inclusive sample-level MC audit, multipage
  per-region density tables, and definitions/formulas for all reported field
  families. First-use definitions now appear immediately below the table in
  which each quantity is introduced, and headings share pages with their
  corresponding tables. The report contains no ABCD prediction or closure
  result.
- Added one unit-area density-shape plot after each of the eight per-region
  tables. Each plot compares Data, target `CC_nue`, summed neutrino background,
  and total standard-neutrino MC, and is followed by the exact normalization
  formula and flow policy. The revised report is 29 pages.
- Froze the sideband-only sparse-tail policy as
  `density_sndsw_scifi_abcd_merged_v1`. It uses six common regular bins with
  edges `[0, 500, 1000, 1500, 2000, 3000, 40000]`; every one of SB1--SB7 has
  at least 50 raw events in each bin. The selection did not use the signal
  region, an ABCD prediction, or a closure result.
- Implemented and ran the arithmetic density-binned closure for the two
  validation constructions only. The calculation uses the frozen six-bin
  policy, exposure-aware ABCD arithmetic, approximate independent-Poisson
  uncertainties, and diagnostic `CC_nue` leakage. It does not calculate a
  nominal signal-region closure, subtract signal, apply a nonclosure
  correction, retune bins, or implement a likelihood.
- At the user's explicit request, implemented and ran a separate, explicitly
  unblinded arithmetic diagnostic for the two original ABCD constructions.
  It reuses the already frozen six density bins and labels the observed signal
  region as development data. No signal subtraction, correction, retuning, or
  likelihood fit was performed.
- Implemented and ran the requested first six-bin cabinetry likelihood
  diagnostic for nominal Baseline 2. The model fits one global unconstrained
  ABCD background scale `alpha_bkg` and the nonnegative `CC_nue` signal
  strength `mu` simultaneously to the already unblinded signal-region data.
  It includes the workspace, fit, pre/post-fit plots, correlation matrix,
  profiled one-dimensional scans, and a two-dimensional likelihood surface.
- Extended that same unchanged fit into a seven-page LaTeX/PDF report. The
  report records the model, variable definitions, six-bin input and result
  tables, runtime/tool setup, a distinct raw-ABCD before-fit plot, cabinetry
  pre-fit and post-fit plots, both profile-likelihood scans, the two-parameter
  likelihood surface, the correlation plot, and the interpretation boundary.
- Corrected the before-fit, pre-fit, and post-fit comparison plots to use
  `density_scifi_sndsw` as the physical x-axis with the frozen unequal edges
  `[0, 500, 1000, 1500, 2000, 3000, 40000]`, rather than six equal-width bin
  indices. The cabinetry predictions, fitted parameters, and likelihood are
  unchanged.
- Implemented and produced two isolated high-density simultaneous A/B/C/D
  diagnostic reports: cut-and-count edges `[5000,40000]` and exploratory
  shape edges `[5000,7000,40000]`. Both consume the validated Step-3 table,
  include standard `CC_nue` leakage in A/B/C/D, fix unvalidated nonclosure
  `kappa=1`, and leave all existing baseline outputs untouched.
- Implemented a separate expected-only reblinded-development workflow. Its
  sanitized input rule reads observed data only for B/C/D and MC for A/B/C/D;
  the A data histogram and summary are absent from the Snakemake DAG and the
  sanitized table has blank A observations. It builds control-only and full
  Asimov workspaces, performs expected discovery inference, produces plots
  and two structured reports, and writes a checksum freeze manifest. No real
  signal-region fit or unblinding target was run.

## Findings and decisions

- Proceed with both baselines defined in the baseline plan. Classify each as
  closed, usable only with an independently validated correction, or rejected.
- Baseline 1 uses the veto x US plane inside the SciFi fiducial volume:
  `A=signal`, `B=SB4`, `C=SB1`, `D=SB5`; validate with
  `A'=SB2`, `B'=SB6`, `C'=SB3`, `D'=SB7`.
- Baseline 2 uses the fiducial transfer between composite veto-US categories:
  `A=signal`, `B=SB2`, `C=SB5`, `D=SB7`; validate with
  `A'=SB1`, `B'=SB3`, `C'=SB4`, `D'=SB6`.
- Existing inclusive counts demonstrate that Baseline 1 fails factorization by
  orders of magnitude. It should be retained only to document rejection.
- Baseline 2 is less pathological but has about 49% validation nonclosure. It
  is not currently suitable for a final signal-region prediction.
- Build the first statistical prototype as a four-channel, one-bin `pyhf`
  workspace because the parameterization is transparent. Use `cabinetry` after
  the model is verified to construct/steer the reproducible binned HistFactory
  workspace, fits, scans, pulls, significance, and limits.
- A likelihood propagates statistical and nuisance-parameter uncertainty but
  cannot repair an invalid ABCD independence assumption.
- The Region V1 feature inputs are ready for data and neutrino MC. Every
  catalogued feature path exists on EOS and is non-empty, and every catalogue
  `feature_available` flag is true. Sparse MC groups can have fewer than ten
  written partitions because empty partitions are intentionally omitted.
- `CC_nue` is the target signal sample. `CC_numu`, `NC_nue`, and `NC_numu`
  are separately modelled neutrino backgrounds; the corresponding
  EventBuilder samples are alternate productions, not additional yields to
  add to the standard samples.
- The catalogue event totals are raw selected MC entries, not 2024 expected
  yields. Signal-contamination tables still require the authoritative event
  weights, with each yield calculated as `sum(w)` and MC statistical variance
  as `sum(w^2)`.
- At the earlier planning checkpoint, no further ABCD inference was
  authorized. The subsequently requested validation and explicitly unblinded
  nominal calculations remain diagnostics, not a final background or signal
  result.
- The Region V2 plotting workflow uses
  `inspection_event_weight = target_luminosity / source_luminosity` with a
  run-9715 target of `0.6102 fb^-1`. That inspection-specific normalization
  is not used for Region V1 ABCD counts. The new Region V1 density histograms
  contain raw unit-weight data counts and store exposure as metadata.
- The initial 500-unit density binning is now versioned and reproducible. It
  remains the materialization binning. The frozen closure-study binning is
  `density_sndsw_scifi_abcd_merged_v1`: `[0,500)`, `[500,1000)`,
  `[1000,1500)`, `[1500,2000)`, `[2000,3000)`, and `[3000,40000)`, with
  underflow separate and overflow already folded into the final bin.
- The merge policy requires at least 50 raw events in every selected sideband
  per regular bin. The conservative four-count design proxy is therefore at
  most `sqrt(4/50) = 0.2829`, below the frozen 0.30 target. This is only a
  bin-design bound; it is not a closure calculation or the uncertainty model
  for the final likelihood. The edges must not be retuned after closure is
  inspected.
- The report's normalized plots are shape-only diagnostics. For each curve
  `c`, regular-bin fractions are `f[c,i] = y[c,i] / sum_j(y[c,j])`, normalized
  independently over the 80 regular bins. Data use raw counts; MC use the
  table's luminosity-normalized yields. Underflow is excluded and overflow is
  already folded into the final bin. Absolute counts and rates remain in the
  tables and are not inferred from these plots.
- The complete updated source metadata is the normalization authority:
  `MC_neutrino_2024_ve_metadata.csv` has 976 rows at 15 luminosity units per
  row (`14640.0` total), and `MC_neutrino_2024_vm_metadata.csv` has 890 rows
  at 15 units per row (`13350.0` total). Do not sum only non-empty Region V1
  partition rows, because sparse samples intentionally omit empty partitions.
- Representative standard-neutrino feature trees have no event-level weight,
  luminosity, flux, or cross-section branch. The implemented weight is
  therefore constant within each region and source flavour:
  `target_data_exposure / source_mc_luminosity`, with unit event physics
  weight. Electron-flavour weights are `0.0013511394076467214` without veto
  and `0.00013390158826741803` with veto; muon-flavour weights are
  `0.0014816989459137079` and `0.00014684039342584268`, respectively.
- Formal validation closure confirms that Baseline 1 is rejected: its direct
  inclusive prediction is about 106 times the observed validation count, and
  every frozen density bin has `kappa` far below one.
- Baseline 2 is not closed as currently defined. Its direct inclusive
  `kappa` is `0.512230`, and its binned values vary strongly with density,
  from `0.495166` in the first bin through values above one in four middle
  bins and `0.743605` in the final bin. These arithmetic diagnostics do not
  authorize a correction or a nominal signal-region prediction.
- The reported uncertainties and pulls use independent-Poisson delta-method
  propagation. They are closure-screening diagnostics, not profile-likelihood
  confidence intervals.
- The original constructions also fail closure against the already inspected
  signal-region data. Baseline 1 predicts about 195 times the observed count;
  Baseline 2 predicts about 2.56 times the observed count. This unblinded
  comparison cannot be used to tune a correction and later presented as an
  independent blind measurement.
- `L_A`, `L_B`, `L_C`, and `L_D` are the effective real-data exposures for
  the four regions, not event counts or fit parameters. The implemented
  prediction applies ABCD to rates:
  `A_pred=(B*C/D)*(L_A*L_D)/(L_B*L_C)`. No-veto regions have exposure
  `19.780680927948`; veto-active regions have `1.960319252235`.
- For every current nominal and validation mapping, the combined exposure
  factor `(L_A*L_D)/(L_B*L_C)` is exactly one because the paired exposure
  ratios cancel. Therefore an additional luminosity normalization cannot fix
  the observed nonclosure. Unit-area normalization remains shape-only and is
  invalid for Poisson yield estimation.
- Decision: do not normalize an ABCD prediction directly to observed
  signal-region data, because that would force closure by construction and
  remove signal sensitivity. A correction may only be transported from
  independent validation after demonstrating stability and secondary
  closure. No correction is currently authorized or implemented.
- For the explicitly requested first fit, use one global `alpha_bkg` shared by
  all six density bins and one shared `mu`; do not introduce freely floating
  per-bin background scales. This v1 model intentionally has no ABCD-template,
  MC-statistical, or systematic nuisance parameters and is diagnostic only.
- The fitted `mu` is effectively at its physical lower boundary and the
  goodness-of-fit p-value underflows to zero. The global scale matches the
  total count, but it does not make the ABCD density shape compatible with the
  data. Therefore the fit cannot be interpreted as a signal measurement,
  discovery significance, or limit.
- For report terminology, `before fit` is explicitly defined as the raw ABCD
  background-only prediction at `alpha_bkg=1, mu=0`; standard `pre-fit` is
  the configured nominal signal-plus-background model at
  `alpha_bkg=1, mu=1`; and `post-fit` uses the simultaneous maximum-likelihood
  estimates. Cabinetry draws all three statistical comparisons. ROOT was used
  in upstream histogram materialization, not to refit or redraw this model.
- The comparison plots must preserve the physical density-bin widths. The
  final `[3000,40000)` tail therefore occupies its actual x range; equal-width
  categorical bins are not used for these three plots.
- The high-density fit must be interpreted as diagnostic. The cut-and-count
  model is saturated with zero goodness-of-fit degrees of freedom and drives
  `u[0]` to its lower boundary. The two-bin model has one degree of freedom
  and goodness-of-fit `p=0.008064`, rejecting the requested factorized shape.
  Neither result supports a discovery claim or limit.
- The technically masked workflow is labelled `REBLINDED_DEVELOPMENT`, not a
  historically blind analysis, because A-region data were inspected in prior
  development. Its default `kappa=1` is explicitly `UNVALIDATED`, its 30%
  other-neutrino normalization constraints are explicitly `PLACEHOLDER`, MC
  statistical uncertainties are not yet included, and observed inference is
  disabled. Expected significance values are therefore software/model
  diagnostics only.

## Results and evidence

- Authoritative inclusive real-data counts and exposures:

  | Region | `n_events` | Exposure | Written partitions |
  | --- | ---: | ---: | ---: |
  | Signal | 158707 | 19.780680927948 | 100 |
  | SB1 | 40300 | 19.780680927948 | 100 |
  | SB2 | 447152 | 19.780680927948 | 100 |
  | SB3 | 73186 | 19.780680927948 | 100 |
  | SB4 | 20348106 | 1.960319252235 | 96 |
  | SB5 | 26510 | 1.960319252235 | 96 |
  | SB6 | 18928315 | 1.960319252235 | 96 |
  | SB7 | 29142 | 1.960319252235 | 96 |

- `selected_entries` in the existing `density_sndsw_scifi` summary exactly
  matches the metadata `n_events` totals above. The density histogram
  `raw_integral` does not match because its configured range excludes
  underflow/overflow.
- Diagnostic inclusive closure calculations:

  | Construction | Prediction | Observation | Kappa |
  | --- | ---: | ---: | ---: |
  | Baseline 1 nominal SR | 30932805 | 158707 | 0.005131 |
  | Baseline 1 outside validation | 47535779 | 447152 | 0.009407 |
  | Baseline 2 nominal SR | 406767 | 158707 | 0.390167 |
  | Baseline 2 mixed validation | 78676 | 40300 | 0.512230 |

- Realized standard neutrino-MC feature counts after Region V1 selection:

  | Region | `CC_nue` | `CC_numu` | `NC_nue` | `NC_numu` |
  | --- | ---: | ---: | ---: | ---: |
  | Signal | 8072 | 20015 | 2475 | 5817 |
  | SB1 | 183 | 12 | 50 | 295 |
  | SB2 | 8886 | 22946 | 3106 | 6833 |
  | SB3 | 148 | 23 | 54 | 277 |
  | SB4 | 12249 | 22862 | 3667 | 6472 |
  | SB5 | 527 | 26 | 118 | 488 |
  | SB6 | 12899 | 22700 | 3642 | 6761 |
  | SB7 | 281 | 39 | 86 | 336 |

- The four EventBuilder neutrino groups also have available, existing,
  non-empty feature files in every region. Their counts were checked
  separately and must not be combined with the standard samples without an
  explicit production choice.
- The versioned audit outputs are:
  - `/afs/cern.ch/work/z/zhibin/snd-ml/evaluation_region_partitions/abcd/partition_v1_region_v1/abcd_input_audit_v1/input_audit.csv`
  - `/afs/cern.ch/work/z/zhibin/snd-ml/evaluation_region_partitions/abcd/partition_v1_region_v1/abcd_input_audit_v1/input_audit.md`
  Both were written on 2026-09-30 at 20:17 Europe/Zurich and report `PASS`.

- The versioned real-data density outputs are under:
  `/afs/cern.ch/work/z/zhibin/snd-ml/evaluation_region_partitions/abcd/partition_v1_region_v1/density_sndsw_scifi_v1/`.
  It contains eight ROOT files under `hists/` and eight one-row CSV files
  under `summaries/`.
- Independent ROOT/CSV verification of the generated histograms:

  | Region | Events | Underflow | Overflow folded into last bin | Status |
  | --- | ---: | ---: | ---: | --- |
  | Signal | 158707 | 0 | 0 | PASS |
  | SB1 | 40300 | 0 | 0 | PASS |
  | SB2 | 447152 | 0 | 0 | PASS |
  | SB3 | 73186 | 0 | 0 | PASS |
  | SB4 | 20348106 | 0 | 104 | PASS |
  | SB5 | 26510 | 0 | 0 | PASS |
  | SB6 | 18928315 | 0 | 5 | PASS |
  | SB7 | 29142 | 0 | 0 | PASS |

- Every generated histogram has 80 bins from 0 to 40000, and its integral
  including preserved flow equals the authoritative catalogue event count.
- The same output directory now contains 32 normalized standard-neutrino MC
  ROOT files and 32 one-row CSV summaries. Each ROOT file contains
  `h_density_sndsw_scifi_raw`, `h_density_sndsw_scifi_weighted`, and
  `h_density_sndsw_scifi_sumw2` plus embedded normalization/closure metadata.
  Independent validation opened every file and found all 32 statuses `PASS`.
- Total luminosity-normalized standard-neutrino MC yields by region are:

  | Region | Total weighted MC |
  | --- | ---: |
  | Signal | 52.5257145033 |
  | SB1 | 0.769697058377 |
  | SB2 | 60.3263766869 |
  | SB3 | 0.717439844119 |
  | SB4 | 6.43859377962 |
  | SB5 | 0.161842486653 |
  | SB6 | 6.54093100225 |
  | SB7 | 0.104207030429 |
- The Step 3 table is saved at
  `/afs/cern.ch/work/z/zhibin/snd-ml/evaluation_region_partitions/abcd/partition_v1_region_v1/density_sndsw_scifi_v1/tables/analysis_table_v1.csv`.
  It has 656 rows and 46 columns: eight regions times one underflow row, 80
  regular-bin rows, and one inclusive row. Every row has source validation
  status `PASS`. No ABCD prediction, kappa, closure, or merged-bin column is
  present.
- The human-readable report products are saved alongside the CSV as
  `analysis_table_v1.tex` and `analysis_table_v1.pdf`. The revised PDF is 29
  A4 pages and 921042 bytes. It displays each MC yield with its MC-statistical
  uncertainty `sqrt(sum(w^2))`. Definitions for the inclusive quantities are
  directly below Table 1, normalization and MC-statistical quantities are
  directly below Table 2, and bin-level quantities are directly below the
  first per-bin table. Section/subsection headings are no longer stranded on
  separate empty pages. Eight PNG shape plots are saved under
  `normalized_distribution_plots_v1/` and embedded after their corresponding
  region tables; each has its normalization explanation directly beneath it.
  The complete data dictionary remains at the end.
- The frozen merge-policy evidence is saved under
  `/afs/cern.ch/work/z/zhibin/snd-ml/evaluation_region_partitions/abcd/partition_v1_region_v1/density_sndsw_scifi_v1/merged_binning_v1/`
  as `sideband_count_audit.csv` and `sideband_count_audit.md`. The CSV has 42
  rows (six bins times seven sidebands), contains no signal/prediction/closure
  fields, and reports `PASS` throughout. Per-bin minimum sideband counts are
  25132, 548, 133, 66, 50, and 52.
- The validation-only closure outputs are saved under
  `/afs/cern.ch/work/z/zhibin/snd-ml/evaluation_region_partitions/abcd/partition_v1_region_v1/density_sndsw_scifi_v1/validation_closure_v1/`
  as `validation_closure.csv`, `.md`, `.tex`, and `.pdf`, with two PNG and two
  vector-PDF plots under `plots/`. The report has seven A4 pages.
- Inclusive validation results are:

  | Construction | Summary | Observation | Prediction | Kappa | Fractional nonclosure |
  | --- | --- | ---: | ---: | ---: | ---: |
  | Baseline 1 outside-fiducial | Direct inclusive | 447152 | 47535778.7 | 0.00940664 | -99.059% |
  | Baseline 1 outside-fiducial | Sum of binned predictions | 447152 | 49933273.2 | 0.00895499 | -99.105% |
  | Baseline 2 mixed-category | Direct inclusive | 40300 | 78675.6 | 0.512230 | -48.777% |
  | Baseline 2 mixed-category | Sum of binned predictions | 40300 | 77186.0 | 0.522115 | -47.788% |

- Baseline 2 per-bin `kappa` values over the frozen bins are `0.495166`,
  `1.023045`, `1.254592`, `1.267338`, `1.407881`, and `0.743605`. Baseline 1
  per-bin values range from `0.0120041` down to `0.000463635`.
- Inclusive target-signal leakage is negligible relative to the raw data in
  all validation roles: the largest fraction is `0.002685%` in Baseline 1
  region A (SB2). It is reported only and is not subtracted.
- The nominal diagnostic outputs are saved under
  `/afs/cern.ch/work/z/zhibin/snd-ml/evaluation_region_partitions/abcd/partition_v1_region_v1/density_sndsw_scifi_v1/nominal_closure_v1/`
  as `nominal_closure.csv`, `.md`, `.tex`, and `.pdf`, with two PNG and two
  vector-PDF plots under `plots/`. The report has seven A4 pages.
- Original-construction results are:

  | Construction | Summary | Signal-region observation | Prediction | Kappa | Fractional nonclosure |
  | --- | --- | ---: | ---: | ---: | ---: |
  | Baseline 1 nominal | Direct inclusive | 158707 | 30932805.4 | 0.00513070 | -99.487% |
  | Baseline 1 nominal | Sum of binned predictions | 158707 | 34417386.6 | 0.00461124 | -99.539% |
  | Baseline 2 nominal | Direct inclusive | 158707 | 406766.8 | 0.390167 | -60.983% |
  | Baseline 2 nominal | Sum of binned predictions | 158707 | 411433.9 | 0.385741 | -61.426% |

- Baseline 1 nominal per-bin `kappa` values are `0.0064575`, `0.00199582`,
  `0.00102331`, `0.000718780`, `0.000363649`, and `0.000400867`. Baseline 2
  values are `0.373561`, `0.535625`, `0.532265`, `0.614749`, `0.494630`, and
  `0.368421`.
- The inclusive normalized `CC_nue` MC yield in the signal region is
  `10.9064`, or `0.006872%` of the 158707 observed events. This diagnostic
  yield is reported in the output and was not subtracted.
- Applying the Baseline 2 direct-inclusive validation factor
  `kappa_VR=0.512230` only as a diagnostic would change the nominal prediction
  from `406766.8` to approximately `208400`, still 31% above the observed
  `158707`. Equivalently, the corrected closure factor would be about `0.762`.
  This confirms that the validation normalization does not transfer reliably.
- The cabinetry fit outputs are saved under
  `/afs/cern.ch/work/z/zhibin/snd-ml/evaluation_region_partitions/abcd/partition_v1_region_v1/density_sndsw_scifi_v1/signal_region_fit_v1/`.
  They include `workspace.json`, `model_inputs.csv`, `fit_results.json`, the
  two one-dimensional scan CSVs, `likelihood_2d.csv`, `fit_report.md`, and 12
  PNG/PDF plot files under `plots/`.
- The realized fit result is `alpha_bkg = 0.385739245 +/- 0.000968279` and
  `mu = 3.13448e-05 +/- 2.06155` from the local Hesse estimate. `mu` is
  classified as effectively at the configured lower boundary, so the Hesse
  uncertainty is not a symmetric physical interval; use the saved profile
  likelihood scan for interpretation. The fitted background total is
  `158706.211`, the fitted signal total is `0.000342`,
  `corr(alpha_bkg,mu) = -0.000655`, twice negative log-likelihood is
  `1856.42506`, and cabinetry reports goodness-of-fit p-value `0.0`.
- The saved likelihood products contain 81 `mu` scan points, 89 resolved
  `alpha_bkg` scan points, and an 81 by 81 two-dimensional grid (6561 rows).
  The signal scan minimum is at `mu=0` within optimizer tolerance.

- The cited ABCD guide requires at least a nominal ABCD plane plus an
  independent validation construction. If a nonclosure correction or
  uncertainty is derived, it requires secondary validation.
- Representative output timestamps: the density summary and plot were written
  2026-06-16; the authoritative combined partition metadata was written
  2026-06-17. The inclusive summary counts still match the metadata exactly.
- Commands and outcomes:
  - Task-only checkpoint commit:
    `924bf9e3b816f66488e99be2e2b2e2e0d705d624`
    (`docs: checkpoint ABCD background estimation baseline`). It contains the
    baseline plan, likelihood guide, and active task record. The worktree had
    no unrelated changes and was clean immediately after this commit.
  - `git status --short --branch`, `git rev-parse HEAD`, and `git log -5`:
    branch was at starting commit `5793ce8`; only task documentation was
    untracked.
  - Targeted `rg`, `find`, and `sed` inspection of Region V1 configs, rules,
    workflow documentation, and existing plot summaries: identified the
    components and 119 available feature summaries.
  - `condor_q -nobatch`: zero jobs for user `zhibin`.
  - Streamed the cited PDF with
    `curl -L ...ABCDGuide_draft18Oct18.pdf | pdftotext -layout - -`: verified
    the arithmetic formula, likelihood construction, validation, nonclosure,
    and secondary-validation guidance.
  - Activated `snakemake/envs/env_lcg.sh` in one shell with hostname, working
    directory, interpreter, Python version, and import preflights. Read-only
    Python CSV aggregation reproduced the metadata counts/exposures and the
    four diagnostic calculations above.
  - Added the audit rule and first dry-ran its output target. Treating the
    authoritative catalogue as a Snakemake input initially exposed an unsafe
    dependency expansion of 802 jobs; no such jobs were executed. Moving the
    catalogue path to a rule parameter reduced the DAG to exactly one local
    audit job and prevents this audit target from regenerating partitions.
  - Ran the focused synthetic audit test under the documented LCG environment:
    `python3 evaluation_region_partitions/test_abcd_region_input_audit.py`.
    The passing fixture succeeded, the intentional count-mismatch fixture
    returned nonzero as required, and the test ended with
    `Region V1 ABCD input audit synthetic test: OK`.
  - Ran the single local Snakemake target through `snakemake/snakefile` with
    one core. Log:
    `.snakemake/log/2026-09-30T201714.122490.snakemake.log`. It scheduled only
    `region_v1_abcd_input_audit`, completed `1 of 1 steps`, and wrote the two
    versioned audit outputs. A subsequent dry-run reported nothing to do.
  - `python3 -m py_compile` passed for the audit script and focused test;
    `git diff --check` passed.
  - On 2026-10-01, activated the same LCG 107 environment with preflight and
    used a read-only Python `csv`/`os.path` audit of
    `partition_v1_region_v1.csv`. For real data and the eight standard plus
    EventBuilder neutrino groups, all catalogued feature paths existed and
    had nonzero size in all eight regions. No output file was written.
  - `condor_q -nobatch` on 2026-10-01 at about 01:07 Europe/Zurich reported
    zero jobs for `zhibin`.
  - Inspected the Region V2 veto-inspection config and histogram/plot scripts.
    Confirmed the 80-bin `[0, 40000)` density definition, overflow folding,
    raw versus `inspection_event_weight` histograms, and unit-area plotting
    normalization. The Region V2 constant luminosity weight was deliberately
    not reused.
  - Under LCG 107 (Python 3.11.9, ROOT 6.34.02, PyYAML 6.0.1), ran
    `python3 -m py_compile` for the new builder/test and
    `python3 evaluation_region_partitions/test_abcd_density_hist.py`. The
    synthetic test passed the seven-event flow/count fixture and confirmed an
    intentional catalogue mismatch fails.
  - Dry-run command:
    `snakemake --snakefile snakemake/snakefile region_v1_abcd_density_hists --cores 8 --dry-run --printshellcmds`.
    It scheduled exactly eight `build_region_v1_abcd_density_hist` jobs plus
    the aggregate target and no partition-production jobs.
  - Production command:
    `snakemake --snakefile snakemake/snakefile region_v1_abcd_density_hists --cores 4 --printshellcmds --rerun-incomplete`.
    It completed 9 of 9 local steps. Log:
    `.snakemake/log/2026-10-01T024951.126989.snakemake.log`.
  - Independently opened all eight ROOT files under LCG, verified their
    binning, histogram objects, integrals, CSV status, count closure, and
    exposure closure. A final dry-run reported all requested files present
    and up to date.
  - Under LCG 107 (Python 3.11.9, ROOT 6.34.02, PyYAML 6.0.1), compiled the
    normalized-MC builder/test and ran
    `python3 evaluation_region_partitions/test_abcd_density_mc_hist.py`.
    The seven-event fixture passed with weight `0.2`, `sum(w)=1.4`, and
    `sum(w^2)=0.28`; an intentional source-luminosity mismatch failed as
    required.
  - The dedicated Snakemake executable at
    `/afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake`
    dry-ran `region_v1_abcd_density_mc_hists` as exactly 32 builder jobs plus
    one aggregate job and no upstream partition jobs. The default user-site
    launcher was incomplete (`humanfriendly` missing), and launching the
    dedicated executable from an already activated LCG shell caused a Python
    runtime mismatch; neither failed attempt built a DAG or wrote outputs.
  - Production command:
    `/afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake --snakefile snakemake/snakefile region_v1_abcd_density_mc_hists --cores 4 --printshellcmds --rerun-incomplete`.
    It completed 33 of 33 local steps. Log:
    `.snakemake/log/2026-10-01T031735.883886.snakemake.log`.
  - Independently opened all 32 MC ROOT files and checked all three histogram
    objects and embedded closure flags; checked all 32 CSV summaries and
    aggregated the yields above. A clean-shell final dry-run reported all
    requested files present and up to date.
  - Under LCG 107, compiled and ran
    `python3 evaluation_region_partitions/test_abcd_density_analysis_table.py`.
    The synthetic fixture reconstructed data and all four MC samples exactly;
    an intentionally corrupted weighted bin failed before output as required.
  - Dry-ran `region_v1_abcd_density_analysis_table`; the DAG contained exactly
    one local table job and no histogram or partition regeneration.
  - Production command:
    `/afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake --snakefile snakemake/snakefile region_v1_abcd_density_analysis_table --cores 1 --printshellcmds --rerun-incomplete`.
    The finalized run completed 1 of 1 steps and wrote 656 validated rows.
    Log: `.snakemake/log/2026-10-01T033751.195380.snakemake.log`.
  - An independent CSV audit verified 656 rows, 46 columns, 82 rows per
    region, all `PASS` statuses, explicit electron/muon source luminosities,
    and zero closure-result columns. A final clean-shell dry-run was a no-op.
  - Generated a temporary LaTeX/PDF preview, compiled it twice with system
    `pdflatex`, and inspected representative extracted pages. After improving
    density-edge formatting and definition-section line breaking, the final
    tables use readable labels such as `28,500--29,000`.
  - Dry-ran and ran the one-job
    `region_v1_abcd_density_analysis_report` target. The production log is
    `.snakemake/log/2026-10-01T034824.820385.snakemake.log`. `pdfinfo` reports
    31 A4 pages; `pdftotext` checks confirmed the inclusive table, per-bin
    table, and definitions for exposure, raw MC, source luminosity, `sum(w)`,
    `sum(w^2)`, effective entries, and validation status. The final dry-run
    was a no-op.
  - Revised the report layout so landscape sections begin before their
    headings and added first-use definition blocks directly beneath Tables 1,
    2, and 3. Regenerated the one-job report target; production log:
    `.snakemake/log/2026-10-01T035854.766679.snakemake.log`. The first launch
    attempt stopped before DAG construction because an LCG-activated shell
    was incorrectly mixed with the Snakemake Conda interpreter
    (`ModuleNotFoundError: encodings`); it wrote no output. The successful
    clean-shell run kept Snakemake and the rule's LCG environment separate,
    completed 1 of 1 steps, and produced a 21-page, 394392-byte A4 PDF.
    `pdftotext -layout` confirmed the table/definition ordering and table-page
    headings; the final dry-run reported all outputs up to date.
  - Added normalized-plot generation to the report script and declared the
    plot directory as a Snakemake output. An isolated preview initially caught
    an over-escaped Matplotlib math label; after correction, Python compilation,
    eight PNG renders, and two `pdflatex` passes succeeded. The focused dry-run
    scheduled exactly one report job. Production completed 1 of 1 steps; log:
    `.snakemake/log/2026-10-01T040707.517033.snakemake.log`. `pdfinfo` reports
    29 A4 pages and 921042 bytes; eight PNG files were present, and
    `pdftotext -layout` found eight plot headings and eight normalization
    explanations. The final dry-run was a no-op.
  - Inspected only SB1--SB7 raw regular-bin counts from the Step 3 table and
    compared deterministic minimum-count thresholds of 10, 25, and 50. Froze
    50 because `sqrt(4/50)=0.2829` satisfies the predeclared 0.30 design target.
  - Under LCG 107 (Python 3.11.9, PyYAML 6.0.1), compiled and preview-ran
    `audit_abcd_density_merge_policy.py`; it reproduced all six frozen bins
    and wrote a 42-row audit. The focused Snakemake dry-run scheduled exactly
    one local job and no upstream production. Production completed 1 of 1
    steps; log: `.snakemake/log/2026-10-01T042128.439013.snakemake.log`.
  - An independent output check verified exactly seven sideband labels, six
    bins, minimum count 50, all statuses `PASS`, and no closure, prediction,
    kappa, or signal fields. The final dry-run was a no-op.
  - Previewed the validation closure report under LCG 107. The first LaTeX
    pass exposed an interval cell immediately following `\\midrule` being
    parsed as an optional rule argument; bracing the interval text fixed it.
    The corrected preview passed two LaTeX runs with no errors or overfull
    warnings.
  - Dry-ran `region_v1_abcd_density_validation_closure`; it scheduled exactly
    one local closure job and no upstream regeneration. Production command:
    `env -u PYTHONHOME -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake --snakefile snakemake/snakefile region_v1_abcd_density_validation_closure --cores 1 --printshellcmds --rerun-incomplete`.
    It completed 1 of 1 steps. Log:
    `.snakemake/log/2026-10-01T043902.939333.snakemake.log`.
  - Independently audited all 16 CSV rows under LCG 107: exactly two expected
    baselines, six bins plus direct-inclusive and binned-sum rows each, only
    SB1--SB7 mappings, all statuses `PASS`, and the frozen edges unchanged.
    Recomputed transfer factors, predictions, variances, uncertainties,
    `kappa`, fractional nonclosure, and pulls with a maximum relative
    difference of `2.31e-16`. The final Snakemake dry-run was a no-op.
  - Generalized the closure calculator with two strictly checked modes. An
    isolated LCG preview compiled both seven-page reports, and byte comparison
    of the validation CSV against the previous production CSV showed that its
    16 numeric rows were unchanged.
  - The focused nominal dry-run scheduled exactly one job. Production command:
    `env -u PYTHONHOME -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake --snakefile snakemake/snakefile region_v1_abcd_density_closures --cores 2 --printshellcmds --rerun-incomplete`.
    It refreshed the validation report, generated the nominal report, and
    completed 3 of 3 local steps. Log:
    `.snakemake/log/2026-10-01T045538.739193.snakemake.log`.
  - Independently audited all 16 nominal CSV rows: both exact nominal
    mappings, A equal to the signal region throughout, six frozen bins plus
    direct-inclusive and binned-sum rows, and all statuses `PASS`. Recomputed
    every prediction, variance, uncertainty, transfer factor, `kappa`,
    nonclosure, and pull with maximum relative difference `2.16e-16`.
    `pdfinfo` reports a seven-page A4 PDF, extracted text contains the
    unblinding warning, four expected plots are present, and the final
    aggregate dry-run was a no-op.
  - Checkpoint verification on 2026-10-01 at about 05:16 Europe/Zurich:
    `git status --short --branch`, `git rev-parse HEAD`, and `git log -3`
    confirmed clean commit `edc1e5210313a5da97a8e95906ea38b0ad935edd` before
    this documentation update. `pdfinfo` reconfirmed the nominal report as a
    seven-page A4 PDF; both closure CSVs have one header plus 16 data rows.
    A clean-shell dry-run of `region_v1_abcd_density_closures` reported
    nothing to do.
  - Under LCG 107, preflight recorded Python 3.11.9, cabinetry 0.5.2, pyhf
    0.7.6, Matplotlib 3.8.3, NumPy 1.26.4, and PyYAML 6.0.1. An isolated
    preview compiled the new script and completed the workspace, fit, both
    one-dimensional scans, 6561-point two-dimensional scan, Markdown report,
    and 12 plots.
  - The focused Snakemake dry-run scheduled exactly one
    `region_v1_abcd_density_signal_region_fit` job and no upstream work.
    Production command:
    `env -u PYTHONHOME -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake --snakefile snakemake/snakefile region_v1_abcd_density_signal_region_fit --cores 1 --printshellcmds --rerun-incomplete`.
    It completed 1 of 1 local steps. Log:
    `.snakemake/log/2026-10-01T110247.237306.snakemake.log`.
  - Independent output validation reconstructed the three template sums,
    loaded and schema-validated the saved pyhf workspace, checked its one
    channel, two samples, two normfactors, six bins, and POI, reproduced the
    saved likelihood at the best-fit point, and compared `alpha_bkg` with the
    analytic zero-signal normalization. All scan grids were finite,
    nonnegative, and had their configured sizes; all 12 plots were nonempty.
    A bare SciPy-backed `pyhf.infer.mle.fit` cross-check remained at its
    initial point for this large-count model, while cabinetry's supported
    Minuit-backed path and direct likelihood checks agreed. The final focused
    Snakemake dry-run was a no-op and `git diff --check` passed.
  - Added `fit_report.tex` and `fit_report.pdf` to the same versioned output
    directory and added `before_fit_data_model.{pdf,png}` to the plot
    directory. The final directory contains nine non-plot artifacts and 14
    nonempty plot files: seven vector PDFs and seven PNGs.
  - The reporting production run used the same clean-environment Snakemake
    command and completed 1 of 1 local steps. Final successful log:
    `.snakemake/log/2026-10-01T112302.645524.snakemake.log`. Preflight recorded
    LCG 107 Python 3.11.9, cabinetry 0.5.2, pyhf 0.7.6, Matplotlib 3.8.3,
    NumPy 1.26.4, PyYAML 6.0.1, and `/usr/bin/pdflatex`.
  - Independent report validation found seven A4 pages, readable extracted
    text for every required section, all requested tables and plot headings,
    exact reproduction of `alpha_bkg=0.3857392452578279`,
    `mu=3.134482636180059e-05`, and goodness-of-fit `p=0`. The scan CSVs
    contain 81, 89, and 6561 rows as configured. The final target dry-run was
    a no-op and `git diff --check` passed.
  - The physical-axis correction completed 1 of 1 local steps; final log:
    `.snakemake/log/2026-10-01T113535.219832.snakemake.log`. Extracted text
    from all three comparison PDFs contains `density_scifi_sndsw` and the
    40000 endpoint, the report states that true unequal widths are used, and
    the final Snakemake dry-run was a no-op.
  - The new `region_v1_abcd_density_high_density_reports` dry-run scheduled
    exactly one local job and no upstream regeneration. Production completed
    1 of 1 steps; final log:
    `.snakemake/log/2026-10-01T120011.237368.snakemake.log`.
  - Cut-and-count inputs are `A=59`, `B=80`, `C=36`, `D=18`, arithmetic
    `A_pred=160.0 +/- 49.5`, and standard `CC_nue` A yield `9.70118`. The
    simultaneous fit returns `mu=7.12809 +/- 0.60454`, has zero goodness-of-fit
    degrees of freedom, and places `u[0]` at zero; its covariance warnings and
    saturated status are reported explicitly.
- The exploratory two-bin fit returns `mu=2.53404 +/- 1.10213` with one
    goodness-of-fit degree of freedom and `p=0.00806437`. The report states
    that the factorized model is incompatible with the observations.
  - Independent validation schema-checked both saved pyhf workspaces,
    reproduced the exact saved likelihood values (`25.2797262` and
    `45.9862770`), verified 81 finite nonnegative scan points per analysis,
    ten nonempty plots per output tree, readable five- and six-page PDFs, and
    all expected warning/parameter/result text. The new and frozen baseline
    targets jointly dry-run as a no-op.
- An isolated real-input test under `/tmp/abcd_blinded_real_test` created 320
  sanitized rows with zero A observations, valid control/background/signal
  pyhf workspaces, eight plots per analysis, and two four-page PDFs. With the
  explicitly provisional model, the Asimov injected-`mu=1` asymptotic
  sensitivities were `Z=0.216406` for `[5000,40000]` and `Z=0.510738` for
  `[5000,7000,40000]`. These are not production or observed results.
- Task-only implementation commit: `11c714a`
  (`feat: add blinded ABCD expected workflow`).

## Modified files

- `snakemake/metadata/configs/region_evaluation/abcd_density_blinded_inference_v1.yaml`
  - Added the reblinded status, A/B/C/D mapping, two high-density binnings,
    provisional nonclosure/systematic declarations, fit settings, blinding
    guards, and isolated expected-report outputs.
- `evaluation_region_partitions/make_abcd_blinded_inputs.py`
  - Added a ROOT histogram reader that deliberately never resolves an A-data
    path, validates B/C/D data plus A/B/C/D standard-neutrino MC, and writes a
    sanitized 320-row table with blank A observations.
- `evaluation_region_partitions/run_abcd_blinded_expected.py`
  - Added B/C/D control fitting with `mu=0`, residual-ABCD plus explicit
    other-neutrino components, background-only and signal-injected Asimov
    workspaces, asymptotic expected `p0`/`Z`, scans, plots, structured
    Markdown/LaTeX reports, and a checksum freeze manifest.
- `evaluation_region_partitions/test_abcd_blinded_expected.py`
  - Added synthetic aggregation/workspace/inference/report checks and an
    intentional failure when any A observation appears.
- `snakemake/rules/region_partitions.smk`
  - Added `region_v1_abcd_density_blinded_inputs` and
    `region_v1_abcd_density_blinded_expected_report`. The expected DAG names
    only the three B/C/D real-data products and all four regions' MC products.

- `docs/analysis/regions/REGION_V1_ABCD_BASELINE_PLAN.md`
  - Added the approved two-baseline plan, validation mappings, signal leakage,
    density binning, required outputs, and interpretation rules; recorded the
    frozen sideband-only merged edges and no-retuning rule.
- `docs/analysis/regions/REGION_V1_ABCD_STATISTICAL_LIKELIHOOD_GUIDE.md`
  - Added the from-scratch likelihood derivation, current arithmetic results,
    profile-likelihood outline, and `cabinetry` / `pyhf` workflow boundary.
- `docs/tasks/active/abcd-bkg-estimation.md`
  - Added the continuing checkpoints, including the final normalization
    interpretation and decision recorded here.
- `snakemake/metadata/configs/region_evaluation/abcd_density_signal_region_fit_v1.yaml`
  - Added the frozen source mapping, six-bin model, global background and
    signal parameters, fit/scan settings, scope boundary, and output paths;
    now also declares the physical observable name and LaTeX/PDF outputs.
- `evaluation_region_partitions/fit_abcd_density_signal_region.py`
  - Added strict source validation, workspace construction, cabinetry fit,
    pre/post-fit predictions, goodness-of-fit, correlation and likelihood
    scans, deterministic plots, and atomic machine-readable/report outputs.
    It now also creates the raw-ABCD before-fit prediction and the complete
    LaTeX report, passing the physical observable and unequal bin edges to
    cabinetry visualization.
- `snakemake/rules/region_partitions.smk`
  - Added the focused one-job
    `region_v1_abcd_density_signal_region_fit` target using the documented LCG
    environment and the existing nominal-closure CSV as its sole data input;
    the rule now preflights `pdflatex` and compiles the report twice in an
    isolated temporary directory.
- `snakemake/metadata/configs/region_evaluation/abcd_density_high_density_fits_v1.yaml`
  - Added the two frozen high-density edge sets, standard-signal choice,
    simultaneous-model contract, diagnostic boundary, and isolated outputs.
- `evaluation_region_partitions/fit_abcd_high_density_reports.py`
  - Added strict Step-3 aggregation, four-channel HistFactory workspace
    construction, signal leakage, simultaneous fits, scans, physical-axis
    plots, machine-readable outputs, and two complete LaTeX reports.
- `evaluation_region_partitions/test_abcd_high_density_reports.py`
  - Added focused aggregation, ABCD arithmetic, workspace-dimension/schema,
    POI/sample/channel, and zero-control-count failure checks.
- `snakemake/rules/region_partitions.smk`
  - Added the one-job `region_v1_abcd_density_high_density_reports` target and
    isolated two-pass compilation of both PDFs.
- `snakemake/rules/region_partitions.smk`
  - Added the single-purpose `region_v1_abcd_input_audit` rule. The catalogue
    is deliberately a parameter rather than an input so an audit cannot
    trigger partition regeneration.
- `snakemake/metadata/configs/region_evaluation/abcd_input_audit_v1.yaml`
  - Added frozen Region V1 real-data counts, exposures, fractions, and
    partition expectations plus versioned output paths.
- `evaluation_region_partitions/audit_abcd_region_inputs.py`
  - Added catalogue/schema/region/count/exposure/partition/availability checks
    and atomic CSV/Markdown reporting.
- `evaluation_region_partitions/test_abcd_region_input_audit.py`
  - Added an end-to-end passing fixture and an intentional count-failure case.
- `docs/analysis/regions/ABCD_LIKELIHOOD_STEPS_AND_REFERENCES.md`
  - Added a compact likelihood/ABCD workflow and primary reading list.
- `snakemake/metadata/configs/region_evaluation/abcd_density_sndsw_scifi_v1.yaml`
  - Added the frozen initial density binning, raw-count normalization, flow
    policies, region expectations, output location, and validation contract.
- `evaluation_region_partitions/make_abcd_density_hist.py`
  - Added the catalogue-driven, one-region raw ROOT histogram builder with
    atomic ROOT/CSV outputs and embedded audit metadata.
- `evaluation_region_partitions/test_abcd_density_hist.py`
  - Added synthetic tests for binning, underflow preservation, overflow
    folding, count closure, embedded metadata, and intentional failure.
- `snakemake/rules/region_partitions.smk`
  - Added the eight-job `region_v1_abcd_density_hists` aggregate and focused
    per-region builder rule without a dependency that can regenerate source
    partitions.
- `snakemake/metadata/configs/region_evaluation/abcd_density_neutrino_mc_v1.yaml`
  - Added standard-neutrino samples, source luminosities, region exposures,
    normalization formula, shared binning, outputs, and validation contract.
- `evaluation_region_partitions/make_abcd_density_mc_hist.py`
  - Added complete-source luminosity validation, raw/weighted/sumw2
    histograms, atomic outputs, provenance, and closure metadata.
- `evaluation_region_partitions/test_abcd_density_mc_hist.py`
  - Added a weighted-yield/flow/sumw2 test and intentional luminosity failure.
- `snakemake/rules/region_partitions.smk`
  - Added the 32-job MC aggregate and focused per-region/per-sample rule while
    keeping the catalogue as a parameter to prevent upstream regeneration.
- `snakemake/metadata/configs/region_evaluation/abcd_density_analysis_table_v1.yaml`
  - Added the versioned Step 3 table schema, sample roles/prefixes, output,
    flow policy, and reconstruction tolerances.
- `evaluation_region_partitions/make_abcd_density_analysis_table.py`
  - Added source ROOT/summary validation, table materialization, exact
    reconstruction checks, derived MC totals/fractions, and atomic CSV output.
- `evaluation_region_partitions/test_abcd_density_analysis_table.py`
  - Added a four-sample synthetic pass case and corrupted-weight failure case.
- `snakemake/rules/region_partitions.smk`
  - Added the single-job `region_v1_abcd_density_analysis_table` target with
    all existing data/MC histogram products as explicit inputs.
- `snakemake/metadata/configs/region_evaluation/abcd_density_analysis_report_v1.yaml`
  - Added the versioned report title, source-table contract, scope statement,
    LaTeX/PDF output names, and the unit-area plot/flow/output contract.
- `evaluation_region_partitions/make_abcd_density_analysis_report.py`
  - Added validated CSV loading and LaTeX rendering for inclusive, MC-audit,
    per-bin, and complete definition sections; later moved headings inside
    their landscape table pages and placed concise definitions immediately
    below the first table that introduces each quantity. It now also validates
    and generates eight atomic unit-area PNG plots and embeds each plot plus
    its normalization explanation after the corresponding table.
- `snakemake/rules/region_partitions.smk`
  - Added the one-job `region_v1_abcd_density_analysis_report` target with
    two-pass isolated `pdflatex` compilation; the rule now declares the plot
    directory and preflights Matplotlib.
- `snakemake/metadata/configs/region_evaluation/abcd_density_merged_binning_v1.yaml`
  - Added the selected/excluded regions, deterministic minimum-50 merge
    algorithm, six frozen regular bins, flow policy, freeze declarations, and
    versioned audit outputs.
- `evaluation_region_partitions/audit_abcd_density_merge_policy.py`
  - Added sideband-only source validation, deterministic edge reconstruction,
    minimum-count and conservation checks, and atomic CSV/Markdown evidence.
- `snakemake/rules/region_partitions.smk`
  - Added the focused `region_v1_abcd_density_merge_policy` rule, keeping the
    existing Step 3 table as a parameter so this audit cannot regenerate
    upstream products.
- `snakemake/metadata/configs/region_evaluation/abcd_density_validation_closure_v1.yaml`
  - Added the validation-only mappings, frozen input/output contracts,
    arithmetic definitions, and prohibitions on signal-region use,
    subtraction, correction, bin retuning, and likelihood inference.
- `evaluation_region_partitions/calculate_abcd_density_validation_closure.py`
  - Added strict input and scope validation, exposure-aware binned and
    inclusive arithmetic, approximate statistical propagation, signal-leakage
    diagnostics, atomic CSV/Markdown/LaTeX outputs, and two closure plots in
    raster and vector formats.
- `snakemake/rules/region_partitions.smk`
  - Added the single-job
    `region_v1_abcd_density_validation_closure` target. The frozen merge audit
    is an explicit input, while the already materialized Step 3 table remains
    a parameter so this action cannot trigger upstream regeneration.
- `snakemake/metadata/configs/region_evaluation/abcd_density_validation_closure_v1.yaml`
  - Added an explicit `unblinded_signal_region: false` assertion.
- `snakemake/metadata/configs/region_evaluation/abcd_density_nominal_closure_v1.yaml`
  - Added the two original ABCD mappings, explicit unblinding declaration,
    development-data boundary, frozen inputs, arithmetic contract, and
    versioned nominal output paths.
- `evaluation_region_partitions/calculate_abcd_density_validation_closure.py`
  - Generalized strict scope validation and report wording to support either
    the original validation-only mode or the explicit nominal-unblinded mode.
    Signal-region bins are reconstructed directly from the Step 3 table while
    the frozen merge audit remains authoritative for every sideband.
- `snakemake/rules/region_partitions.smk`
  - Added the focused `region_v1_abcd_density_nominal_closure` rule and the
    `region_v1_abcd_density_closures` aggregate target. Neither can regenerate
    the Step 3 table or upstream histogram/partition products.

## Validation

- At the earlier documentation-only checkpoint, no code tests were required;
  the implementation added since then has the focused validation listed
  below.
- Verified the saved likelihood guide by reading its opening and final
  sections.
- Verified the inclusive counts independently from the authoritative partition
  metadata and the existing evaluation summary.
- The frozen Region V1 input audit reports `PASS` for all eight data regions;
  every count difference is zero and every exposure difference is within the
  configured absolute tolerance of `1e-9`.
- The focused synthetic test, Python compilation, one-job Snakemake execution,
  final no-op dry-run, and `git diff --check` all passed.
- A separate physical-file check confirmed all catalogued real-data and
  neutrino-MC feature ROOT files exist on EOS and have nonzero size.
- The density builder synthetic test and Python compilation passed.
- The workflow dry-run contained exactly eight histogram jobs; the production
  target completed, independent ROOT/CSV checks passed for all regions, and
  the final dry-run was a no-op.
- The normalized-MC synthetic test and compilation passed. The MC workflow
  dry-run contained exactly 32 builder jobs, production completed all 33
  steps, independent ROOT/CSV checks passed for all 32 products, and the
  final clean-shell dry-run was a no-op.
- The Step 3 table synthetic test and compilation passed. The one-job dry-run,
  production run, independent schema/content audit, exact ROOT/summary
  reconstruction checks, `git diff --check`, and final no-op dry-run passed.
- The report generator compiled, the LaTeX source passed two `pdflatex`
  passes, representative PDF text was readable, requested definitions were
  present immediately below their first-use tables, PDF metadata was valid,
  and empty heading pages remained removed. All eight unit-area plots passed
  finite/nonnegative input, positive-total, and sum-to-one checks; eight PNGs
  were embedded, all eight normalization explanations were present, and the
  final report dry-run was a no-op.
- The merge-policy auditor compiled and passed an isolated preview, the
  production one-job target, an independent schema/content audit, count
  conservation and minimum-count checks, `git diff --check`, and a final
  no-op dry-run. No closure quantity was produced.
- The closure calculator compiled and its isolated report preview passed two
  LaTeX runs. The focused dry-run scheduled one job, production completed one
  job, and the final dry-run was a no-op. Independent formula recomputation
  matched every numeric result to relative precision better than `2.31e-16`;
  the PDF, CSV schema/scope, six frozen bins, output plots, and all `PASS`
  statuses were verified. `git diff --check` passed.
- Both calculator modes compiled and passed isolated execution and two LaTeX
  passes. The prior validation CSV remained byte-identical. Production
  completed all three aggregate steps; independent nominal arithmetic agreed
  to `2.16e-16` relative precision; PDF/text/plot checks passed; the final
  aggregate dry-run was a no-op; and `git diff --check` passed.
- The fit script compiled, the isolated preview and production target passed,
  and independent workspace/input/likelihood/scan/plot validation passed. The
  final dry-run was a no-op. The zero goodness-of-fit value is a scientific
  model failure, not a workflow or optimizer failure.
- The report extension compiled, regenerated the fit reproducibly, completed
  two `pdflatex` passes, and produced a readable seven-page A4 PDF. Independent
  checks verified all nine declared non-plot artifacts, 14 nonempty plot
  files, exact fit values, configured scan-grid sizes, required report
  sections, `git diff --check`, and a final no-op dry-run.
- All three corrected comparison PDFs expose `density_scifi_sndsw` on the
  x-axis from 0 to 40000 and use the configured variable edges. The regenerated
  seven-page report passed two LaTeX runs, text checks, `git diff --check`, and
  a final no-op dry-run; numerical fit results were unchanged.
- The high-density scripts compiled and the focused synthetic test passed.
  Production generated and compiled both reports. Independent artifact,
  schema, exact-likelihood, scan, plot-count, PDF metadata/text, baseline
  isolation, `git diff --check`, and final no-op DAG checks passed. The
  optimizer covariance warnings for the saturated cut-count model are a
  documented scientific limitation rather than a missing output.
- The blinded builder/runner/test compiled under LCG 107. The focused
  synthetic test passed expected inference and rejected an injected A
  observation. A Snakemake summary scheduled only the sanitized-input and
  expected-report rules and showed no signal-region real-data dependency.
- The isolated real-input builder and expected runner passed. Independent
  checks verified 320 sanitized rows, blank A observations, valid workspace
  schemas, the locked observed-result section, four PDF plus four PNG plots
  per analysis, a manifest with `unblinding_authorized=false`, and two-pass
  compilation of both four-page PDFs. `git diff --check` passed. Minuit
  covariance warnings were observed and are consistent with the saturated,
  weakly constrained diagnostic model.

## Running jobs

- This task has no scheduler jobs. Its histograms were produced by local
  Snakemake jobs.
- At the final check on 2026-10-01 at 13:28 Europe/Zurich,
  `condor_q -nobatch` reported zero jobs for the user. This ABCD task submitted
  no scheduler jobs.

## Current Git state

- Branch: `task/abcd-bkg-estimation`.
- HEAD before this checkpoint: `6ef08d5`
  (`feat: add high-density ABCD fit reports`).
- Blinded-workflow implementation commit: `11c714a`.
- Before updating this task record, the worktree contained five task-scoped
  implementation paths and no unrelated user changes. This task record is the
  sixth task-scoped path intended for the checkpoint commit.

## Blockers and open questions

- Baseline 1 is already demonstrably non-closing and is expected to be
  rejected after the formal baseline evaluation.
- Baseline 2 requires quantitative density- and run-dependent validation and
  an independent secondary validation before any nonclosure correction can be
  transported to the signal region.
- Baseline 1 fails validation decisively and must not be transported to the
  signal region.
- Baseline 2 has strong inclusive nonclosure and density dependence. Run-block
  and secondary validation are still required before making the formal
  closure decision; no correction is authorized.
- Both original ABCD constructions fail the explicitly unblinded diagnostic.
  These signal-region comparisons are development data and are not an
  independent measurement.
- Direct signal-region normalization is scientifically circular. The
  validation-derived Baseline 2 normalization also fails transport, leaving a
  roughly 31% overprediction, so it is not accepted as a correction.
- The first global-scale fit has unacceptable shape goodness-of-fit and a
  boundary signal estimate. Adding significance or limits to this model is
  not justified. A decision is required before either redesigning the
  background or constructing the full simultaneous four-region diagnostic.

## Exact next action

Review the reblinded implementation and choose whether to first replace the
unvalidated `kappa=1` and placeholder 30% neutrino uncertainties, or to run
the expected-only production target as an explicitly provisional diagnostic.
Do not implement or run observed unblinding until that decision is recorded.
