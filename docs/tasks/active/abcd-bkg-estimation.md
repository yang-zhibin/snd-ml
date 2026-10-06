# Task: abcd-bkg-estimation

Status: active
Last updated: 2026-10-02 Europe/Zurich

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

## 2026-10-01 strict one-bin reblinding checkpoint

### Verified findings and decisions

- The primary analysis is now exactly one counting bin,
  `density_scifi_sndsw` in `[5000, 40000]`; density-bin indices and the former
  two-bin expected fit are absent. `[5000, 7000, 40000]` is retained only as a
  disabled future extension in configuration.
- The blinded expected DAG reads real data from B, C, D and the independent
  validation plane only. It does not declare, resolve, open, serialize, or
  otherwise consume the signal-region A observation. The sanitized one-row
  schema rejects `A_data`, `count_a`, `observed_a`, and equivalent fields.
- The likelihood uses one shared residual-background parameterization,
  `L_D rho`, `L_B rho u`, `L_C rho v`, and `L_A rho u v kappa`, plus explicit
  CC-nue signal and CC-numu/NC-nue/NC-numu components. These explicit MC
  components are subtracted when the residual ABCD initialization is formed,
  so they are not counted twice.
- The control-only B+C+D fit fixes `mu=0`. This is explicitly conditional
  because the measured CC-nue contamination fractions are 12.25% in B,
  0.182% in C, and 0.189% in D; B exceeds the configured 1% negligible
  threshold. The separately gated final A+B+C+D fit releases `mu` and profiles
  every nuisance again rather than freezing the control-fit estimates.
- The provisional nonclosure constraint is derived without A data from the
  independent Baseline-2 validation plane. For `[5000,40000]`, its residual
  result is `kappa = 0.398767 +/- 0.139976` (relative uncertainty 0.351022),
  including validation-data Poisson, MC-statistical, and configured 30% MC
  normalization contributions. Transport of this constraint and the 30% MC
  normalization uncertainties remain explicitly provisional.
- The isolated real-input expected-only run (not production) gave a
  control-post-fit A background of 65.5042 +/- 26.4625 events, expected
  `mu=1` discovery `Z=0.0656`, and median background-only 95% CL upper limit
  `mu=6.9609`. These are sensitivity diagnostics, not observed results.
- The unblinding authorization file remains `authorized: false`. No observed
  A fit and no production target were run in this checkpoint.

### Modified files

- Replaced
  `snakemake/metadata/configs/region_evaluation/abcd_density_blinded_inference_v1.yaml`
  with the one-bin
  `abcd_density_blinded_counting_v2.yaml` and added the disabled
  `abcd_density_unblind_authorization_v2.yaml` gate.
- Reworked `evaluation_region_partitions/make_abcd_blinded_inputs.py` and
  `run_abcd_blinded_expected.py`; added `abcd_blinded_model.py` and
  `run_abcd_unblinded.py`; expanded
  `test_abcd_blinded_expected.py`; and split expected/unblinded targets in
  `snakemake/rules/region_partitions.smk`.

### Commands and results

- Under the documented LCG 107 environment, preflight recorded host
  `lxplus913.cern.ch`, Python 3.11.9, ROOT 6.34.02, NumPy 1.26.4, pyhf 0.7.6,
  cabinetry 0.5.2, and PyYAML 6.0.1. `python3 -m py_compile` passed for all
  five Python implementation/test files.
- `python3 evaluation_region_partitions/test_abcd_blinded_expected.py`
  passed. It checks the forbidden-A dependency/schema boundary, four one-bin
  channels, independent-kappa modifier, no explicit-MC double counting,
  control/Asimov outputs, a downward fluctuation with one-sided `q0=0`, denial
  by the disabled authorization, authorized synthetic A+B+C+D inference, and
  full nuisance profiling in five toy fits. SciPy/Minuit boundary and
  non-positive covariance warnings were emitted for this deliberately weak,
  saturated synthetic model; the assertions completed successfully.
- The documented Snakemake executable dry-ran
  `region_v1_abcd_density_blinded_expected_report`: exactly the sanitized
  input and expected report rules were scheduled, and its real-data inputs
  were the seven sideband/validation regions only. A separate summary dry-run
  of `region_v1_abcd_density_unblind_one_bin` confirmed that the signal-region
  real-data ROOT/CSV inputs appear only in that explicit target.
- `git diff --check` passed. Implementation checkpoint commit:
  `4d4c354` (`fix: enforce strict blinded ABCD counting workflow`).

## 2026-10-01 one-bin closure-table report checkpoint

### Verified findings and decisions

- The expected report now contains a single `[5000,40000)` closure row in the
  style of `nominal_closure_v1/nominal_closure.pdf`. Because this workflow is
  blinded, the row is explicitly the independent validation plane and uses
  primed notation; it does not expose or use signal-region A data.
- The production validation row is: residual
  `A'_obs = 12.9459605`, `A'_pred = 32.4649396 +/- 6.9362650`,
  `T = 1.47872254 +/- 0.00346355`,
  `kappa_val = 0.39876743 +/- 0.13997597`, fractional nonclosure
  `-60.1233%`, and pull `-2.49686`.
- Definitions and calculation formulas are printed immediately below the
  table for raw count `N_R'`, explicit neutrino MC `Q_R'`, residual
  `R_R'`, exposure `L_R'`, transfer `T`, prediction `A'_pred`, propagated
  uncertainties, `kappa_val`, nonclosure, and pull. The report also defines
  the likelihood quantities `b_R`, `rho`, `u`, `v`, `nu_R`, `mu`, `S_R`,
  `q_k,R`, and `gamma_k`, plus the input-table quantities.
- A report-only formula typo was corrected from `b_B=L_D rho u` to the
  implemented `b_B=L_B rho u`.

### Modified files and commit

- `evaluation_region_partitions/make_abcd_blinded_inputs.py` now materializes
  the validation residuals, variances, exposures, transfer, prediction,
  uncertainties, nonclosure, and pull.
- `evaluation_region_partitions/run_abcd_blinded_expected.py` serializes those
  quantities and renders the Markdown/LaTeX closure table and definitions.
- `evaluation_region_partitions/test_abcd_blinded_expected.py` covers the new
  schema, payload, table, and definition text.
- Task-only implementation commit: `431931e`
  (`feat: add one-bin closure table to blinded report`).

### Commands, results, and outputs

- LCG 107 compilation and the full focused synthetic test passed. The test
  again exercised strict A masking, expected inference, the authorization
  gate, simultaneous synthetic inference, and toy fits; the known weak-model
  Minuit covariance warnings remained non-fatal.
- An isolated real-sideband preview reproduced the closure values, expected
  `Z=0.06557599`, and expected 95% upper limit `mu=6.9608563` without reading
  observed A. Independent arithmetic matched every serialized closure value
  to relative/absolute tolerance `1e-12`.
- The preview and production LaTeX sources passed two `pdflatex` passes. The
  final PDF is four-page A4 and its extracted text contains the closure row,
  definitions, expected-only notice, and locked observed-results statement.
- Production command:
  `env -u PYTHONHOME -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake --snakefile snakemake/snakefile region_v1_abcd_density_blinded_expected_report --cores 1 --printshellcmds --rerun-incomplete`.
  Both local jobs completed. The production freeze manifest records clean Git
  commit `431931e`, `freeze_status=FROZEN`, and
  `signal_region_observation_used=false`; all recorded artifact SHA256 hashes
  were independently verified.
- Final PDF:
  `/afs/cern.ch/work/z/zhibin/snd-ml/evaluation_region_partitions/abcd/partition_v1_region_v1/density_sndsw_scifi_v1/blinded_counting_v2/expected/expected_report.pdf`.
  The corresponding Markdown, LaTeX, JSON, workspaces, scans, covariance, and
  plots are in the same `expected/` directory.
- The final aggregate dry-run reported that every requested file is present
  and up to date. No observed/unblinded target was run.

## 2026-10-01 expanded definitions and methodology checkpoint

### Report changes and references

- Replaced the large definition paragraphs with more than 30 individual
  entries. Every entry has separate `Definition`, `Why it is used`, and
  `Interpretation` text. The entries cover the ABCD likelihood, input table,
  validation closure, nuisance parameters, and expected-inference quantities.
- Added eight explicit procedure steps: freeze/blind, materialize inputs,
  validation closure, likelihood construction, conditional B+C+D fit,
  Asimov construction, profile-likelihood inference, and freeze/unblinding
  boundary. Every step contains `Why`, `How`, and applicable formulas or
  numerical substitutions.
- Added primary references and in-text citations: HistFactory
  (CERN-OPEN-2012-016, including its ABCD example), Choi and Oh's ABCD
  extrapolation paper (arXiv:1906.10831), Cowan--Cranmer--Gross--Vitells
  (arXiv:1007.1727), and the peer-reviewed pyhf paper
  (doi:10.21105/joss.02823).

### Modified files, validation, and production

- Modified `evaluation_region_partitions/run_abcd_blinded_expected.py` and
  `evaluation_region_partitions/test_abcd_blinded_expected.py`. Task-only
  implementation commit: `f9db4a5`
  (`docs: expand blinded report methodology and definitions`).
- LCG compilation and the complete focused synthetic test passed. The test
  now requires at least 30 entries of each explanation type, all eight steps,
  and all four bibliography records, in addition to the existing likelihood,
  blinding, authorization, and toy-fit checks. Known weak-model optimizer
  warnings remained non-fatal.
- An isolated preview compiled in two LaTeX passes with no overfull/underfull
  boxes, undefined citations, or LaTeX warnings. Extracted text verified all
  explanation labels, steps 1 and 8, the reference list, and the locked
  observed-results statement.
- Production reran only
  `region_v1_abcd_density_blinded_expected_report`; the sanitized input was
  unchanged. The final PDF is seven-page A4, and the numerical expected and
  closure results are unchanged. The new freeze manifest records clean commit
  `f9db4a5`, `freeze_status=FROZEN`, and no signal-region observation use; all
  recorded hashes passed verification.
- The final aggregate dry-run is a no-op. No observed/unblinded target was
  run.

## Running jobs

- This task has no scheduler jobs. Its histograms were produced by local
  Snakemake jobs.
- At the final check on 2026-10-01 at 14:48 Europe/Zurich,
  `condor_q -nobatch` reported zero jobs for the user. This ABCD task submitted
  no scheduler jobs.

## 2026-10-01 original-binning transfer-factor comparison checkpoint

### Implementation and scope

- Added an explicitly unblinded PyROOT diagnostic comparing the exposure-
  corrected `Signal/SB2` and `SB4/SB6` inside/outside transfer factors in the
  same canvas, with their double ratio in a lower panel.
- The calculation consumes the four historical ROOT histograms behind the
  original Region V1 comparison plot. It validates and preserves their exact
  80-bin ROOT axis over `[100,40000]` (actual width `498.75`) and excludes
  underflow and overflow, matching that plot rather than the later frozen
  six-bin closure policy.
- Each individual transfer graph now uses its own statistical support. A
  transfer is shown whenever its denominator is positive; a zero numerator is
  retained as a zero estimate with an asymmetric upper interval. Only the
  lower ratio-of-ratios requires all four raw counts to be positive. All 80
  bins remain in the CSV with an explicit per-quantity status.
- The individual transfer-factor uncertainties are 68.27% exact conditional
  intervals for the ratio of two independent Poisson rates, constructed by
  conditioning on the total, evaluating the Clopper--Pearson binomial
  interval with `ROOT.TEfficiency.ClopperPearson`, and transforming it back to
  the exposure-corrected rate ratio. The lower-panel uncertainty is the
  68.27% four-Poisson profile-likelihood interval (`-2 Delta log L = 1`), which
  remains tractable for the million-event veto samples and is asymmetric in
  sparse bins. The former symmetric `1/sqrt(N)` delta approximation is no
  longer used.
- The diagnostic is labelled `Explicitly unblinded development diagnostic`.
  It does not subtract signal, derive a nonclosure correction, or authorize
  transport into a final result.

### Files and outputs

- Added
  `evaluation_region_partitions/plot_abcd_original_binning_transfer_comparison.py`,
  its focused synthetic test, and
  `snakemake/metadata/configs/region_evaluation/abcd_original_binning_transfer_comparison_v1.yaml`.
- Added the one-job Snakemake target
  `region_v1_abcd_original_binning_transfer_factor_comparison` in
  `snakemake/rules/region_evaluation.smk`. Historical EOS histogram paths are
  passed as frozen parameters so this diagnostic cannot rebuild or overwrite
  the requested original inputs.
- Generated `transfer_factor_comparison.{csv,root,pdf,png}` under
  `evaluation_region_partitions/plots/partition_v1_region_v1/hist_features_v1__evaluation_options_v1/density_sndsw_scifi/none/none/transfer_factor_signal_sb2_vs_sb4_sb6_v1/`.
  The CSV has 80 rows. The ROOT graphs contain 27 valid `Signal/SB2` points,
  77 valid `SB4/SB6` points, and 22 ratio-of-ratios points for which all four
  counts are positive. The ROOT file also contains the canvas, frames, and
  scope/flow metadata. The regenerated PDF is one page and the PNG passed
  visual inspection.
- At user request, the regenerated upper transfer-factor panel uses the fixed
  range `[0,40]`. This keeps every central estimate visible (maximum about
  `32.67`) while clipping only very large sparse-bin upper interval bars; the
  annotation states that clipping explicitly.

### Commands and results

- The documented LCG 107 preflight passed with Python `3.11.9`, ROOT
  `6.34.02`, SciPy `1.15.0`, and PyYAML `6.0.1`.
- `python3 evaluation_region_partitions/test_abcd_original_binning_transfer_comparison.py`
  passed in the activated LCG environment, including the undefined-bin policy,
  zero-numerator interval policy, distinct graph supports, and persisted ROOT
  graph checks.
- The task-recorded Snakemake executable was used with `PYTHONHOME` and
  `PYTHONPATH` unset. The final dry-run contained exactly the new plotting
  rule; the production target completed locally with one job; the subsequent
  dry-run reported everything present and up to date. An initial exact
  conditional odds-ratio implementation was stopped after it proved
  impractical for the million-count veto samples; no process or partial
  output remains from that attempt.
- No HTCondor or other scheduler job was submitted.

## 2026-10-01 six signal-ratio comparisons checkpoint

- Added the six requested exposure-corrected comparisons: `Signal/SB2` versus
  `SB4/SB6` and `SB1/SB3`; `Signal/SB1` versus `SB4/SB5` and `SB2/SB3`;
  and `Signal/SB4` versus `SB1/SB5` and `SB2/SB6`. Each occupies one full page
  of one PDF. Equal closure quantities are adjacent in the final page order:
  `S1,S6`, `S2,S4`, then `S3,S5`.
- Every individual ratio uses its own positive-denominator support. Each lower
  ratio-of-ratios requires all four counts to be positive. At user request,
  these six pages use the standard symmetric independent-Poisson propagation
  from `sqrt(N)` for both individual transfer factors and ratios of transfer
  factors, rather than the exact/profile intervals retained by the earlier
  standalone plot.
- Plot text was simplified: the y-axis reads `Transfer factor`; the upper-left
  heading names the transfer direction (`Inside / outside`, `Has US / no US`,
  or `No veto / veto`); legends spell out the complete selections of both
  regions in each ratio; and the plotted unblinding/error-method annotations
  were removed. The original 80-bin `[100,40000]` density axis and exposure
  correction are unchanged.
- The page layout now reserves a separate header band for the title, subtitle,
  and two complete legend entries, so the legend cannot cover the transfer-
  factor data. Axis scaling is selected from the calculated central values:
  the upper panel switches to logarithmic scale when its positive-value
  dynamic range exceeds 100, while the lower panel switches when its smallest
  positive ratio is below 0.05 or its dynamic range exceeds 100. Zero-valued
  individual transfers on logarithmic pages are retained as downward-triangle
  markers at the plotting floor; the CSV value remains exactly zero.
- At user request, a new front page overlays the signal region and all seven
  sideband density distributions in the original 80-bin presentation. Every
  histogram is normalized to the common reference exposure
  `L_ref=19.78 fb^-1` with factor `L_ref/L_region`; the displayed integrals
  reproduce the original comparison (`SB4=1.17097e8`, `SB6=1.05044e8`,
  `SB5=95453.2`, `SB7=82689.1`, `SB2=174188`, `SB1=24847.1`,
  `SB3=32965.9`, and `Signal=104185`). The page uses the original logarithmic
  event-yield presentation and is stored as `c_region_distributions` together
  with its scaled histograms in the ROOT output.
- The first ratio comparison, `S1` (`Signal/SB2` versus `SB4/SB6`), now has an
  explicit `top_scale: linear` override and retains its configured `[0,40]`
  range. It is PDF page 2 after insertion of the distribution overview.
- Added `evaluation_region_partitions/plot_abcd_signal_ratio_comparisons.py`,
  its focused synthetic test, the versioned YAML configuration, and the named
  Snakemake target `region_v1_abcd_signal_ratio_comparisons`. Outputs are
  `signal_ratio_comparisons.{csv,root,pdf}` under the new
  `.../density_sndsw_scifi/none/none/signal_ratio_comparisons_v1/` directory.
- The production CSV contains 480 rows. Valid lower-panel bin counts are
  `S1=22`, `S2=15`, `S3=15`, `S4=15`, `S5=15`, and `S6=22`. Independent
  verification found `S1=S6`, `S2=S4`, and `S3=S5` bin by bin to tolerance
  `1e-12`, as required for the alternative orientations of the same three
  cube faces.
- The LCG 107 preflight, focused synthetic test, `git diff --check`, ROOT
  canvas/graph inspection, seven-page `pdfinfo` check, and rendered front-page
  plus `S1` preview passed. Production uses logarithmic upper panels for `S6`,
  `S3`, and `S5`, and logarithmic lower panels for `S3` and `S5`; `S1`, `S2`,
  and `S4` remain linear. The header legends do not overlap the histograms and
  the previously compressed small ratios are visible. ROOT `6.34.02` cannot
  safely construct an empty `TGraphErrors(0)` through this PyROOT overload,
  so empty zero-marker selections are skipped instead; the focused test and
  production both pass with that guard. The final PDF has seven portrait pages,
  the CSV has 481 lines including its header, and the ROOT file opens without
  a zombie flag. The final Snakemake dry-run is a no-op. The production target
  completed as one local job; no scheduler job is running or was submitted.

## Current Git state

- Branch: `task/abcd-bkg-estimation`.
- HEAD before this implementation: `5028eb3`
  (`docs: checkpoint expanded blinded analysis report`).
- The transfer-comparison and six-comparison scripts, tests, configurations,
  Snakemake rule changes, and this task-record update are uncommitted task
  changes. Generated plot products are ignored outputs. No unrelated tracked
  changes were present.

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

Review the completed DS--SciFi efficiency and density report with the user and
obtain a decision to accept this extremely low-rejection working point or
redefine it. Do not start post-selection ABCD closure or change the frozen cut
until that decision is recorded.

## 2026-10-02 DS--SciFi rejection implementation checkpoint

### Frozen decision and boundary

- The user's instruction to implement the saved plan is recorded as
  acceptance of its recommended inclusive DS definition: all four DS stations
  active, every `count_dsN <= 2`, and `count_ds <= 5`.
- The SciFi component requires five XY-active stations and every station count
  below 20. The common base remains `count_scifi > 35`.
- Veto and US variables are forbidden from the tag and remain Region V1 ABCD
  axes. The existing V7 selection, partition products, baseline histograms,
  and reports are not modified.

### Implemented files

- Added the frozen configuration
  `snakemake/metadata/configs/region_evaluation/abcd_ds_scifi_muon_rejection_v1.yaml`.
- Added the pure selection/configuration helper, full-source data shard
  worker, validated shard merger, neutrino-MC acceptance worker, and PyROOT
  report generator under `evaluation_region_partitions/`.
- Added focused selection-boundary and synthetic end-to-end tests. The latter
  covers two data shards, count and histogram-flow closure, region-total and
  luminosity closure, exact Clopper--Pearson intervals, MC normalization with
  `sum(w^2)`, combined CSV/ROOT outputs, and multipage PDF generation.
- Added the isolated aggregate target
  `region_v1_abcd_ds_scifi_muon_rejection_v1` to
  `snakemake/rules/region_partitions.smk`. It comprises 100 full-data shards,
  32 region/sample MC jobs, one merge, and one report job. It is not part of a
  broad default target. The frozen partition catalogue is passed as a
  parameter so Snakemake cannot rebuild its producer DAG.

### Commands and verified results

- LCG 107 preflight: host `lxplus9112.cern.ch`, Python `3.11.9`, ROOT
  `6.34.02`, and PyYAML `6.0.1`.
- `python3 -m py_compile` passed for all five implementation modules.
- `python3 evaluation_region_partitions/test_abcd_ds_scifi_selection.py`
  passed.
- `python3 -u -X faulthandler evaluation_region_partitions/test_abcd_ds_scifi_rejection.py`
  passed. Its synthetic counts closed as `8 = 4 + 4`; MC retained yield was
  `2.0` after the configured `2/4` luminosity weight; all requested report
  products were nonempty.
- The first production DAG dry-run exposed an unwanted 800-job Region V1
  partition regeneration. No jobs were run. The catalogue dependency was
  severed from that producer and retained as an explicitly frozen parameter.
- The final dry-run passed with exactly 135 DAG nodes: 100 data-shard jobs,
  32 MC jobs, one merge, one report, and the aggregate target. No canonical
  partition-building job appears.

### Outputs and running jobs

- Expected directory:
  `/afs/cern.ch/work/z/zhibin/snd-ml/evaluation_region_partitions/abcd/partition_v1_region_v1/ds_scifi_muon_rejection_v1/`.
- Final products will be `efficiencies.csv`, `data_efficiencies.csv`,
  `data_full.root`, `ds_scifi_muon_rejection.root`,
  `ds_scifi_muon_rejection.pdf`, and `report.md`, plus validated data-shard and
  MC intermediates.
- No scheduler job has yet been submitted at this checkpoint.

### Git state

- Branch `task/abcd-bkg-estimation`, HEAD `5028eb3`.
- The DS--SciFi files and rule changes are uncommitted task work alongside the
  already recorded uncommitted transfer-comparison work. No commit was made,
  and no unrelated user change was altered.

## 2026-10-02 DS--SciFi production completion checkpoint

### Scheduler execution

- The first two HTCondor submission attempts failed before any job entered the
  scheduler: the profile's submit helper required the removed legacy
  `htcondor` Python binding and then tried to import Snakemake from system
  Python. Both retry loops were stopped; `condor_q` confirmed zero jobs.
- Updated only the worktree profile helper
  `snakemake/.config/snakemake/htcondor/grid-submit.py` to use the installed
  `htcondor2` binding and to parse Snakemake's standard JSON properties header
  directly. The submit semantics, credentials, resources, and CERN accounting
  group are unchanged. System-Python compilation and object construction
  checks passed.
- Production command:
  `env -u PYTHONHOME -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake --snakefile snakemake/snakefile --profile snakemake/.config/snakemake/htcondor region_v1_abcd_ds_scifi_muon_rejection_v1 --jobs 132 --rerun-incomplete --quiet progress rules`.
- HTCondor data/MC clusters were the contiguous range `16337251` through
  `16337382`; merge was `16338129`; the first report was `16338289`. Logs are
  under `/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/htcondor_log/<jobid>_<uuid>/`.
  All jobs completed, and the final `condor_q -totals` reported zero jobs.
- The first report was valid but its 32-row MC table was too dense. The report
  generator was corrected to use two 16-row MC pages and concise readable cut
  definitions, its synthetic end-to-end test passed again, and only the report
  rule was regenerated locally. The final aggregate dry-run is a no-op.

### Validated production results

- The full common-exposure scan used 100 shards and exactly
  `19.780680927948 fb^-1`. All count, finite-value, histogram-flow,
  region-union, fingerprint, and luminosity checks passed.
- Total data after `count_scifi > 35`: `381486372` events; `6693` tagged;
  `381479679` retained. The removed fraction is `1.75445323641e-5` and the
  keep efficiency is `0.999982455468`.
- Per-region tagged counts are: Signal `0`, SB1 `0`, SB2 `13`, SB3 `0`, SB4
  `3249`, SB5 `3`, SB6 `3418`, and SB7 `10`. Exact 68.27% Clopper--Pearson
  intervals are stored in `data_efficiencies.csv` and the report.
- All 32 neutrino-MC region/sample populations have zero tagged events and
  keep efficiency exactly `1` at available MC precision. In the signal
  region, raw counts are `8072` CC nue, `2475` NC nue, `20015` CC numu, and
  `5817` NC numu.
- The combined table has 41 `PASS` rows and one selection fingerprint,
  `0507841dbbdd561e`. `data_full.root` and
  `ds_scifi_muon_rejection.root` open successfully with 62 and 41 keys.
- The final report has 13 pages and includes the cut definition, data and MC
  tables, retained all-region density overlay, and eight per-region
  before/tagged/retained distributions with retention panels.

### Final products

- Directory:
  `/afs/cern.ch/work/z/zhibin/snd-ml/evaluation_region_partitions/abcd/partition_v1_region_v1/ds_scifi_muon_rejection_v1/`.
- Human-readable outputs: `ds_scifi_muon_rejection.pdf` and `report.md`.
- Machine-readable outputs: `efficiencies.csv`, `data_efficiencies.csv`,
  `data_full.root`, and `ds_scifi_muon_rejection.root`, plus the 100 data
  shard and 32 MC ROOT/CSV pairs.

### Final verification and Git state

- Final module compilation, focused selection test, synthetic end-to-end
  test, ROOT-open checks, 13-page `pdfinfo`, `git diff --check`, aggregate
  no-op dry-run, and empty scheduler check passed.
- Branch remains `task/abcd-bkg-estimation` at HEAD `5028eb3`. All task
  implementation and previously recorded ratio-comparison changes remain
  uncommitted; no commit was created and no unrelated work was modified.

## 2026-10-02 DS--SciFi sequential-cutflow report checkpoint

### Implemented scope

- Added one shared ordered six-stage cutflow definition without changing the
  frozen tag: `count_scifi > 35`, five SciFi XY-active stations, every SciFi
  station below 20 hits, all four DS stations active, every DS station at most
  two hits, and total DS hits at most five.
- Data shards and every MC region/sample row now persist all six cumulative raw
  counts in their CSV and ROOT metadata. The workers and data merger require a
  monotonic cutflow, require the first count to equal `n_before`, and require
  the final count to equal `n_tagged`.
- The report aggregates the full-source data total and the configured signal
  sample (`CC_nue`) over all eight mutually exclusive regions. It adds one PDF
  and Markdown table for each, with remaining counts, events removed at the
  step, step efficiency, and cumulative efficiency. It also writes
  `cutflow_data` and `cutflow_signal_mc` ROOT histograms.
- The report generator disables ROOT statistics boxes and uses shorter region
  descriptions plus a smaller legend to address the previously observed plot
  overlap and clipping. The frozen selection and fingerprint are unchanged.

### Modified files

- `evaluation_region_partitions/abcd_ds_scifi_selection.py`
- `evaluation_region_partitions/make_abcd_ds_scifi_rejection_data_shard.py`
- `evaluation_region_partitions/merge_abcd_ds_scifi_rejection_data.py`
- `evaluation_region_partitions/make_abcd_ds_scifi_rejection_mc.py`
- `evaluation_region_partitions/make_abcd_ds_scifi_rejection_report.py`
- `evaluation_region_partitions/test_abcd_ds_scifi_selection.py`
- `evaluation_region_partitions/test_abcd_ds_scifi_rejection.py`

### Commands and verified results

- Documented LCG 107 runtime: Python `3.11.9`, ROOT `6.34.02`, PyYAML
  `6.0.1`.
- `python3 -m py_compile` passed for all seven affected implementation and test
  modules.
- The focused selection test passed and verifies the ordered stage keys and
  equivalence of the final cumulative expression to base plus the frozen tag.
- The synthetic end-to-end test passed. Both data and signal MC have the
  expected cutflow `[8, 8, 8, 8, 4, 4]`; the final count equals `n_tagged`.
  The synthetic report contains seven pages, both Markdown sections, and the
  two cutflow ROOT objects.
- A temporary real signal-region `CC_nue` run passed with cutflow
  `[8072, 117, 0, 0, 0, 0]`, reproducing the independent diagnosis.
- A temporary real full-source data shard-0 run passed with cutflow
  `[3739875, 3396793, 546828, 532724, 125925, 73]`; its first and final values
  exactly match the previously verified `n_before` and `n_tagged`.
- The ordinary aggregate Snakemake dry-run contains exactly 135 jobs: 100 data
  shards, 32 MC jobs, one merge, one report, and the aggregate target. It
  contains no partition rebuild.
- `git diff --check` passed. No scheduler jobs are present for the user.

### Output and Git state

- Existing production products still use the earlier schema and 13-page
  report. They were not overwritten or regenerated in this checkpoint.
- Branch `task/abcd-bkg-estimation` remains at `5028eb3`. The task changes are
  uncommitted alongside the previously recorded ratio-comparison work. No
  commit was created.

### Exact next action

Run the named aggregate target
`region_v1_abcd_ds_scifi_muon_rejection_v1` to regenerate the 100 data shards,
32 MC products, merged summaries, and expected 15-page report; then validate
the two production cutflow tables and visually inspect the revised PDF. Do not
start post-selection ABCD closure in the same action.


## 2026-10-02 exact loose SciFi--DS projection pre-production checkpoint

### Decision and implemented scope

- The requested new report uses exactly the SciFi and DS components of the V7
  loose selection while preserving Veto and US for the Region V1 ABCD axes:
  `count_scifi > 35`; all five SciFi stations have X and Y activity; each
  SciFi station has fewer than 20 hits; and all four DS stations are active.
- It intentionally does not apply the stricter experimental requirements of
  at most two hits per DS station or `count_ds <= 5`.
- Added the separate configuration
  `snakemake/metadata/configs/region_evaluation/abcd_ds_scifi_loose_projection_v1.yaml`
  and aggregate target `region_v1_abcd_ds_scifi_loose_projection_v1`. The new
  output directory is
  `/afs/cern.ch/work/z/zhibin/snd-ml/evaluation_region_partitions/abcd/partition_v1_region_v1/ds_scifi_loose_projection_v1/`,
  so the existing strict result is not changed.
- To limit scheduler load, the workflow uses 10 single-threaded full-data
  shards and only the requested `CC_nue` signal sample in the eight Region V1
  regions. The report code now derives its definition page and cutflow tables
  from the configured stages, so it cannot label the loose result with the
  omitted strict DS cuts.

### Verification before production

- LCG 107 preflight used Python `3.11.9`, ROOT `6.34.02`, and PyYAML `6.0.1`.
- Compilation, the focused selection test, and the synthetic end-to-end
  strict-plus-loose test passed. The loose synthetic cutflow is
  `[8, 8, 8, 8]`, its final count equals `n_tagged`, and its seven-page report
  contains no per-station or total DS multiplicity claim.
- The dedicated Snakemake executable dry-run passed with exactly 21 nodes:
  10 data workers, 8 `CC_nue` region workers, one data merge, one report, and
  the aggregate target. No partition-production rule is present.
- Branch `task/abcd-bkg-estimation` remains at HEAD `5028eb3`; the existing
  task-related dirty changes are preserved. `condor_q` shows no running jobs.

### Planned production action

- Submit only `region_v1_abcd_ds_scifi_loose_projection_v1` through the
  repository HTCondor profile with at most 10 concurrent jobs, then validate
  its CSV/ROOT count closure and visually inspect its PDF. Do not regenerate
  the strict workflow or begin post-selection ABCD closure.


## 2026-10-02 loose-projection partition-input direction change

- The user identified that the frozen Region V1 feature partitions already
  contain every density, SciFi, and DS branch required by the loose cut. This
  was verified directly for real data. Catalogue coverage is 100 files and
  `19.780680927948 fb^-1` for each no-veto region, and 96 files and
  `1.960319252235 fb^-1` for each veto region.
- The original full-source clusters `16354305`, `16354307`, and
  `16354309`--`16354316` were removed by the user, but the surviving
  Snakemake controller retried them as `16358202`, `16358263`, `16358266`,
  `16358269`, `16358278`, `16358283`, `16358286`, `16358292`, `16358298`,
  and `16358303`. With the user's instruction to implement and generate the
  partition-based report, controller PID `3269380` and all ten replacements
  were explicitly stopped. A subsequent scoped `condor_q` was empty.
- Frozen decision: replace only the separate loose target's full-source data
  scan with eight direct `real_data` region-partition evaluations. Keep the
  existing eight partition-based `CC_nue` evaluations, validate catalogue
  counts and per-region exposure, merge the data products, and label the
  aggregate data cutflow as a mixed-exposure stored-partition diagnostic.
  Preserve the strict production workflow and its products.

### Exact next action

Implement the partition-based loose data worker/merge/rules and focused
synthetic test; run compilation, tests, and a 19-node dry-run before generating
and validating the separate loose PDF. Do not start post-selection ABCD
closure.


## 2026-10-02 partition-based loose workflow pre-production checkpoint

### Implemented and verified

- Generalized `make_abcd_ds_scifi_rejection_mc.py` into a validated Region V1
  partition-population worker while preserving its MC behavior. For real data
  it requires frozen partition/region versions, unique available paths,
  catalogue/tree event-count equality, all required branches, the configured
  per-region catalogue exposure, base-selection closure, finite density,
  monotonic cutflow, count closure, and histogram closure.
- Added `merge_abcd_ds_scifi_rejection_data_partitions.py`, which requires all
  eight real-data regions, preserves their configured exposures, creates the
  report's region and aggregate histograms, and explicitly labels the aggregate
  as mixed exposure rather than a physics yield at one luminosity.
- Replaced only the loose Snakemake block with eight real-data partition jobs,
  eight `CC_nue` partition jobs, one merge, one report, and one aggregate node.
  The frozen catalogue remains a parameter, so its producer cannot enter the
  DAG. The strict target and products are unchanged.
- Updated the report to show data exposure and use a configurable data-scope
  label. The loose report states that the aggregate combines configured
  exposures. Added the required catalogue identity to both strict and loose
  configurations because the shared partition worker validates it.
- `git diff --check`, compilation, the selection test, and the full synthetic
  strict-plus-loose ROOT/CSV/PDF test passed. The loose test exercises direct
  real-data partitions, the new merge, `CC_nue`, and a seven-page report.
- The production dry-run contains exactly 19 nodes: 8 data regions, 8 `CC_nue`
  regions, one merge, one report, and the aggregate. It contains no full-source
  shard or partition-production rule.
- A real SB1 preflight passed: catalogue/tree count `40300`, exposure
  `19.7806809279 fb^-1`, cutflow `[40300, 141, 59, 0]`, tagged `0`, retained
  `40300`, fingerprint `d2d51ef52665b47c`, and status `PASS`.
- The stale Snakemake lock from the stopped controller was cleared only after
  confirming the controller and scoped HTCondor jobs were absent.

### Production action

Run only `region_v1_abcd_ds_scifi_loose_projection_v1` locally through
Snakemake with four cores, matching the earlier partition-histogram workflow;
then audit CSV/ROOT closure, PDF text/pages, and representative rendered pages.


## 2026-10-02 partition-based loose report completed

### Generated products

- Generated the separate loose-cut report at
  `/afs/cern.ch/work/z/zhibin/snd-ml/evaluation_region_partitions/abcd/partition_v1_region_v1/ds_scifi_loose_projection_v1/ds_scifi_loose_projection.pdf`.
  The PDF has 14 pages and contains the data and `CC_nue` cutflow tables,
  rejection efficiencies, and before/after density distributions.
- Supporting products are `report.md`, `data_efficiencies.csv`,
  `efficiencies.csv`, `data_full.root`, the per-region `data_regions/` and
  `mc/` products, and the merged `ds_scifi_loose_projection.root` in the same
  output directory.
- The loose selection is exactly `count_scifi > 35`, all five SciFi stations
  with X and Y activity, fewer than 20 hits in every SciFi station, and all
  four DS stations active. Veto and US remain available as ABCD axes. The
  stricter DS multiplicity cuts are absent from the PDF.

### Commands and verified results

- Production completed locally through the named Snakemake target with four
  cores: 19/19 nodes (eight data regions, eight `CC_nue` regions, one merge,
  one report, and the aggregate target). The final report regeneration log is
  `.snakemake/log/2026-10-02T151847.147390.snakemake.log`.
- All nine rows in `data_efficiencies.csv` and all 17 rows in
  `efficiencies.csv` have status `PASS`, fingerprint
  `d2d51ef52665b47c`, monotonic cutflows, count closure, and histogram
  closure. The merged ROOT files contain 67 data keys and 43 report keys.
- Stored data totals are 40,051,418 before, 5,290,008 tagged, and 34,761,410
  retained. This aggregate mixes the configured `19.7806809279 fb^-1`
  no-veto exposure and `1.96031925223 fb^-1` veto exposure, so it is a stored
  partition diagnostic rather than a yield at one luminosity.
- Data keep efficiencies are 0.999899 (Signal), 1.000000 (SB1), 0.998504
  (SB2), 1.000000 (SB3), 0.862245 (SB4), 0.999849 (SB5), 0.868649 (SB6),
  and 0.999794 (SB7). No available `CC_nue` event is tagged in any region.
- Visual inspection confirmed that the definition, distribution, and final
  efficiency-table pages are readable after shortening the PDF-facing titles
  and formatting the exposure column. `pdfinfo` reports 14 pages. Text audit
  finds none of the omitted strict DS requirements.
- The final Snakemake dry-run reports all outputs present and up to date.
  `git diff --check` passes. Scoped `condor_q` is empty; there are no running
  jobs for this workflow.

### Modified files and Git state

- The implementation files are the partition worker
  `make_abcd_ds_scifi_rejection_mc.py`, the new data-partition merger
  `merge_abcd_ds_scifi_rejection_data_partitions.py`, the report generator,
  its synthetic tests, the loose/strict configuration metadata, and the loose
  target block in `snakemake/rules/region_partitions.smk`.
- Branch `task/abcd-bkg-estimation` remains at `5028eb3`. These task changes
  remain uncommitted alongside previously recorded task-related work; no
  commit was created.

### Exact next action

Have the user review the generated loose-cut PDF and decide whether to accept
this selection/report. Do not begin post-selection ABCD closure or modify the
selection before that decision.


## 2026-10-02 loose-report table and plot formatting revision

### Implemented presentation changes

- The sequential-cutflow tables now report actual removal efficiencies:
  `(N_(i-1)-N_i)/N_(i-1)` for the newly applied step and
  `(N_0-N_i)/N_0` cumulatively. Their headers are explicitly `Removed step
  eff.` and `Removed cumulative eff.` and use a two-line layout to avoid
  overlap.
- The PDF data-efficiency table no longer contains the exposure column. It
  shows the eight physical Region V1 populations; event counts are scaled to
  the common reference luminosity `19.780680927948 fb^-1` and formatted in
  scientific notation with three significant digits. Per-region removal
  fractions and exact 68.27% Clopper--Pearson keep-efficiency intervals remain
  unchanged. The mixed-exposure aggregate row is omitted because after
  reweighting it has no single exact binomial interval.
- Removed the `Neutrino-MC acceptance after rejection` page from the PDF. The
  validated MC CSV/ROOT products and Markdown information are preserved.
- The retained-region overlay now reserves a separate right-hand pad for the
  legend. Region labels contain definitions only, without event totals, and
  the plot margin preserves the final `40000` x-axis label.

### Verification and outputs

- Under LCG 107, compilation and the full focused synthetic strict-plus-loose
  end-to-end test passed twice after the semantic and final layout changes.
  The synthetic report expectation is now six pages and explicitly checks
  that the removed MC-acceptance page is absent.
- Regenerated only the report rule and aggregate target through Snakemake;
  no data scan, partition worker, or batch submission ran. Final log:
  `.snakemake/log/2026-10-02T154939.765576.snakemake.log`.
- The production PDF is now 13 pages. Text inspection confirms the normalized
  luminosity label, removal headings, eight definition-only legend entries,
  and absence of the MC-acceptance page. Rendered pages 2, 4, and 5 were
  visually inspected and are readable without overlap or clipping.
- The final Snakemake dry-run reports all products present and up to date;
  `git diff --check` passes; scoped `condor_q` is empty.
- Branch `task/abcd-bkg-estimation` remains at `5028eb3`. The report generator,
  focused test, and this task record are uncommitted alongside the previously
  recorded task-related changes. No commit was created.

### Exact next action

Have the user inspect the revised 13-page PDF. Do not change the selection or
begin post-selection ABCD closure until the user provides the next instruction.


## 2026-10-02 final six-page loose-report presentation

### Implemented changes

- Removed the `Interval: 68.27% Clopper--Pearson (nested binomial)` row from
  the PDF definition page. Statistical interval values remain available in
  the CSV and Markdown products.
- Removed the `Keep efficiency [68% CI]` column from the PDF data-rejection
  table. The remaining event-count columns stay normalized to
  `L_ref = 19.780680927948 fb^-1` and use three significant digits.
- Restored a wide comparison frame using separate `1500 x 800` retained and
  removed canvases. Both legends are outside the plotting frame and now show
  each histogram's total event count after its complete region definition.
- Removed all eight individual before/tagged/retained PDF pages. Replaced them
  with one combined removed-event overlay, giving the final PDF six pages.
- Verified from the ROOT inputs that every removed event is below the previous
  original-plot lower boundary of 100: the original tagged histograms contain
  all tagged counts in underflow. The removed page therefore uses the existing
  validated `h_analysis_*_tagged` histograms over `[0,40000]`; it shows all
  rejected events in the first `[0,500]` bin without folding or redistributing
  entries. The retained page continues to use the original `[100,40000]`
  plotting histograms.

### Verification, outputs, and Git state

- LCG 107 compilation and the focused strict-plus-loose synthetic end-to-end
  test passed after the final canvas and histogram-range fixes. The test
  requires six pages and verifies removal of the interval text, keep-efficiency
  column, MC-acceptance page, and individual region-page titles, plus presence
  of both combined overlays.
- Regenerated only the report and aggregate Snakemake nodes. Final log:
  `.snakemake/log/2026-10-02T161620.435901.snakemake.log`.
- `pdfinfo` reports six pages. PDF text contains the retained and removed
  titles and complete count-bearing legend entries, while the removed interval,
  keep-efficiency, and individual-page text are absent. Pages 1, 4, 5, and 6
  were visually inspected; tables, axes, and both external legends are clean.
- Final Snakemake dry-run is a no-op, `git diff --check` passes, and scoped
  `condor_q` is empty.
- Branch `task/abcd-bkg-estimation` remains at `5028eb3`. The report generator,
  focused test, and task record remain uncommitted with the previously recorded
  task changes. No commit was created.

### Exact next action

Have the user inspect the final six-page PDF. Do not alter the selection or
begin post-selection ABCD closure until the user gives a new instruction.


## 2026-10-02 removal-tag cutflow semantic correction

### Correction

- The previously displayed `Removed step eff.` and `Removed cumulative eff.`
  values were complements of the removal-tag efficiencies and therefore
  described events failing the tag requirements. Corrected the definitions to
  `N_i/N_(i-1)` for removal-tag step efficiency and `N_i/N_0` for removal-tag
  cumulative efficiency, because the cumulative cutflow constructs the final
  removal tag.
- Renamed the `N_(i-1)-N_i` count column from `Removed` to
  `Failing requirement`; those events fail the newly applied removal-tag
  requirement and are not themselves removed from the analysis at that step.
- Updated both PDF headers and the Markdown formulas. Added an explicit
  numerical regression probe using counts `[100,80,20]`, requiring step
  efficiencies `[--,80%,25%]` and cumulative efficiencies
  `[100%,80%,20%]`.

### Verified production values

- Region V1 data counts are `[40051418,33658402,5432006,5290008]`.
  Correct removal-tag step efficiencies are `[--,84.0380%,16.1386%,97.3859%]`
  and cumulative efficiencies are `[100%,84.0380%,13.5626%,13.2080%]`.
  The final cumulative value equals `5290008/40051418`.
- Aggregated `CC_nue` counts are `[43245,12821,0,0]`. Correct step
  efficiencies are `[--,29.6474%,0%,n/a]` and cumulative efficiencies are
  `[100%,29.6474%,0%,0%]`. The last step is undefined because its previous
  denominator is zero.

### Validation, output, and state

- LCG 107 compilation and the focused strict-plus-loose synthetic end-to-end
  test passed after updating one obsolete heading assertion.
- Regenerated only the report and aggregate Snakemake nodes. Log:
  `.snakemake/log/2026-10-02T170056.406435.snakemake.log`.
- The production PDF remains six pages. Pages 2 and 3 were rendered and
  visually inspected; extracted text reproduces every verified count and
  efficiency above without overlap.
- Final Snakemake dry-run is a no-op, `git diff --check` passes, and scoped
  `condor_q` is empty. Branch HEAD remains `5028eb3`; task changes are still
  uncommitted and no commit was created.

### Exact next action

Have the user inspect corrected PDF pages 2 and 3. Do not begin the proposed
new ABCD-region development until the user gives the next instruction.


## 2026-10-02 proposed Veto edge-muon selection recorded

### Decision

- Saved the requested selection definition in
  `docs/analysis/regions/REGION_V1_VETO_EDGE_MUON_SELECTION.md`.
- The selection keeps edge-muon candidates. With both horizontal Veto planes
  available, they must identify the same physical y edge. A single available
  y plane is used alone. Missing y or x information is neutral so that the
  other orientation decides, but an explicit `any_valid` guard rejects events
  with no valid Veto position from the edge-muon sample.
- The 10 cm edge thresholds are derived from the frozen 2024 detector
  geometry and are recorded with their physical plane boundaries.
- This checkpoint is documentation only. No analysis code, configuration,
  generated output, job, or existing baseline was changed or run.

### Git state

- The worktree already contained the task-related uncommitted changes recorded
  above. This checkpoint adds the new selection note and updates this task
  record; no commit was created.

### Exact next action

Have the user review and approve the saved Veto edge-muon definition. Do not
implement or evaluate it until the user explicitly requests that next action.


## 2026-10-02 Veto edge-muon workflow implemented and tested

### Implemented scope

- Added the isolated configuration
  `snakemake/metadata/configs/region_evaluation/abcd_veto_edge_muon_selection_v1.yaml`
  and aggregate target
  `region_v1_abcd_veto_edge_muon_selection_v1`.
- Extended the existing partition worker and data merger with a separate
  `veto_edge_muon` mode. The frozen DS--SciFi path and its selection
  fingerprint are unchanged.
- The selected/kept population is the edge tag itself. Output CSVs expose the
  unambiguous fields `n_selected_edge`, `n_nonedge`,
  `edge_efficiency`, `sumw_selected_edge`, and
  `sumw2_selected_edge`; the backend `tagged`/complement objects remain
  only for compatibility with the validated histogram machinery.
- Added exclusive Veto-position availability counts for both orientations,
  y only, x only, and no position, plus diagnostics for the individual y-plane
  availability cases. The four orientation categories must sum exactly to the
  input population.
- Added the dedicated PyROOT report generator
  `evaluation_region_partitions/make_abcd_veto_edge_report.py`. Its
  eight-page report contains the geometry/logic definition, data and
  `CC_nue` cutflows, availability table, luminosity-normalized data table,
  signal-MC acceptance, selected edge-muon density overlay, and complementary
  non-edge overlay. It reports point efficiencies only and contains no
  confidence-interval presentation.
- Added
  `evaluation_region_partitions/test_abcd_veto_edge_selection.py`, covering
  same-edge, opposite-edge, central-position, one-y-plane, missing-y,
  missing-x, and all-missing cases, plus count/category/histogram closure and
  ROOT/CSV/Markdown/PDF production.
- The workflow consumes the existing Region V1 feature partitions directly.
  It does not rescan full source data, rebuild partitions, or submit HTCondor
  jobs. The existing no-Veto populations are expected to select zero edge
  muons because `any_valid` is false there.

### Commands and results

- Under the documented LCG 107 environment, Python compilation passed for all
  modified/new Python files. PyROOT was 6.34.02 and PyYAML was 6.0.1.
- The new synthetic end-to-end test passed and produced the expected
  `10 before -> 6 selected edge muons + 4 non-edge`, availability counts
  `[6 both orientations, 2 y-only, 1 x-only, 1 no-position]`, and an
  eight-page PDF with no Clopper--Pearson text.
- The pre-existing strict-plus-loose DS--SciFi synthetic end-to-end regression
  test also passed, confirming that the old selection path remains intact.
- `git diff --check` passed.
- The first two dry-run launcher attempts failed before DAG construction
  because the incomplete user-local Snakemake launcher could not import
  `snakemake` after environment stripping and then lacked `humanfriendly`.
  No job ran and no output was created.
- The documented Snakemake executable succeeded:
  `env -u PYTHONHOME -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake --snakefile snakemake/snakefile region_v1_abcd_veto_edge_muon_selection_v1 --cores 4 --dry-run --printshellcmds`.
  The DAG has exactly 19 nodes: eight data-region workers, eight `CC_nue`
  workers, one data merge, one report, and the aggregate target.

### Scheduler and Git state

- A scoped scheduler query found no running ABCD/Veto-edge job. No job ID
  exists because production has not been started.
- Branch HEAD remains `5028eb3`. The worktree contains the previously
  recorded uncommitted task changes plus this workflow; unrelated-looking
  pre-existing changes were preserved. No commit was created.

### Exact next action

Run the local production target, and only that target, with:

`env -u PYTHONHOME -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake --snakefile snakemake/snakefile region_v1_abcd_veto_edge_muon_selection_v1 --cores 4 --printshellcmds --rerun-incomplete`

Do not begin interpretation, threshold changes, or a new ABCD construction
until the generated products have been audited.


## 2026-10-02 central-muon rejection redefinition implemented

### Frozen logic

- Added a separate `veto_central_muon_rejection` mode and preserved the
  earlier edge-selection prototype.
- A central muon requires central y information and a valid central x
  measurement simultaneously. If both y planes are valid, both must be
  central; if only one y plane is valid, that plane decides y. Missing all y
  information or missing x makes `central_muon=false`.
- The rejected population is `central_muon = y_central && x_central`; the
  kept analysis population is `!central_muon`.
- The exact central intervals are open:
  Veto 1 y `(24.835139,46.867571)` cm, Veto 2 y
  `(22.784356,44.816788)` cm, and Veto 3 x
  `(-37.370290,-15.755074)` cm.

### Report presentation

- Added
  `evaluation_region_partitions/make_abcd_veto_central_report.py` and the
  versioned configuration
  `abcd_veto_central_muon_rejection_v1.yaml`.
- Page 1 shows the user's exact logical definition as one monospaced code
  block; it does not split the definition into separate cut rows.
- Page 2 contains one row only for the aggregate `central_muon` cut:
  before, central-muon count, point efficiency, and kept count. It contains no
  sequential component cuts or confidence interval.
- The remaining pages contain availability, per-region data, `CC_nue`,
  kept-density, and rejected-central-density diagnostics. The report has seven
  pages.

### Validation and state

- The focused synthetic test produced `10 before -> 3 central rejected + 7
  kept`, including missing-orientation, central/edge, inconsistent-y, and
  exact-boundary cases. Availability closed as
  `[7 both, 1 y-only, 1 x-only, 1 none]`.
- PDF text/page checks confirmed the exact-definition title, the single
  `central_muon` efficiency page, absence of component-cut headings,
  absence of Clopper--Pearson text, and seven pages.
- The earlier edge-selection and DS--SciFi end-to-end regression tests both
  passed. Python compilation and `git diff --check` passed.
- The named-target dry-run succeeded with exactly 19 local nodes: eight data
  regions, eight `CC_nue` regions, one merge, one report, and the aggregate.
- No production job was started and a scoped scheduler query found no related
  job. Branch HEAD remains `5028eb3`; no commit was created.

### Exact next action

Run only the redefined central-muon production target:

`env -u PYTHONHOME -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake --snakefile snakemake/snakefile region_v1_abcd_veto_central_muon_rejection_v1 --cores 4 --printshellcmds --rerun-incomplete`

Do not run the earlier edge-selection target or begin interpretation until the
central-rejection outputs have been audited.
