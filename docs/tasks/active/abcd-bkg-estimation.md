# Task: abcd-bkg-estimation

Status: active
Last updated: 2026-10-01 02:53 Europe/Zurich

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
- No further ABCD inference was authorized or started. The inclusive
  arithmetic values remain diagnostics from already inspected development
  data, not a final background or signal result.
- The Region V2 plotting workflow uses
  `inspection_event_weight = target_luminosity / source_luminosity` with a
  run-9715 target of `0.6102 fb^-1`. That inspection-specific normalization
  is not used for Region V1 ABCD counts. The new Region V1 density histograms
  contain raw unit-weight data counts and store exposure as metadata.
- The initial 500-unit density binning is now versioned and reproducible. It
  is suitable for materializing counts, but sparse-tail merging must still be
  decided before a final binned likelihood.

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

## Modified files

- `docs/analysis/regions/REGION_V1_ABCD_BASELINE_PLAN.md`
  - Added the approved two-baseline plan, validation mappings, signal leakage,
    density binning, required outputs, and interpretation rules.
- `docs/analysis/regions/REGION_V1_ABCD_STATISTICAL_LIKELIHOOD_GUIDE.md`
  - Added the from-scratch likelihood derivation, current arithmetic results,
    profile-likelihood outline, and `cabinetry` / `pyhf` workflow boundary.
- `docs/tasks/active/abcd-bkg-estimation.md`
  - Added this checkpoint.
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

## Running jobs

- This task has no scheduler jobs. Its density histograms were produced by
  local Snakemake jobs.
- At the final check, `condor_q -nobatch` showed unrelated running job
  `16300791.0` from worktree `snd-ml-3d-cnn-testbeam`; it was not submitted
  or modified by this task.

## Current Git state

- Branch: `task/abcd-bkg-estimation`.
- HEAD before this checkpoint: `3e053cd` (`docs: record ABCD checkpoint
  commit`).
- Before updating this task record, the worktree contained five uncommitted
  task-scoped paths; this task-record update is the sixth. No unrelated user
  changes were present.
- This stopping checkpoint is intended to commit those task-scoped changes
  together with this updated task record. The resulting commit SHA and final
  cleanliness check are reported in the stopping handoff.
- HEAD at the start of the density implementation was
  `e5b3abf63a4face736232b8714f091baf9f382ce`.
- Before the density checkpoint commit, the worktree contains four new or
  modified implementation paths plus this task-record update, all scoped to
  this task; no unrelated changes are present.

## Blockers and open questions

- Baseline 1 is already demonstrably non-closing and is expected to be
  rejected after the formal baseline evaluation.
- Baseline 2 requires quantitative density- and run-dependent validation and
  an independent secondary validation before any nonclosure correction can be
  transported to the signal region.
- The initial 500-unit `density_sndsw_scifi` binning and flow policy are
  frozen. The final sparse-tail merge policy remains pending until the
  per-bin control-region statistics are tabulated.

## Exact next action

Audit the authoritative normalization and event-weight source for the standard
`CC_nue`, `CC_numu`, `NC_nue`, and `NC_numu` Region V1 MC samples only.
Record the formula and required branches/metadata for `sum(w)` and
`sum(w^2)`; do not generate MC histograms, materialize ABCD tables, calculate
new closure results, or implement a likelihood in the same action.
