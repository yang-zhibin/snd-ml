# Task: abcd-bkg-estimation

Status: active
Last updated: 2026-09-30 16:34 Europe/Zurich

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

## Modified files

- `docs/analysis/regions/REGION_V1_ABCD_BASELINE_PLAN.md`
  - Added the approved two-baseline plan, validation mappings, signal leakage,
    density binning, required outputs, and interpretation rules.
- `docs/analysis/regions/REGION_V1_ABCD_STATISTICAL_LIKELIHOOD_GUIDE.md`
  - Added the from-scratch likelihood derivation, current arithmetic results,
    profile-likelihood outline, and `cabinetry` / `pyhf` workflow boundary.
- `docs/tasks/active/abcd-bkg-estimation.md`
  - Added this checkpoint.

## Validation

- No code tests were run because this checkpoint contains documentation and
  read-only analysis only.
- Verified the saved likelihood guide by reading its opening and final
  sections.
- Verified the inclusive counts independently from the authoritative partition
  metadata and the existing evaluation summary.

## Running jobs

- None. `condor_q -nobatch` reported zero jobs for `zhibin`; there are no job
  IDs, logs, or expected batch outputs associated with this task.

## Blockers and open questions

- Baseline 1 is already demonstrably non-closing and is expected to be
  rejected after the formal baseline evaluation.
- Baseline 2 requires quantitative density- and run-dependent validation and
  an independent secondary validation before any nonclosure correction can be
  transported to the signal region.
- The final `density_sndsw_scifi` binning and sparse-tail merge policy are not
  yet frozen.

## Exact next action

Implement baseline-plan step 1 only: add a focused Region V1 ABCD input-audit
command to the existing Snakemake workflow that reads the authoritative
`partition_v1_region_v1.csv`, verifies per-region exposure and inclusive event
count closure for all eight real-data regions, and writes a versioned CSV plus
Markdown report. Do not implement the ABCD likelihood or start production in
the same action.
