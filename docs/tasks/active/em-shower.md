# Task: em-shower

Status: active
Last updated: 2026-10-06 03:57 Europe/Zurich

## Goal

I would like to develop a method to identify the EM shower signature from the electronic data

## Work location

- Worktree: `/afs/cern.ch/user/z/zhibin/worktrees/snd-ml-em-shower`
- Branch: `task/em-shower`
- Starting commit: `5793ce82cfc4a1f0bd11c551e2032266ac53c504`
- Chat/session ID: `not recorded`

## Discovered scope

- Prototype geometry-aware clustering for SciFi SiPM QDC distributions before
  adding shower-profile features to the production feature files.
- Preserve the existing renderer and reuse its geometry and hit extraction in
  a separate diagnostic script.
- Cluster independently in each of five horizontal and five vertical planes.

## Task-specific constraints

- A cluster is a maximal sequence without a missing electronics-order channel.
- Physical channel-63/64, SiPM, and mat gaps do not split a cluster when the
  channel indices on their two sides are consecutive.
- Invalid hits do not enter clusters and remain visible as red crosses.
- Expose auditable per-cluster hit and QDC summaries.

## Completed checkpoints

- Inspected the current code, workflow, existing output, branch, and worktree.
- Implemented a separate cluster-coloured SciFi QDC event-display prototype.
- Added synthetic boundary, gap, validity, duplicate, ranking, and closure tests.
- Rendered and validated one real 2024 MC CC electron-neutrino event.
- Generated and validated three signal-selection displays for each of CC nue,
  CC numu, NC nue, and NC numu (12 events total).
- Changed the largest-QDC cluster to fixed amber (`#ffbf00`) in every plane,
  retaining its black outline and recording the color in the JSON audit.
- Regenerated and validated all ten saved signal events per class (40 events
  total) with the fixed-color convention.
- Implemented the agreed two-cluster-per-orientation scalar schema in the
  production feature converter, including all-cluster, remaining-cluster,
  station-profile, shape, fragmentation, and vertical-horizontal summaries.
- Moved clustering into a shared ROOT-free module and changed the diagnostic
  event display to use that same implementation.
- Added an output validator, unit coverage, schema metadata, feature
  documentation, and explicit Snakemake dependencies.
- Ran and validated a 48-event direct-input canary on 2024 MC neutrino data.
- Completed the planned code-correctness test campaign: starting-commit
  regression, external-cutflow and Hit3D alignment, deep formula/closure audit,
  four-class MC sanity comparison, display cross-check, and workflow parsing.
- Implemented a separate versioned 2024 MC-neutrino production and evaluation
  workflow for CC nue, CC numu, NC nue, and NC numu, without submitting it.
- Added three nested analysis scopes, curated unit-area 1D clustering-feature
  comparisons, and selected 2D QDC/hit, smoothness, and fragmentation studies.
- Validated the new histogram and plotting paths on the existing 195-event
  four-class canary sample.
- Added the scalar `mc_incoming_energy` truth branch to the feature converter
  and validated it on 48 selected 2024 MC electron-neutrino events.

## Findings and decisions

- Use `layer_channel = channel + 128 * sipm + 512 * mat`, spanning 0--1535.
- Split clusters only when successive valid, finite-QDC hits differ by more
  than one in `layer_channel`.
- Rank the largest-QDC cluster by QDC sum, breaking ties by hit count and then
  lower first channel; also report the independently largest-hit cluster.
- Clamp negative QDC to zero for plotted heights and QDC summaries.
- Produce horizontal and vertical PDF pages with five plane plots each,
  cluster colours, hardware boundaries, largest-QDC outlines, and invalid hits.
- Always fill the largest-QDC cluster amber (`#ffbf00`); use `tab20` for other
  clusters and red crosses for invalid hits.
- Keep clustering ROOT-free for unit tests; rendering reuses the established
  geometry and hit-extraction functions.
- This prototype clusters raw valid digis. Production feature integration must
  instead apply the agreed clustering to the feature writer's selected,
  time-filtered valid SciFi hits.
- `mc_incoming_energy` is a double filled from `MCTrack[0].GetEnergy()` for MC
  events with at least one truth track. It remains at the standard `-999`
  sentinel for real data or MC events without a truth track. Feature-only
  region partitions preserve it automatically because they copy all input
  feature branches.

## Agreed `digi_2_features.py` cluster-feature plan

### Scope and exclusions

- Cluster independently in each `(station, orientation)` plane, then rank all
  clusters from the five planes of an orientation together.
- Save two fixed cluster slots for `vertical` and two for `horizontal`; rank by
  descending integrated QDC, then descending hit count, then lower station.
- The two saved clusters may come from the same station.
- Find every cluster internally for totals and summaries, but do not save
  per-cluster vectors or individual clusters after rank 2.
- Do not save first/last channel, peak fraction, normalized QDC width, QDC
  balance, or any combined-top-2 QDC/hit/fraction branches.
- `qdc_per_hit` is the arithmetic mean channel QDC within the relevant cluster
  or collection; do not duplicate it under the name `qdc_mean`.

### Per-saved-cluster branches

For `scifi_{vertical|horizontal}_cluster{1|2}_*`, save:

```text
exists
station
nhits
qdc
qdc_per_hit
plane_hit_fraction
plane_qdc_fraction
orientation_hit_fraction
orientation_qdc_fraction
qdc_std
qdc_cv
qdc_roughness
```

For cluster channel QDC values `q_i`, hit count `N`, cluster sum `Q`, plane
totals `(H_p, Q_p)`, and orientation totals `(H_o, Q_o)`:

```text
nhits = N
qdc = Q = sum(q_i)
qdc_per_hit = Q / N
plane_hit_fraction = N / H_p
plane_qdc_fraction = Q / Q_p
orientation_hit_fraction = N / H_o
orientation_qdc_fraction = Q / Q_o
qdc_std = sqrt(sum((q_i - qdc_per_hit)^2) / N)
qdc_cv = qdc_std / qdc_per_hit
qdc_roughness = mean(abs(q_{i+1} - q_i)) / qdc_per_hit
```

Set one-hit-cluster roughness to zero. Low `qdc_cv` and low roughness describe
a smooth plateau; larger values describe a more uneven channel-QDC profile.

### Relations between cluster 1 and cluster 2

For each orientation, save:

```text
cluster2_to_cluster1_hit_ratio
cluster2_to_cluster1_qdc_ratio
cluster2_to_cluster1_qdc_per_hit_ratio
top2_station_delta
top2_station_separation
top2_same_station
top2_qdc_downstream_asymmetry
top2_hit_downstream_asymmetry
```

Calculations:

```text
hit_ratio = cluster2_nhits / cluster1_nhits
qdc_ratio = cluster2_qdc / cluster1_qdc
qdc_per_hit_ratio = cluster2_qdc_per_hit / cluster1_qdc_per_hit
station_delta = cluster2_station - cluster1_station
station_separation = abs(station_delta)
same_station = int(station_delta == 0)
downstream_qdc_asymmetry = (Q_downstream - Q_upstream) / (Q_downstream + Q_upstream)
downstream_hit_asymmetry = (H_downstream - H_upstream) / (H_downstream + H_upstream)
```

Positive downstream asymmetry means growth toward later stations; negative
asymmetry means decreasing activity. Set both asymmetries to zero when the two
clusters share a station and use `top2_same_station` to identify that case.

### Remaining-cluster and general orientation branches

Do not save combined-top-2 branches. Summarize clusters after the two saved
slots directly:

```text
scifi_{orientation}_other_cluster_count
scifi_{orientation}_other_cluster_nhits
scifi_{orientation}_other_cluster_qdc
scifi_{orientation}_other_cluster_qdc_per_hit
scifi_{orientation}_other_cluster_hit_fraction
scifi_{orientation}_other_cluster_qdc_fraction
```

These are calculated from all orientation clusters excluding cluster ranks 1
and 2. The fractions use the complete orientation hit and QDC totals.

For each orientation also save:

```text
cluster_count
active_plane_count
active_station_count
total_nhits
total_qdc
qdc_per_hit
cluster_nhits_mean
cluster_nhits_std
cluster_nhits_cv
cluster_total_qdc_mean
cluster_total_qdc_std
cluster_total_qdc_cv
cluster_qdc_per_hit_mean
cluster_qdc_per_hit_std
cluster_qdc_per_hit_cv
effective_hit_cluster_count
effective_qdc_cluster_count
```

The `*_mean/std/cv` values describe the distribution across all clusters in
that orientation. `total_qdc / total_nhits` weights every hit equally, whereas
`cluster_qdc_per_hit_mean` gives every cluster equal weight. For cluster hit or
QDC fractions `f_i`, calculate the effective count as `1 / sum(f_i^2)`; it is
near one for one dominant cluster and grows with fragmentation.

### Five-station profiles within each orientation

Using all clusters, not only the two saved clusters, save:

```text
hit_frac1 ... hit_frac5
qdc_frac1 ... qdc_frac5
cluster_count_frac1 ... cluster_count_frac5
hit_mean_station
hit_std_station
qdc_mean_station
qdc_std_station
cluster_mean_station
cluster_std_station
max_hit_station
max_qdc_station
max_cluster_count_station
hit_station_slope
qdc_station_slope
hit_station_curvature
qdc_station_curvature
```

Station fractions divide the station quantity by the corresponding total over
stations 1--5. Weighted station moments use
`mean=sum(station*weight)/sum(weight)` and the corresponding weighted standard
deviation. Fit the five normalized station values versus station number with a
linear term for slope and a quadratic term for curvature. Positive slope means
increasing downstream; negative slope means decreasing. Negative quadratic
curvature describes rise-then-fall behavior, while positive curvature describes
fall-then-rise behavior.

### Vertical-horizontal and full-event summaries

Save comparisons constructed from the complete orientation totals:

```text
scifi_vertical_hit_fraction
scifi_vertical_qdc_fraction
scifi_vertical_cluster_fraction
scifi_orientation_hit_asymmetry
scifi_orientation_qdc_asymmetry
scifi_orientation_cluster_asymmetry
```

For quantity `X`, use `vertical_fraction=X_v/(X_v+X_h)` and
`asymmetry=(X_v-X_h)/(X_v+X_h)`. Do not add a combined-top-2 asymmetry.

Save complete-event totals based on all clusters:

```text
scifi_cluster_total_count
scifi_cluster_total_nhits
scifi_cluster_total_qdc
scifi_cluster_qdc_per_hit
scifi_other_clusters_count
scifi_other_clusters_nhits
scifi_other_clusters_qdc
scifi_other_clusters_qdc_per_hit
scifi_other_clusters_hit_fraction
scifi_other_clusters_qdc_fraction
```

Event `other_clusters_*` combines the clusters remaining after excluding the
two saved slots in each orientation; it does not require storing combined-top-2
features.

### Empty values, closure, and validation

- Missing cluster slot: `exists=0`, station `-999`, sums/fractions/ratios zero,
  and unavailable shape quantities `-999`.
- Zero denominator: return zero for fractions, ratios, CV, and asymmetry.
- No-data station moments or fit coefficients: `-999`.
- Require exact hit closure and floating-tolerance QDC closure for every plane,
  orientation, and event.
- Unit-test physical die, SiPM, and mat boundaries; real missing channels;
  ranking/ties; one/zero-cluster cases; same-station top two; shape formulas;
  five-station slope/curvature; empty conventions; and closure.

## Results and evidence

- Smoke tested `cbmsim` entry 3 from 2024 MC path
  `/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/2024/nu12/volume_volTarget/1/`.
- Temporary outputs: `/tmp/scifi_sipm_cluster_event3.pdf` and
  `/tmp/scifi_sipm_cluster_event3.json`.
- PDF has exactly two pages and rasterized successfully. Interactive image
  viewing was blocked by the session's `/var/tmp` sandbox mount issue.
- Closure: 1,287 recorded hits, 1,093 valid finite-QDC hits, 1,093 clustered
  hits, QDC sum 5,030.449309498072, and 186 clusters. All per-plane count and
  QDC closure differences are zero.
- Four-class output root:
  `/afs/cern.ch/work/z/zhibin/snd-ml/evaluation_region_partitions/mc_neutrino_cluster_event_display_v1`.
- Each class has ten individual two-page PDFs with JSON audits and one combined
  20-page PDF. There are 84 output files totalling 5,407,474 bytes.

## Modified files

- `evaluation_region_partitions/scifi_sipm_cluster_event_display.py` (new)
- `evaluation_region_partitions/test_scifi_sipm_cluster_event_display.py` (new)
- `evaluation_region_partitions/prepare_em_shower_neutrino_metadata.py` (new)
- `evaluation_region_partitions/combine_partition_metadata.py` (new)
- `evaluation_region_partitions/make_group_hist2d.py` (new)
- `evaluation_region_partitions/plot_group_hist2d.py` (new)
- `evaluation_region_partitions/test_em_shower_neutrino_workflow.py` (new)
- `evaluation_region_partitions/plot_comparison.py`
- `convertData/scifi_plane_clustering.py` (new)
- `convertData/test_build_region_partitions.py` (new)
- `convertData/test_scifi_plane_clustering.py` (new)
- `convertData/validate_scifi_cluster_feature_file.py` (new)
- `convertData/digi_2_features.py`
- `convertData/FEATURE_DESCRIPTION.md`
- `snakemake/rules/production.smk`
- `snakemake/rules/passing_muon_track_features.smk`
- `snakemake/rules/passing_muon_track_partitions_v2_dq.smk`
- `snakemake/rules/passing_muon_transform_comparison_v8.smk`
- `snakemake/rules/em_shower_neutrino_evaluation.smk` (new)
- `snakemake/metadata/configs/em_shower/production_v1.yaml` (new)
- `snakemake/metadata/configs/em_shower/regions_v1.yaml` (new)
- `snakemake/metadata/configs/em_shower/partitions_v1.yaml` (new)
- `snakemake/metadata/configs/em_shower/hist_features_v1.yaml` (new)
- `snakemake/metadata/configs/em_shower/evaluation_options_v1.yaml` (new)
- `snakemake/snakefile`
- `docs/tasks/active/em-shower.md` (updated)

## Validation

- Used `/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/envs/env_sndsw_nue.sh`.
- Preflight: Python 3.9.25; ROOT, SndlhcGeo, NumPy, and Matplotlib imports passed.
- Direct unit-test invocation: 9 tests passed.
- Real-event render, two-page `pdfinfo`, and JSON closure assertions passed.
- New-file whitespace checks passed. No commit was created.
- One `python3 -m unittest` attempt failed only because that invocation omitted
  the sibling script directory from `sys.path`; direct invocation passed.
- All 40 event PDFs have two pages and all four combined PDFs have 20 pages.
  All 40 audits record `#ffbf00`, ten planes, and exact hit-count/QDC closure.
- Corrected clustering unit suite: 10 tests passed.
- Shared event-display unit suite: 9 tests passed.
- Production-runtime canary used the June25 SNDSW Python with SNDSW_ROOT on
  PYTHONPATH; all 48 input entries were written.
- The output validator checked the schema marker, finite values, internal
  summaries, and closure against the existing count_scifi and qdc_scifi
  branches for all 48 events.
- Verified (runId, eventId, eventIndex) against the raw cbmsim tree for all 48
  direct-input canary entries.
- Canary output: /tmp/scifi_orientation_cluster_canary.root.
- git diff --check passed. No production job or commit was created.
- Starting-commit regression on the 48-event nu12 partition found all 251
  pre-existing branches unchanged and exactly 182 planned scalar branches
  added.
- External-cutflow conversion on 2024 nu12 partition 1 wrote 48 entries; every
  (runId, eventId, eventIndex) matched the existing feature and Hit3D files.
- Direct-input nu14 partition 1 wrote and validated 147 entries. Together with
  nu12 partition 1, the deep audit covered 195 events: 39 CC nue, 106 CC numu,
  9 NC nue, and 41 NC numu.
- All 195 events passed slot/remaining-cluster and station-fraction closure,
  schema exclusions, ROOT dtype, ranking, ratio, sentinel, shape, station-fit,
  and orientation-comparison checks.
- The saved raw display and production features for nu12 eventIndex 3 agreed
  on both ranked clusters in both orientations. Production used 1,079 selected
  hits versus 1,093 raw valid display hits, consistent with their documented
  selection difference.
- Small-sample physics sanity results were directionally consistent with the
  EM-shower hypothesis: CC nue had higher mean QDC/hit (5.37), larger leading
  QDC fractions (0.64/0.66), lower roughness (0.28/0.27), and lower effective
  QDC cluster counts (2.76/2.47) than the other tested classes.
- Snakemake 9.17.2 parsed the complete workflow. No jobs were launched.
- Two diagnostic PyROOT one-liners initially mishandled TFile lifetimes and
  segfaulted; corrected reruns with explicit file lifetimes passed. An initial
  profile-fit audit also treated the documented empty-orientation -999 sentinel
  as zero; the corrected audit passed.
- Source-metadata mapping produced exactly 1,866 unique rows in a local pilot:
  976 for `2024_ve` and 890 for `2024_vm`.
- A concrete feature target dry-run resolved only the dedicated isolated rule;
  no job ran. A concrete CC nue partition dry-run constructed 979 jobs (976
  feature validations, one readiness marker, one metadata map, one partition)
  and exited successfully; no job ran.
- The first aggregate dry-run exposed redundant repeated validation edges and
  was interrupted without running jobs. Replacing them with one readiness
  marker per source subfolder removed the redundant dependency fan-out.
- New root-free workflow tests: 3 passed. Existing clustering tests: 10 passed.
  Existing event-display tests: 9 passed. Python compilation, YAML loading, and
  `git diff --check` passed.
- The 195-event local four-class pilot contained 39 CC nue, 106 CC numu, 9 NC
  nue, and 40 NC numu events after truth splitting (one nu14 event had neither
  requested CC/NC code in the split).
- Four 1D grouped histograms passed selection/integral checks. The comparison
  plotter initially exposed that config defaults were not merged for plot
  normalization; the fix was verified with all four scaled integrals equal to
  one. The pilot 1D PDF has five pages.
- Four 2D grouped histograms, profiles, summary CSV, and a two-page comparison
  PDF were produced successfully under `/tmp`.
- The new `/tmp/mc_incoming_energy_canary.root` contains 48 entries. Every
  `mc_incoming_energy` value is finite and agrees with the corresponding raw
  `MCTrack[0].GetEnergy()` within `1e-9` relative tolerance. The branch type is
  `Double_t`; the first event has energy `1051.6178061137196` GeV.
- Regression against the preceding external-cutflow canary found all 433
  existing feature branches unchanged for all 48 events, with
  `mc_incoming_energy` as the only added branch. The clustering output validator
  also passed all 48 entries.

## Checkpoint: implementation ready for review

Recorded at 2026-10-05 18:29 Europe/Zurich.

### Verified Git state before checkpoint

- Worktree: `/afs/cern.ch/user/z/zhibin/worktrees/snd-ml-em-shower`
- Branch: `task/em-shower`
- HEAD before checkpoint: `5793ce82cfc4a1f0bd11c551e2032266ac53c504`
- All modified and untracked paths are listed in **Modified files** and belong
  to this task; no unrelated user changes were found.
- `git diff --check` exited successfully.

### Checkpoint commit

- Implementation commit:
  `299906f6971db28f0c4e856d1dcd7603ca861e4c`
  (`Add SciFi shower clustering feature workflow`).
- The implementation commit contains 25 task files with 3,036 insertions and
  3 deletions. Generated ROOT files, plots, scheduler logs, and `/tmp` canaries
  are not committed.
- Immediately after the implementation commit, `git status --short` was empty
  and `condor_q -nobatch` still returned no jobs.

### Reproducible commands and results

Root-free tests in the documented LCG environment:

```bash
python3 evaluation_region_partitions/test_em_shower_neutrino_workflow.py
python3 convertData/test_scifi_plane_clustering.py
python3 evaluation_region_partitions/test_scifi_sipm_cluster_event_display.py
```

Results: 3 workflow tests, 10 clustering tests, and 9 event-display tests
passed.

The final 48-event MC-energy canary used:

```bash
/cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python \
  convertData/digi_2_features.py \
  -p /eos/experiment/sndlhc/users/zhibin/MC_neutrino/2024_ve/1/nueAnalysisFilter_MC_neutrino_2024_ve_1.root \
  -d /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/2024/nu12/volume_volTarget/1/sndLHC.Genie-TGeant4_dig.root \
  -g /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/2024/nu12/volume_volTarget/1/geofile_full.Genie-TGeant4.root \
  -o /tmp/mc_incoming_energy_canary.root \
  -t MC_neutrino --tree-name cbmsim
/cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python \
  convertData/validate_scifi_cluster_feature_file.py \
  /tmp/mc_incoming_energy_canary.root
```

Result: 48/48 entries written and validated. A PyROOT audit compared every
`mc_incoming_energy` value to the corresponding raw
`MCTrack[0].GetEnergy()` and compared every pre-existing branch to
`/tmp/scifi_cluster_external_cutflow.root`; all checks passed.

The dedicated feature rule was checked without execution using:

```bash
snakemake --snakefile snakemake/snakefile --dry-run --cores 1 \
  /eos/experiment/sndlhc/users/zhibin/snd-ml/em_shower_neutrino_v1/features/2024_ve/1/feature_scifi_cluster_v1_MC_neutrino_2024_ve_1.root.validated
```

Result: the DAG selected the metadata preparation and dedicated feature rule;
no job ran. A concrete CC nue partition dry-run also exited successfully with
979 planned jobs and no execution.

The intended feature-production submission command, not yet run, is:

```bash
/afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake \
  --snakefile snakemake/snakefile \
  --profile snakemake/.config/snakemake/htcondor \
  --jobs 100 --rerun-incomplete --latency-wait 120 \
  em_shower_mc_neutrino_features_v1
```

This requires the `task/em-shower` worktree until the checkpoint is integrated;
the `develop` checkout does not yet contain the target.

## Running jobs

- None. `condor_q -nobatch` returned no job IDs at checkpoint time.

## Blockers and open questions

- None.

## Exact next action

Wait for explicit user instruction to rerun the versioned
`em_shower_mc_neutrino_features_v1` target from the `task/em-shower` worktree
via Snakemake/HTCondor. Do not submit jobs yet.

## Checkpoint: local runtime benchmark and submission-resource fix

Recorded at 2026-10-05 20:24 Europe/Zurich.

### Failure diagnosis and decision

- The failed production attempt did not reach HTCondor. The first
  `em_shower_prepare_source_metadata_v1` submission exited in the profile's
  `grid-submit.py` because that helper requires `resources["runtime"]`, while
  the rule had no runtime resource. The readiness-marker rule had the same
  omission and would have failed later.
- Timed the exact metadata shell path locally, including LCG activation and
  CSV generation, three times: 24.11 s, 12.31 s, and 13.53 s. Peak RSS was
  106,220--107,000 kB. Each run wrote the expected 1,866 source rows under
  `/tmp`.
- Timed the exact readiness-marker shell path locally: 0.02 s and 3,328 kB
  peak RSS.
- Set both rules to `runtime=5 * 60` (300 seconds). This gives the metadata
  path about 12 times the slowest observed wall time and avoids an unrealistically
  tight limit for batch-worker startup.
- Set metadata resources to `mem_mb=1000`, `disk_mb=1000`, and
  `nvidia_gpu=0`; set marker resources to `mem_mb=500`, `disk_mb=100`, and
  `nvidia_gpu=0`.

### Commands and validation results

- Local timing used `/usr/bin/time` around the rule's own `env_lcg.sh`
  activation and `prepare_em_shower_neutrino_metadata.py` command. All three
  invocations exited successfully.
- Snakemake preflight: host `lxplus912.cern.ch`, worktree root as expected,
  Snakemake `9.17.2`.
- Focused non-submitting dry-run:

  ```bash
  /afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake \
    --snakefile snakemake/snakefile --dry-run --cores 1 \
    em_shower_prepare_source_metadata_v1
  ```

  Result: success; Snakemake displayed `runtime=300`, `mem_mb=1000`,
  `disk_mb=1000`, and `nvidia_gpu=0` for the metadata job.
- A non-submitting aggregate dry-run was manually interrupted while enumerating
  the 1,866-file DAG; it launched no rule and submitted no job. The focused
  parse/dry-run had already validated the edited Snakefile and the resource
  values needed by the submit helper.
- `git diff --check` passed.

### Current Git and scheduler state

- Branch: `task/em-shower`.
- HEAD remains the task-record checkpoint commit
  `f8a8f7122c6f1dd3369651556fdf2ca82bff9483`.
- Uncommitted task-only changes:
  `snakemake/rules/em_shower_neutrino_evaluation.smk` and this task file.
- No checkpoint commit was created for this resource fix.
- `condor_q -nobatch` reported zero jobs for user `zhibin`; there are no
  running job IDs or expected production outputs from this test.

### Exact next action

Wait for explicit user instruction to resume
`em_shower_neutrino_evaluation_v1` through the HTCondor profile. Do not submit
the production workflow yet.

## Current checkpoint pointer

As of 2026-10-06 10:51 CEST, the latest verified state is the checkpoint
`scifi35-only energy and shower-feature evaluation` immediately below. Its
implementation and local validation are complete; no production jobs are
running.

### Exact next action

Wait for explicit user instruction to run the versioned
`em_shower_neutrino_evaluation_v1` aggregate target through Snakemake/HTCondor.
Do not submit it yet.

## Checkpoint: scifi35-only energy and shower-feature evaluation

Recorded at 2026-10-06 10:51 CEST.

### Implemented decisions

- Restricted the four-class comparison to `scifi35`, whose configured region
  selection is `count_scifi > 35`. The Snakefile now derives the scheduled
  regions from `evaluation_options_v1.yaml` and rejects undefined requested
  regions, instead of always scheduling every region definition.
- Added `mc_incoming_energy` and `log10(mc_incoming_energy)` as 1D MC-truth
  control distributions. Both require `mc_incoming_energy > 0`; energy is not
  proposed as a real-data classifier input.
- Added nine energy-conditioned 2D studies: energy versus total SciFi hits,
  total SciFi QDC, all-cluster QDC per hit, vertical/horizontal leading-cluster
  QDC fraction, vertical/horizontal leading-cluster roughness, and
  vertical/horizontal effective QDC cluster count. Energy is the x axis so
  each saved `ProfileX` is the mean shower observable versus incoming energy.
- Set versions to `em_shower_hist_features_energy_v2` and
  `em_shower_evaluation_scifi35_energy_v2`. Versioned the 2D plot output path
  as well as the already-versioned histogram paths to prevent stale plots.
- Measured positive incoming-energy maxima and quantiles in the available
  partitions before choosing a common 0--6500 GeV range. A separate scan found
  class maxima of 4,332--5,961 hits and 27,803--35,784 QDC; the new 2D axes use
  0--7000 hits and 0--40000 QDC. This removed the overflow seen with the first
  1000-hit/10000-QDC draft.

### Files and validation

- Modified:
  `snakemake/metadata/configs/em_shower/hist_features_v1.yaml`,
  `snakemake/metadata/configs/em_shower/evaluation_options_v1.yaml`,
  `snakemake/rules/em_shower_neutrino_evaluation.smk`, and
  `evaluation_region_partitions/test_em_shower_neutrino_workflow.py`.
- The complete configured scope is 28 1D features and 16 2D pairs, including
  two energy 1D features and nine energy-conditioned pairs, for the four
  classes `CC_nue`, `CC_numu`, `NC_nue`, and `NC_numu` in `scifi35` only.
- Focused tests in LCG 107 passed: six tests across
  `test_em_shower_neutrino_workflow.py` and
  `test_build_region_partitions.py`; Python compilation and
  `git diff --check` also passed.
- Built all four local `incoming_energy_vs_qdc_scifi` canaries from the
  available production partitions and regenerated the two-page density/profile
  PDF under `/tmp/em_shower_energy_canary_v2`. Visible 2D integrals exactly
  matched selected entries: 38,870 (`CC_nue`), 88,623 (`CC_numu`), 13,198
  (`NC_nue`), and 24,523 (`NC_numu`). Raw/log-energy 1D and compound roughness
  canaries had also completed successfully.
- A non-executing Snakemake parse with Snakemake 9.17.2 succeeded and listed
  the aggregate evaluation, histogram, partition, and plot targets. No
  production job was run or submitted.
- Available `scifi35` partition-row metadata currently covers 38 of 40 groups:
  `CC_nue/part002` and `NC_numu/part001` are missing and will be scheduled by
  a future aggregate run.

### Current Git and scheduler state

- Branch: `task/em-shower`; HEAD remains
  `f8a8f7122c6f1dd3369651556fdf2ca82bff9483`.
- No commit was created. The task-scoped files above, the earlier partition
  alias/environment/resource fixes, their regression test, this task file,
  and generated workflow metadata remain uncommitted/untracked as shown by
  `git status`.
- `condor_q -nobatch` reported zero jobs for user `zhibin`; there are no
  running job IDs.

### Exact next action

Wait for explicit user instruction to run the versioned
`em_shower_neutrino_evaluation_v1` aggregate target through Snakemake/HTCondor.
Do not submit it yet.

## Checkpoint: inclusive partition alias fix and local validation

Recorded at 2026-10-06 03:57 Europe/Zurich.

- The evaluation submission planned 565 jobs. All 40 `scifi35` and all 40
  `signal_region` partition rows completed, while all 40 `inclusive_selected`
  jobs failed on both attempts before histogramming began.
- Representative worker logs showed
  `runtime_error: Unknown column: "feat.true"`. The shared identifier aliaser
  treated every token in the selection as a feature branch, including the ROOT
  literal `true` from the inclusive expression.
- Changed `alias_expression_identifiers` to accept the actual feature-column
  schema and alias only identifiers present in that schema. Both cutflow and
  Hit3D friend-tree call sites now supply `feature_columns`; literals and
  function names remain unchanged.
- Added `convertData/test_build_region_partitions.py` with regression cases
  for `(true) && pdgCode` and `true && int(count_scifi > 0)`. Both tests passed
  in LCG 107; Python compilation and `git diff --check` passed.
- An exact single-target Snakemake attempt was interrupted after 595.19 s in
  its global mtime inventory before any rule ran. The rule's exact shell
  payload was then executed directly to isolate the requested functional test.
- Local test: `inclusive_selected / CC_numu / part006`, using 89 source files
  and four RDF threads. It completed successfully in 95.73 s with 996,540 kB
  peak RSS and selected 8,850 events.
- Created the production test output
  `/eos/experiment/sndlhc/users/zhibin/snd-ml/em_shower_neutrino_v1/partitions/em_shower_partition_v1/em_shower_regions_v1/inclusive_selected/CC_numu/scifi_cluster_partition_inclusive_selected__CC_numu__part006.root`
  (46 MB) and its row
  `evaluation_region_partitions/em_shower_neutrino_v1/metadata/partition_rows/inclusive_selected__CC_numu__part006.csv`.
- PyROOT validation found 8,850 entries in both `sndData` and
  `cutFlowSummary`, 439 `sndData` branches, and a matching one-row metadata
  event count, region, and particle group.
- Branch remains `task/em-shower` at
  `f8a8f7122c6f1dd3369651556fdf2ca82bff9483`. No commit was created. The
  shared-code fix, its test, earlier rule/task changes, and generated workflow
  metadata remain uncommitted/untracked as shown by `git status`.
- `condor_q -nobatch` reported zero jobs for user `zhibin`.

### Exact next action

Wait for explicit user instruction to resume
`em_shower_neutrino_evaluation_v1` through the HTCondor profile. Do not submit
the production workflow yet.

## Checkpoint: SNDSW environment isolation fix

Recorded at 2026-10-05 21:14 Europe/Zurich.

- The local failure log showed that the feature job exited before producing
  its requested output, but Snakemake's log did not retain the underlying
  worker stderr.
- Reproduced the environment fault by placing the Snakemake Conda Python 3.13
  environment first on `PATH`: the explicit June25 Python then loaded
  incompatible Conda libraries and failed during interpreter startup with
  `ImportError: cannot import name 'text_encoding' from 'io'`.
- Updated `em_shower_mc_neutrino_feature_v1` to follow the established
  `process_features` isolation pattern: remove Miniconda and LCG entries from
  `PATH`, unset `PYTHONHOME`, `PYTHONPATH`, `LD_LIBRARY_PATH`, `ROOTSYS`, and
  `ROOT_INCLUDE_PATH`, then source `env_sndsw_nue.sh` and restore the intended
  `SNDSW_ROOT` Python path.
- A deliberately contaminated environment preflight now succeeds with system
  Python 3.9.25 and imports `ROOT` 6.36.04 plus `SndlhcGeo`.
- A non-submitting dry-run of the concrete `2024_ve/1` validation-marker
  target succeeded, selected exactly one feature job, and rendered the new
  cleanup block in the shell command.
- `git diff --check` passed. No full feature conversion was run and no output
  file or marker was created by this validation.
- Branch remains `task/em-shower` at
  `f8a8f7122c6f1dd3369651556fdf2ca82bff9483`. The rule and this task file have
  uncommitted task-only changes; no commit was created.
- `condor_q -nobatch` reported zero jobs for user `zhibin`.

### Exact next action

Wait for explicit user instruction to resume
`em_shower_neutrino_evaluation_v1` through the HTCondor profile. Do not submit
the production workflow yet.

## Checkpoint: one local energy-histogram job

Recorded at 2026-10-06 11:02 CEST.

- The production `partitions.csv` does not exist yet, so asking Snakemake for
  the production histogram path would also require prerequisite jobs. To keep
  the requested test to exactly one job, ran the exact
  `em_shower_neutrino_hist2d_v1` payload against the verified temporary
  38-partition metadata in LCG 107.
- Tested `incoming_energy_vs_qdc_scifi` for `scifi35/CC_numu`; the command
  completed with exit code 0 in 19.42 seconds and 677,240 kB peak RSS.
- Validated `/tmp/em_shower_energy_single_job_test.root`: it is readable and
  contains `hist2d`, a 65-bin `profile_x`, and 88,623 selected entries. The
  visible 2D integral is also 88,623, so no selected event overflowed.
- No production output was written and no HTCondor job was submitted.
  `condor_q -nobatch` reported zero jobs for user `zhibin`.
- Branch remains `task/em-shower` at
  `f8a8f7122c6f1dd3369651556fdf2ca82bff9483`; the previously recorded
  task-scoped changes remain uncommitted and no commit was created.

### Exact next action

Wait for explicit user instruction to submit
`em_shower_neutrino_evaluation_v1` through the HTCondor profile. Do not submit
it yet.

## Checkpoint: corrected default-resource CLI ordering

Recorded at 2026-10-06 11:13 CEST.

- Diagnosed the reported `dictionary update sequence element #4` exception as
  CLI parsing: `--default-resources` accepts a variable-length list and, when
  it was the last option, consumed `em_shower_neutrino_evaluation_v1` as a
  fifth resource expression without an equals sign.
- Verified that placing `--dry-run` immediately after the four resource
  assignments terminates that list. The corrected command passed parsing and
  reached `Building DAG of jobs...`.
- The non-executing DAG inventory was manually interrupted after several
  minutes because CERN filesystem existence checks were again slow. This was
  not the original parser error and no rule ran.
- `condor_q -nobatch` reported zero jobs for user `zhibin`.

### Exact next action

Wait for the user to run the corrected HTCondor command, with another option
placed directly after the default-resource assignments. Do not submit it from
this session.

## Checkpoint: combine job killed by runtime-unit conversion

Recorded at 2026-10-06 11:36 CEST.

- The submission planned 224 jobs. Both missing partition jobs completed, so
  all 40 `scifi35` partition rows and ROOT partitions are now present.
- `em_shower_combine_partition_metadata_v1` failed on job 16409371 and its
  automatic retry 16409372. Both Condor event logs say
  `SYSTEM_PERIODIC_REMOVE due to wall time exceeded allowed max`; neither was
  a memory, disk, input, or Python failure. The jobs were assigned 3,000 MB
  memory and disk but only `runtime=60`, and were removed about 68 seconds
  after execution began. Snakemake then removed the incomplete combined CSV.
- Verified the unit mismatch in the installed Snakemake 9.17.2 parser:
  command-line `runtime=3600` is treated as 3,600 seconds and normalized to
  the numeric value 60 (minutes), while the repository's legacy
  `grid-submit.py` passes the numeric value directly to CERN `MaxRuntime` as
  seconds. `runtime=60h` and `runtime=3600m` both normalize to the raw value
  3,600 expected by this profile.
- A focused dry run was stopped during the known slow filesystem inventory;
  no rule ran. `condor_q -nobatch` reported zero jobs for user `zhibin`.
- No code or configuration fix was made during this diagnostic request. Git
  remains on `task/em-shower` at
  `f8a8f7122c6f1dd3369651556fdf2ca82bff9483` with the previously recorded
  task-scoped changes.

### Exact next action

Wait for explicit user direction either to implement explicit resources for
the combine/2D/plot rules (the robust fix) or to retry with the temporary CLI
workaround `runtime=60h`. Do not submit a job yet.

## Checkpoint: code audit identifies missing rule resources

Recorded at 2026-10-06 11:39 CEST.

- Rechecked the code after the user noted this failure had not occurred in
  earlier workflows. The primary defect is in the EM-shower Snakefile: four
  executable rules have no explicit `resources:` block:
  `em_shower_combine_partition_metadata_v1`,
  `em_shower_neutrino_plot_v1`, `em_shower_neutrino_hist2d_v1`, and
  `em_shower_neutrino_plot2d_v1`.
- `git show` and `git blame` confirm these omissions were present in the
  workflow's initial commit `299906f6`; they were not introduced by the
  current energy-feature edits. Earlier attempts stopped in feature or
  partition production before reaching the combine/plot stages, so the
  omission had not yet produced this failure.
- Comparable maintained rules declare explicit runtime, memory, disk, and GPU
  resources. For example, `combine_region_partition_metadata` uses
  `runtime=30 * 60`, `mem_mb=1000`, `disk_mb=1000`, and `nvidia_gpu=0`, and
  `region_eval_feature_comparison_plot` also declares all four resources.
- The CLI runtime normalization explains the observed 60-second fallback but
  should not be relied upon to repair incomplete production rule definitions.
  The correct fix is to add explicit resources to all four executable rules,
  then validate their rendered resources before resubmission.
- No files other than this task record were changed during the audit. No job
  was submitted; `condor_q -nobatch` reported zero jobs for user `zhibin`.

### Exact next action

Wait for explicit user authorization to add and validate explicit resources
for the four affected EM-shower rules. Do not resubmit the workflow yet.

## Checkpoint: explicit resources fixed and validated

Recorded at 2026-10-06 11:43 CEST.

- Added explicit single-threaded, CPU-only resources to all four affected
  executable rules in `em_shower_neutrino_evaluation.smk`:
  - combine metadata: 1,800 s, 1,000 MB memory, 1,000 MB disk;
  - 1D plotting: 1,200 s, 2,000 MB memory, 1,000 MB disk;
  - 2D histogramming: 2,700 s, 3,000 MB memory, 3,000 MB disk;
  - 2D plotting: 1,200 s, 2,000 MB memory, 1,000 MB disk.
  All four set `nvidia_gpu=0`.
- Added `test_batch_rules_have_explicit_resources` to
  `test_em_shower_neutrino_workflow.py`, covering the exact resources of all
  four rules so the omission cannot silently recur.
- Validation in LCG 107: seven focused tests passed; Python compilation and
  `git diff --check` passed. Snakemake 9.17.2 parsed the full Snakefile and
  listed the EM-shower aggregate targets successfully.
- A complete text audit of the EM-shower Snakefile found zero executable
  `shell`/`run`/`script`/`wrapper` rules without explicit resources.
- No local production rule or HTCondor job was run. `condor_q -nobatch`
  reported zero jobs for user `zhibin`.
- Branch remains `task/em-shower` at
  `f8a8f7122c6f1dd3369651556fdf2ca82bff9483`; no commit was created and the
  previously recorded task-scoped changes remain uncommitted.

### Exact next action

Wait for explicit user instruction to rerun
`em_shower_neutrino_evaluation_v1` through the HTCondor profile. The command no
longer needs `--default-resources` for this workflow. Do not submit it yet.

## Checkpoint: histogram fan-out EOS directory failure

Recorded at 2026-10-06 11:57 CEST.

- The corrected workflow planned 222 jobs. The metadata-combine job received
  the intended `runtime=1800` and completed successfully; its 40-row output
  plus header remains at
  `evaluation_region_partitions/em_shower_neutrino_v1/metadata/partitions.csv`.
- Snakemake then submitted a wave of 100 1D histogram jobs. An audit of the
  105 new Condor log directories found 25 jobs that reached execution and
  failed, one completed payload, and 80 jobs removed by the user.
- All 25 executed failures have the same worker-side exception before the
  histogram script starts: `WorkflowError: Failed to create output directory`
  followed by `FileNotFoundError` for the new versioned hierarchy under
  `/eos/experiment/sndlhc/users/zhibin/snd-ml/em_shower_neutrino_v1/hists/`.
  This is not a feature-expression, ROOT, memory, disk, or runtime failure.
- One worker successfully created and wrote
  `mc_incoming_energy/none/none/scifi35/CC_numu.root` with 88,623 selected
  entries. Its success, alongside failures on other hosts, demonstrates that
  the credentials and payload are valid but direct concurrent construction of
  a new deep EOS directory tree from many worker nodes is unreliable.
- The current rules declare deep EOS paths directly as Snakemake outputs.
  Nested worker Snakemake therefore tries to create those directories before
  entering the shell payload. The robust workflow fix is to prepare all exact
  histogram and plot parent directories once on the submit host before the
  fan-out (with an AFS marker dependency); scratch staging/XRootD can then be
  added for atomic output transfer if required. Merely changing resources
  cannot fix this failure.
- The user stopped Snakemake and removed remaining Condor jobs.
  `condor_q -nobatch` now reports zero jobs for user `zhibin`. No code was
  changed during this diagnostic request; only this task record was updated.

### Exact next action

Wait for explicit user authorization to implement and locally validate a
single submit-host output-directory preparation rule required by every
histogram and plot job. Do not resubmit the workflow yet.

## Checkpoint: submit-host output-directory preparation implemented

Recorded at 2026-10-06 12:14 CEST.

- Added versioned marker
  `evaluation_region_partitions/em_shower_neutrino_v1/metadata/output_dirs/em_shower_hist_features_energy_v2__em_shower_evaluation_scifi35_energy_v2.prepared`.
- Added `em_shower_prepare_evaluation_output_dirs_v1` as a Snakemake
  `localrules` rule. It deterministically derives, deduplicates, creates, and
  records every histogram and plot parent directory before batch fan-out.
  It has explicit CPU-only resources (`runtime=600`, `mem_mb=500`,
  `disk_mb=100`, `nvidia_gpu=0`).
- Added the marker as an input dependency of the 1D histogram, 1D plot, 2D
  histogram, and 2D plot rules. A version/config change produces or refreshes
  the marker before those jobs can start.
- The configured 220 output jobs share 88 unique parent directories: 44 EOS
  histogram directories and 44 AFS plot directories. The initial test
  expectation incorrectly used the job count; live validation caught this,
  and the regression assertion was corrected to 88 unique directories.
- Executed only the preparation target locally. It completed successfully in
  about 18 seconds, wrote 88 unique absolute paths, and an independent check
  found all 88 directories present and writable. A second invocation reported
  `Nothing to be done`, confirming idempotency.
- Added regression coverage for the local rule, its explicit resources, all
  four marker dependencies, and the 88-directory count. Eight focused tests,
  Python compilation, `git diff --check`, and the Snakemake 9.17.2 full-file
  parse passed.
- An exact local Snakemake histogram target was stopped while spending several
  minutes in the known global EOS mtime inventory; no payload had started.
  The rule's exact payload was then run directly into the previously failing
  prepared path for `scifi_cluster_qdc_per_hit`, `scifi35/CC_nue`. It exited 0
  in 15.06 seconds with 676,060 kB peak RSS and wrote 43,245 selected entries;
  selected, raw-integral, and visible-integral counts all agree.
- The successful production test output is
  `/eos/experiment/sndlhc/users/zhibin/snd-ml/em_shower_neutrino_v1/hists/em_shower_partition_v1_em_shower_regions_v1/em_shower_hist_features_energy_v2__em_shower_evaluation_scifi35_energy_v2/scifi_cluster_qdc_per_hit/none/none/scifi35/CC_nue.root`.
- No HTCondor job was submitted. `condor_q -nobatch` reports zero jobs for
  user `zhibin`. Branch remains `task/em-shower` at
  `f8a8f7122c6f1dd3369651556fdf2ca82bff9483`; no commit was created.

### Exact next action

Wait for explicit user instruction to rerun
`em_shower_neutrino_evaluation_v1` through the HTCondor profile. The prepared
marker and the two already completed histogram ROOT files will be reused. Do
not submit the workflow yet.

## Checkpoint: scifi35 energy evaluation completed

### Verified findings

- The latest Snakemake run completed successfully with `221/221` steps at
  `2026-10-06 13:26:45`.
- Completion log:
  `.snakemake/log/2026-10-06T125926.958441.snakemake.log`.
- `condor_q -nobatch` showed no remaining jobs for the user after completion.
- The completed `scifi35` evaluation produced:
  - 112 one-dimensional histogram ROOT files (28 features x 4 classes),
  - 64 two-dimensional histogram ROOT files (16 feature pairs x 4 classes),
  - 28 one-dimensional comparison PDFs and 28 CSV summaries,
  - 16 two-dimensional comparison PDFs and 16 CSV summaries.
- Histogram roots:
  - `/eos/experiment/sndlhc/users/zhibin/snd-ml/em_shower_neutrino_v1/hists/em_shower_partition_v1_em_shower_regions_v1/em_shower_hist_features_energy_v2__em_shower_evaluation_scifi35_energy_v2`
  - `/eos/experiment/sndlhc/users/zhibin/snd-ml/em_shower_neutrino_v1/hists/2d/em_shower_hist_features_energy_v2__em_shower_evaluation_scifi35_energy_v2`
- Plot roots:
  - `evaluation_region_partitions/em_shower_neutrino_v1/plots/em_shower_partition_v1_em_shower_regions_v1/em_shower_hist_features_energy_v2__em_shower_evaluation_scifi35_energy_v2`
  - `evaluation_region_partitions/em_shower_neutrino_v1/plots/2d/em_shower_hist_features_energy_v2__em_shower_evaluation_scifi35_energy_v2`

### Current Git state

- Branch: `task/em-shower`.
- Starting/current checkpoint commit remains
  `f8a8f7122c6f1dd3369651556fdf2ca82bff9483`.
- No checkpoint commit was created; this record and generated plot products
  remain in the existing task worktree state.

### Exact next action

Wait for explicit user instruction to inspect or interpret the completed
distributions. Do not begin another workflow action automatically.

## Plan: add MC muon up to the EM-shower comparison

Recorded at 2026-10-06 CEST. This is a design checkpoint only; no
MC-muon-up implementation or production job was started.

### Objective and interpretation

- Add the upward-going MC-muon background as one additional comparison
  sample alongside `CC_nue`, `CC_numu`, `NC_nue`, and `NC_numu`.
- The intended plots therefore contain five unit-area curves. `MC_muon_up`
  remains separate from the neutrino CC/NC truth classification.
- Produce current detector and SciFi-clustering features and reusable
  histogram products for MC muon up, but render only:
  - `density_sndsw_scifi`, and
  - `scifi_cluster_qdc_per_hit`.

### Verified existing inputs and constraints

- The maintained MC-muon-up definition expects 15 source feature files from
  four productions: `scoring_1.8`, `scoring_2`, `scoring_2.5`, and
  `scoring_at449m`.
- Source digi products are under
  `/eos/experiment/sndlhc/MonteCarlo/MuonBackground/muons_up/`.
- The configured source catalogue is
  `snakemake/metadata/updated/MC_muon_up_metadata.csv`; it must be generated or
  restored before production because it is not currently present in this
  worktree.
- `convertData/digi_2_features_mc_muon.py` is the required compatibility
  entry point. It delegates to the current `digi_2_features.py` while handling
  legacy `FairEventHeader` inputs.
- The existing em-shower source loader is neutrino-specific: it assumes
  numeric partitions, neutrino cut-flow paths, and CC/NC source mappings.
  MC muon up must not simply be appended to that loader.
- The current rule uses one feature list for both histogram creation and plot
  creation. The implementation must split reusable histogram features from
  the smaller plot-feature subset.

### Planned data flow

```text
15 MC-muon-up digi files
  -> current MC-muon-compatible feature converter
  -> versioned, validated MC-muon-up feature files
  -> MC_muon_up partitions with abs(pdgCode) == 13 and count_scifi > 35
  -> histograms for all applicable shared detector/clustering features
  -> two five-sample unit-area comparison PDFs and CSV summaries
```

### Implementation logic

1. Add a dedicated MC-muon-up source configuration rather than mixing the
   records into the neutrino-only `(subfolder, integer partition)` registry.
   Preserve the four production labels and require exactly 15 source rows.
2. Add a versioned MC-muon-up feature rule using
   `digi_2_features_mc_muon.py`, the metadata-provided digi/geometry/tree, and
   `data_type=MC_muon`. Write into an isolated em-shower output directory so
   generic/shared feature products are not overwritten. Validate every file
   with the cluster-feature validator.
3. Normalize the validated MC-muon rows to the metadata schema consumed by
   `build_region_partitions.py`, then combine their partition metadata with
   the existing neutrino partition metadata for comparison only.
4. Add a partition group with a new stable particle ID:

   ```yaml
   MC_muon_up:
     source_data_type: MC_muon
     source_subfolders: [up]
     particle_id: 5
     particle_family: muon
     event_filter: "std::abs(pdgCode) == 13"
   ```

   Apply the existing `scifi35` region expression, so the final MC selection
   is `std::abs(pdgCode) == 13 && count_scifi > 35`.
5. Keep neutrino classes and MC-muon comparison groups as separate rule
   concepts. Existing four-class feature, partition, and histogram products
   must remain reusable and must not be regenerated merely because the muon
   comparison is enabled.
6. Produce MC-muon histograms for all applicable shared detector and
   clustering observables. Do not schedule the two neutrino-energy features
   or the nine incoming-energy 2D pairs for MC muons; their semantics are not
   valid for this sample. The current configuration has 27 shared 1D features
   after excluding the two energy observables, plus seven non-energy 2D pairs
   if reusable 2D products are desired.
7. Introduce an explicit plot subset:

   ```yaml
   plot_features:
     - density_sndsw_scifi
     - scifi_cluster_qdc_per_hit
   ```

   Histogram aggregates may include all applicable features, while the plot
   aggregate requests only these two PDFs and their summaries.
8. Each requested plot must consume five histogram inputs in this order:
   `CC_nue`, `CC_numu`, `NC_nue`, `NC_numu`, `MC_muon_up`. Add a distinct
   muon label/style and write the five-sample plots to a new namespace so the
   completed four-class plots are preserved.
9. Keep unit-area normalization. Pool the four MC-muon productions only as an
   explicitly labelled unweighted diagnostic event sum; do not present the
   result as an absolute flux or physics-rate prediction.

### Planned aggregate targets

- `em_shower_mc_muon_up_features_v1`: produce and validate all 15 feature
  files.
- `em_shower_mc_muon_up_partitions_v1`: create the selected MC-muon
  `scifi35` partitions and metadata.
- `em_shower_mc_muon_up_hists_v1`: create all applicable shared MC-muon
  histograms without rebuilding completed neutrino histograms.
- `em_shower_neutrino_muon_up_plots_v1`: create only the two requested
  five-sample PDFs and CSV summaries.

### Validation plan

1. Assert exactly 15 source rows and all four scoring productions.
2. Run a one-file feature canary and verify the current schema, especially
   `density_sndsw_scifi` and `scifi_cluster_qdc_per_hit`.
3. Validate event identity and uniqueness of `(runId, eventId, eventIndex)`.
4. Confirm every selected MC-muon event satisfies
   `std::abs(pdgCode) == 13 && count_scifi > 35`.
5. Compare partition entry counts with histogram raw and selected counts.
6. Verify each requested PDF has exactly five curves and its summary has five
   group rows.
7. Dry-run each aggregate and confirm the plot aggregate schedules only the
   two requested feature plots and does not rebuild completed neutrino
   products.

### Current Git state

- Branch: `task/em-shower`.
- HEAD: `f8a8f7122c6f1dd3369651556fdf2ca82bff9483`.
- The worktree already contains uncommitted em-shower task changes and
  generated evaluation products; they must be preserved. No checkpoint commit
  was created for this plan-only update.

### Exact next action

Wait for explicit user authorization to implement the MC-muon-up workflow
plan. Do not modify workflow code, generate metadata, run a feature canary, or
submit jobs yet.

## Checkpoint: MC-muon-up comparison workflow implemented

Recorded at 2026-10-06 CEST.

### Implemented files and decisions

- Added `snakemake/metadata/configs/em_shower/mc_muon_up_v1.yaml` with the
  15-file/four-production source contract, isolated feature root, 27 shared 1D
  features, seven shared 2D pairs, and the two requested plot features.
- Added
  `snakemake/metadata/configs/em_shower/evaluation_options_muon_up_v1.yaml`
  for the five unit-area curves. Its new plot namespace reads the completed
  `em_shower_evaluation_scifi35_energy_v2` histogram namespace.
- Added `evaluation_region_partitions/prepare_em_shower_mc_muon_up_metadata.py`.
  It filters and validates the source catalogue, maps all four scoring
  productions, assigns stable SHA-256-derived source keys, and writes isolated
  feature paths.
- Added `MC_muon_up` as particle ID 5 with
  `std::abs(pdgCode) == 13`; the existing `scifi35` region adds
  `count_scifi > 35`.
- Extended `validate_scifi_cluster_feature_file.py` with repeatable
  `--required-branch` checks.
- Added a `histogram_input_version` override to `plot_comparison.py`, keeping
  histogram lookup and plot-output versioning separate.
- Added checkpoint-backed rules and named aggregates for MC-muon metadata,
  15 isolated feature files, ten partitions, 27 1D and seven 2D muon
  histograms, combined comparison metadata, and exactly two five-sample plots.
  Existing four-class aggregates and outputs remain separate and reusable.
- Added `test_em_shower_mc_muon_up_workflow.py` and updated the existing
  neutrino workflow assertions for the density feature and extra partition
  rule.

### Validation results

- LCG preflight selected Python 3.11.9, ROOT 6.34.02, pandas 2.2.2, and
  PyYAML 6.0.1.
- Python compilation passed for all new/modified Python entry points.
- Five focused MC-muon workflow tests passed, including a synthetic 15-row
  metadata normalization test.
- Six existing em-shower workflow tests passed.
- Full Snakemake `--list-target-rules` parsing exited successfully and listed
  all new aggregate targets.
- `git diff --check` passed.
- A final aggregate dry-run was intentionally interrupted during the known
  slow global EOS inventory phase after several minutes. It executed no rule
  and submitted no job. No dry-run syntax or DAG error appeared before it was
  stopped, and no Snakemake process remains.

### Current Git state

- Branch: `task/em-shower`.
- HEAD remains `f8a8f7122c6f1dd3369651556fdf2ca82bff9483`.
- No commit was created. Pre-existing task changes and generated evaluation
  products were preserved.

### Exact next action

Wait for explicit user instruction to materialize
`/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/MC_muon_up_metadata.csv`
through the existing metadata workflow, then dry-run one MC-muon feature
target or the new aggregate. Do not start feature production automatically.

## Checkpoint: direct SciFi QDC-per-hit feature implemented

Recorded at 2026-10-06 CEST.

### Implementation

- Added the scalar feature branch `scifi_qdc_per_hit_direct`.
- Added `direct_scifi_qdc_per_hit(hits)`, which independently loops over the
  selected SciFi hits, sums non-negative QDC, counts accepted stations 1--5,
  and returns `total_qdc / total_nhits` or zero for an empty selection.
- Kept the existing hit selection, `process_count_and_qdc`, and clustering
  calculations unchanged. The new calculation consumes the same final
  selected/time-filtered `SciFi_hits` list.
- Extended the ROOT feature validator to require the branch and verify it
  event by event against both `qdc_scifi / count_scifi` and
  `scifi_cluster_qdc_per_hit`.
- Added a separate 1D histogram definition, documentation, and focused unit
  and workflow test coverage.

### Modified files for this checkpoint

- `convertData/scifi_plane_clustering.py`
- `convertData/digi_2_features.py`
- `convertData/test_scifi_plane_clustering.py`
- `convertData/validate_scifi_cluster_feature_file.py`
- `convertData/FEATURE_DESCRIPTION.md`
- `snakemake/metadata/configs/em_shower/hist_features_v1.yaml`
- `evaluation_region_partitions/test_em_shower_neutrino_workflow.py`
- `evaluation_region_partitions/test_em_shower_mc_muon_up_workflow.py`
- `docs/tasks/active/em-shower.md`

### Validation

- LCG 107 preflight: Python 3.11.9, PyYAML 6.0.1, pandas 2.2.2.
- Clustering/direct-calculation suite: 11 tests passed.
- Neutrino workflow suite: 6 tests passed.
- MC-muon-up workflow suite: 5 tests passed.
- Production SNDSW preflight: June25 Python 3.9.25, ROOT 6.36.04; converter
  import and Python compilation passed.
- Reused the documented 2024 `nu12` partition-1 canary. All 48 selected events
  were written to `/tmp/scifi_qdc_per_hit_direct_canary.root`; the validator
  passed all 48 entries and both direct-ratio equality checks.
- The direct loop used 0.018 seconds total for 48 events in the canary
  (approximately 0.37 ms/event).
- `git diff --check` passed. No production job was submitted and no commit was
  created.

### Current Git state

- Branch: `task/em-shower`.
- HEAD: `f8a8f7122c6f1dd3369651556fdf2ca82bff9483`.
- The worktree retains the pre-existing em-shower and MC-muon-up uncommitted
  changes and generated evaluation products; they were preserved.

### Exact next action

Wait for explicit user instruction to rerun or submit feature production with
the new branch. Existing feature ROOT files predate
`scifi_qdc_per_hit_direct` and must be regenerated before its histogram can be
produced. Do not start production or histogram jobs automatically.

## Checkpoint: local outputs migrated from AFS user to AFS work

Recorded at 2026-10-06 CEST.

### Destination and scope

- New authoritative local-output root:
  `/afs/cern.ch/work/z/zhibin/snd-ml-outputs/em_shower_neutrino_v1`.
- Removed the old output tree from
  `/afs/cern.ch/user/z/zhibin/worktrees/snd-ml-em-shower/evaluation_region_partitions/em_shower_neutrino_v1`
  after copy and validation.
- Kept a temporary recovery copy at
  `/tmp/zhibin/em_shower_neutrino_v1.afs-user-backup-20261006`.
- EOS roots for neutrino features, MC-muon features, partitions, and
  histograms were not changed.
- Snakemake's worktree-local `.snakemake` state was not migrated.

### Code and configuration changes

- Set the production `workspace_root`, partition metadata path, evaluation
  metadata paths, and neutrino/muon plot roots to the absolute AFS-work root.
- Added `_em_shower_local_path` to accept both absolute and repository-relative
  local paths without incorrectly prefixing an absolute path with the repo
  root.
- Updated the MC-muon prepared, partition, and combined metadata paths.
- Added focused tests asserting that local outputs use the work volume.

Modified files for this checkpoint:

- `snakemake/metadata/configs/em_shower/production_v1.yaml`
- `snakemake/metadata/configs/em_shower/partitions_v1.yaml`
- `snakemake/metadata/configs/em_shower/evaluation_options_v1.yaml`
- `snakemake/metadata/configs/em_shower/mc_muon_up_v1.yaml`
- `snakemake/metadata/configs/em_shower/evaluation_options_muon_up_v1.yaml`
- `snakemake/rules/em_shower_neutrino_evaluation.smk`
- `evaluation_region_partitions/test_em_shower_neutrino_workflow.py`
- `evaluation_region_partitions/test_em_shower_mc_muon_up_workflow.py`
- `docs/tasks/active/em-shower.md`

### Migration and validation evidence

- Before transfer: 212 files and 8,780,032 file bytes under the AFS-user
  output tree.
- Initial `rsync -a` copy matched all 212 relative files by SHA-256 and exact
  byte count.
- Rewrote the old output-root prefix in 118 copied text metadata/marker files.
- Source and destination CSVs both contained 2,522 total lines; all plot files
  remained byte-identical by SHA-256.
- The migrated consolidated metadata has 40 rows and every
  `source_metadata_csv` value points to the migrated work-volume metadata.
- No old output-root string remains in the configs, rule, or migrated tree.
- Focused LCG tests passed: 7 neutrino workflow tests and 6 MC-muon workflow
  tests.
- Snakemake target parsing completed and listed all nine relevant aggregate
  targets.
- A target-specific dry-run recognized the migrated absolute plot target and
  reached `Building DAG of jobs...`; it was stopped during the known slow EOS
  inventory phase. It ran no rule and submitted no job. The lingering local
  process was terminated and no Snakemake lock remains.
- Final authoritative destination: 212 files, 8.8 MB.
- AFS user usage changed from 4,445,633 KiB to 4,436,715 KiB; AFS work usage
  changed from 84,825,037 KiB to 84,834,663 KiB.
- `git diff --check` passed. No commit was created.

### Current Git state

- Branch: `task/em-shower`.
- HEAD: `f8a8f7122c6f1dd3369651556fdf2ca82bff9483`.
- Pre-existing task changes were preserved; the generated output directory is
  no longer present in the worktree.

### Exact next action

Wait for explicit user instruction before running or submitting feature,
partition, histogram, or plot production. Do not remove the temporary `/tmp`
recovery copy unless the user asks or a later verified production milestone
makes it unnecessary.

## Checkpoint: requested 1D plot y axes updated

Recorded at 2026-10-06 CEST.

### Changes

- Set both vertical and horizontal leading-cluster hit plots to logarithmic y
  scale with `y_max: 1.0`.
- Set `y_max: 0.3` for vertical and horizontal leading-cluster QDC-per-hit,
  leading-cluster orientation-QDC-fraction, and other-cluster QDC-fraction
  plots.
- Set the combined event-level `scifi_other_clusters_qdc_fraction` plot to
  `y_max: 0.3` for consistency.
- Added a focused configuration regression test covering all nine affected
  feature plots.

### Validation and outputs

- The focused LCG workflow suite passed all 8 tests.
- Regenerated exactly nine four-class PDFs and summaries directly from the
  existing ROOT histograms; no feature, partition, or histogram job ran.
- Outputs are under
  `/afs/cern.ch/work/z/zhibin/snd-ml-outputs/em_shower_neutrino_v1/plots/em_shower_partition_v1_em_shower_regions_v1/em_shower_hist_features_energy_v2__em_shower_evaluation_scifi35_energy_v2`.
- Every regenerated PDF is non-empty and has five pages; every summary is
  non-empty and has eight data rows.
- Parsed YAML validation confirmed the inherited horizontal settings and all
  requested y maxima. `git diff --check` passed.
- No Snakemake job was run or submitted and no commit was created.

### Exact next action

Wait for explicit user instruction. Do not regenerate additional plots or run
histogram/feature production automatically.

## Checkpoint: one-job feature regeneration canary passed

Recorded at 2026-10-06 CEST.

### Exact tested action

- Forced one local feature job for `2024_ve/1` with one core:
  `/afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake --snakefile snakemake/snakefile --cores 1 --force --rerun-incomplete --latency-wait 120 /eos/experiment/sndlhc/users/zhibin/snd-ml/em_shower_neutrino_v1/features/2024_ve/1/feature_scifi_cluster_v1_MC_neutrino_2024_ve_1.root`.
- Snakemake first refreshed the production source manifest because the migrated
  production configuration was newer. The refreshed manifest contains 1,866
  rows: 976 `2024_ve` and 890 `2024_vm`.
- The selected converter wrote 48 of 48 events in about 11.1 seconds and the
  validator passed all 48 entries with schema
  `scifi_orientation_clusters_v1`.
- ROOT audit: 48 entries, 435 branches, and
  `scifi_qdc_per_hit_direct` is present. Its maximum difference from both
  `qdc_scifi / count_scifi` and `scifi_cluster_qdc_per_hit` is zero for this
  canary.
- Output:
  `/eos/experiment/sndlhc/users/zhibin/snd-ml/em_shower_neutrino_v1/features/2024_ve/1/feature_scifi_cluster_v1_MC_neutrino_2024_ve_1.root`.
- Log:
  `.snakemake/log/2026-10-06T160451.249035.snakemake.log`.
- Snakemake emitted a non-fatal candidate-rule `InputFunctionException` for
  the generic `process_features` rule while building the DAG. Rule ordering
  selected `em_shower_mc_neutrino_feature_v1`, and the workflow completed
  successfully.
- `git diff --check` passed. No code was changed, no HTCondor job was
  submitted, and no commit was created.

### Current Git state

- Branch: `task/em-shower`.
- HEAD: `f8a8f7122c6f1dd3369651556fdf2ca82bff9483`.
- Pre-existing task changes remain uncommitted and were preserved.

### Exact next action

Wait for the user to submit the full aggregate target
`em_shower_mc_neutrino_features_v1` through the existing HTCondor profile; do
not submit it automatically.

## Checkpoint: ready for full feature-regeneration dry run

Recorded at 2026-10-06 CEST before stopping.

### Verified findings and decisions

- The local forced canary for `2024_ve/1` completed successfully: 48 of 48
  selected events were written and validated, with 435 branches and schema
  `scifi_orientation_clusters_v1`.
- `scifi_qdc_per_hit_direct` is present and matched both
  `qdc_scifi / count_scifi` and `scifi_cluster_qdc_per_hit` exactly in the
  canary.
- The refreshed source manifest contains 1,866 inputs: 976 `2024_ve` and 890
  `2024_vm`.
- The full production target remains
  `em_shower_mc_neutrino_features_v1`; forcing rule
  `em_shower_mc_neutrino_feature_v1` is the selected method to regenerate
  every feature file.
- The Snakemake invocation must be run from
  `/afs/cern.ch/user/z/zhibin/worktrees/snd-ml-em-shower` because its
  Snakefile and HTCondor profile arguments are repository-relative.
- Rule-defined HTCondor resources are sufficient; no
  `--default-resources` argument is required.
- No production dry run or HTCondor submission was started during this
  checkpoint.

### Modified files in the task-only checkpoint

- Feature and validation code:
  `convertData/FEATURE_DESCRIPTION.md`,
  `convertData/build_region_partitions.py`,
  `convertData/digi_2_features.py`,
  `convertData/scifi_plane_clustering.py`,
  `convertData/test_build_region_partitions.py`,
  `convertData/test_scifi_plane_clustering.py`, and
  `convertData/validate_scifi_cluster_feature_file.py`.
- Evaluation code and tests:
  `evaluation_region_partitions/plot_comparison.py`,
  `evaluation_region_partitions/prepare_em_shower_mc_muon_up_metadata.py`,
  `evaluation_region_partitions/test_em_shower_mc_muon_up_workflow.py`, and
  `evaluation_region_partitions/test_em_shower_neutrino_workflow.py`.
- Workflow/configuration:
  `snakemake/rules/em_shower_neutrino_evaluation.smk` and the six files
  under `snakemake/metadata/configs/em_shower/` named
  `evaluation_options_muon_up_v1.yaml`,
  `evaluation_options_v1.yaml`, `hist_features_v1.yaml`,
  `mc_muon_up_v1.yaml`, `partitions_v1.yaml`, and
  `production_v1.yaml`.
- Continuity record: `docs/tasks/active/em-shower.md`.

### Commands and results

- Successful canary command:
  `/afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake/bin/snakemake --snakefile snakemake/snakefile --cores 1 --force --rerun-incomplete --latency-wait 120 /eos/experiment/sndlhc/users/zhibin/snd-ml/em_shower_neutrino_v1/features/2024_ve/1/feature_scifi_cluster_v1_MC_neutrino_2024_ve_1.root`.
  It completed in about 11.1 seconds and its validator passed all 48 entries.
- Canary log:
  `.snakemake/log/2026-10-06T160451.249035.snakemake.log`.
- `condor_q zhibin -af ClusterId ProcId JobStatus JobBatchName` returned no
  jobs.
- `pgrep -af '[s]nakemake'` returned no active Snakemake process.
- `git diff --check` passed.

### Current Git state before the checkpoint commit

- Branch: `task/em-shower`.
- Starting HEAD: `f8a8f7122c6f1dd3369651556fdf2ca82bff9483`.
- All 19 changed or untracked paths listed above belong to this task; no
  unrelated work was found.
- Running job IDs: none.

### Exact next action

From `/afs/cern.ch/user/z/zhibin/worktrees/snd-ml-em-shower`, run the
previously supplied Snakemake command with `--dry-run`,
`--forcerun em_shower_mc_neutrino_feature_v1`, and aggregate target
`em_shower_mc_neutrino_features_v1`. Do not submit the non-dry-run production
until that dry run has been reviewed.
