# Task: em-shower

Status: active
Last updated: 2026-10-05 18:29 Europe/Zurich

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

Wait for explicit user instruction to submit the versioned
`em_shower_mc_neutrino_features_v1` target from the `task/em-shower` worktree
via Snakemake/HTCondor. Do not submit jobs yet.
