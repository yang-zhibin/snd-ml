# Code and scientific analysis review

## A. Review metadata

- Review date: 2026-10-22 (Europe/Zurich).
- Branch: `task/unified-shower-features`.
- Reviewed HEAD: `42e0f09582533437b5b2a2532ca0b860d93b027d`.
- Comparison base: `main` merge-base `8986cb8dc14ab2a500c3414671d3369c20aec36b`. This is an old common ancestor (993 changed paths), so the review also examines current code and the branch's integration history rather than treating every diff as new task work.
- Working tree at start: 5 tracked modifications and 7 untracked implementation/config/test files, plus this newly created report. Uncommitted changes are in scope; they belong to the pre-existing task work and will not be changed by this review.
- Scope: the current branch's maintained conversion, orchestration, region/statistical analysis, data-driven dataset, and training/evaluation workflows, plus legacy code where it remains on an active execution path. Generated binary products and vendored dependencies are outside source review except as interface evidence.
- Overall status: **IN PROGRESS**.

## B. Project overview

The project processes SND@LHC detector and simulated events for feature construction, control/signal-region physics studies, and machine-learning classification. Digi ROOT files, geometry, cutflow/index products, and metadata catalogues feed Snakemake workflows. Conversion writes feature ROOT (`sndData`) and Hit3D products; downstream code builds region partitions and distributions or five-fold ML datasets. Scientific inference depends on correct event selection, truth/domain provenance, source exposure/weights, control-to-signal transfer, and uncertainty treatment. Feature/Hit3D identity is `(runId, eventId, eventIndex)`, with entry alignment preserved or checked explicitly. Runtime stacks are distinct: LCG for metadata/ROOT analysis, a rule-specific SNDSW release for detector conversion, and a validated PyTorch environment/container for ML. Long production uses Snakemake and HTCondor.

Current task-specific work adds a unified SciFi shower schema, source manifests and validation receipts, separate data/MC distributions, and downstream histogram bundles. Region V1 ABCD and other partition analyses share the branch. The branch is under active development; a successful dry-run is evidence for DAG construction, not correctness of produced physics outputs.

## C. Review plan and coverage

| Stage | Inspection focus | Status |
| --- | --- | --- |
| 0 | Repository state, authoritative task/project documentation, branch topology, workflow inventory, review report | Complete |
| 1 | Unified conversion, source selection/provenance, feature/Hit3D identity, manifests, validators | In progress |
| 2 | Snakemake DAG, checkpoints, batching, paths, completion/validation contracts | In progress: unified feature and real-data distribution rules inspected |
| 3 | Testbeam and real-data histogram bundles, plots, normalization, coverage and cohort selection | In progress: V2/V4 bundle and plot paths inspected |
| 4 | Region partitions, ABCD/statistical inference, uncertainty and transfer-factor propagation | In progress: one-bin likelihood and observed entry point inspected |
| 5 | Data-driven dataset folds, labels, weights, training/evaluation metrics and leakage | In progress: fold sampler and ROOT reader inspected |
| 6 | Cross-component interfaces and representative safe validation checks | Pending |
| 7 | Consolidate findings, severity, limitations, and final review status | Pending |

Review order follows source identity and selection into downstream distributions and statistical/ML consumers. Each stage will record inspected files, execution paths, validated findings, unverified questions, and a next task before moving on. The 993-path diff against the old `main` ancestor includes legacy and generated files; active execution paths take priority, while skipped paths will be named in limitations.

## D. Findings summary

| ID | Title | Severity | Component | Confidence | Verification | Resolution |
| --- | --- | --- | --- | --- | --- | --- |
| CR-001 | Accepted lowercase MC type bypasses event metadata and MC hit policy | P2 | Feature converter | High | Confirmed conditional path | Open |
| CR-002 | Downstream snapshots accept feature files without checking their validation checksum | P2 | Snapshot and histogram inputs | High | Confirmed conditional path | Open |
| CR-003 | Coverage gate accepts same-count snapshots with unapproved sources | P2 | Snapshot coverage | High | Confirmed conditional path | Open |
| CR-004 | Region partition friend-tree path does not enforce declared event identity | P2 | Region partitions | High | Confirmed conditional path | Open |
| CR-005 | Final feature coverage omits selection-stage failures from its denominator | P2 | Real-data coverage | High | Confirmed conditional path | Open |
| CR-006 | ABCD boundary regression requires exact floating-point zero | P3 | Statistical regression test | High | Reproduced twice | Open |

Counts: P0 0; P1 0; P2 5; P3 1. These counts reflect completed inspection only.

## E. Detailed findings

### CR-001 — Accepted lowercase MC type bypasses event metadata and MC hit policy

- **Severity:** P2.
- **Location:** `convertData/source_provenance.py:51-59`; `convertData/digi_2_features.py:235-246`, `:1767-1822`.
- **Description:** Source-option validation lowercases the sample type and accepts `mc_neutrino`, `mc_muon`, or `mc_testbeam` for their domain IDs. The converter later dispatches MC events with the case-sensitive test `'MC' in args.type`, and `is_mc_sample` uses the same test. For an accepted lowercase MC type, neither the MC nor real-data event metadata branch runs.
- **Impact:** A direct converter invocation using the accepted lowercase spelling writes entries with reset sentinel values for `isMC`, `eventId`, incoming truth, and `source_energy_gev`/`source_pdg_code`, and skips MC-specific hit-time policy. Current production manifests spell their MC types `MC_*`, so the traced Snakemake path does not trigger this issue. A subsequent unified feature validator should reject at least the source sentinels, but conversion itself does not fail or signal the mismatch.
- **Evidence:** Read-only LCG reproduction: `validate_source_options(SimpleNamespace(type="mc_neutrino", source_domain_id=1, source_pdg_code=None, source_energy_gev=None, source_uid="example"))` returned successfully; the event-loop predicates then evaluate `"MC" in "mc_neutrino" == False` and `"real" in "mc_neutrino" == False`. `snakemake/metadata/configs/unified_shower_features_v1.yaml` currently uses uppercase `MC_neutrino`, `MC_muon`, and `MC_testbeam`.
- **Recommended fix:** Normalize sample type once in the converter and dispatch on explicit normalized values; use the same normalized predicate for MC hit filtering and reject any type not handled by the event loop. Add a targeted test for both accepted spellings and source scalar values.
- **Confidence:** High. **Verification:** Confirmed conditional path by predicate reproduction and code trace; no ROOT product was generated.

### CR-002 — Downstream snapshots accept feature files without checking their validation checksum

- **Severity:** P2.
- **Location:** `snakemake/metadata/build_unified_partial_feature_snapshots.py:34-63,90-124`; `evaluation_region_partitions/unified_feature_distributions.py:229-258`; `convertData/validate_unified_shower_feature_file.py:703-724`.
- **Description:** The validator records `feature_sha256` and `feature_size_bytes` for the validated file. Snapshot construction accepts a canonical file or shard if the feature and marker exist and the marker says it passed with the matching source UID and path; it never compares either recorded digest or size to the file. The shared histogram loader repeats the same status/UID/path checks, and the V2 bundle builder reads those rows directly. The coverage validator hashes source and snapshot manifests, not feature contents.
- **Impact:** If a feature ROOT file is replaced or altered after validation while retaining its path and marker, the partial snapshot still declares it available and downstream histograms can silently use contents different from those validated. The inspected current path does not establish that this has happened to existing products.
- **Evidence:** End-to-end static trace: `validate_unified_shower_feature_file.py` writes the digest at lines 703 and 724; no `feature_sha256` reference occurs in the snapshot builder, coverage validator, shared histogram loader, or V2 histogram bundle path. The bundle's `TChain.Add` checks readability, not digest identity. No product was modified for reproduction.
- **Recommended fix:** Compare the published file's SHA-256 (and optionally byte size) to its validation receipt when freezing snapshots, including each shard. Keep a digest in the frozen snapshot or verify again immediately before histogram construction; add a targeted stale/replaced-file rejection test.
- **Confidence:** High. **Verification:** Confirmed conditional path by complete receipt-to-bundle code trace; actual file tampering or product mismatch has not been observed.

### CR-003 — Coverage gate accepts same-count snapshots with unapproved sources

- **Severity:** P2.
- **Location:** `evaluation_region_partitions/validate_unified_feature_snapshot_coverage.py:21-27,50-99`; `snakemake/unified_real_data_distributions_v2.smk:132-193`.
- **Description:** The coverage gate checks only the number of unique snapshot source UIDs against `available_source_count`. It verifies the checksum of the source manifest but never compares snapshot UIDs to its rows or to per-source availability records. It hashes the snapshot *after* accepting it, without an expected snapshot digest in the coverage input. The V2 histogram workflow then passes this snapshot directly to the bundle loader.
- **Impact:** A stale or substituted snapshot with the same unique-source count can pass coverage even if it omits an approved available source and includes a different source. If the substituted row points to an independently valid feature and receipt, histogram production proceeds with the wrong cohort, silently changing scientific shapes or counts. Current snapshots were checked and do not exhibit this mismatch.
- **Evidence:** Static execution trace through the gate and V2 Snakemake inputs. A read-only audit of current frozen No-Veto and Veto-10% snapshots found 1,109 unique UIDs and zero UIDs outside their respective 11,151/11,150-row source manifests; neither snapshot had duplicate feature-unit UIDs. This confirms current membership, not gate enforcement. No files were modified for a tampering test.
- **Recommended fix:** Require every snapshot UID to exist in the checksummed source manifest, verify expected available UID membership or a recorded snapshot digest from snapshot creation, and check per-source shard unit uniqueness/ranges before issuing the coverage receipt. Add a same-count swapped-UID rejection test.
- **Confidence:** High. **Verification:** Confirmed conditional path by gate-to-bundle trace and current-data membership audit; no incorrect current product observed.

### CR-004 — Region partition friend-tree path does not enforce declared event identity

- **Severity:** P2.
- **Location:** `snakemake/metadata/configs/partitions/partition_v1.yaml:13-18,25-36,92-94`; `convertData/build_region_partitions.py:574-585,790-810,905-915`; `snakemake/rules/region_partitions.smk:1408-1439`.
- **Description:** The frozen partition config declares entrywise alignment on `(runId,eventId,eventIndex)`, `require_same_entries: true`, and `fail_on_mismatch: true`. The builder never reads that `alignment` block. For Hit3D output it joins the feature tree as a ROOT friend by entry number and filters with feature values. Its optional preflight compares only entry counts, and the current config sets `check_entry_counts: false`.
- **Impact:** If same-length source feature and Hit3D files have different entry order or identity, the Hit3D partition can silently inherit region and particle selection from a different event. This could corrupt labels/regions for a manually enabled Hit3D partition. Current `partition_v1.yaml` has `products.hit3d.enabled: false`, so the standard Region V1 aggregate does not take this path; a current mismatched product has not been observed.
- **Evidence:** Read-only trace of the config, builder, and Hit3D rule. `check_entry_counts` compares only `GetEntries()`; `hit3d_chain.AddFriend(feature_chain, "feat")` uses ordinal alignment; no source identity comparison or post-snapshot identity verification appears on this path. The maintained project invariant requires this check where aligned products are materialized.
- **Recommended fix:** Before friend-based filtering, compare the three identity fields for every source entry and fail on any mismatch; also validate output feature/Hit3D identities entrywise when both products are requested. Make the declared alignment block drive these checks, with a same-count permuted-input regression test.
- **Confidence:** High. **Verification:** Confirmed missing guard and conditional failure path by code/config trace; no active mismatched file demonstrated.

### CR-005 — Final feature coverage omits selection-stage failures from its denominator

- **Severity:** P2.
- **Location:** `snakemake/metadata/prepare_unified_real_data_feature_manifests.py:45-59,190-242`; `evaluation_region_partitions/validate_unified_feature_snapshot_coverage.py:50-99`; `snakemake/unified_real_data_distributions_v2.smk:55-65,132-158`.
- **Description:** The feature-manifest builder drops a selection source with a missing or failed selection receipt and records the reason only in `source_coverage.json`. The downstream V2 coverage gate receives the already-filtered source manifests and feature coverage, but no selection manifest or `source_coverage.json`. Its `complete` policy can therefore certify 100% of the reduced source manifest even when eligible upstream sources never entered it.
- **Impact:** A nominally complete real-data feature/plot run can omit selection-stage sources without failing the final coverage gate. If missing sources are nonrandom by run or detector conditions, counts and shapes can be biased. Current V2 plotting is explicitly labeled partial, and no complete V2 result has been claimed.
- **Evidence:** Current `source_coverage.json` reports 11,368 selection sources, 11,359 available, and 9 `missing_selection_validation`. The two nonempty feature source manifests contain 11,151 and 11,150 rows after selection validation and zero-selected exclusions. The V2 workflow references feature coverage and snapshots but not source coverage. This is an observed upstream omission and confirmed conditional final-gate gap; individual absent receipts were not re-audited.
- **Recommended fix:** Make the downstream coverage receipt consume the selection-stage coverage and require all eligible selection sources to be validated for a `complete` result, with explicit handling of deliberately excluded/zero-selected sources. Include source-stage status counts and selection-manifest checksum in the final plot manifest.
- **Confidence:** High. **Verification:** Confirmed by current coverage artifact and end-to-end dependency trace; no bias in an existing plot has been measured.

### CR-006 — ABCD boundary regression requires exact floating-point zero

- **Severity:** P3. **Location:** `evaluation_region_partitions/test_abcd_blinded_expected.py:188`.
- **Description:** The synthetic downward-fluctuation test asserts `downward_q0 == 0.0` after numerical optimization. In the documented LCG runtime (pyhf 0.7.6), two independent runs gave q0 = 1.646599017135486e-07 and fitted mu = 2.347799140310793e-12, failing this assertion before the remainder of the inference checks could execute.
- **Impact:** A numerically negligible boundary residual stops the regression suite. This is not evidence of a meaningful false physics significance.
- **Recommended fix:** Use a justified numerical tolerance for q0 together with the existing fitted-mu boundary check, or explicitly implement and test a consistent numerical boundary convention in the statistic helper. Preserve the test for substantive nonzero q0.
- **Confidence:** High. **Verification:** Reproduced with the original unmodified test and a separate synthetic invocation. A tolerance-adjusted in-memory diagnostic is recorded separately below; it does not count as an original-test pass.

## F. Validation and testing

- Read-only LCG preflight on `lxplus978.cern.ch`: Python 3.11.9, ROOT 6.34.02, NumPy 1.26.4, pandas 2.2.2, PyYAML 6.0.1. The lowercase-MC predicate reproduction passed; no conversion or product was written.
- Current real-data partial snapshots contain 1,109 source UIDs per cohort and share 1,108. Both one-sided source UIDs appear in the opposite source manifest, indicating availability rather than zero-selection caused the mismatch. This remains an interpretation limitation, not a confirmed defect.
- In the inspected ABCD path, histogram overflow is folded into the last bin, matching the downstream high-end integration. No ABCD defect is confirmed yet.

- The partial no-Veto/Veto snapshots cover roughly 10% of source manifests and differ by one source in each direction. Quantify run/exposure balance and plot sensitivity before interpreting their shape difference as a full-2024 conclusion.
- The current frozen source manifests contain 11,151 No-Veto and 11,150 Veto-10% rows. Their snapshots contain 1,109 unique sources each, with no out-of-manifest UIDs or duplicate feature units. A checksum spot-check of the first feature file in each snapshot matched both receipt SHA-256 and byte size (226,718 and 8,425,550 bytes); this samples only two files.
- Read `AGENTS.md`, `docs/project/OVERVIEW.md`, `docs/project/CURRENT_STATUS.md`, and `docs/project/ENVIRONMENT_AND_SNAKEMAKE.md`.
- Inspected `git status --short`, branch/HEAD, branch list, merge-base, file inventory, and diff statistics. At that initial checkpoint no tests or production commands had run; the later execution results are in section J.
- Existing task record reports a successful 2026-10-22 dry-run of `unified_real_data_distribution_features_v1`, resolving 22,301 feature rules and one aggregate. This is prior evidence, not a test performed in this review.
- A read-only ROOT schema check of the first validated file in each current real-data snapshot found 438 branches, all 37 branches required by the V2 histogram definitions, and 77/3,811 entries respectively. This is a two-file spot-check, not full cohort validation.
- The frozen ABCD config's unblinding authorization remains `authorized: false`; no observed signal-region A data or inference outputs were opened in this review.
- Representative V4 testbeam ROOT schema checks under LCG: the frozen MC snapshot has 75 feature units, including five 100-GeV positive-pion sources; its first group file has 600 events and all 36 required branches. The frozen data snapshot has 22 feature units (including shards); its 100-GeV electron group has one source, 50,000 events, and all 36 required branches. Each checked file has 438 branches. This does not validate all files or values.
- The V4 testbeam bundle has explicit deterministic chain-prefix event caps, ROOT flow closure for each TH1, and a 37-object bundle validator. Its two plot cohorts separate data and MC; the data cohort labels its H8 Fe-pion versus H4 W-electron comparison as cross-geometry diagnostic. A chain-prefix sample is potentially order-sensitive, so the plots should not be interpreted as random event samples.
- The ML input path uses fold-specific training normalization for the configured model family. The shared reader selects SciFi detector type 1, rejects invalid/nonfinite hit vectors and negative QDC, and keeps truth/energy as metadata outside model input. Five-fold training samples with replacement; natural validation enumerates held-out events. These are code-path checks, not a trained-model validation.
- A fresh `condor_q zhibin -af ClusterId ProcId JobStatus JobBatchName Cmd Args` returned no queued/running jobs. HEAD remains `42e0f09582533437b5b2a2532ca0b860d93b027d`; no new job or product was started.
- Existing one-bin ABCD expected output under `/afs/cern.ch/work/z/zhibin/snd-ml/evaluation_region_partitions/abcd/partition_v1_region_v1/density_sndsw_scifi_v1/blinded_counting_v2/` has `freeze_status: FROZEN`, a clean Git status in its historical freeze record, and `signal_region_observation_used: false`; its expected result says `PROVISIONAL_EXPECTED_ONLY` and `REBLINDED_DEVELOPMENT`. It reports a provisional kappa of about 0.399, expected asymptotic Z of about 0.066, and background-Asimov upper limit on mu of about 6.96. These are historical expected outputs, not a current validated physics conclusion; the config explicitly labels kappa transport unvalidated and MC normalization uncertainties placeholders. The current authorization remains false, and observed A data were not opened.
- The ABCD code path fits B/C/D with mu fixed to zero for a conditional prediction, builds mu=0 and mu=1 Asimov A+B+C+D workspaces, then uses a simultaneous fit with all nuisances profiled for the guarded observed path. The current profile interval uses the one-parameter 68% threshold, matching the configured 0.68 level; other confidence levels are not implemented by the helper.
- The feature validator checks every feature entry against selected raw-entry indices in uncapped production, checks duplicate `(runId,eventId,eventIndex)`, validates the source domain and scalar provenance, and compares Hit3D identity entrywise when a Hit3D file is supplied. The unified real-data feature rule does not supply Hit3D, consistent with its feature-only product; this does not repair the separate Region V1 friend-tree gap in CR-004.
- The five-fold ML view builder assigns exactly the held-out physical fold to validation in each CV view; the reader carries `dataset_file_uid`, physical source entry, and available event identities to predictions. Evaluation merges additive confusion/loss statistics and demands a complete contiguous shard-index sequence. These data-driven ML inputs use their own legacy feature/Hit3D catalog and are not a downstream consumer of the new `feature_scifi_cluster_v1_path` products.
- External ML dataset diagnostics under `/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/ml_dataset/data_driven_3d_cnn_v1/` report `valid: true`, 11,291 logical rows, 7 training classes, 7,291 fold groups, and 56,455 five-fold view rows. An independent read-only pandas audit found zero duplicate catalog file UIDs, zero fold groups assigned to multiple folds, zero duplicate `(cv_fold,dataset_file_uid)` rows, zero wrong validation assignments, and zero train/validation fold-group overlaps in all five views. Validation event totals range from 8.14M to 9.66M; these are natural fold populations, not model performance. The diagnostics warn that fold 0 has no independent MC passing-muon or muon-DIS source group and that high-density MC training bins may repeat events 76-166 times under balanced sampling. No trained-model calibration or scored predictions were audited.

## G. Unverified issues and limitations

- Only the baseline inventory stage is complete; correctness of the full branch is not yet established.
- ROOT products and raw events have been spot-checked as recorded below; a full production-data scan and historical HTCondor log audit remain unperformed.
- The comparison base is historically distant and contains generated/legacy material. Active call paths will determine relevant coverage.
- The new unified feature distribution workflows use `feature_scifi_cluster_v1_path`; the established Region V1 ABCD partitioner still reads the legacy `feature_path` column. The current review therefore treats these as separate downstream branches until an explicit integration path is verified.
- Stage 5 has inspected dataset rules, fold sampling and construction, density/entry index production, dataset validation, external fold diagnostics, the ROOT-backed reader, model-specific collation, training confusion-matrix metrics, and shard evaluation/merge. Full training execution and calibration remain unreviewed.

## H. Prioritized action plan

Fix CR-001 with normalized, exhaustive sample-type dispatch and a lowercase-MC regression test. Bind snapshots and histogram inputs to the validated feature-file digest (CR-002). Require source membership and frozen snapshot identity in the coverage gate (CR-003). Enforce the declared three-field event alignment before friend-tree Region V1 Hit3D filtering (CR-004). Carry selection-stage completeness into final real-data coverage (CR-005). Continue science-critical selection/identity and normalization inspection, then statistical and ML consumers. Do not make implementation changes as part of this inspection-only review.

## I. Review progress and continuation notes

- Last completed stage: 0. Stages 1, 2, 3, 4, and 5 are partial and remain open.
- Inspected: project guidance and Git baseline; real-data selection/index producers and validator; unified converter selection, provenance, feature calculations, identity and validator; source-manifest/snapshot builders; unified feature and real-data V1/V2 Snakemake rules; V2/V4 bundle and plot paths; ABCD one-bin model, observed entry point, blinded-input builder, closure/fit entry points and histogram overflow; data-driven dataset rules, fold and index builders, dataset validator, sampler, and ROOT-backed reader. Coverage is partial except for smaller files; no stage beyond inventory is declared complete.
- Pending: remaining focused tests and full orchestration; full-cohort histogram value checks and distribution interpretation; complete ABCD statistical outputs and current-product checks; full ML training/evaluation execution; cross-component review. Representative testbeam branch schema and model-specific collation have now been inspected, but do not close their stages.
- Unresolved investigations: partial-snapshot cohort balance and shape sensitivity; per-source reason for the nine absent selection receipts; remaining source/Hit3D identity conditions; ABCD transport assumptions; ML training/evaluation population closure.
- Exact next task: finish cross-component cohort/identity reconciliation and review severity against the current production paths. Then decide whether remaining unverified execution paths require another review session. Keep the review inspection-only.

## J. Requested preventive checks — 2026-10-22

Scope: user explicitly requested execution of the recommended checks. Source/config/test files remain unchanged. Fixtures and test products were confined to automatically cleaned temporary directories; no production conversions or scheduler jobs were launched. Observed production signal-region A data were not read.

Runtime: lxplus929.cern.ch; LCG 107 Python 3.11.9, ROOT 6.34.02, NumPy 1.26.4, pandas 2.2.2, pyhf 0.7.6. Raw detector spot-check used `env_sndsw_nue.sh` and the real-data rule's explicit `/cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python`, reporting ROOT 6.36.04.

### Existing regression results

After activation, with `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, and `convertData`, `evaluation_region_partitions`, `snakemake/metadata` on PYTHONPATH:

```text
python3 -m unittest -q test_real_data_2024_selection test_scifi_plane_clustering test_unified_shower_features test_validate_unified_shower_feature_shards test_build_region_partitions test_unified_partial_feature_snapshots test_unified_feature_distributions test_unified_feature_distributions_root test_unified_real_data_histogram_bundles test_unified_testbeam_histogram_bundles
```

Result: **54 tests passed in 15.248 seconds**. These include deterministic sampling vectors, empty/zero-QDC and channel-boundary clustering, independent numerical feature expectations, provenance, selected-entry ordering, shard merging/completeness, histogram construction and plotting. Two additional existing manifest tests were invoked with isolated temporary directories and passed: smallest-canary selection and rejection of an index checksum mismatch.

### Deliberately invalid fixture results

| Check | Observed result | Interpretation |
| --- | --- | --- |
| Valid feature receipt baseline | Accepted | Baseline works |
| Change feature bytes and size while retaining old digest receipt | Accepted | CR-002 dynamically confirmed |
| Remove feature receipt | Rejected as missing_validation_record | Guard works |
| Valid source/snapshot baseline | Accepted | Baseline works |
| Substitute unapproved snapshot UID with unchanged unique count | Accepted | CR-003 dynamically confirmed |
| Change checksummed source manifest | Rejected | Guard works |
| Reverse Hit3D identities while preserving two-entry count | Count preflight accepted | CR-004 dynamically confirmed for the count-only guard |
| Remove one of two upstream selection receipts | Upstream records one missing source; downstream complete gate accepts reduced one-source manifest | CR-005 dynamically confirmed |
| Fill ROOT TH1 at underflow, lower edge, interior edge, upper edge with weights 2,3,4,5 | Bin contents [2,3,4,5], sumw=14, sumw2=54 | Expected ROOT flow and weight accounting confirmed |

The invalid-file checks used synthetic temporary bytes/trees, never production corruption. The Hit3D check exercises the actual preflight; it does not run full region partition production.

### Existing-data spot-checks

- First current real-data snapshot file per cohort: all 77 No-Veto and 3,811 Veto-10% entries match their selection index order exactly, have unique three-field identities, satisfy their veto condition, and reproduce direct QDC/hit from qdc_scifi/count_scifi for positive hit counts. All Veto entries also satisfy an independently computed SHA-256 bucket-zero selection.
- Five raw events per cohort: raw event-header run/event IDs match feature identities at the recorded raw entry, and independently scanning valid MuFilter system-1 hits reproduces the feature veto classification. This checks 10 raw events, not detector-level recomputation of every shower feature.
- ML catalogue/view audit repeated: 11,291 unique logical files, 56,455 CV view rows; every view has exactly the catalogue UID set, no duplicated (CV fold, file UID), no physical group assigned multiple folds, no train/validation group overlap, and all validation rows belong to the held-out fold. Training-only normalization reduction was verified in code; trained-model prediction closure and calibration remain unexecuted.
- Fresh scheduler query returned no jobs. The original five tracked modifications and seven implementation/config/test untracked paths were preserved.

### ABCD regression status

The original `python3 evaluation_region_partitions/test_abcd_blinded_expected.py` failed at the exact-zero boundary assertion (CR-006). Independent diagnosis reproduced q0=1.646599017135486e-07 and mu=2.347799140310793e-12. A diagnostic execution with only this assertion relaxed in memory to `0 <= q0 < 1e-6` passed all remaining synthetic expected/observed inference, blinding and authorization assertions; no test file was edited. The optimizer emitted bound-clipping and covariance-repair warnings, so this diagnostic pass does not establish fit robustness or calibrated interval coverage. All observed values in this synthetic test are invented fixtures, not production data.

These checks strengthen the evidence but do not certify the full production cohort, detector geometry/calibration, ABCD transport assumption, or trained classifier performance.

## K. Concrete fix design — 2026-11-04

Re-read the report and affected current code at HEAD `42e0f09582533437b5b2a2532ca0b860d93b027d`. The six recorded defects remain present. This section specifies proposed implementation and acceptance tests; it does not mark a finding resolved. No source changes, tests, or production jobs were performed during this fix-design pass. Section J remains the dated execution evidence.

### CR-001: use one normalized sample-type policy throughout conversion

Reuse `sample_selection.normalized_sample_type` instead of adding another normalization convention. Define explicit MC/data classification for supported types and use it consistently in provenance validation, event metadata dispatch, hit timing/filtering and the `selectHits` MC argument. Replace every case-sensitive MC/real dispatch on the active converter path, including the paths around lines 1234-1242 and 1767-1814; changing only `is_mc_sample` is insufficient. Validate the type before opening output files. Preserve supported legacy domain-0 types (including muon-DIS and passing-muon data) rather than narrowing the accepted set to unified production domains. Inspect any converter wrapper which calls these helpers.

Acceptance: uppercase/lowercase and whitespace variants produce the same MC classification, event identity, incoming/source truth and hit-selection policy; neutrino and non-neutrino timing distinctions remain unchanged; unsupported types fail before writing. Production's canonical uppercase spelling must retain its current behavior. Full conversion comparison belongs to a separately authorized small canary after unit checks.

### CR-002: bind every consumed feature unit to its validated bytes

Add a shared verification helper used by snapshot canonical/shard inspection and `unified_feature_distributions.load_validated_rows`. Require a well-formed SHA-256 and byte size in the receipt; compare size first, then stream-hash the actual file. Copy the validated digest/size and receipt digest into each frozen snapshot unit. Before histogram construction, verify the file against the frozen digest, not merely a newly replaced receipt. Digest failures must abort consumers even when missing-source handling is enabled; they must not silently become allowed omissions. Review `_check_row` fallback so a corrupt canonical product is not hidden by accepting shards without an explicit recorded policy.

Acceptance: changed-size and same-size byte replacement, missing digest, malformed receipt, replaced receipt and one corrupt shard fail; valid canonical and complete shard inputs pass. Update existing synthetic fixtures that currently use empty files and receipts without checksums. Use streaming reads and bounded workers to control EOS I/O; do not substitute mtime/size for the digest. Verification immediately before use assumes published files are immutable during the read; stronger enforcement requires reading a verified immutable/staged copy.

### CR-003: validate exact snapshot identity and membership

At snapshot publication, store the finalized CSV SHA-256 in the coverage record and record expected available source UIDs (or a checksummed per-source availability table). At the gate, verify that stored digest before accepting the snapshot. Parse both source and snapshot manifests: require correct family, unique source-manifest UIDs, snapshot UIDs contained in the source universe, and exact equality with the recorded available set. For complete policy, require equality with the full eligible nonempty source set. Check unique feature-unit IDs and paths, canonical/shard exclusivity, contiguous shard ranges and total selected-entry coverage; repeated source UIDs are valid for distinct shards and must not be rejected indiscriminately. Declare all consumed manifests as Snakemake inputs so changes invalidate the receipt.

Acceptance: swap an approved or unapproved UID without changing counts, alter a feature path, duplicate a unit, mix canonical and shard forms, overlap or omit a shard, or change the CSV after freezing: each must fail. A valid partial snapshot and a complete sharded snapshot must pass. Old coverage records lack the expected snapshot hash and require an explicit validated refresh, not a silent permissive default. A hash computed only after acceptance is insufficient.

### CR-004: enforce source-pair alignment before ROOT friend filtering

In `build_region_partitions.py`, read the declared alignment contract and require all identity branches, equal per-file entry counts and entrywise equality of `(runId,eventId,eventIndex)` before `AddFriend` whenever Hit3D output is requested. Stream only the identity columns in bounded chunks. Report source paths, first mismatching entry and both identities. The old optional count-only flag must not bypass identity verification on this path. Preserve source-pair order when building chains; repeated identifiers across different source files are allowed by the config. When producing both outputs, verify their identities again before publishing completed products; ensure multithreaded snapshot ordering cannot invalidate the output contract.

Acceptance: a valid pair passes; equal-count permutation, one wrong identity, missing identity branch and unequal count fail before output publication. Feature-only operation remains usable without Hit3D. A tiny synthetic full partition test should assert the exact selected output identities, beyond testing the preflight alone. Keep the currently disabled Hit3D aggregate disabled unless separately requested.

### CR-005: carry selection-stage accounting through the final coverage gate

Have the source-manifest builder consume the selection policy/exclusion configuration and emit checksummed per-source status records, including explicit exclusions and valid zero-selected sources. Pass that selection coverage, selection manifest and relevant config into the downstream gate as declared Snakemake inputs. Require every eligible source to have successful selection validation for complete mode; valid zero-selected sources satisfy selection completeness but need no feature product. Require complete features for every eligible nonempty source. Reconcile identities, not only totals, at each stage. Embed both stage denominators, exclusions and failures in the final plot manifest.

The current selection config explicitly excludes nine UIDs, while the source coverage currently calls missing receipts failures. The repair must preserve this documented exclusion policy and distinguish 'complete for eligible sources' from 'complete for all 11,368 original sources'. Verify the actual excluded UID set against missing records when implementing; do not assume equality merely because both counts are nine. Retain each exclusion's reason and policy digest. In partial mode, allow only explicitly configured missing conditions and label both stages partial; invalid receipts/checksums remain fatal.

Acceptance: an unexpected missing/failed selection receipt prevents complete certification even when all remaining features exist. A documented exclusion and a validated zero-selected source are accounted for correctly. Changed exclusion policy or source membership invalidates the receipt. The existing two-source adversarial example must no longer pass as complete.

### CR-006: make the regression assertion numerically meaningful

Change the test's exact equality to a finite, nonnegative absolute bound, initially `0.0 <= downward_q0 < 1e-6`, retaining the independent near-zero fitted-mu assertion. Document that the observed residual was 1.65e-7 and that this tolerance is a numerical regression threshold, not an inference-accuracy guarantee. Add/retain a signal-excess case with clearly positive q0 so the test cannot pass if the statistic is always zero. Reject non-finite or substantively nonzero downward results.

Run the original test unmodified by the harness after the source assertion is actually repaired, and repeat under the documented runtime to check stability. Avoid altering production q0, clipping all small fitted signals, loosening optimizer tolerances or suppressing covariance warnings merely to pass this test. The covariance/optimizer warnings seen in the prior diagnostic require separate convergence and coverage study before physics interpretation.

### Implementation order and product handling

Implement CR-002/003/005 as a coordinated provenance/coverage change, then CR-004, CR-001 and CR-006 (the last two can be small isolated patches). Run targeted positive and negative tests for each, then the existing focused regressions. Include any new shared helper in the relevant Snakemake input dependencies. After separately authorized deployment, refresh validation/snapshot/coverage artifacts under the new contract. Do not regenerate all feature ROOT files by default: matching files can be verified and retained, while failures require investigation. No automatic migration should replace a receipt's old digest with the current file's digest without revalidation.

## L. Inspected implementation plan — 2026-11-04

Requested scope: inspect each fix, test where needed, and retain an implementation plan. This section refines K; no finding is closed and no implementation is applied. Only this report is edited. Existing workspace changes are preserved. No production conversion, scheduler submission, deployment, or commit is part of this work. The focused regressions and adversarial tests in J remain historical evidence, not tests rerun today.

### 1. CR-001 — normalize classification without changing selection physics

Files: `convertData/sample_selection.py`, `source_provenance.py`, `digi_2_features.py`, and the `digi_2_features_mc_muon.py` wrapper; corresponding selection/provenance tests.

Implementation steps:
1. Inventory the supported CLI types from active callers and legacy domain-0 consumers; include `MC_ve`, `mc_muondis`, and `passing_muon_real_data`. Centralize normalization/classification in the existing ROOT-free `sample_selection.py`. Reject unsupported types before any output is opened, after the compatibility inventory is complete.
2. Use that classification for provenance, event metadata dispatch, real-data veto/US rejection, `is_mc_sample`, and the `selectHits` MC argument. Audit all dispatch sites, not just the helper.
3. Preserve the current neutrino timing exception: `applies_mc_25ns_hit_filter` also examines input/output paths for `mc_neutrino`. Replacing it with a type-only test would change legacy behavior and is outside this repair. Preserve muon calibration and FairEventHeader compatibility.
4. Add case/whitespace parameterized tests for every supported family and legacy alias, including domain-0 compatibility and invalid-domain/type combinations. Assert identical identity/truth dispatch and hit policy, not only classification booleans. Test the wrapper's lowercase accepted type through the common policy.

Completion evidence: focused unit tests plus a separately authorized small SNDSW conversion comparison for canonical versus normalized spelling, checking ordered identities and feature values. Pure helper tests alone cannot establish detector conversion parity.

### 2. CR-002 — verify file content at publication and consumption

Files: partial-snapshot builder, `unified_feature_distributions.py`, shard validator/merger, their tests, and relevant workflow inputs.

Implementation steps:
1. Extract a ROOT-free digest/receipt helper into a shared import location supported by both metadata and evaluation entry points (a proposed location is `convertData/feature_integrity.py`, with explicit PYTHONPATH/wrapper support). The existing shard merger already verifies SHA-256, but importing that module introduces ROOT; do not import it into the lightweight gate just to reuse hashing.
2. Require receipt identity/path, digest syntax, and integer nonnegative size; verify size then streaming SHA-256. Preserve existing shard receipt invariants. The current shard-completeness record includes digest but not byte size: read the individual validated receipt and carry its size, or version the completeness schema to include it. Never infer an expected size from the untrusted current file.
3. Freeze each unit's source/family, path, selected range, file digest/size, receipt path and receipt digest into the snapshot contract. At consumption compare against that frozen contract as well as validating the current receipt.
4. Distinguish absent canonical output from an invalid existing canonical output. Allow shard fallback only under a stated policy; checksum/identity failures must not be hidden by fallback or `allow_missing_validated_sources`.
5. Add the helper as a declared workflow dependency. Stream with bounded concurrency; document the immutable-product assumption during verification and reading.

Acceptance matrix: valid canonical and complete shards pass; same-size replacement, changed-size replacement, missing/malformed digest, invalid size, replaced receipt, and corrupt shard fail. Update synthetic receipts to describe their actual fixture bytes. Reuse and extend existing shard tests rather than duplicating the merger's validation logic.

### 3. CR-003 — make coverage a frozen membership contract

Files: `snakemake/metadata/build_unified_partial_feature_snapshots.py`, `evaluation_region_partitions/validate_unified_feature_snapshot_coverage.py`, coverage tests, and `snakemake/unified_real_data_distributions_v2.smk`.

Implementation steps:
1. Version the coverage schema. Store the final snapshot CSV hash, source-manifest hash, and exact available UID set or hashed availability ledger. Derive counts from sets, rather than accepting independent count fields as evidence.
2. Require unique source-manifest UIDs and validate every snapshot row against its source's family and provenance. Require the observed snapshot UID set to equal the frozen available set and to be a subset of the eligible source universe. Under complete policy it must equal the required nonempty source set.
3. Validate unique unit identities/paths, canonical-versus-shard exclusivity, ordered contiguous shard ranges, and final selected-entry total. Multiple valid shards for one source are permitted; equal-count UID substitutions are not.
4. Publish CSV/ledger first and the coverage record last. Multiple-file publication is not atomic: hash disagreement during an interrupted update must fail closed. Reject old records with a clear refresh instruction.
5. Declare every source/snapshot/ledger/policy/helper actually consumed as workflow input. Keep the standalone downstream workflow downstream; do not implicitly import production rules to satisfy these dependencies.

Acceptance matrix: valid complete canonical, valid complete sharded, and explicitly partial cohorts pass. Unknown UID, approved-but-unavailable UID swap, altered path, duplicate unit, mixed canonical/shards, missing/overlapping shard and post-publication CSV modification fail. Exercise workflow invalidation when a source manifest or policy changes.

### 4. CR-004 — protect both input and output entry order

Files: `convertData/build_region_partitions.py`, `snakemake/rules/region_partitions.smk`, and partition tests.

Implementation steps:
1. Enforce the configured identity contract before friend filtering whenever Hit3D is requested: branches exist, per-source counts match, and every ordered `(runId,eventId,eventIndex)` agrees. Stream identity columns; report the first mismatch with both paths and values. The optional count-only flag cannot disable this guard. Do not require global uniqueness across distinct MC source files.
2. Make identity-aligned snapshots deterministic. The code calls `ROOT.EnableImplicitMT` for positive `rdf_threads` and writes trees in separate snapshot actions; input alignment does not prove output alignment. Start with serial execution for aligned products unless explicit ordering is implemented and tested. Account for an already enabled ROOT implicit-MT state, not merely omission of another enable call. Validate feature/cutflow pairing too where positional friendship is used.
3. Validate selected output identities before publishing metadata. Cover both combined output mode and the actual workflow's split feature/Hit3D rules. The Hit3D rule currently depends on a feature metadata row and reruns selection: compare its new output against the already published feature partition, with the same frozen source ordering and selection configuration. Handle a legitimately empty partition explicitly.
4. Make the published feature artifact or its verified manifest an effective dependency, so metadata alone cannot conceal replacement. Check that the rule executes the reviewed worktree script: its current path is constructed from `PERSONAL_WORK_SPACE`.

Acceptance matrix: aligned pairs and empty pairs pass; missing identity, unequal counts, same-count permutation and a single wrong key fail before publication. Allow repeated keys across separate sources. A tiny end-to-end fixture must assert exact selected feature/Hit3D sequences in combined and split-rule modes, including a thread request greater than one. Feature-only workflows remain usable; enabling the disabled Hit3D aggregate is not part of this fix.

### 5. CR-005 — reconcile upstream eligibility per selection profile

Files: `snakemake/metadata/prepare_unified_real_data_feature_manifests.py`, selection/snapshot coverage schemas, final coverage gate and plot manifest, associated workflow inputs/tests.

Implementation steps:
1. Emit a checksummed ledger for the original selection universe, with explicit excluded UID/reason, validation status and selected count for each profile. Bind the selection manifest and exclusion policy hashes. Reject exclusion UIDs outside the original universe.
2. Define eligible UIDs as original minus excluded. Validated and failed/missing status sets must partition eligible UIDs without duplication. Within each profile, define required-feature UIDs as successfully validated sources with positive selected count. A source may have zero No-Veto events and nonzero Veto events: zero-selection handling must be per profile, not a single source-wide flag.
3. Require the feature source manifest to equal that profile's required-feature set. For complete certification require all eligible sources selection-validated, then complete features for every required source. A validated zero count satisfies selection accounting without a feature file.
4. Carry both stage denominators, excluded/zero/missing counts, exact membership evidence and policy hashes into the final gate and plot metadata. Permit only specifically configured missing conditions in partial mode; invalid metadata/checksums are fatal. Report upstream selection incompleteness separately from downstream feature incompleteness.
5. Reconcile the actual nine configured excluded UIDs with the missing-selection set before migration. Equal totals are not evidence of equal membership; this reconciliation remains outstanding and must not be described as completed.

Acceptance matrix: the J two-source/missing-receipt example must fail complete certification. Explicit exclusion, profile-specific zero selection, and valid complete inputs pass. Unknown exclusion, changed policy, duplicate source, missing eligible receipt, and swapped manifest membership fail. Label completeness for eligible sources distinctly from coverage of the original 11,368-source universe.

### 6. CR-006 — repair the assertion and test its discriminatory power

File: `evaluation_region_partitions/test_abcd_blinded_expected.py`; production statistical code stays unchanged.

Implementation steps:
1. Require finite q0 with `0 <= q0 < 1e-6` and finite fitted mu near zero for the existing downward fixture. Keep the independent mu assertion; do not clip production q0 or suppress optimizer warnings.
2. Use a model-consistent signal-injected Asimov dataset as the positive control: obtain suggested parameters, set the signal parameter to 5, then call `model.expected_data(parameters)` so signal contamination in all channels and auxiliary observations is represented. Require a clearly positive statistic and recovery of injected mu within a documented tolerance.
3. After implementation, run the original complete synthetic regression file directly, then repeat the focused boundary/positive controls in the documented runtime. A numerical assertion repair does not establish optimizer robustness or inference coverage.

Today's bounded probes used LCG 107, Python 3.11.9, pyhf 0.7.6 on lxplus915, with user-site imports and bytecode writing disabled. The actual unmodified `build_workspace` and `q0_value` were called on invented data; no production signal-region observation was read. The downward fixture again returned q0=1.646599017135486e-7, mu=2.347799140310793e-12, satisfying the proposed tolerance. Two trial positive controls that changed only A to 200 or 500 failed the proposed q0>1 assertion: q0 was respectively 1.3336727633372902e-7 and 2.6568778110913627e-7, with mu effectively zero. These failed fixture designs are recorded rather than counted as passes; they do not by themselves demonstrate a production inference defect. Use the consistent injection described above instead.

### Delivery sequence, migration, and stop conditions

1. Implement the shared integrity contract and tests (CR-002), then the coordinated snapshot/selection membership schema and gate changes (CR-003/005). Keep each intermediate consumer compatible with an explicitly versioned contract or fail with a migration instruction.
2. Implement the partition alignment repair (CR-004), then normalized conversion dispatch (CR-001), and the independent numerical test repair (CR-006).
3. Run each acceptance matrix and the focused regression suite recorded in J using its documented environment. Run the repaired ABCD test directly. Inspect named-target dry runs for dependency invalidation and intended script paths before any production action.
4. Refresh receipts/ledgers/snapshots only through validation of the existing products and source identity. Do not replace an old digest with current bytes merely to make a check pass. Keep matching ROOT products; investigate mismatches before deciding whether conversion is required. No full-cohort conversion or training is implied by this plan.
5. Close a finding only with code diff, passing negative and positive tests, and relevant integration evidence. Record unexecuted canaries, unresolved exclusion reconciliation, and fit warnings separately. The full raw-to-downstream scientific review remains incomplete as stated in I.

Exact next action for a future implementation session: after explicit implementation authorization, add the ROOT-free integrity helper and CR-002 canonical/shard corruption regression cases, preserving unrelated changes. Stop this inspection session after saving and checking this plan; do not start that implementation now.

### Final probe result and validation limits

The model-consistent injection returned q0=0.03717362382782419 and fitted mu=5.001926136627247. An initial q0>1 trial also failed for this fixture; that threshold demands more separation than this model provides. The final proposed positive-control assertions are finite q0>0.01 and finite `abs(mu_hat - 5) < 0.01`. Both passed in two repeated calls with identical results. This threshold is a regression guard against an always-zero statistic, not a discovery-significance requirement or a physics sensitivity claim. Together with the downward-fixture checks it supports the test repair; no production statistic or test file was modified.

Reproduction: activate `snakemake/envs/env_lcg.sh` with nounset disabled, then enable strict Bash; set `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, and add `evaluation_region_partitions` to PYTHONPATH. Import `config,input_row` from `test_abcd_blinded_expected` and `ROLES,build_workspace,q0_value` from `abcd_blinded_model`; set the pyhf NumPy backend. Build the four-channel workspace with synthetic A/B/C/D=70/80/36/18, copy `model.config.suggested_init()`, set its POI entry to 5, and pass `model.expected_data(parameters)` to `q0_value` twice. For the downward check build A/B/C/D=0/80/36/18 and use workspace data. These were inline probes with no retained fixtures.

Final workspace inspection retained the same five pre-existing tracked modifications and the same untracked paths. `git diff --check` passed for tracked changes; the untracked review report was separately checked for its six plan subsections. No full regression suite, conversion canary, partition production, scheduler action or implementation was run in this session. CR-001 through CR-006 remain open pending implementation and the acceptance evidence specified above.

## M. Implemented fixes and verification — 2026-11-04

The user authorized implementation of section L. CR-001 through CR-006 now have code changes and regression evidence in this worktree; the inspection-only restriction is superseded for this implementation. Production migration and detector conversion canaries have not been run. HEAD remains `42e0f09582533437b5b2a2532ca0b860d93b027d`; changes are uncommitted and the pre-existing work was preserved.

| Finding | Implemented behavior | Verification |
| --- | --- | --- |
| CR-001 | Shared normalized MC/real classifier used by provenance validation, event dispatch, veto/US handling and hit selection. Explicit supported legacy aliases include MC, MC_ve, MC_vm, MC_data, MC_muonDIS and passing-muon data. Invalid types fail before converter outputs open. Path-dependent neutrino timing exception retained. | All supported types tested with case/whitespace variants; unknown types rejected; actual extracted converter predicates tested for timing parity. Existing provenance/selection tests pass. Full SNDSW raw conversion parity remains a deployment canary. |
| CR-002 | New ROOT-free `convertData/feature_integrity.py` validates receipt identity, digest and byte size and streams the feature checksum. Canonical/shard snapshots freeze file and receipt digests; consumers reject replacement. Existing corrupt canonical files cannot fall back to shards. Snapshot publication aborts on integrity failure. | Valid canonical/shards pass; same-size and changed-size corruption, malformed/missing digest, invalid sizes, replaced receipt and corrupted shard fail. Shard completeness digest and per-receipt ranges are reconciled. |
| CR-003 | Coverage schema 2 freezes snapshot CSV identity and exact source-status membership. `snapshot_contract.py` checks source provenance, available UID sets, unique units/paths, canonical/shard exclusivity and contiguous shard coverage. Duplicate coverage families and invalid statuses fail. | Membership substitution, path/family alteration, duplicate units, modified CSV and inconsistent statuses fail; valid complete canonical/sharded and configured partial cases pass. Old schemas fail with refresh instructions. |
| CR-004 | Mandatory streaming three-field identity checks precede Hit3D friendship. Serial ROOT snapshots preserve order even if implicit MT was already enabled. Selected outputs are checked before metadata publication. Split Hit3D requires feature reference metadata, validates its file digest, selection, source order and count, then compares output identities. Feature/cutflow counts are also checked before positional friendship. | Actual ROOT combined and split partition filtering produces the expected ordered identities with `rdf_threads=2`; existing MT is disabled. Same-count permutations, wrong key, missing branch, unequal count and replaced published feature fail. Empty pair passes. The Hit3D aggregate remains disabled. |
| CR-005 | Source-manifest builder consumes exclusion policy and emits a versioned per-source ledger, preserving per-profile selected counts including zero. Snapshot coverage binds this upstream ledger. Final gate reconciles exact eligible/nonempty membership, profile counts and upstream totals; embeds both selection and feature denominators. Current partial policy allows missing features only; upstream incomplete selection remains fatal. | Missing upstream receipt cannot be hidden by a reduced feature manifest; profile-specific zero and explicit exclusion cases pass; changed policy fails. Read-only inventory of all 11,368 selection receipt paths found exactly nine absent; their UID set exactly equals the nine configured exclusions. This inventory establishes availability, not fresh validation of receipt contents. |
| CR-006 | Downward q0 must be finite and in [0,1e-6), with finite near-zero fitted mu. Model-consistent mu=5 injection requires q0>0.01 and mu recovery within 0.01. No production inference clipping or tolerance change. | The complete synthetic blinded/unblinded regression passes directly. Optimizer bound-clipping and covariance-repair warnings remain; this is not a fit-robustness or physics-coverage certification. |

### Final commands and results

LCG activation: `set +u; source snakemake/envs/env_lcg.sh`, followed by strict Bash, `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, and repository `convertData`, `evaluation_region_partitions`, `snakemake/metadata` on PYTHONPATH. Preflight recorded lxplus915, Python 3.11.9, ROOT 6.34.02 and pyhf 0.7.6.

```text
python3 -m unittest -q test_real_data_2024_selection test_scifi_plane_clustering test_unified_shower_features test_validate_unified_shower_feature_shards test_build_region_partitions test_unified_partial_feature_snapshots test_unified_feature_distributions test_unified_feature_distributions_root test_unified_real_data_histogram_bundles test_unified_testbeam_histogram_bundles test_review_contracts
```

Final result: **66 tests passed in 13.517 seconds** (54 existing focused tests plus 12 new adversarial/integration tests). Log: `/tmp/zhibin/review_fix_final_tests.log`. Two metadata tests also passed in 0.75 seconds using `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q snakemake/metadata/test_prepare_unified_real_data_feature_manifests.py -p no:cacheprovider`. `python3 evaluation_region_partitions/test_abcd_blinded_expected.py` completed with `Strict one-bin blinded/unblinded ABCD synthetic checks: OK`. All changed Python sources parsed successfully. `git diff --check` passed.

Snakemake was activated from `/afs/cern.ch/work/z/zhibin/miniconda3/envs/snakemake` (Python 3.13.12, Snakemake 9.17.2). Checks:

- Real-data V2 coverage target dry-run: exactly one coverage job; explicit source manifests, selection manifest/policy/coverage and helper dependencies appear in its inputs.
- Testbeam V4 histogram aggregate dry-run: exactly 61 jobs (20 audits, 20 bundles, 20 validators, one aggregate), with no selection or feature production.
- Main Snakefile `--list-target-rules`: passed; output retained in `/tmp/zhibin/review_fix_targets.txt`. Region builder rules now resolve the script/helper from this workflow's checkout instead of the unrelated main workspace.

### Migration and remaining limits

No production files, receipts, snapshots, partitions, histograms or models were regenerated. No scheduler job or commit was created. Before downstream production, refresh source coverage and snapshots through their validators under schema 2; old frozen snapshots lack the new file/receipt digest columns and intentionally fail consumption. Refresh split partition metadata through validated partition generation when required; do not fill its new digest field from an unvalidated replacement. Retain matching existing feature files; conversion is not automatically required by this change.

Content verification adds streaming I/O and serial partition snapshots trade throughput for deterministic ordering. Checks assume published products remain immutable during verification and consumption. Legacy cutflow trees without explicit event identity still rely on their converter-established order plus the new count check; the strict three-field comparison is enforced for feature/Hit3D pairs. Full detector conversion parity, production rollout, full-cohort histogram closure and statistical fit robustness remain outside this completed implementation/test pass. The broader scientific review in I is not declared complete.

Exact next action: wait for the user's instruction on validated metadata migration or a detector conversion canary. Do not automatically refresh production metadata, regenerate features, or start histogram/ML production.

## N. Commit acceptance verification — 2026-11-04

User requested fresh testing against the plan and a commit if successful. Reinspection identified and repaired two additional boundary cases before accepting the change: canonical snapshot units now require shard index -1, range start 0 and the declared selected-event total, with layout/status agreement; split Hit3D cannot bypass its reference check through an empty source list when feature metadata is nonempty. The real-data coverage rule now explicitly activates LCG for its YAML dependency.

The exact staged tree `0da3057fc409a0b071d65369d2ff4a12ccca03d5` was exported with `git checkout-index` to `/tmp/zhibin/review_fix_commit_k1c9aeaw`. All verification below ran from that export, with module paths checked to ensure tests used staged code. Subsequent staged changes are documentation only.

- Combined focused/adversarial/integration suite: **67 tests passed in 17.324 s**. This includes the new canonical range/status regression and the empty-input split-partition rejection inside the full ROOT integration test.
- Metadata tests: **2 passed in 0.69 s**. Total counted tests: **69**.
- Full `test_abcd_blinded_expected.py`: passed with exit 0 and its synthetic-check success message. The previously recorded optimizer/covariance warnings persist; they were not suppressed or treated as proof of fit robustness.
- All staged Python files parse successfully. `git diff --cached --check` passed.
- Real-data coverage dry-run: one coverage job. Testbeam bundle dry-run: 61 downstream jobs. Main Snakefile target listing passed. No jobs executed or submitted.
- Read-only check against current production coverage: rejected the obsolete schema with the expected migration instruction; no output receipt was written.

Runtime: LCG 107, Python 3.11.9, ROOT 6.34.02, pyhf 0.7.6, lxplus915; Snakemake environment Python 3.13.12/Snakemake 9.17.2. The first export-test preflight failed because the harness replaced LCG's PYTHONPATH; preserving the activated path fixed the harness, after which all checks above passed. No packages or shared environments were changed.

Logs are `/tmp/zhibin/review_fix_commit_tests.log`, `review_fix_commit_abcd.log`, `review_fix_commit_real_dryrun.log`, `review_fix_commit_testbeam_dryrun.log`, and `review_fix_commit_targets.log`. Test commands match M, with the additional canonical test included automatically by `test_review_contracts`.

Commit scope includes the related real-data V2 workflow, wrappers/configuration and required dataset-neutral bundle/plot support on which CR-003/005 depend. The independent pre-existing testbeam energy-summary axis adjustment is deliberately left unstaged and unchanged in the working tree. Production migration and detector conversion parity remain pending as documented in M; successful commit acceptance does not certify the broader scientific review.
