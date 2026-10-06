# Unified Shower Feature Operation Plan

Status: proposed

Last updated: 2026-10-06

Scope: Execute the approved design for a backward-compatible unified feature
schema across 2024 neutrino MC, MC muon-up, testbeam MC/data, and skimmed
collision data, followed by validated partition, histogram, and plotting
workflows.

Related design:
[Unified Shower Feature Production and Region Evaluation Plan](UNIFIED_SHOWER_FEATURE_PRODUCTION_AND_REGION_EVALUATION_PLAN.md).

## Operating principles

- Complete one bounded action and its validation before starting the next.
- Do not merge from or into a dirty worktree.
- Keep `convertData/digi_2_features.py` as the single owner of feature
  calculations; adapters may change input handling but must not duplicate the
  algorithms.
- Do not overwrite canonical legacy `feature_path` products.
- Preserve or validate `(runId, eventId, eventIndex)` and source-file identity.
- Run one real canary for every distinct input contract before full production.
- Use named Snakemake aggregate targets and HTCondor for long production.
- Record commands, logs, job IDs, outputs, results, and Git state in the active
  task file at every milestone.

## Phase 0: freeze the existing worktrees

### Step 1: checkpoint the EM-shower documentation

1. In the `task/em-shower` worktree, review the proposed design plan, this
   operation plan, and the proposed-plans index.
2. Run `git diff --check`.
3. Confirm no generated output is staged.
4. Create a documentation-only checkpoint commit.
5. Record the commit SHA and confirm the worktree is clean.

Exit gate: the saved plans are committed and `task/em-shower` is clean.

### Step 2: checkpoint the ABCD branch

1. In the `task/abcd-bkg-estimation` worktree, classify every modified and
   untracked path as task work or unrelated user work.
2. Reconcile the active task record with actual central-muon reports,
   scheduler state, and outputs.
3. Run the focused ABCD tests recorded by that task.
4. Run `git diff --check`.
5. Commit only the verified ABCD task paths.
6. Confirm the worktree is clean and no related HTCondor/Snakemake job is
   active.

Exit gate: ABCD has a task-only checkpoint SHA and a clean worktree.

### Step 3: freeze the testbeam source boundary

1. Treat `task/3d-cnn-testbeam` as read-only reference material.
2. Record the exact source commits and files that own:
   - identity-cutflow generation;
   - testbeam data-quality selection policy;
   - PDG/energy override handling;
   - geometry/source catalogue validation; and
   - focused regression tests.
3. Record current testbeam production completeness and failures.
4. Do not merge the whole testbeam branch.

Exit gate: the small set of behavior to port is explicit and reproducible.

## Phase 1: create the integration worktree

### Step 4: create the integration branch

1. Create `task/unified-shower-features` from the checkpointed ABCD branch.
2. Attach it to a dedicated worktree, provisionally:
   `/afs/cern.ch/user/z/zhibin/worktrees/snd-ml-unified-shower-features`.
3. Record the starting commit and initial clean status in a new active task
   file.

Exit gate: a clean, isolated integration worktree exists.

### Step 5: merge the EM-shower branch

1. Merge the complete `task/em-shower` history; do not copy individual source
   files manually.
2. Resolve conflicts only after comparing both sides and preserving existing
   ABCD behavior.
3. Do not port testbeam changes in the same action.

### Step 6: validate the merged baseline

Run and record:

1. Python compilation/import checks in the documented environment.
2. SciFi clustering tests.
3. EM-neutrino workflow tests.
4. MC-muon workflow tests.
5. Existing ABCD focused tests.
6. Snakemake parsing/list-target checks for both workflows.
7. `git diff --check` and final Git status.

Exit gate: all existing behavior passes before new dataset support begins.

## Phase 2: freeze schema, paths, labels, and source scope

### Step 7: freeze the production version

Use an explicit version such as `unified_shower_features_v1`. Add a new
metadata output-path column, provisionally:

```text
feature_scifi_cluster_v1_path
```

Do not change the existing `feature_path` meaning.

### Step 8: freeze output roots

Use isolated feature roots for:

```text
mc_neutrino
mc_muon_up
real_data_2024_skim
testbeam_mc_2024
testbeam_data_2024
```

Store large ROOT products on EOS. Store metadata, summaries, and plots on the
AFS work volume rather than the AFS user volume.

### Step 9: freeze common source fields

Retain `mc_incoming_energy` and define exact ROOT types and sentinel behavior
for:

```text
source_energy_gev
source_pdg_code
source_domain_id
```

- Neutrino and muon MC: populate from MC truth.
- Testbeam MC: cross-check truth against the catalogue.
- Testbeam data: populate from validated beam metadata.
- Collision real data: use documented unavailable values.

Catalogue beam energy must not be described as MC truth.

### Step 10: freeze canonical sample labels

Use one vocabulary:

```text
real_data
CC_nue
CC_numu
NC_nue
NC_numu
MC_muon_up
testbeam_mc_electron
testbeam_mc_pion
testbeam_data_electron
testbeam_data_pion
```

Do not use both `muon` and `MC_muon_up` for one production.

### Step 11: freeze testbeam scope

1. Start with sources passing the existing valid-W2 geometry audit.
2. Produce a scope report by domain, particle, energy, geometry, and exclusion
   reason.
3. Decide and record whether v1 covers every eligible W2 metadata row or only
   the validated full-campaign physical-source catalogue.
4. Do not mix the `Fe_1wall` geometry into the v1 comparison.

Exit gate: schema, paths, labels, source rows, and exclusions are versioned and
reviewed.

## Phase 3: implement adapters, manifests, and targets

### Step 12: keep one feature implementation

All established and SciFi clustering/profile features remain implemented in
`convertData/digi_2_features.py`.

### Step 13: preserve the MC-muon adapter

Keep `digi_2_features_mc_muon.py` limited to its legacy event-header
compatibility. It must continue to delegate calculations to the main
converter.

### Step 14: add testbeam input support

Port the established testbeam behavior:

- identity cutflow;
- real-testbeam bypass of collision-only quality requirements;
- paired PDG/energy overrides;
- source-domain and geometry provenance.

Prefer common converter options, provisionally:

```text
--selection-mode
--source-pdg-code
--source-energy-gev
--source-domain-id
```

Required paired overrides must fail when only one value is supplied.

### Step 15: prepare one manifest per dataset family

Each row should record:

```text
source metadata file and row
digi and geometry paths
cutflow path
input tree
sample type and selection mode
source PDG, energy, and domain
output feature path
validation marker
expected event count
```

Require unique source and output identities.

### Step 16: add isolated aggregate targets

Implement full-production targets:

```text
unified_features_mc_neutrino_v1
unified_features_mc_muon_up_v1
unified_features_real_data_2024_skim_v1
unified_features_testbeam_mc_2024_v1
unified_features_testbeam_data_2024_v1
```

Add one separately named canary target per family. Canary targets must not
implicitly depend on full production.

Exit gate: manifests are deterministic and dry runs show the expected source
counts without executing jobs.

## Phase 4: static and synthetic validation

### Step 17: validate backward compatibility

For fixed representative events:

1. Compare legacy branch names and ROOT scalar types.
2. Verify no established branch was removed or renamed.
3. Compare legacy feature values before and after integration.
4. Confirm new branches are strictly additive unless an explicitly documented
   bug fix requires a value change.

### Step 18: validate the cluster schema

Require:

- cluster hit totals equal `count_scifi`;
- cluster QDC totals equal `qdc_scifi`;
- direct QDC per hit equals `qdc_scifi / count_scifi`;
- direct and cluster QDC-per-hit values agree;
- leading and other clusters close to the totals;
- all required scalar values are finite; and
- the schema marker is correct.

### Step 19: run sample-contract tests

Cover:

- neutrino CC/NC classification;
- the legacy MC-muon header;
- collision-data quality selection;
- testbeam-data bypass of collision-only requirements;
- paired testbeam PDG and energy overrides;
- missing/inconsistent testbeam metadata;
- event identity;
- empty inputs and no-SciFi-hit events.

Exit gate: compilation, focused tests, Snakemake parsing, and
`git diff --check` pass.

## Phase 5: run five real canaries

For each canary, perform exactly:

```text
target-specific dry run
-> one local conversion job
-> schema and identity validation
-> branch/provenance audit
-> runtime and memory record
-> task checkpoint
```

Do not start the next canary until the current result passes.

### Step 20: MC-neutrino canary

Repeat the already validated 48-event `2024_ve/1` source on the integration
branch. Confirm the unified schema, energy, classification, identity, and
cluster closure.

### Step 21: MC-muon-up canary

Use one of the 15 sources. Confirm legacy-header behavior, muon truth,
cluster closure, runtime, and memory.

### Step 22: collision real-data canary

Use one manageable 2024 skim source. Confirm collision quality cuts, retained
Veto/US populations, `isMC == 0`, truth sentinels, and cluster closure.

### Step 23: testbeam-MC canary

Use one valid-W2 source. Confirm truth/catalogue agreement, geometry,
identity-cutflow behavior, source provenance, and cluster closure.

### Step 24: testbeam-data canary

Use one valid-W2 source. Confirm no collision-only branch requirement,
catalogue PDG/energy, `isMC == 0`, source-domain provenance, and cluster
closure.

Exit gate: all five input contracts pass real-data integration tests.

## Phase 6: full feature production

For every family use this fixed sequence:

```text
full target dry run
-> review job count/resources
-> HTCondor production
-> monitor and diagnose failures
-> validate every intended output
-> write completeness report
-> final no-op dry run
-> task checkpoint
```

Stop the campaign for unexplained failures, missing markers, identity or
closure errors, source/output count disagreement, or quota risk.

### Step 25: MC muon-up

Process and validate all 15 sources first because this is the smallest complete
campaign and exercises the compatibility adapter.

### Step 26: MC neutrino

Process 976 electron-flavour and 890 muon-flavour sources, for 1,866 intended
outputs in total.

### Step 27: testbeam MC

Process only the frozen valid-W2 scope and report coverage by particle and
energy.

### Step 28: testbeam data

Process only sources with complete validated particle, energy, and geometry
metadata.

### Step 29: 2024 skimmed collision data

Process the 11,368-source catalogue last. Set concurrency, runtime, memory,
and disk requests from the real-data canary evidence.

## Phase 7: collision Region V1 partitions

### Step 30: create a new partition version

Create a versioned catalogue that reads only the new unified feature paths and
does not overwrite `partition_v1_region_v1`.

Include:

```text
real_data
CC_nue
CC_numu
NC_nue
NC_numu
MC_muon_up
```

### Step 31: preserve the initial Region V1 policy

Use `count_scifi > 35` and the eight combinations of:

```text
Veto/no Veto
US/no US
SciFi fiducial inside/outside
```

Preserve the existing exposure/fraction policy for the first reproducibility
pass. If a full-exposure shape-only view is later desired, make a separate
version rather than silently changing this one.

### Step 32: validate collision partitions

Require source/partition count closure, correct particle filters, mutually
exclusive regions, preserved identities, no silent omissions, and matching
exposure totals.

## Phase 8: testbeam partitions

### Step 33: create an isolated testbeam partition configuration

Use axes such as:

```text
domain
particle
beam energy
geometry
optional beam-entry position
```

Do not label testbeam groups as Signal or SB1--SB7.

### Step 34: validate matched comparisons

Require support for:

- MC electron versus MC pion at fixed energy;
- data electron versus data pion;
- MC versus data for matched particle and energy; and
- energy dependence within a fixed particle.

Do not combine unmatched energy mixtures without explicitly labelling them.

## Phase 9: histograms and plots

### Step 35: freeze the shared histogram configuration

Include the established features, cluster/profile features, energy variables,
and agreed two-dimensional correlations in one versioned configuration.

Required one-dimensional observables include:

- `count_scifi`, `qdc_scifi`, and `density_sndsw_scifi`;
- `scifi_qdc_per_hit_direct` and `scifi_cluster_qdc_per_hit`;
- cluster counts;
- leading-cluster hits, QDC per hit, and QDC fraction;
- other-cluster QDC fraction;
- QDC variation and roughness;
- effective cluster count;
- station-profile mean, slope, and curvature; and
- applicable energy variables.

Required two-dimensional observables include:

- total QDC versus hit count;
- leading-cluster QDC versus leading-cluster hits;
- roughness versus cluster QDC per hit;
- fragmentation versus leading-cluster fraction; and
- energy versus principal shower observables.

### Step 36: produce collision-region plots

Produce both:

1. particle groups compared within one fixed region; and
2. regions compared for one fixed particle group.

Use unit-area normalization for the first shape study. Keep absolute-rate plots
separate and apply explicit exposure/luminosity weights.

### Step 37: produce testbeam plots

Produce electron/pion, MC/data, and energy-dependence comparisons. Treat
testbeam event counts as beam-sample statistics rather than collision
luminosity.

### Step 38: validate plot products

Require nonempty ROOT histograms and summary CSVs, selected-entry and flow
closure, expected PDF page counts, correct axes/log settings, readable legends,
and visual inspection of representative plots from every comparison family.

## Phase 10: final review and handoff

### Step 39: review normalization and blinding labels

Clearly distinguish unit-area shapes, luminosity-normalized yields,
mixed-exposure diagnostics, testbeam event counts, and already-unblinded
collision signal-region plots. Do not tune on the real signal region and later
describe the result as blind.

### Step 40: write the completion report

Record Git commits, manifest counts, successful/failed jobs, HTCondor IDs,
runtime/storage, schema, partitions, normalization, outputs, validation, and
known limitations.

### Step 41: final checkpoint

Require a clean integration worktree, no active or held jobs, no-op dry runs
for every aggregate target, all focused tests passing, `git diff --check`
passing, and a task-only checkpoint commit.

Only then should the integration branch be considered for merging into
`develop`.

## Exact first operation

1. Review and commit the proposed plan documents on `task/em-shower`.
2. Checkpoint and commit the dirty `task/abcd-bkg-estimation` worktree.

Do not create the integration branch or merge code until both checkpoints are
complete.
