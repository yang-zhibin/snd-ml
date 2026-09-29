# SNDSW EventBuilder `sndEventFilter` / `sndCutFlowTree` Fix Plan

## Context

We are producing an EventBuilder-based MC neutrino comparison sample:

1. Start from the original MC neutrino raw file.
2. Run `run_MCEventBuilder.py` to create `newRaw`.
3. Run `run_digiSND.py` to create `newDigi`.
4. Run the usual downstream production:
   - `process_nueAnalysis`
   - `process_features`
   - `process_hit3d`
   - region partitions
   - region evaluation/comparison

The failing step is `process_nueAnalysis`, which calls `sndEventFilter` on the EventBuilder `newDigi` file.

Example failing input:

```text
/eos/experiment/sndlhc/users/zhibin/MC_neutrino_EventBuilder/2024_ve/1/newDigi_EventBuilder_MC_neutrino_2024_ve_1.root
```

Example failing output:

```text
/eos/experiment/sndlhc/users/zhibin/MC_neutrino_EventBuilder/2024_ve/1/nueAnalysisFilter_EventBuilder_MC_neutrino_2024_ve_1.root
```

## Observed Failure

The relevant runtime log is:

```text
Running sndEventFilter
Using sndEventFilter pipeline:
/afs/cern.ch/work/z/zhibin/public/SndBuild/sw/rhel9_x86-64/sndsw/master-local12/analysis/analyses/snd_analysis_2024_0mu/pipelines/nueFilterMoriondOrder_withEventLevelOutput.h

Error in <TChain::LoadTree>: Cannot find tree with name rawConv in file ...
[INFO] Got input tree
Error in <TTree::SetBranchStatus>: unknown branch -> EventHeader
...
[FATAL] cutFlowTree ERROR: EventHeader not found in ROOT globals
terminate called after throwing an instance of 'std::runtime_error'
```

The first warning about `rawConv` is not the real fatal problem. For MC input, `sndEventFilter` is supposed to try `rawConv` first, then fall back to `cbmsim`.

The fatal error is:

```text
cutFlowTree ERROR: EventHeader not found in ROOT globals
```

## Important File Structure Check

The EventBuilder `newDigi` file is not simply missing the event header. It has a normal MC-style tree:

```text
cbmsim
```

and it has an event header branch:

```text
EventHeader.
```

This was checked with:

```bash
rootls -t /eos/experiment/sndlhc/users/zhibin/MC_neutrino_EventBuilder/2024_ve/1/newDigi_EventBuilder_MC_neutrino_2024_ve_1.root
```

The original MC digi file also has `cbmsim`. Therefore, the immediate issue is not that the EventBuilder digi has no `cbmsim`, and not that it has no event header branch.

## Relevant SNDSW Code

The pipeline used by `process_nueAnalysis` is:

```text
/afs/cern.ch/work/z/zhibin/public/SndBuild/sw/rhel9_x86-64/sndsw/master-local12/analysis/analyses/snd_analysis_2024_0mu/pipelines/nueFilterMoriondOrder_withEventLevelOutput.h
```

In that pipeline, MC/data mode is decided by whether `rawConv` exists in ROOT globals:

```cpp
bool isMC = true;
if (gROOT->GetListOfGlobals()->FindObject("rawConv")) isMC = false;
```

The pipeline later adds:

```cpp
pipeline.push_back( new snd::analysis_processes::cutFlowTree());
pipeline.push_back( new snd::analysis_processes::cutFlowHist());
```

The fatal error comes from:

```text
analysis/processes/sndCutFlowTree.cxx
```

Relevant code:

```cpp
inputTree_ = dynamic_cast<TTree*>(gROOT->GetListOfGlobals()->FindObject("rawConv"));
if (!inputTree_) inputTree_ = dynamic_cast<TTree*>(gROOT->GetListOfGlobals()->FindObject("cbmsim"));
if (!inputTree_) LOG(FATAL) << "cutFlowTree ERROR: input tree not found in ROOT globals";

eventHeader_ = dynamic_cast<SNDLHCEventHeader*>(gROOT->GetListOfGlobals()->FindObject("EventHeader"));
if (!eventHeader_) eventHeader_ = dynamic_cast<SNDLHCEventHeader*>(gROOT->GetListOfGlobals()->FindObject("EventHeader."));
if (!eventHeader_) LOG(FATAL) << "cutFlowTree ERROR: EventHeader not found in ROOT globals";
```

So `sndCutFlowTree` expects the event header object to be present in ROOT globals under one of these names:

```text
EventHeader
EventHeader.
```

## Root Cause

The root cause is a ROOT global-object naming mismatch between:

```text
analysis/core/sndEventFilter.cxx
```

and:

```text
analysis/processes/sndCutFlowTree.cxx
```

In `sndEventFilter.cxx`, input selection is:

```cpp
bool isMC = false;
TChain * ch = new TChain("rawConv");
ch->Add(input.c_str());
if (ch->GetEntries() == 0){
  delete ch;
  ch = new TChain("cbmsim");
  ch->Add(input.c_str());
  if (ch->GetEntries() > 0) {
    isMC = true;
  } else {
    LOG(FATAL) << "Didn't find rawConv or cbmsim in input file";
  }
}
LOG(INFO) << "Got input tree";
```

This part is fine: it supports `cbmsim`.

Then `sndEventFilter.cxx` reads the event header:

```cpp
SNDLHCEventHeader * event_header = new SNDLHCEventHeader();
ch->SetBranchAddress("EventHeader", &event_header);
ch->GetEntry(0);
if (event_header->GetEventNumber() == -1) {
  ch->SetBranchAddress("EventHeader.", &event_header);
  ch->GetEntry(0);
  if (event_header->GetEventNumber() == -1) {
    LOG(FATAL) << "Invalid event header"; 
  }
}
rootGlobals->Add(event_header);
```

The last line is the fragile part:

```cpp
rootGlobals->Add(event_header);
```

This adds the object to ROOT globals using the object's current internal ROOT name. It does not guarantee that the object can be found later as:

```cpp
FindObject("EventHeader")
```

or:

```cpp
FindObject("EventHeader.")
```

For the EventBuilder `cbmsim` file, `sndEventFilter` is apparently not publishing the event header object under either expected name. Therefore `sndCutFlowTree` cannot find it, even though the input file has an `EventHeader.` branch and `sndEventFilter` read it.

## Why This Is Not Primarily A Snakemake Bug

The Snakemake rule is passing:

```text
--input <EventBuilder newDigi ROOT>
--geofile <geo file>
--output <nueAnalysisFilter ROOT>
--pipeline <nueFilterMoriondOrder_withEventLevelOutput.h>
```

That is consistent with the intended `sndEventFilter` interface.

The failing logic happens inside `sndEventFilter` and `sndCutFlowTree` after input tree discovery.

## Best Fix Strategy

The best fix is to repair the producer-side contract in:

```text
analysis/core/sndEventFilter.cxx
```

not to make `sndCutFlowTree.cxx` guess more names.

The clean invariant should be:

```text
sndEventFilter publishes shared branch objects into ROOT globals under stable canonical names.
All cuts and processes consume those canonical names.
```

This is already implied by comments in `sndEventFilter.cxx`, but not enforced.

## Proposed Patch

### 1. Explicitly Name EventHeader Before Adding To ROOT Globals

Current code:

```cpp
rootGlobals->Add(event_header);
```

Replace with:

```cpp
event_header->SetName("EventHeader");
rootGlobals->Add(event_header);
```

This guarantees that `sndCutFlowTree.cxx` can find:

```cpp
gROOT->GetListOfGlobals()->FindObject("EventHeader")
```

### 2. Explicitly Name SciFi Hits, MuFilter Hits, And MC Tracks

Current code:

```cpp
TClonesArray * scifiDigiHitCollection = new TClonesArray("sndScifiHit", 3000);
ch->SetBranchAddress("Digi_ScifiHits", &scifiDigiHitCollection);
rootGlobals->Add(scifiDigiHitCollection); // Name: sndScifiHits

TClonesArray * muFilterDigiHitCollection = new TClonesArray("MuFilterHit", 470);
ch->SetBranchAddress("Digi_MuFilterHits", &muFilterDigiHitCollection);
rootGlobals->Add(muFilterDigiHitCollection); // Name: MuFilterHits

TClonesArray * MCTracks; 
if (isMC) {
  MCTracks = new TClonesArray("ShipMCTrack", 5000);
  ch->SetBranchAddress("MCTrack", &MCTracks);
  rootGlobals->Add(MCTracks); // Name: ShipMCTracks
}
```

The comments say the intended names, but the code does not enforce them.

Change to:

```cpp
TClonesArray * scifiDigiHitCollection = new TClonesArray("sndScifiHit", 3000);
ch->SetBranchAddress("Digi_ScifiHits", &scifiDigiHitCollection);
scifiDigiHitCollection->SetName("sndScifiHits");
rootGlobals->Add(scifiDigiHitCollection);

TClonesArray * muFilterDigiHitCollection = new TClonesArray("MuFilterHit", 470);
ch->SetBranchAddress("Digi_MuFilterHits", &muFilterDigiHitCollection);
muFilterDigiHitCollection->SetName("MuFilterHits");
rootGlobals->Add(muFilterDigiHitCollection);

TClonesArray * MCTracks; 
if (isMC) {
  MCTracks = new TClonesArray("ShipMCTrack", 5000);
  ch->SetBranchAddress("MCTrack", &MCTracks);
  MCTracks->SetName("ShipMCTracks");
  rootGlobals->Add(MCTracks);
}
```

This should prevent the next likely failures after fixing `EventHeader`, because other cuts/processes expect:

```text
sndScifiHits
MuFilterHits
ShipMCTracks
```

### 3. Optional Defensive Debugging In `sndCutFlowTree.cxx`

This is optional, not the main fix.

Before the fatal:

```cpp
if (!eventHeader_) LOG(FATAL) << "cutFlowTree ERROR: EventHeader not found in ROOT globals";
```

one could print the available ROOT globals:

```cpp
gROOT->GetListOfGlobals()->Print();
```

or log a clearer message:

```cpp
if (!eventHeader_) {
  gROOT->GetListOfGlobals()->Print();
  LOG(FATAL) << "cutFlowTree ERROR: EventHeader not found in ROOT globals";
}
```

This makes debugging easier, but the proper structural fix is still in `sndEventFilter.cxx`.

## Why Not Fix Only `sndCutFlowTree.cxx`?

It would be possible to make `sndCutFlowTree.cxx` search for more object names, but that would be a weaker fix.

`sndCutFlowTree` is a consumer. It should not have to guess arbitrary internal object names. The producer, `sndEventFilter`, should publish a stable interface.

Fixing only the consumer would also leave other pipeline components vulnerable to the same issue.

Examples:

```text
sndSciFiHitFilter expects sndScifiHits
MuFilter cuts expect MuFilterHits
cutFlowTree expects ShipMCTracks for MC truth
```

Therefore the best plan is:

```text
Set stable global names once in sndEventFilter.
Keep downstream consumers simple and explicit.
```

## Validation Plan

After patching and rebuilding/loading the patched `sndsw`, rerun one failing job:

```bash
cd /afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake

snakemake -s snakefile \
  /eos/experiment/sndlhc/users/zhibin/MC_neutrino_EventBuilder/2024_ve/1/nueAnalysisFilter_EventBuilder_MC_neutrino_2024_ve_1.root \
  -n
```

Then run for real with the usual profile:

```bash
snakemake -s snakefile \
  /eos/experiment/sndlhc/users/zhibin/MC_neutrino_EventBuilder/2024_ve/1/nueAnalysisFilter_EventBuilder_MC_neutrino_2024_ve_1.root \
  --profile <your_profile>
```

Expected behavior:

1. It may still print a warning about missing `rawConv`; that is okay for MC.
2. It should fall back to `cbmsim`.
3. It should no longer crash with:

```text
cutFlowTree ERROR: EventHeader not found in ROOT globals
```

4. It should produce:

```text
nueAnalysisFilter_EventBuilder_MC_neutrino_2024_ve_1.root
```

5. The output file should contain:

```text
cutFlowSummary
```

Check with:

```bash
rootls -t /eos/experiment/sndlhc/users/zhibin/MC_neutrino_EventBuilder/2024_ve/1/nueAnalysisFilter_EventBuilder_MC_neutrino_2024_ve_1.root
```

Then continue:

```bash
snakemake -s snakefile eventbuilder_production -n
```

and finally:

```bash
snakemake -s snakefile eventbuilder_production --profile <your_profile>
```

## Summary

The EventBuilder `newDigi` file has `cbmsim` and an event header branch. `sndEventFilter` can detect `cbmsim`, but it does not reliably publish the branch objects under the canonical ROOT global names expected by `sndCutFlowTree` and other pipeline processes.

The best fix is to patch `sndEventFilter.cxx` so that, before adding shared objects to `gROOT->GetListOfGlobals()`, it explicitly sets:

```text
EventHeader
sndScifiHits
MuFilterHits
ShipMCTracks
```

as the ROOT object names.

This makes the producer/consumer contract explicit and should fix the `cutFlowTree ERROR: EventHeader not found in ROOT globals` failure for EventBuilder MC.
