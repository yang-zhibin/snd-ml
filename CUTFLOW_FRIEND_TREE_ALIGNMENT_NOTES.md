# Cutflow Friend Tree Alignment Notes

## Problem

The region cutflow workflow can spend a long time at:

```text
[cutflow_efficiency] building sndData and friend-tree indices on runId/eventId
```

This happens in `evaluation_region_partitions/cutflow_efficiency.py` when it
builds ROOT indices for both trees:

```python
chain.BuildIndex("runId", "eventId")
friend_chain.BuildIndex("runId", "eventId")
```

For large aggregations with hundreds of region partition files, this step can
be much slower than adding files or counting cuts.

## Current Conservative Logic

The current evaluation script treats `sndData` and `cutFlowSummary` as friend
trees matched by:

```text
runId, eventId
```

This is conservative because it does not assume that entry `i` in `sndData`
matches entry `i` in `cutFlowSummary`.

The cost is that ROOT has to build lookup indices over large `TChain`s.

## Upstream Production Logic

The upstream code strongly suggests that the trees are intended to be
entry-aligned.

In `convertData/digi_2_features.py`, feature entries are written by looping
over a selected `cutFlowSummary` entry list:

```python
cutflow_entry = elist.GetEntry(i)
preSelect_tree.GetEntry(cutflow_entry)
...
kept_cutflow_entries.Enter(cutflow_entry)
...
new_tree.Fill()
```

At the end, the matching `cutFlowSummary` entries are copied from the same
kept entry list:

```python
preSelect_tree.SetEntryList(kept_cutflow_entries)
cutflow_selected = preSelect_tree.CopyTree("")
cutflow_selected.SetName("cutFlowSummary")
```

This means the source feature ROOT files should have:

```text
sndData entry i <-> cutFlowSummary entry i
```

In `convertData/build_region_partitions.py`, region partitions are then built
by filtering `sndData`:

```python
feature_rdf = ROOT.RDataFrame(feature_chain).Filter(selection_expr)
```

and filtering `cutFlowSummary` with the same region selection, using `sndData`
as a friend:

```python
cutflow_chain.AddFriend(feature_chain, "feat")
cutflow_rdf = ROOT.RDataFrame(cutflow_chain)
cutflow_rdf = cutflow_rdf.Filter(cutflow_selection_expr)
```

Both trees are then snapshotted into the same partition ROOT file:

```python
Snapshot("sndData", ..., mode="RECREATE")
Snapshot("cutFlowSummary", ..., mode="UPDATE")
```

## Important Caveat

Region partition production currently enables ROOT implicit multithreading:

```yaml
performance:
  rdf_threads: 4
```

Snakemake passes this to `build_region_partitions.py`, which calls:

```python
ROOT.EnableImplicitMT(4)
```

With separate `RDataFrame.Snapshot` calls under implicit MT, output entry order
is less obvious than in single-threaded mode. The selected event set should be
the same, but before relying on entry-order friend matching, the produced
partition files should be validated.

## What Disabling Implicit MT Would Change

If region partition production is run single-threaded, for example with:

```yaml
performance:
  rdf_threads: 0
```

or by otherwise avoiding `ROOT.EnableImplicitMT`, the entry-order alignment
argument becomes much stronger.

Single-threaded snapshots should preserve deterministic source order:

```text
feature_rdf.Filter(selection).Snapshot("sndData", ...)
cutflow_rdf.Filter(same selection).Snapshot("cutFlowSummary", ...)
```

Expected result:

```text
sndData entry 0 == cutFlowSummary entry 0
sndData entry 1 == cutFlowSummary entry 1
...
```

Tradeoff: partition production may be slower, and existing partition files
produced with implicit MT would still need validation or regeneration.

## Recommended Validation Before Changing Cutflow Code

Before removing `BuildIndex`, validate representative region partition files.

For each file, compare entry-by-entry:

```text
sndData.GetEntries() == cutFlowSummary.GetEntries()
sndData.runId[i] == cutFlowSummary.runId[i]
sndData.eventId[i] == cutFlowSummary.eventId[i]
```

If available and useful, also compare:

```text
sndData.eventIndex[i]
cutFlowSummary.entry[i] or another original-entry branch
```

Validate several categories:

```text
real_data
CC_nue
CC_nue_EventBuilder
muonDIS or muon
one no-veto region
one has-veto region
one inside-fiducial region
one outside-fiducial region
```

## Possible Future Implementation

If validation confirms entry alignment, add an explicit cutflow config option:

```yaml
metadata:
  tree_name: sndData
  friend_tree_name: cutFlowSummary
  friend_match_mode: entry_aligned
```

Then in `cutflow_efficiency.py`:

```text
friend_match_mode: indexed
  build runId/eventId indices before AddFriend

friend_match_mode: entry_aligned
  skip BuildIndex and call AddFriend directly
```

This keeps the current conservative behavior available while allowing the fast
entry-aligned path for validated region partition files.

## Bottom Line

The upstream logic is designed as if `sndData` and `cutFlowSummary` are
entry-aligned. The only reason not to immediately skip `BuildIndex` is the
current use of ROOT implicit MT during partition snapshots. Validate the
existing files first, or regenerate partitions single-threaded, then switch
cutflow evaluation to entry-aligned friend matching.
