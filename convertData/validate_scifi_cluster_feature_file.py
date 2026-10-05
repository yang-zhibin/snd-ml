#!/usr/bin/env python3
"""Validate the fixed SciFi clustering schema and event-level closure."""
import argparse
import math
import sys

import ROOT

from scifi_plane_clustering import SCHEMA_VERSION, scalar_feature_dtypes

def close(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-7)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("feature")
    parser.add_argument("--tree",default="sndData")
    parser.add_argument("--hit3d")
    parser.add_argument("--hit3d-tree",default="hit3D")
    parser.add_argument("--max-events",type=int,default=0)
    args=parser.parse_args()
    source=ROOT.TFile.Open(args.feature)
    if not source or source.IsZombie(): raise RuntimeError("cannot open "+args.feature)
    tree=source.Get(args.tree)
    if not tree: raise RuntimeError("missing tree "+args.tree)
    names={branch.GetName() for branch in tree.GetListOfBranches()}
    missing=[name for name,_ in scalar_feature_dtypes() if name not in names]
    if missing: raise RuntimeError("missing cluster branches: "+", ".join(missing))
    marker=source.Get("metadata_scifi_cluster_schema")
    if not marker or marker.GetTitle()!=SCHEMA_VERSION:
        raise RuntimeError("missing or wrong SciFi cluster schema marker")
    hit_tree=None; hit_source=None
    if args.hit3d:
        hit_source=ROOT.TFile.Open(args.hit3d)
        hit_tree=hit_source.Get(args.hit3d_tree) if hit_source else None
        if not hit_tree: raise RuntimeError("missing Hit3D tree")
        if hit_tree.GetEntries()!=tree.GetEntries(): raise RuntimeError("feature/Hit3D entry-count mismatch")
    limit=tree.GetEntries() if args.max_events<=0 else min(tree.GetEntries(),args.max_events)
    for entry in range(limit):
        tree.GetEntry(entry)
        if hit_tree:
            hit_tree.GetEntry(entry)
            feature_id=(int(tree.runId),int(tree.eventId),int(tree.eventIndex))
            hit_id=(int(hit_tree.runId),int(hit_tree.eventId),int(hit_tree.eventIndex))
            if feature_id!=hit_id: raise RuntimeError("identity mismatch at entry {}".format(entry))
        vc=int(tree.scifi_vertical_cluster_count); hc=int(tree.scifi_horizontal_cluster_count)
        vh=int(tree.scifi_vertical_total_nhits); hh=int(tree.scifi_horizontal_total_nhits)
        vq=float(tree.scifi_vertical_total_qdc); hq=float(tree.scifi_horizontal_total_qdc)
        checks=((tree.scifi_cluster_total_count,vc+hc,"cluster count"),
                (tree.scifi_cluster_total_nhits,vh+hh,"hit count"),
                (tree.scifi_cluster_total_qdc,vq+hq,"QDC"),
                (tree.scifi_cluster_total_nhits,tree.count_scifi,"existing SciFi hit"),
                (tree.scifi_cluster_total_qdc,tree.qdc_scifi,"existing SciFi QDC"),
                (tree.scifi_other_clusters_count,
                 tree.scifi_vertical_other_cluster_count+tree.scifi_horizontal_other_cluster_count,
                 "other count"),
                (tree.scifi_other_clusters_nhits,
                 tree.scifi_vertical_other_cluster_nhits+tree.scifi_horizontal_other_cluster_nhits,
                 "other hits"),
                (tree.scifi_other_clusters_qdc,
                 tree.scifi_vertical_other_cluster_qdc+tree.scifi_horizontal_other_cluster_qdc,
                 "other QDC"))
        for observed,expected,label in checks:
            if not close(observed,expected):
                raise RuntimeError("{} closure at entry {}".format(label,entry))
        for name,_ in scalar_feature_dtypes():
            if not math.isfinite(float(getattr(tree,name))):
                raise RuntimeError("non-finite {} at entry {}".format(name,entry))
    print("validated {} entries; schema={}".format(limit,SCHEMA_VERSION))
    source.Close()
    if hit_source: hit_source.Close()

if __name__=="__main__":
    try: main()
    except Exception as error:
        print("ERROR: {}".format(error),file=sys.stderr)
        raise
