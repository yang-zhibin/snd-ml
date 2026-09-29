#!/usr/bin/env python3
import argparse
import math

import ROOT


def valid_time(value):
    try:
        value = float(value)
    except Exception:
        return False
    return math.isfinite(value) and value > -900.0


def iter_mufilter_all_times(hit):
    try:
        raw_times = hit.GetAllTimes()
    except TypeError:
        try:
            raw_times = hit.GetAllTimes(False)
        except Exception:
            raw_times = {}
    except Exception:
        raw_times = {}

    try:
        iterator = raw_times.items()
    except AttributeError:
        iterator = raw_times

    for item in iterator:
        try:
            channel, value = item
        except Exception:
            try:
                channel, value = item[0], item[1]
            except Exception:
                continue
        yield int(channel), float(value)


def time_range(values):
    values = [float(value) for value in values if valid_time(value)]
    if not values:
        return None
    return min(values), max(values), max(values) - min(values), len(values)


def update_counter(counters, key, values, tolerance):
    result = time_range(values)
    if result is None:
        return None

    counters[f"{key}_events"] += 1
    if result[2] <= tolerance:
        counters[f"{key}_same"] += 1
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check whether digi hit times are identical within each event."
    )
    parser.add_argument("digi_file", help="Input digi ROOT file")
    parser.add_argument("--tree", default="cbmsim", help="Input tree name")
    parser.add_argument("--max-events", type=int, default=0, help="Maximum events to scan, 0 means all")
    parser.add_argument("--examples", type=int, default=10, help="Number of example events to print")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Same-time tolerance in ns")
    parser.add_argument("--list-branches", action="store_true", help="Print branches before scanning")
    args = parser.parse_args()

    root_file = ROOT.TFile.Open(args.digi_file)
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {args.digi_file}")

    tree = root_file.Get(args.tree)
    if not tree:
        raise RuntimeError(f"Tree '{args.tree}' not found in {args.digi_file}")

    if args.list_branches:
        print("Branches:")
        for branch in tree.GetListOfBranches():
            print(f"  {branch.GetName()}")

    n_entries = int(tree.GetEntries())
    n_scan = n_entries if args.max_events <= 0 else min(n_entries, args.max_events)
    print(f"Input: {args.digi_file}")
    print(f"Tree: {args.tree}")
    print(f"Entries: {n_entries}")
    print(f"Scanning: {n_scan}")

    counters = {
        "scifi_events": 0,
        "scifi_same": 0,
        "mufilter_gettime_events": 0,
        "mufilter_gettime_same": 0,
        "mufilter_alltimes_events": 0,
        "mufilter_alltimes_same": 0,
        "all_events": 0,
        "all_same": 0,
    }

    examples_printed = 0

    for entry in range(n_scan):
        tree.GetEntry(entry)

        scifi_times = []
        mufilter_get_times = []
        mufilter_all_times = []

        if hasattr(tree, "Digi_ScifiHits"):
            for hit in tree.Digi_ScifiHits:
                if hit.isValid():
                    scifi_times.append(float(hit.GetTime()))

        if hasattr(tree, "Digi_MuFilterHits"):
            for hit in tree.Digi_MuFilterHits:
                if not hit.isValid():
                    continue

                mufilter_get_times.append(float(hit.GetTime()))
                channel_times = [value for _, value in iter_mufilter_all_times(hit)]
                if channel_times:
                    mufilter_all_times.extend(channel_times)
                else:
                    mufilter_all_times.append(float(hit.GetTime()))

        all_times = scifi_times + mufilter_all_times

        scifi_range = update_counter(counters, "scifi", scifi_times, args.tolerance)
        mufilter_get_range = update_counter(
            counters,
            "mufilter_gettime",
            mufilter_get_times,
            args.tolerance,
        )
        mufilter_all_range = update_counter(
            counters,
            "mufilter_alltimes",
            mufilter_all_times,
            args.tolerance,
        )
        all_range = update_counter(counters, "all", all_times, args.tolerance)

        should_print = examples_printed < args.examples and (
            all_range is not None
            or scifi_range is not None
            or mufilter_get_range is not None
            or mufilter_all_range is not None
        )
        if should_print:
            print(f"event {entry}")
            if scifi_range is not None:
                print(
                    "  SciFi GetTime:       "
                    f"n={scifi_range[3]} min={scifi_range[0]} max={scifi_range[1]} range={scifi_range[2]}"
                )
            if mufilter_get_range is not None:
                print(
                    "  MuFilter GetTime:    "
                    f"n={mufilter_get_range[3]} min={mufilter_get_range[0]} "
                    f"max={mufilter_get_range[1]} range={mufilter_get_range[2]}"
                )
            if mufilter_all_range is not None:
                print(
                    "  MuFilter GetAllTimes:"
                    f"n={mufilter_all_range[3]} min={mufilter_all_range[0]} "
                    f"max={mufilter_all_range[1]} range={mufilter_all_range[2]}"
                )
            if all_range is not None:
                print(
                    "  all detector times:  "
                    f"n={all_range[3]} min={all_range[0]} max={all_range[1]} range={all_range[2]}"
                )
            examples_printed += 1

    print("Summary:")
    for label, key in [
        ("SciFi GetTime", "scifi"),
        ("MuFilter GetTime", "mufilter_gettime"),
        ("MuFilter GetAllTimes/fallback", "mufilter_alltimes"),
        ("All detector times", "all"),
    ]:
        n_events = counters[f"{key}_events"]
        n_same = counters[f"{key}_same"]
        fraction = n_same / n_events if n_events else 0.0
        print(f"  {label}: same-time events {n_same}/{n_events} ({fraction:.6f})")

    root_file.Close()


if __name__ == "__main__":
    main()

# python3 /afs/cern.ch/user/z/zhibin/work/snd-ml/on_going_work/hit_time_raw_check/check_digi_hit_time.py \
#   /eos/experiment/sndlhc/users/cvilela/muonDIS_prod_20240826/muonDIS_2001/sndLHC.muonDIS-TGeant4_digCPP.root \
#   --max-events 1000 \
#   --examples 10
