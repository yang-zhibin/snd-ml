import argparse
from collections import Counter

import ROOT


REQUIRED_VECTOR_BRANCHES = [
    "hit_x",
    "hit_y",
    "hit_z",
    "hit_qdc",
    "hit_station",
    "hit_detType",
    "hit_ix",
    "hit_iy",
    "hit_iz",
    "hit_index_valid",
    "hit_index_type",
    "hit_v_channel",
    "hit_h_channel",
    "hit_v_qdc",
    "hit_h_qdc",
]

INDEX_TYPE_NAMES = {
    0: "unknown",
    1: "scifi_crossed",
    2: "us_voxel",
    3: "ds_crossed",
    4: "ds_single_orientation_voxel",
    5: "scifi_vertical_only",
    6: "scifi_horizontal_only",
    7: "ds_vertical_only",
    8: "ds_horizontal_only",
}

DET_TYPE_NAMES = {
    1: "SciFi",
    2: "US",
    3: "DS",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Check hit3D index branches and basic invariants.")
    parser.add_argument("path", help="Input hit3D ROOT file")
    parser.add_argument("--tree", default="hit3D", help="Tree name")
    parser.add_argument("--max-events", type=int, default=None, help="Limit number of events to inspect")
    parser.add_argument("--show-branches", action="store_true", help="Print all tree branch names")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any check fails")
    return parser.parse_args()


def format_counter(counter, names):
    lines = []
    for key, value in sorted(counter.items()):
        label = names.get(key, "unknown_code")
        lines.append(f"  {key:>3}  {label:<32} {value}")
    return "\n".join(lines) if lines else "  (none)"


def main():
    args = parse_args()

    root_file = ROOT.TFile.Open(args.path)
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open ROOT file: {args.path}")

    tree = root_file.Get(args.tree)
    if not tree:
        raise KeyError(f"Tree '{args.tree}' not found in {args.path}")

    branch_names = [branch.GetName() for branch in tree.GetListOfBranches()]
    if args.show_branches:
        print("Branches:")
        for name in branch_names:
            print(f"  {name}")

    missing = [name for name in REQUIRED_VECTOR_BRANCHES if name not in branch_names]
    n_entries = int(tree.GetEntries())
    n_check = n_entries if args.max_events is None else min(n_entries, int(args.max_events))

    det_type_counts = Counter()
    index_type_counts = Counter()
    bad_lengths = []
    bad_crossed = []
    bad_single_orientation = []
    bad_voxel = []
    total_hits = 0

    for entry in range(n_check):
        tree.GetEntry(entry)
        arrays = {name: getattr(tree, name) for name in REQUIRED_VECTOR_BRANCHES if name not in missing}
        lengths = {name: len(value) for name, value in arrays.items()}
        if len(set(lengths.values())) > 1:
            bad_lengths.append((entry, lengths))
            continue

        if not lengths:
            continue

        total_hits += next(iter(lengths.values()))
        for det_type, index_type, index_valid, ix, iy, iz in zip(
            tree.hit_detType,
            tree.hit_index_type,
            tree.hit_index_valid,
            tree.hit_ix,
            tree.hit_iy,
            tree.hit_iz,
        ):
            det_type = int(det_type)
            index_type = int(index_type)
            index_valid = int(index_valid)
            ix = int(ix)
            iy = int(iy)
            iz = int(iz)

            det_type_counts[det_type] += 1
            index_type_counts[index_type] += 1

            if index_type in (1, 3):
                if index_valid != 1 or ix < 0 or iy < 0 or iz < 0:
                    bad_crossed.append((entry, det_type, index_type, index_valid, ix, iy, iz))
            elif index_type in (5, 6, 7, 8):
                if index_valid != 0:
                    bad_single_orientation.append((entry, det_type, index_type, index_valid, ix, iy, iz))
            elif index_type in (2, 4):
                if index_valid != 1 or ix < 0 or iy < 0 or iz < 0:
                    bad_voxel.append((entry, det_type, index_type, index_valid, ix, iy, iz))

    root_file.Close()

    print(f"File: {args.path}")
    print(f"Tree: {args.tree}")
    print(f"Entries: {n_entries}")
    print(f"Checked entries: {n_check}")
    print(f"Total checked hits: {total_hits}")
    print(f"Missing required vector branches: {missing}")
    print("")
    print("hit_detType counts:")
    print(format_counter(det_type_counts, DET_TYPE_NAMES))
    print("")
    print("hit_index_type counts:")
    print(format_counter(index_type_counts, INDEX_TYPE_NAMES))
    print("")
    print(f"Bad vector-length events: {len(bad_lengths)}")
    print(f"Bad crossed-index hits: {len(bad_crossed)}")
    print(f"Bad single-orientation hits: {len(bad_single_orientation)}")
    print(f"Bad voxel-index hits: {len(bad_voxel)}")

    for label, values in (
        ("bad vector-length event", bad_lengths),
        ("bad crossed-index hit", bad_crossed),
        ("bad single-orientation hit", bad_single_orientation),
        ("bad voxel-index hit", bad_voxel),
    ):
        for item in values[:5]:
            print(f"Example {label}: {item}")

    failed = bool(missing or bad_lengths or bad_crossed or bad_single_orientation or bad_voxel)
    if failed:
        print("Result: FAILED")
        if args.strict:
            raise SystemExit(1)
    else:
        print("Result: OK")


if __name__ == "__main__":
    main()
