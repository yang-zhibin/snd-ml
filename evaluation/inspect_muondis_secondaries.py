#!/usr/bin/env python3

import argparse
import csv
import glob
import math
import os
import re
from array import array
from collections import Counter, defaultdict

import ROOT

import find_had_scale_factor as common

ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)
ROOT.TH1.SetDefaultSumw2()


PARTICLE_GROUP_ORDER = [
    "gamma",
    "neutron_all",
    "neutral_kaon",
    "charged_kaon",
    "charged_pion",
    "proton",
    "muon",
    "electron",
    "other_neutral",
    "other_charged",
    "unknown",
]

CHARGE_GROUP_ORDER = ["charged", "neutral", "unknown"]
CHARGED_PARTICLE_GROUPS = [
    "charged_kaon",
    "charged_pion",
    "proton",
    "muon",
    "electron",
    "other_charged",
]
NEUTRAL_PARTICLE_GROUPS = [
    "gamma",
    "neutron_all",
    "neutral_kaon",
    "other_neutral",
]
GROUP_COLORS = {
    "charged": ROOT.kRed + 1,
    "neutral": ROOT.kBlue + 1,
    "unknown": ROOT.kGray + 1,
    "gamma": ROOT.kOrange + 7,
    "neutron_all": ROOT.kAzure + 1,
    "neutral_kaon": ROOT.kCyan + 2,
    "charged_kaon": ROOT.kMagenta + 1,
    "charged_pion": ROOT.kGreen + 2,
    "proton": ROOT.kRed - 4,
    "muon": ROOT.kViolet + 1,
    "electron": ROOT.kOrange - 3,
    "other_neutral": ROOT.kBlue - 7,
    "other_charged": ROOT.kPink + 6,
}
PDG_COLORS = [
    ROOT.kRed + 1,
    ROOT.kBlue + 1,
    ROOT.kGreen + 2,
    ROOT.kMagenta + 1,
    ROOT.kOrange + 7,
    ROOT.kViolet + 1,
    ROOT.kCyan + 2,
    ROOT.kAzure + 1,
    ROOT.kPink + 6,
    ROOT.kTeal + 3,
    ROOT.kSpring + 5,
    ROOT.kYellow + 2,
    ROOT.kGray + 2,
]
ENERGY_BRANCH_CANDIDATES = [
    "secondary_energy",
    "secondary_E",
    "secondary_e",
    "secondary_kinetic_energy",
    "secondary_ke",
]


def parse_cut_list(cuts_text):
    cuts = [int(token) for token in cuts_text.replace(",", " ").split()]
    if not cuts:
        raise ValueError("No cut indices were provided")

    missing = [idx for idx in cuts if idx not in common.cut_info]
    if missing:
        raise KeyError(f"Cut indices not defined in cut_info: {missing}")

    return cuts


def build_cut_steps(cuts):
    steps = []
    cumulative = []
    for idx in cuts:
        cumulative.append(idx)
        cut_key = "_".join(str(item) for item in cumulative)
        cut_label = " ".join(str(item) for item in cumulative)
        steps.append(
            {
                "cut_key": cut_key,
                "cut_label": cut_label,
                "base_cut": cut_label,
            }
        )
    return steps


def tree_has_branch(chain, branch_name):
    return bool(chain.GetListOfBranches().FindObject(branch_name))


def make_rdataframe(chain):
    return ROOT.RDataFrame(chain)


def rdf_has_column(df, branch_name):
    try:
        df.GetColumnType(branch_name)
        return True
    except Exception:
        return False


def resolve_energy_branch(chain, requested):
    if requested and requested != "auto":
        return requested if tree_has_branch(chain, requested) else None

    for candidate in ENERGY_BRANCH_CANDIDATES:
        if tree_has_branch(chain, candidate):
            return candidate

    return None


def pdg_name(pdg_code):
    particle = ROOT.TDatabasePDG.Instance().GetParticle(int(pdg_code))
    if particle:
        return str(particle.GetName())
    return f"unknown_{int(pdg_code)}"


def charge_group(pdg_code):
    particle = ROOT.TDatabasePDG.Instance().GetParticle(int(pdg_code))
    if not particle:
        return "unknown"

    try:
        charge = float(particle.Charge())
    except Exception:
        return "unknown"

    return "neutral" if abs(charge) < 1e-12 else "charged"


def particle_group(pdg_code):
    pdg = int(pdg_code)
    apdg = abs(pdg)

    if pdg == 22:
        return "gamma"
    if apdg == 2112:
        return "neutron_all"
    if apdg in {130, 310, 311}:
        return "neutral_kaon"
    if apdg == 321:
        return "charged_kaon"
    if apdg == 211:
        return "charged_pion"
    if apdg == 2212:
        return "proton"
    if apdg == 13:
        return "muon"
    if apdg == 11:
        return "electron"

    group = charge_group(pdg)
    if group == "neutral":
        return "other_neutral"
    if group == "charged":
        return "other_charged"
    return "unknown"


def make_muondis_chain(input_dir, tree_name):
    candidates = sorted(glob.glob(os.path.join(input_dir, "hist_MC_muonDIS*.root")))
    files = []
    chain = ROOT.TChain(tree_name)

    for filepath in candidates:
        added = chain.Add(filepath)
        if added <= 0:
            print(f"[skip] could not add {filepath}")
            continue
        files.append(filepath)
        print(f"[ok] {os.path.basename(filepath)}")

    if not files:
        raise RuntimeError(
            "No muonDIS files found. Expected files like hist_MC_muonDIS_Max10-1.root "
            "in the input directory."
        )

    print(f"[chain] muonDIS: {len(files)} files, entries={chain.GetEntries()}")
    return chain


def rdf_take_type(df, branch_name, fallback_types):
    try:
        column_type = str(df.GetColumnType(branch_name))
    except Exception:
        column_type = ""

    normalized = re.sub(r"\s+", " ", column_type.strip())
    normalized = normalized.replace("Int_t", "int")
    normalized = normalized.replace("UInt_t", "unsigned int")
    normalized = normalized.replace("Float_t", "float")
    normalized = normalized.replace("Double_t", "double")
    compact = normalized.replace(" ", "")
    if compact:
        if compact.startswith("ROOT::VecOps::RVec<"):
            return normalized
        if compact.startswith("ROOT::RVec<"):
            return normalized.replace("ROOT::RVec<", "ROOT::VecOps::RVec<", 1)
        if compact.startswith("std::vector<"):
            return normalized
        if compact.startswith("vector<"):
            return f"std::{normalized}"

    return fallback_types[0]


def take_vector(node, branch_name, type_name):
    return node.Take[type_name](branch_name)


def identifiers_in_expr(expr):
    return set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr or ""))


def missing_branches_for_expr(chain, expr):
    known_functions = {"abs", "fabs", "sqrt", "pow", "log", "log10", "exp", "min", "max"}
    missing = []
    for identifier in sorted(identifiers_in_expr(expr)):
        if identifier in known_functions:
            continue
        if not tree_has_branch(chain, identifier):
            missing.append(identifier)
    return missing


def build_selection(chain, base_cut, extra_cut):
    selected_exprs = []
    cut_names = []
    skipped_cuts = []

    if base_cut:
        for token in base_cut.replace(",", " ").split():
            idx = int(token)
            if idx not in common.cut_info:
                raise KeyError(f"Cut index {idx} not found in cut_info")
            name, expr = common.cut_info[idx]
            missing = missing_branches_for_expr(chain, expr)
            if missing:
                skipped_cuts.append(
                    {
                        "cut_index": idx,
                        "cut_name": name,
                        "expr": expr,
                        "missing_branches": ",".join(missing),
                    }
                )
                print(
                    f"[warn] cut {idx} ({name}) cannot be applied; "
                    f"missing branches: {', '.join(missing)}"
                )
                continue
            cut_names.append(name)
            selected_exprs.append(f"({expr})")

    if extra_cut and extra_cut.strip():
        missing = missing_branches_for_expr(chain, extra_cut)
        if missing:
            raise RuntimeError(
                f"Extra cut cannot be applied because branches are missing: {', '.join(missing)}"
            )
        selected_exprs.append(f"({extra_cut.strip()})")
        cut_names.append("extra_cut")

    selection = " && ".join(selected_exprs) if selected_exprs else "1"
    return selection, cut_names, skipped_cuts


def count_secondaries(df, selection, cut_key, energy_branch=None, max_events=None):
    node = df if selection == "1" else df.Filter(selection)
    count_result = None
    if max_events is not None:
        count_result = node.Count()
        node = node.Range(int(max_events))

    pdg_type = rdf_take_type(
        df,
        "secondary_pdg",
        ["ROOT::VecOps::RVec<int>", "std::vector<int>"],
    )
    pdg_result = take_vector(node, "secondary_pdg", pdg_type)

    nsecondary_result = None
    nsecondary_type = None
    if rdf_has_column(df, "nSecondary"):
        nsecondary_type = rdf_take_type(df, "nSecondary", ["int"])
        nsecondary_result = take_vector(node, "nSecondary", nsecondary_type)

    energy_result = None
    energy_type = None
    if energy_branch:
        energy_type = rdf_take_type(
            df,
            energy_branch,
            [
                "ROOT::VecOps::RVec<float>",
                "ROOT::VecOps::RVec<double>",
                "std::vector<float>",
                "std::vector<double>",
            ],
        )
        energy_result = take_vector(node, energy_branch, energy_type)

    pdg_arrays = pdg_result.GetValue()
    nsecondary_values = nsecondary_result.GetValue() if nsecondary_result else None
    energy_arrays = energy_result.GetValue() if energy_result else None

    processed_events = len(pdg_arrays)
    selected_events = int(count_result.GetValue()) if count_result else processed_events
    print(f"[select] {cut_key:35s}: {selected_events} events")
    print(f"[rdf] secondary_pdg type: {pdg_type}")
    if nsecondary_type:
        print(f"[rdf] nSecondary type: {nsecondary_type}")
    if energy_type:
        print(f"[rdf] {energy_branch} type: {energy_type}")
    if max_events is not None and processed_events < selected_events:
        print(f"[limit] {cut_key:35s}: processing first {processed_events} of {selected_events} selected events")

    pdg_particle_counts = Counter()
    pdg_event_counts = Counter()
    group_particle_counts = Counter()
    group_event_counts = Counter()
    charge_particle_counts = Counter()
    charge_event_counts = Counter()
    energy_by_group = defaultdict(list)
    energy_by_pdg = defaultdict(list)
    secondary_counts_per_event = []

    total_secondary_particles = 0
    energy_events_used = 0
    energy_mismatch_events = 0
    energy_particles_used = 0

    for event_index, event_pdgs_vector in enumerate(pdg_arrays):
        pdgs = [int(pdg) for pdg in event_pdgs_vector]
        if nsecondary_values is not None:
            secondary_counts_per_event.append(int(nsecondary_values[event_index]))
        else:
            secondary_counts_per_event.append(len(pdgs))

        event_pdgs = set(pdgs)
        event_groups = set()
        event_charge_groups = set()

        for pdg in pdgs:
            group = particle_group(pdg)
            charge = charge_group(pdg)
            pdg_particle_counts[pdg] += 1
            group_particle_counts[group] += 1
            charge_particle_counts[charge] += 1
            event_groups.add(group)
            event_charge_groups.add(charge)
            total_secondary_particles += 1

        for pdg in event_pdgs:
            pdg_event_counts[pdg] += 1
        for group in event_groups:
            group_event_counts[group] += 1
        for charge in event_charge_groups:
            charge_event_counts[charge] += 1

        if energy_arrays is not None:
            energies = [energy for energy in energy_arrays[event_index]]
            if len(energies) != len(pdgs):
                energy_mismatch_events += 1
                continue

            energy_events_used += 1
            for pdg, energy in zip(pdgs, energies):
                try:
                    energy = float(energy)
                except Exception:
                    continue
                if not math.isfinite(energy) or energy <= 0:
                    continue
                energy_by_group[particle_group(pdg)].append(energy)
                energy_by_pdg[pdg].append(energy)
                energy_particles_used += 1

    return {
        "cut_key": cut_key,
        "selection": selection,
        "selected_events": selected_events,
        "processed_events": processed_events,
        "total_secondary_particles": total_secondary_particles,
        "pdg_particle_counts": pdg_particle_counts,
        "pdg_event_counts": pdg_event_counts,
        "group_particle_counts": group_particle_counts,
        "group_event_counts": group_event_counts,
        "charge_particle_counts": charge_particle_counts,
        "charge_event_counts": charge_event_counts,
        "energy_by_group": energy_by_group,
        "energy_by_pdg": energy_by_pdg,
        "secondary_counts_per_event": secondary_counts_per_event,
        "energy_events_used": energy_events_used,
        "energy_mismatch_events": energy_mismatch_events,
        "energy_particles_used": energy_particles_used,
    }


def pdg_rows(result):
    total_particles = result["total_secondary_particles"]
    processed_events = result["processed_events"]
    rows = []

    for pdg, count in result["pdg_particle_counts"].items():
        event_count = result["pdg_event_counts"].get(pdg, 0)
        rows.append(
            {
                "pdg_code": pdg,
                "pdg_name": pdg_name(pdg),
                "particle_group": particle_group(pdg),
                "charge_group": charge_group(pdg),
                "particle_count": count,
                "particle_fraction": count / total_particles if total_particles else 0.0,
                "event_count": event_count,
                "event_fraction": event_count / processed_events if processed_events else 0.0,
            }
        )

    rows.sort(key=lambda row: (-row["particle_count"], row["pdg_code"]))
    return rows


def charge_rows(result):
    total_particles = result["total_secondary_particles"]
    processed_events = result["processed_events"]
    rows = []

    for group in CHARGE_GROUP_ORDER:
        count = result["charge_particle_counts"].get(group, 0)
        event_count = result["charge_event_counts"].get(group, 0)
        rows.append(
            {
                "charge_group": group,
                "particle_count": count,
                "particle_fraction": count / total_particles if total_particles else 0.0,
                "event_count": event_count,
                "event_fraction": event_count / processed_events if processed_events else 0.0,
            }
        )

    return rows


def write_pdg_txt(outdir, step, result, rows):
    path = os.path.join(outdir, f"secondary_pdg_table__cuts_{step['cut_key']}.txt")
    with open(path, "w") as fout:
        fout.write(f"cut_key: {step['cut_key']}\n")
        fout.write(f"cut_label: {step['cut_label']}\n")
        fout.write(f"base_cut: {result.get('base_cut', step['base_cut'])}\n")
        fout.write(f"extra_cut: {result.get('extra_cut', '')}\n")
        fout.write(f"extra_cut_key: {result.get('extra_cut_key', '')}\n")
        fout.write(f"selection: {result['selection']}\n")
        fout.write(f"selected_events: {result['selected_events']}\n")
        fout.write(f"processed_events: {result['processed_events']}\n")
        if result.get("skipped_cuts"):
            fout.write("skipped_cuts:\n")
            for skipped in result["skipped_cuts"]:
                fout.write(
                    "  "
                    f"{skipped['cut_index']} {skipped['cut_name']}: "
                    f"missing {skipped['missing_branches']} ; expr={skipped['expr']}\n"
                )
        fout.write(f"total_secondary_particles: {result['total_secondary_particles']}\n")
        fout.write(f"energy_events_used: {result['energy_events_used']}\n")
        fout.write(f"energy_mismatch_events: {result['energy_mismatch_events']}\n\n")

        headers = (
            f"{'pdg_code':>12}  {'pdg_name':<24}  {'particle_group':<16}  "
            f"{'charge':<8}  {'particle_count':>15}  {'particle_frac':>14}  "
            f"{'event_count':>12}  {'event_frac':>12}\n"
        )
        fout.write(headers)
        fout.write("-" * (len(headers) - 1) + "\n")
        for row in rows:
            fout.write(
                f"{row['pdg_code']:12d}  {row['pdg_name']:<24}  {row['particle_group']:<16}  "
                f"{row['charge_group']:<8}  {row['particle_count']:15d}  "
                f"{row['particle_fraction']:14.8e}  {row['event_count']:12d}  "
                f"{row['event_fraction']:12.8e}\n"
            )
    print(f"[write] {path}")


def write_charge_txt(outdir, step, result, rows):
    path = os.path.join(outdir, f"charge_summary__cuts_{step['cut_key']}.txt")
    with open(path, "w") as fout:
        fout.write(f"cut_key: {step['cut_key']}\n")
        fout.write(f"cut_label: {step['cut_label']}\n")
        fout.write(f"base_cut: {result.get('base_cut', step['base_cut'])}\n")
        fout.write(f"extra_cut: {result.get('extra_cut', '')}\n")
        fout.write(f"extra_cut_key: {result.get('extra_cut_key', '')}\n")
        fout.write(f"selection: {result['selection']}\n")
        fout.write(f"selected_events: {result['selected_events']}\n")
        fout.write(f"processed_events: {result['processed_events']}\n")
        if result.get("skipped_cuts"):
            fout.write("skipped_cuts:\n")
            for skipped in result["skipped_cuts"]:
                fout.write(
                    "  "
                    f"{skipped['cut_index']} {skipped['cut_name']}: "
                    f"missing {skipped['missing_branches']} ; expr={skipped['expr']}\n"
                )
        fout.write(f"total_secondary_particles: {result['total_secondary_particles']}\n\n")
        fout.write(
            f"{'charge_group':<12}  {'particle_count':>15}  {'particle_frac':>14}  "
            f"{'event_count':>12}  {'event_frac':>12}\n"
        )
        fout.write("-" * 76 + "\n")
        for row in rows:
            fout.write(
                f"{row['charge_group']:<12}  {row['particle_count']:15d}  "
                f"{row['particle_fraction']:14.8e}  {row['event_count']:12d}  "
                f"{row['event_fraction']:12.8e}\n"
            )
    print(f"[write] {path}")


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path}")


def write_all_csv_outputs(outdir, steps, results_by_cut, pdg_rows_by_cut, charge_rows_by_cut):
    pdg_all_rows = []
    charge_all_rows = []
    cut_flow_rows = []

    for step in steps:
        result = results_by_cut[step["cut_key"]]
        for row in pdg_rows_by_cut[step["cut_key"]]:
            pdg_all_rows.append(
                {
                    "cut_key": step["cut_key"],
                    "cut_label": step["cut_label"],
                    "base_cut": result.get("base_cut", step["base_cut"]),
                    "extra_cut": result.get("extra_cut", ""),
                    "extra_cut_key": result.get("extra_cut_key", ""),
                    "selection": result["selection"],
                    **row,
                }
            )

        for row in charge_rows_by_cut[step["cut_key"]]:
            charge_all_rows.append(
                {
                    "cut_key": step["cut_key"],
                    "cut_label": step["cut_label"],
                    "base_cut": result.get("base_cut", step["base_cut"]),
                    "extra_cut": result.get("extra_cut", ""),
                    "extra_cut_key": result.get("extra_cut_key", ""),
                    **row,
                }
            )

        selected_events = result["selected_events"]
        processed_events = result["processed_events"]
        total_particles = result["total_secondary_particles"]
        cut_flow_rows.append(
            {
                "cut_key": step["cut_key"],
                "cut_label": step["cut_label"],
                "base_cut": result.get("base_cut", step["base_cut"]),
                "extra_cut": result.get("extra_cut", ""),
                "extra_cut_key": result.get("extra_cut_key", ""),
                "selection": result["selection"],
                "skipped_cuts": ";".join(
                    f"{item['cut_index']}:{item['cut_name']} missing {item['missing_branches']}"
                    for item in result.get("skipped_cuts", [])
                ),
                "selected_events": selected_events,
                "processed_events": processed_events,
                "total_secondary_particles": total_particles,
                "avg_secondaries_per_event": (
                    total_particles / processed_events if processed_events else 0.0
                ),
                "energy_events_used": result["energy_events_used"],
                "energy_mismatch_events": result["energy_mismatch_events"],
                "energy_particles_used": result["energy_particles_used"],
            }
        )

    write_csv(
        os.path.join(outdir, "secondary_pdg_summary_all_cuts.csv"),
        pdg_all_rows,
        [
            "cut_key",
            "cut_label",
            "base_cut",
            "extra_cut",
            "extra_cut_key",
            "selection",
            "pdg_code",
            "pdg_name",
            "particle_group",
            "charge_group",
            "particle_count",
            "particle_fraction",
            "event_count",
            "event_fraction",
        ],
    )
    write_csv(
        os.path.join(outdir, "charge_summary_all_cuts.csv"),
        charge_all_rows,
        [
            "cut_key",
            "cut_label",
            "base_cut",
            "extra_cut",
            "extra_cut_key",
            "charge_group",
            "particle_count",
            "particle_fraction",
            "event_count",
            "event_fraction",
        ],
    )
    write_csv(
        os.path.join(outdir, "cut_flow_summary.csv"),
        cut_flow_rows,
        [
            "cut_key",
            "cut_label",
            "base_cut",
            "extra_cut",
            "extra_cut_key",
            "selection",
            "skipped_cuts",
            "selected_events",
            "processed_events",
            "total_secondary_particles",
            "avg_secondaries_per_event",
            "energy_events_used",
            "energy_mismatch_events",
            "energy_particles_used",
        ],
    )


def write_energy_summary_csv(outdir, steps, results_by_cut):
    rows = []
    for step in steps:
        result = results_by_cut[step["cut_key"]]
        for group in PARTICLE_GROUP_ORDER:
            energies = sorted(result["energy_by_group"].get(group, []))
            count = len(energies)
            sum_energy = sum(energies)
            mean_energy = sum_energy / count if count else 0.0
            median_energy = 0.0
            if count:
                mid = count // 2
                if count % 2:
                    median_energy = energies[mid]
                else:
                    median_energy = 0.5 * (energies[mid - 1] + energies[mid])
            rows.append(
                {
                    "cut_key": step["cut_key"],
                    "cut_label": step["cut_label"],
                    "base_cut": result.get("base_cut", step["base_cut"]),
                    "extra_cut": result.get("extra_cut", ""),
                    "extra_cut_key": result.get("extra_cut_key", ""),
                    "particle_group": group,
                    "charge_group": group_charge_label(group),
                    "particle_count": count,
                    "mean_energy": mean_energy,
                    "median_energy": median_energy,
                    "sum_energy": sum_energy,
                    "fraction_above_1GeV": fraction_above(energies, 1.0),
                    "fraction_above_10GeV": fraction_above(energies, 10.0),
                    "fraction_above_100GeV": fraction_above(energies, 100.0),
                }
            )

    write_csv(
        os.path.join(outdir, "secondary_energy_summary_all_cuts.csv"),
        rows,
        [
            "cut_key",
            "cut_label",
            "base_cut",
            "extra_cut",
            "extra_cut_key",
            "particle_group",
            "charge_group",
            "particle_count",
            "mean_energy",
            "median_energy",
            "sum_energy",
            "fraction_above_1GeV",
            "fraction_above_10GeV",
            "fraction_above_100GeV",
        ],
    )
    return rows


def fraction_above(sorted_values, threshold):
    if not sorted_values:
        return 0.0
    count = sum(1 for value in sorted_values if value > threshold)
    return count / len(sorted_values)


def group_charge_label(group):
    if group in {"gamma", "neutron_all", "neutral_kaon", "other_neutral"}:
        return "neutral"
    if group in {"charged_kaon", "charged_pion", "proton", "muon", "electron", "other_charged"}:
        return "charged"
    return "unknown"


def selection_labels(step, result):
    base_label = str(step.get("cut_key") or result.get("base_cut") or "none")
    extra_label = str(result.get("extra_cut_key") or "").strip()
    if not extra_label:
        extra_label = "none"
    return base_label, extra_label


def write_secondary_count_distribution_csv(outdir, steps, results_by_cut):
    rows = []
    for step in steps:
        result = results_by_cut[step["cut_key"]]
        counts = Counter(result["secondary_counts_per_event"])
        processed_events = result["processed_events"]
        for n_secondaries in sorted(counts):
            n_events = counts[n_secondaries]
            rows.append(
                {
                    "cut_key": step["cut_key"],
                    "cut_label": step["cut_label"],
                    "base_cut": result.get("base_cut", step["base_cut"]),
                    "extra_cut": result.get("extra_cut", ""),
                    "extra_cut_key": result.get("extra_cut_key", ""),
                    "n_secondaries": n_secondaries,
                    "n_events": n_events,
                    "event_fraction": n_events / processed_events if processed_events else 0.0,
                }
            )

    write_csv(
        os.path.join(outdir, "secondary_count_per_event_distribution_all_cuts.csv"),
        rows,
        [
            "cut_key",
            "cut_label",
            "base_cut",
            "extra_cut",
            "extra_cut_key",
            "n_secondaries",
            "n_events",
            "event_fraction",
        ],
    )


def write_secondary_count_distribution_plots(outdir, steps, results_by_cut):
    ROOT.gStyle.SetOptStat(0)

    for step in steps:
        result = results_by_cut[step["cut_key"]]
        counts_per_event = result["secondary_counts_per_event"]
        if not counts_per_event:
            print(f"[plot] no selected events for cuts {step['cut_key']}; skipping secondary-count plot")
            continue

        max_count = max(counts_per_event)
        n_bins = max_count + 1
        hist = ROOT.TH1D(
            f"h_secondary_count_{step['cut_key']}",
            "Secondary-particle multiplicity per event",
            n_bins,
            -0.5,
            max_count + 0.5,
        )
        hist.SetDirectory(0)
        hist.SetLineColor(ROOT.kBlue + 2)
        hist.SetLineWidth(2)
        hist.SetFillColor(ROOT.kAzure - 9)
        hist.SetFillStyle(1001)

        for count in counts_per_event:
            hist.Fill(count)

        canvas = ROOT.TCanvas(f"c_secondary_count_{step['cut_key']}", "", 850, 650)
        canvas.SetLeftMargin(0.12)
        canvas.SetRightMargin(0.06)
        canvas.SetTopMargin(0.08)
        canvas.SetBottomMargin(0.12)
        canvas.SetLogy()

        hist.GetXaxis().SetTitle("Number of secondary particles per event")
        hist.GetYaxis().SetTitle("Events")
        hist.GetYaxis().SetTitleOffset(1.25)
        hist.SetMinimum(0.5)
        hist.SetMaximum(max(1.0, hist.GetMaximum()) * 20.0)
        hist.Draw("HIST")

        text = ROOT.TText()
        text.SetTextAlign(13)
        text.SetTextSize(0.030)
        base_label, extra_label = selection_labels(step, result)
        text.DrawTextNDC(0.16, 0.84, f"base cut: {base_label}")
        text.DrawTextNDC(0.16, 0.79, f"extra cut: {extra_label}")
        text.DrawTextNDC(0.16, 0.74, f"selected events: {result['selected_events']}")
        text.DrawTextNDC(0.16, 0.69, f"processed events: {result['processed_events']}")
        text.DrawTextNDC(
            0.16,
            0.64,
            f"mean secondaries/event: {result['total_secondary_particles'] / result['processed_events']:.3g}"
            if result["processed_events"]
            else "mean secondaries/event: 0",
        )

        output_path = os.path.join(
            outdir,
            f"secondary_count_per_event__cuts_{step['cut_key']}.pdf",
        )
        canvas.RedrawAxis()
        canvas.Print(output_path)
        print(f"[write] {output_path}")


def make_log_bins(n_bins, x_min, x_max):
    if n_bins <= 0:
        raise ValueError("--energy-bins must be positive")
    if x_min <= 0 or x_max <= 0 or x_max <= x_min:
        raise ValueError("--energy-min and --energy-max must be positive with min < max")

    log_min = math.log10(x_min)
    log_max = math.log10(x_max)
    step = (log_max - log_min) / n_bins
    return array("d", [10 ** (log_min + step * index) for index in range(n_bins + 1)])


def safe_plot_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "none"


def make_energy_hist(name, title, bins, color):
    hist = ROOT.TH1D(name, title, len(bins) - 1, bins)
    hist.SetDirectory(0)
    hist.SetLineColor(ROOT.kBlack)
    hist.SetLineWidth(1)
    hist.SetFillColor(color)
    hist.SetFillStyle(1001)
    return hist


def fill_group_energy_histograms(result, group_order, bins, name_prefix):
    hists = []
    for group in group_order:
        hist = make_energy_hist(
            f"{name_prefix}_{safe_plot_name(group)}",
            "",
            bins,
            GROUP_COLORS.get(group, ROOT.kGray + 1),
        )
        for energy in result["energy_by_group"].get(group, []):
            hist.Fill(float(energy))
        if hist.Integral() > 0:
            hists.append((group, hist))
    return hists


def fill_charge_energy_histograms(result, bins, name_prefix):
    hists = []
    for charge in CHARGE_GROUP_ORDER:
        hist = make_energy_hist(
            f"{name_prefix}_{safe_plot_name(charge)}",
            "",
            bins,
            GROUP_COLORS.get(charge, ROOT.kGray + 1),
        )
        for group in PARTICLE_GROUP_ORDER:
            if group_charge_label(group) != charge:
                continue
            for energy in result["energy_by_group"].get(group, []):
                hist.Fill(float(energy))
        if hist.Integral() > 0:
            hists.append((charge, hist))
    return hists


def color_for_index(index):
    return PDG_COLORS[index % len(PDG_COLORS)]


def combined_pdg_label(abs_pdg):
    abs_pdg = int(abs(abs_pdg))
    particle = ROOT.TDatabasePDG.Instance().GetParticle(abs_pdg)
    antiparticle = ROOT.TDatabasePDG.Instance().GetParticle(-abs_pdg)

    particle_name = str(particle.GetName()) if particle else None
    antiparticle_name = str(antiparticle.GetName()) if antiparticle else None

    if particle_name and antiparticle_name and particle_name != antiparticle_name:
        return f"{particle_name}/{antiparticle_name} [|{abs_pdg}|]"
    if particle_name:
        return f"{particle_name} [{abs_pdg}]"
    if antiparticle_name:
        return f"{antiparticle_name} [|{abs_pdg}|]"
    return f"unknown [|{abs_pdg}|]"


def plot_label(key):
    if isinstance(key, int):
        return combined_pdg_label(key)
    return str(key).replace("_", " ")


def fill_combined_pdg_energy_histograms(result, bins, name_prefix, max_particles=None):
    combined_counts = Counter()
    combined_energies = defaultdict(list)

    for pdg, count in result["pdg_particle_counts"].items():
        combined_counts[abs(int(pdg))] += count

    for pdg, energies in result["energy_by_pdg"].items():
        combined_energies[abs(int(pdg))].extend(energies)

    ordered_abs_pdgs = [
        abs_pdg
        for abs_pdg, _ in combined_counts.most_common()
        if combined_energies.get(abs_pdg)
    ]
    if max_particles is not None:
        ordered_abs_pdgs = ordered_abs_pdgs[:max_particles]

    hists = []
    for color_index, abs_pdg in enumerate(ordered_abs_pdgs):
        hist = make_energy_hist(
            f"{name_prefix}_abs_{safe_plot_name(abs_pdg)}",
            "",
            bins,
            color_for_index(color_index),
        )
        for energy in combined_energies.get(abs_pdg, []):
            hist.Fill(float(energy))
        if hist.Integral() > 0:
            hists.append((abs_pdg, hist))
    return hists


def stacked_maximum(hists):
    if not hists:
        return 1.0
    n_bins = hists[0][1].GetNbinsX()
    max_value = 0.0
    for bin_index in range(1, n_bins + 1):
        total = sum(hist.GetBinContent(bin_index) for _, hist in hists)
        max_value = max(max_value, total)
    return max(max_value, 1.0)


def draw_energy_stack(
    outdir,
    step,
    result,
    hists,
    plot_key,
    title,
    bins,
    logy=True,
):
    output_path = os.path.join(
        outdir,
        f"secondary_energy_stack__{plot_key}__cuts_{step['cut_key']}.pdf",
    )

    ROOT.gStyle.SetOptStat(0)
    canvas = ROOT.TCanvas(f"c_{plot_key}_{step['cut_key']}", "", 900, 750)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.07)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.12)
    canvas.SetLogx()
    if logy:
        canvas.SetLogy()

    stack = ROOT.THStack(f"stack_{plot_key}_{step['cut_key']}", "")
    for _, hist in hists:
        stack.Add(hist)
    ymax = stacked_maximum(hists)

    frame = ROOT.TH1D(f"frame_{plot_key}_{step['cut_key']}", "", len(bins) - 1, bins)
    frame.SetDirectory(0)
    frame.SetTitle(title)
    frame.GetXaxis().SetTitle("Secondary energy [GeV]")
    frame.GetYaxis().SetTitle("Secondary particles / bin")
    frame.GetXaxis().SetTitleSize(0.045)
    frame.GetYaxis().SetTitleSize(0.045)
    frame.GetYaxis().SetTitleOffset(1.25)
    frame.SetMinimum(0.5 if logy else 0.0)
    frame.SetMaximum((50.0 if logy else 1.35) * ymax)
    frame.Draw()

    if hists:
        stack.Draw("HIST SAME")

    legend = ROOT.TLegend(0.58, 0.48, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    for group, hist in reversed(hists):
        label = group.replace("_", " ")
        legend.AddEntry(hist, f"{label} ({hist.Integral():.0f})", "f")
    legend.Draw()

    text = ROOT.TText()
    text.SetTextAlign(13)
    text.SetTextSize(0.030)
    base_label, extra_label = selection_labels(step, result)
    text.DrawTextNDC(0.14, 0.86, f"base cut: {base_label}")
    text.DrawTextNDC(0.14, 0.81, f"extra cut: {extra_label}")
    text.DrawTextNDC(0.14, 0.76, f"selected events: {result['selected_events']}")
    text.DrawTextNDC(0.14, 0.71, f"processed events: {result['processed_events']}")

    canvas.RedrawAxis()
    canvas.Print(output_path)
    print(f"[write] {output_path}")


def draw_energy_overlay(
    outdir,
    step,
    result,
    hists,
    plot_key,
    title,
    bins,
    logy=True,
):
    output_path = os.path.join(
        outdir,
        f"secondary_energy_overlay__{plot_key}__cuts_{step['cut_key']}.pdf",
    )

    ROOT.gStyle.SetOptStat(0)
    many_labels = len(hists) > 10
    canvas_width = 1200 if many_labels else 900
    canvas_height = 850 if many_labels else 750
    canvas = ROOT.TCanvas(f"c_overlay_{plot_key}_{step['cut_key']}", "", canvas_width, canvas_height)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.28 if many_labels else 0.07)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.12)
    canvas.SetLogx()
    if logy:
        canvas.SetLogy()

    ymax = max((hist.GetMaximum() for _, hist in hists), default=1.0)
    frame = ROOT.TH1D(f"frame_overlay_{plot_key}_{step['cut_key']}", "", len(bins) - 1, bins)
    frame.SetDirectory(0)
    frame.SetTitle(title)
    frame.GetXaxis().SetTitle("Secondary energy [GeV]")
    frame.GetYaxis().SetTitle("Secondary particles / bin")
    frame.GetXaxis().SetTitleSize(0.045)
    frame.GetYaxis().SetTitleSize(0.045)
    frame.GetYaxis().SetTitleOffset(1.25)
    frame.SetMinimum(0.5 if logy else 0.0)
    frame.SetMaximum((50.0 if logy else 1.35) * max(ymax, 1.0))
    frame.Draw()

    if many_labels:
        legend = ROOT.TLegend(0.74, 0.16, 0.97, 0.90)
    else:
        legend = ROOT.TLegend(0.58, 0.62, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    if len(hists) > 10:
        legend.SetNColumns(1)
        legend.SetTextSize(0.022)

    draw_hists = []
    for index, (group, source_hist) in enumerate(hists):
        hist = source_hist.Clone(f"{source_hist.GetName()}_overlay")
        hist.SetDirectory(0)
        draw_hists.append(hist)
        color = GROUP_COLORS.get(group, color_for_index(index))
        hist.SetFillStyle(0)
        hist.SetLineColor(color)
        hist.SetLineWidth(2)
        hist.SetMarkerColor(color)
        hist.SetMarkerStyle(20)
        hist.SetMarkerSize(0.8)
        hist.Draw("E1 SAME")
        legend.AddEntry(hist, f"{plot_label(group)} ({hist.Integral():.0f})", "lep")

    legend.Draw()

    text = ROOT.TText()
    text.SetTextAlign(13)
    text.SetTextSize(0.030)
    base_label, extra_label = selection_labels(step, result)
    text.DrawTextNDC(0.14, 0.86, f"base cut: {base_label}")
    text.DrawTextNDC(0.14, 0.81, f"extra cut: {extra_label}")
    text.DrawTextNDC(0.14, 0.76, f"selected events: {result['selected_events']}")
    text.DrawTextNDC(0.14, 0.71, f"processed events: {result['processed_events']}")

    canvas.RedrawAxis()
    canvas.Print(output_path)
    print(f"[write] {output_path}")


def write_energy_distribution_plots(outdir, steps, results_by_cut, energy_bins, energy_min, energy_max):
    bins = make_log_bins(energy_bins, energy_min, energy_max)

    for step in steps:
        result = results_by_cut[step["cut_key"]]
        if result["energy_particles_used"] <= 0:
            print(f"[plot] no secondary energies available for cuts {step['cut_key']}; skipping plots")
            continue

        plot_specs = [
            (
                "charge_groups",
                "Secondary energy: charged vs neutral",
                fill_charge_energy_histograms(result, bins, f"h_charge_{step['cut_key']}"),
            ),
            (
                "particle_groups_all",
                "Secondary energy: all particle groups",
                fill_group_energy_histograms(
                    result,
                    PARTICLE_GROUP_ORDER,
                    bins,
                    f"h_all_groups_{step['cut_key']}",
                ),
            ),
            (
                "particle_groups_charged",
                "Secondary energy: charged particle groups",
                fill_group_energy_histograms(
                    result,
                    CHARGED_PARTICLE_GROUPS,
                    bins,
                    f"h_charged_groups_{step['cut_key']}",
                ),
            ),
            (
                "particle_groups_neutral",
                "Secondary energy: neutral particle groups",
                fill_group_energy_histograms(
                    result,
                    NEUTRAL_PARTICLE_GROUPS,
                    bins,
                    f"h_neutral_groups_{step['cut_key']}",
                ),
            ),
        ]

        for plot_key, title, hists in plot_specs:
            if not hists:
                print(f"[plot] no entries for {plot_key} with cuts {step['cut_key']}; skipping")
                continue
            draw_energy_stack(outdir, step, result, hists, plot_key, title, bins)

        charge_overlay_hists = fill_charge_energy_histograms(
            result,
            bins,
            f"h_charge_overlay_{step['cut_key']}",
        )
        if charge_overlay_hists:
            draw_energy_overlay(
                outdir,
                step,
                result,
                charge_overlay_hists,
                "charge_groups",
                "Secondary energy: charged vs neutral",
                bins,
            )

        all_group_overlay_hists = fill_group_energy_histograms(
            result,
            PARTICLE_GROUP_ORDER,
            bins,
            f"h_all_groups_overlay_{step['cut_key']}",
        )
        if all_group_overlay_hists:
            draw_energy_overlay(
                outdir,
                step,
                result,
                all_group_overlay_hists,
                "particle_groups_all",
                "Secondary energy: all particle groups",
                bins,
            )

        pdg_overlay_hists = fill_combined_pdg_energy_histograms(
            result,
            bins,
            f"h_pdg_overlay_{step['cut_key']}",
        )
        if pdg_overlay_hists:
            draw_energy_overlay(
                outdir,
                step,
                result,
                pdg_overlay_hists,
                "particles_pdg",
                "Secondary energy: particle species, particle/antiparticle combined",
                bins,
            )


def run(args):
    os.makedirs(args.outdir, exist_ok=True)

    if args.cuts:
        print("[warn] --cuts is deprecated; use --base_cut instead")
        base_cut = args.cuts
    else:
        base_cut = args.base_cut or "2 3 4 5 8 9 10"
    cuts = parse_cut_list(base_cut)
    cut_key = "_".join(str(item) for item in cuts)
    step = {
        "cut_key": cut_key,
        "cut_label": " ".join(str(item) for item in cuts),
        "base_cut": " ".join(str(item) for item in cuts),
    }
    steps = [step]
    chain = make_muondis_chain(args.input_dir, args.tree)
    df = make_rdataframe(chain)

    if not tree_has_branch(chain, "secondary_pdg"):
        raise RuntimeError("Branch 'secondary_pdg' was not found in the muonDIS sndData tree")

    energy_branch = resolve_energy_branch(chain, args.secondary_energy_branch)
    if energy_branch:
        print(f"[energy] using branch: {energy_branch}")
    else:
        print(
            "[energy] no secondary energy branch found. "
            "PDG and charge tables will be produced; energy plots will be skipped."
        )

    selection, cut_names, skipped_cuts = build_selection(chain, step["base_cut"], args.extra_cut)
    result = count_secondaries(
        df=df,
        selection=selection,
        cut_key=step["cut_key"],
        energy_branch=energy_branch,
        max_events=args.max_events,
    )
    result["cut_names"] = cut_names
    result["skipped_cuts"] = skipped_cuts
    result["base_cut"] = step["base_cut"]
    result["extra_cut"] = args.extra_cut
    result["extra_cut_key"] = args.extra_cut_key

    results_by_cut = {step["cut_key"]: result}
    pdg_table_rows = pdg_rows(result)
    charge_table_rows = charge_rows(result)
    pdg_rows_by_cut = {step["cut_key"]: pdg_table_rows}
    charge_rows_by_cut = {step["cut_key"]: charge_table_rows}

    write_pdg_txt(args.outdir, step, result, pdg_table_rows)
    write_charge_txt(args.outdir, step, result, charge_table_rows)

    write_all_csv_outputs(args.outdir, steps, results_by_cut, pdg_rows_by_cut, charge_rows_by_cut)
    write_secondary_count_distribution_csv(args.outdir, steps, results_by_cut)
    write_secondary_count_distribution_plots(args.outdir, steps, results_by_cut)

    if energy_branch:
        write_energy_summary_csv(args.outdir, steps, results_by_cut)
        write_energy_distribution_plots(
            args.outdir,
            steps,
            results_by_cut,
            args.energy_bins,
            args.energy_min,
            args.energy_max,
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect muonDIS secondary-particle composition for one explicit base-cut "
            "selection plus an optional extra cut. Produces text, CSV summaries, and "
            "secondary-energy stack plots when an energy branch is available."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="/eos/experiment/sndlhc/users/zhibin/nueAnalysis",
        help="Directory containing hist_{partition}.root files",
    )
    parser.add_argument("--tree", default="sndData", help="TTree name")
    parser.add_argument(
        "--base_cut",
        default=None,
        help='Selection base cut indices, e.g. "2 3 4 5 8 9 10"',
    )
    parser.add_argument(
        "--cuts",
        default="",
        help="Deprecated alias for --base_cut",
    )
    parser.add_argument(
        "--extra_cut",
        default="",
        help='Extra selection appended to the base cut, e.g. "density_sndsw_scifi > 100"',
    )
    parser.add_argument(
        "--extra-cut-key",
        default="",
        help="Named extra-cut key, kept for output bookkeeping compatibility",
    )
    parser.add_argument(
        "--secondary-energy-branch",
        default="auto",
        help=(
            "Vector branch containing secondary energies. Use 'auto' to try common names. "
            "Energy plots are skipped if no branch is found."
        ),
    )
    parser.add_argument(
        "--outdir",
        default="muonDIS_secondary_scan",
        help="Output directory",
    )
    parser.add_argument(
        "--energy-bins",
        type=int,
        default=60,
        help="Number of logarithmic bins for secondary-energy stack plots",
    )
    parser.add_argument(
        "--energy-min",
        type=float,
        default=0.1,
        help="Lower edge in GeV for secondary-energy stack plots",
    )
    parser.add_argument(
        "--energy-max",
        type=float,
        default=10000.0,
        help="Upper edge in GeV for secondary-energy stack plots",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help=(
            "Debug/development limit: process at most this many selected events per cut. "
            "The output still records the full selected_events from the RDataFrame selection."
        ),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
