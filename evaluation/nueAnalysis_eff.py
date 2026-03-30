import os
import argparse
import pandas as pd
import ROOT
from tqdm import tqdm
import re
from dataclasses import dataclass
from pathlib import Path

ROOT.EnableImplicitMT()

@dataclass
class PartitionInfo:
    raw: str
    is_mc: bool
    category: str                  # real_data, kaon, neutron, NC_nue, CC_numu, ...

    year: int | None = None

    # sub-partitioning info for strings like Max20-3
    partition_id: int | None = None
    max_partitions: int | None = None

    energy: str | None = None
    energy_min: int | None = None
    energy_max: int | None = None

    interaction: str | None = None   # NC / CC
    flavor: str | None = None        # nue / numu

    metadata_csv: str | None = None

def build_neutrino_filter(info):
    """
    Build PDG-based filter from PartitionInfo.interaction and PartitionInfo.flavor.
    """

    if info.interaction is None or info.flavor is None:
        return ""

    interaction = info.interaction.upper()
    flavor = info.flavor.lower()

    pdg_map = {
        ("CC", "nue"): 12,
        ("CC", "numu"): 14,
        ("NC", "nue"): 112,
        ("NC", "numu"): 114,
    }

    key = (interaction, flavor)
    if key not in pdg_map:
        raise ValueError(
            f"Unknown interaction/flavor combination: "
            f"interaction={info.interaction}, flavor={info.flavor}"
        )

    pdg = pdg_map[key]

    return f"(pdgCode == {pdg} || pdgCode == {-pdg})"



def process_file_stepwise(root_path, tree_name, cut_groups, is_mc=False):
    """
    Sequential cutflow:
    count(cut_i) = events passing all previous cuts and this cut.
    """

    if not os.path.exists(root_path):
        raise FileNotFoundError(f"ROOT file not found: {root_path}")

    df = ROOT.RDataFrame(tree_name, root_path)

    total_result = df.Count()

    # Flatten cuts in the order they should be applied
    all_cuts = []

    # For MC, skip data-quality cuts entirely
    for group_name, cuts in cut_groups:
        if is_mc and group_name == "dataQuality":
            continue
        all_cuts.extend(cuts)

    # Book counts lazily
    count_results = []
    current = df

    for cut in all_cuts:
        current = current.Filter(f"{cut} == 1", cut)
        count_results.append((cut, current.Count()))

    # Trigger event loop
    total = int(total_result.GetValue())

    result = {"total_events": total}

    prev_count = total
    for cut, res in count_results:
        count = int(res.GetValue())
        rel_eff = count / prev_count if prev_count > 0 else 0.0
        cum_eff = count / total if total > 0 else 0.0

        result[f"{cut}_count"] = count
        result[f"{cut}_rel_eff"] = rel_eff
        result[f"{cut}_cum_eff"] = cum_eff

        prev_count = count

    # For MC, optionally fill skipped data-quality cuts as identical to total
    if is_mc:
        for cut in cut_groups[0][1]:  # dataQuality group assumed first
            result[f"{cut}_count"] = total
            result[f"{cut}_rel_eff"] = 1.0 if total > 0 else 0.0
            result[f"{cut}_cum_eff"] = 1.0 if total > 0 else 0.0

    return result


def parse_partition(partition: str) -> PartitionInfo:
    # real data with explicit partitioning:
    # real_data_2024_Max20-1
    m = re.fullmatch(r"real_data_(\d{4})_Max(\d+)-(\d+)", partition)
    if m:
        year = int(m.group(1))
        max_partitions = int(m.group(2))
        partition_id = int(m.group(3))
        return PartitionInfo(
            raw=partition,
            is_mc=False,
            category="real_data",
            year=year,
            partition_id=partition_id,
            max_partitions=max_partitions,
        )

    # MC kaon / neutron with energy + explicit partitioning:
    # MC_kaon_5-10GeV_Max20-1
    # MC_neutron_5-10GeV_Max20-1
    m = re.fullmatch(r"MC_(kaon|neutron)_(\d+)-(\d+)GeV_Max(\d+)-(\d+)", partition)
    if m:
        particle = m.group(1)
        e_min = int(m.group(2))
        e_max = int(m.group(3))
        max_partitions = int(m.group(4))
        partition_id = int(m.group(5))
        return PartitionInfo(
            raw=partition,
            is_mc=True,
            category=particle,
            energy=f"{e_min}-{e_max}GeV",
            energy_min=e_min,
            energy_max=e_max,
            partition_id=partition_id,
            max_partitions=max_partitions,
        )

    # MC kaon / neutron without partitioning:
    # MC_kaon_10-20GeV
    # MC_neutron_100-150GeV
    m = re.fullmatch(r"MC_(kaon|neutron)_(\d+)-(\d+)GeV", partition)
    if m:
        particle = m.group(1)
        e_min = int(m.group(2))
        e_max = int(m.group(3))
        return PartitionInfo(
            raw=partition,
            is_mc=True,
            category=particle,
            energy=f"{e_min}-{e_max}GeV",
            energy_min=e_min,
            energy_max=e_max,
        )

    # MC neutrino:
    # MC_NC_nue, MC_CC_numu, MC_NC_numu, MC_CC_nue
    m = re.fullmatch(r"MC_(NC|CC)_(nue|numu)", partition)
    if m:
        interaction = m.group(1)
        flavor = m.group(2)
        return PartitionInfo(
            raw=partition,
            is_mc=True,
            category=f"{interaction}_{flavor}",
            interaction=interaction,
            flavor=flavor,
        )

    raise ValueError(f"Unrecognized partition format: {partition}")

def resolve_metadata_csv(info: PartitionInfo, metadata_dir: str | Path) -> Path:
    metadata_dir = Path(metadata_dir)

    if not info.is_mc:
        filename = f"real_data_{info.year}_skim_runs_metadata.csv"
        return metadata_dir / filename

    if info.category == "kaon":
        return metadata_dir / "MC_kaon_FTFP_BERT_metadata.csv"

    if info.category == "neutron":
        return metadata_dir / "MC_neutron_FTFP_BERT_metadata.csv"

    # neutrino samples
    if  "nue" in info.category:
        return metadata_dir / "MC_neutrino_2024_ve_metadata.csv"
    if "numu" in info.category:
        return metadata_dir / "MC_neutrino_2024_vm_metadata.csv"

    raise ValueError(f"Could not resolve metadata CSV for: {info}")


def parse_energy_range_cell(value):
    """
    Accepts things like:
      '(5,10)'
      '(5, 10)'
      '[5, 10]'
      '5-10GeV'
      '5-10'
      tuple/list
    Returns (emin, emax) as ints.
    """
    if pd.isna(value):
        return None

    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])

    s = str(value).strip()

    # try literal parse first: "(5,10)" or "[5, 10]"
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (tuple, list)) and len(parsed) == 2:
            return int(parsed[0]), int(parsed[1])
    except Exception:
        pass

    # fallback: "5-10GeV" or "5-10"
    m = re.fullmatch(r"\(?\s*(\d+)\s*[-,]\s*(\d+)\s*\)?(?:GeV)?", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    raise ValueError(f"Could not parse energy_range value: {value}")


def split_dataframe_partition(df: pd.DataFrame, partition_id: int, n_parts: int) -> pd.DataFrame:
    """
    Partition rows into n_parts approximately equal contiguous chunks.
    partition_id is 1-based.
    """
    if partition_id < 1 or partition_id > n_parts:
        raise ValueError(f"partition_id must be in [1, {n_parts}], got {partition_id}")

    n = len(df)
    start = (partition_id - 1) * n // n_parts
    end = partition_id * n // n_parts
    return df.iloc[start:end].reset_index(drop=True)

def filter_metadata_rows(df: pd.DataFrame, info: PartitionInfo) -> pd.DataFrame:
    """
    Apply row filtering according to partition info.

    Rules
    -----
    - real data with MaxN-i:
        split all rows into N contiguous partitions, take partition i
    - kaon/neutron MC:
        filter by energy_range
        if MaxN-i is present, split the filtered rows into N partitions
    - neutrino MC:
        keep all rows
    """
    selected = df

    # Real data: split directly
    if not info.is_mc:
        if info.partition_id is None or info.max_partitions is None:
            raise ValueError(
                f"Real-data partition must include MaxN-i format, got: {info.raw}"
            )
        selected = split_dataframe_partition(
            df,
            partition_id=info.partition_id,
            n_parts=info.max_partitions,
        )
        return selected.reset_index(drop=True)

    # Kaon / neutron: filter by energy first
    if info.category in {"kaon", "neutron"}:
        if "energy_range" not in df.columns:
            raise KeyError(
                f"'energy_range' column not found in metadata for {info.raw}. "
                f"Available columns: {list(df.columns)}"
            )

        mask = df["energy_range"].apply(
            lambda x: parse_energy_range_cell(x) == (info.energy_min, info.energy_max)
        )
        selected = df.loc[mask].reset_index(drop=True)

        if selected.empty:
            raise ValueError(
                f"No rows matched energy range {info.energy_min}-{info.energy_max} GeV "
                f"for partition {info.raw}"
            )

        # If this MC sample is further partitioned, split after filtering
        if info.partition_id is not None or info.max_partitions is not None:
            if info.partition_id is None or info.max_partitions is None:
                raise ValueError(
                    f"Incomplete partitioning info in {info.raw}: "
                    f"partition_id={info.partition_id}, max_partitions={info.max_partitions}"
                )

            selected = split_dataframe_partition(
                selected,
                partition_id=info.partition_id,
                n_parts=info.max_partitions,
            )

        return selected.reset_index(drop=True)

    # Neutrino MC: currently use all rows
    return df.reset_index(drop=True)



def get_partition_info(partition: str, metadata_dir: str | Path) -> PartitionInfo:
    info = parse_partition(partition)
    info.metadata_csv = str(resolve_metadata_csv(info, metadata_dir))
    return info

def build_tchain_and_lumi(
    info,
    df,
    tree_name="cutFlowSummary",
    input_col="nueAnalysisFilter_path",
    lumi_col="lumi_per_file"
):
    chain = ROOT.TChain(tree_name)

    neutrino_filter = build_neutrino_filter(info)

    lumi_per_file = {}
    total_lumi = 0.0
    n_missing = 0

    for _, row in df.iterrows():
        base = str(row["output_base_path"]).rstrip("/")
        rel = str(row[input_col]).lstrip("/")
        path = f"{base}/{rel}"

        if not os.path.exists(path):
            print(f"[WARNING] File not found: {path}")
            n_missing += 1
            continue

        chain.Add(path)


        lumi_val = 0.0
        if lumi_col in df.columns:
            val = row[lumi_col]
            if val == val:  # NaN check
                lumi_val = float(val)

        lumi_per_file[path] = lumi_val
        total_lumi += lumi_val


    print(f"\nFiles added to TChain: {len(lumi_per_file)}")
    print(f"Missing files: {n_missing}")
    print(f"Total luminosity: {total_lumi}")

    return chain, lumi_per_file, total_lumi

def extract_info_from_partition(
    args,
    tree_name="cutFlowSummary",
    input_col="nueAnalysisFilter_path",
    lumi_col="lumi_per_file",
):
    """
    Parse partition, locate metadata CSV, filter rows, build TChain, and collect lumi.

    Returns
    -------
    info : PartitionInfo
    df_selected : pandas.DataFrame
    chain : ROOT.TChain
    lumi_per_file : dict[str, float]
    total_lumi : float
    """

    
    metadata_dir = args.metadata_dir
    info = get_partition_info(args.partition, metadata_dir)

    df = pd.read_csv(info.metadata_csv)
    df_selected = filter_metadata_rows(df, info)

    print(f"Selected {len(df_selected)} rows from metadata for partition {info.raw}")

    chain, lumi_per_file, total_lumi = build_tchain_and_lumi(
        info,
        df_selected,
        tree_name=tree_name,
        input_col=input_col,
        lumi_col=lumi_col,
    )

    return info, df_selected, chain, lumi_per_file, total_lumi
    


    

def process_efficiency(args, chain, total_lumi, info):
    os.makedirs(args.outdir, exist_ok=True)

    partition = args.partition
    outfile = f"{args.outdir}/eff_{partition}.root"

    df = ROOT.RDataFrame(chain)

    # Apply neutrino filter first
    neutrino_filter = build_neutrino_filter(info)
    if neutrino_filter:
        df = df.Define("pdgCode", "species")
        df = df.Filter(neutrino_filter, "neutrino_filter")

    total_events = int(df.Count().GetValue())

    # Full ordered list of cuts for display
    all_cut_names = [
        "StableBeams",
        "IP1BunchCrossing",
        "PreEvtClockCycle100",
        
        "avgScifiFiducial",
        "USBarsVeto_Top",
        "USBarsVeto_Bottom",
        "noVetoHit",
        
        "consecutiveSciFiHits",
        "SciFiContinuity",
        "USPlaneHit_0_1",
        "SciFiHit35",
        "USQDC_700MC_600Data",
        "NoHitLastDS",
    ]

    # Real filters to apply
    # For MC, skip the first three cuts but keep them in the printed/output table as pass-all
    if info.is_mc:
        cut_exprs = [
            ("avgScifiFiducial", "AvgSFChan == 1"),
            ("USBarsVeto_Top", "USBarsVeto_0_2_000000_1_2_000000 == 1"),
            ("USBarsVeto_Bottom", "USBarsVeto_0_8_000000_1_8_000000 == 1"),
            ("noVetoHit", "NoVetoHits == 1"),
            ("consecutiveSciFiHits", "At_least_two_consecutive_SciFi_planes == 1"),
            ("SciFiContinuity", "SciFiContinuity == 1"),
            ("USPlaneHit_0_1", "USPlanesHit == 1"),
            ("SciFiHit35", "SciFiMinHits == 1"),
            ("USQDC_700MC_600Data", "USQDC == 1"),
            ("NoHitLastDS", "DSVetoCut == 1"),
        ]
    else:
        cut_exprs = [
            ("StableBeams", "StableBeams == 1"),
            ("IP1BunchCrossing", "IP1 == 1"),
            ("PreEvtClockCycle100", "EventDeltat_1_100 == 1"),
            ("avgScifiFiducial", "AvgSFChan == 1"),
            ("USBarsVeto_Top", "USBarsVeto_0_2_000000_1_2_000000 == 1"),
            ("USBarsVeto_Bottom", "USBarsVeto_0_8_000000_1_8_000000 == 1"),
            ("noVetoHit", "NoVetoHits == 1"),
            ("consecutiveSciFiHits", "At_least_two_consecutive_SciFi_planes == 1"),
            ("SciFiContinuity", "SciFiContinuity == 1"),
            ("USPlaneHit_0_1", "USPlanesHit == 1"),
            ("SciFiHit35", "SciFiMinHits == 1"),
            ("USQDC_700MC_600Data", "USQDC == 1"),
            ("NoHitLastDS", "DSVetoCut == 1"),
        ]

    # Book real stepwise counts
    count_map = {}
    current = df
    for cut_name, cut_expr in cut_exprs:
        current = current.Filter(cut_expr, cut_name)
        count_map[cut_name] = int(current.Count().GetValue())

    # Build final displayed cutflow
    cutflow_rows = []
    prev_count = total_events

    for cut_name in tqdm(all_cut_names, desc="Building cutflow"):
        if info.is_mc and cut_name in ["StableBeams", "IP1BunchCrossing", "PreEvtClockCycle100"]:
            count = total_events
        else:
            count = count_map[cut_name]

        rel_eff = count / prev_count if prev_count > 0 else 0.0
        cum_eff = count / total_events if total_events > 0 else 0.0

        cutflow_rows.append((cut_name, count, rel_eff, cum_eff))
        prev_count = count

    # Print table
    print("\n========== CUT FLOW TABLE ==========\n")
    print(f"{'Cut':30s} {'Count':>12s} {'RelEff':>12s} {'CumEff':>12s}")
    for cut_name, count, rel_eff, cum_eff in cutflow_rows:
        print(f"{cut_name:30s} {count:12d} {rel_eff:12.6f} {cum_eff:12.6f}")

    print(f"\n{'Total':30s} {total_events:12d}")
    print(f"{'Total lumi':30s} {float(total_lumi):12.6f}")

    # Save to ROOT
    fout = ROOT.TFile(outfile, "RECREATE")

    h_cutflow = ROOT.TH1D("cutflow_counts", "Cutflow Counts", len(all_cut_names) + 1, 0, len(all_cut_names) + 1)
    h_cutflow.GetXaxis().SetBinLabel(1, "Total")
    h_cutflow.SetBinContent(1, total_events)

    h_cumeff = ROOT.TH1D("cutflow_cumeff", "Cutflow Cumulative Efficiency", len(all_cut_names) + 1, 0, len(all_cut_names) + 1)
    h_cumeff.GetXaxis().SetBinLabel(1, "Total")
    h_cumeff.SetBinContent(1, 1.0 if total_events > 0 else 0.0)

    h_releff = ROOT.TH1D("cutflow_releff", "Cutflow Relative Efficiency", len(all_cut_names) + 1, 0, len(all_cut_names) + 1)
    h_releff.GetXaxis().SetBinLabel(1, "Total")
    h_releff.SetBinContent(1, 1.0 if total_events > 0 else 0.0)

    for i, (cut_name, count, rel_eff, cum_eff) in enumerate(cutflow_rows, start=2):
        h_cutflow.GetXaxis().SetBinLabel(i, cut_name)
        h_cutflow.SetBinContent(i, count)

        h_cumeff.GetXaxis().SetBinLabel(i, cut_name)
        h_cumeff.SetBinContent(i, cum_eff)

        h_releff.GetXaxis().SetBinLabel(i, cut_name)
        h_releff.SetBinContent(i, rel_eff)

    h_cutflow.Write()
    h_cumeff.Write()
    h_releff.Write()

    ROOT.TParameter("double")("total_lumi", float(total_lumi)).Write()
    ROOT.TParameter("int")("total_events", int(total_events)).Write()
    ROOT.TNamed("category", str(getattr(info, "category", ""))).Write()
    ROOT.TNamed("interaction", str(getattr(info, "interaction", ""))).Write()
    ROOT.TNamed("flavor", str(getattr(info, "flavor", ""))).Write()

    fout.Close()
    
def main(args):
    info, df_selected, chain, lumi_per_file, total_lumi = extract_info_from_partition(args)
    
    process_efficiency(args, chain, total_lumi, info)

    
    
    
    
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--outdir",
        dest="outdir",
        help="output diretory",
        default=None
    )
    parser.add_argument(
        "-p", "--partition",
        dest="partition",
        help="partition name, e.g. real_data_2024_1, MC_kaon_10GeV, etc.",
        required=True
    )
    
    parser.add_argument(
        "-d", "--metadata_dir",
        dest="metadata_dir",
        help="metadata directory",
        required=True
    )
    
    args = parser.parse_args()
    main(args)