import os
import argparse
import pandas as pd
import ROOT
from tqdm import tqdm
import re
from dataclasses import dataclass
from pathlib import Path

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

ROOT.EnableImplicitMT()

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

    m = re.fullmatch(r"MC_muonDIS_Max(\d+)-(\d+)", partition)
    if m:
        max_partitions = int(m.group(1))
        partition_id = int(m.group(2))
        return PartitionInfo(
            raw=partition,
            is_mc=True,
            category="muonDIS",
            partition_id=partition_id,
            max_partitions=max_partitions,
        )

    if partition == "MC_muon":
        return PartitionInfo(
            raw=partition,
            is_mc=True,
            category="muon",
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
    
    if info.category == "muon":
        return metadata_dir / "MC_muon_up_metadata.csv"
    
    # neutrino samples
    if  "nue" in info.category:
        return metadata_dir / "MC_neutrino_2024_ve_metadata.csv"
    if "numu" in info.category:
        return metadata_dir / "MC_neutrino_2024_vm_metadata.csv"
    
    if info.category == "muonDIS":
        return metadata_dir / "MC_muonDIS_cvilela_metadata.csv"


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
    if (not info.is_mc) or (info.category == "muonDIS"):
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
    df,
    tree_name="sndData",
    input_col="feature_path",
    lumi_col="lumi_per_file"
):

    chain = ROOT.TChain(tree_name)
    lumi_per_file = {}
    total_lumi = 0.0
    cutflow_chain = ROOT.TChain("cutFlowSummary")
    n_missing = 0

    for _, row in df.iterrows():

        base = str(row["output_base_path"]).rstrip("/")
        rel = str(row[input_col]).lstrip("/")
        path = f"{base}/{rel}"

        # ---------- check file exists ----------
        if not os.path.exists(path):
            print(f"[WARNING] File not found: {path}")
            n_missing += 1
            continue

        # ---------- safe to add ----------
        chain.Add(path)
        cutflow_chain.Add(path)

        lumi_val = 0.0
        if lumi_col in df.columns:
            lumi_val = float(row[lumi_col])

        lumi_per_file[path] = lumi_val
        total_lumi += lumi_val

    chain.AddFriend(cutflow_chain)
    print(f"\nFiles added to TChain: {len(lumi_per_file)}")
    print(f"Missing files: {n_missing}")
    print(f"Total luminosity: {total_lumi}")

    return chain, lumi_per_file, total_lumi
def extract_info_from_partition(
    args,
    tree_name="sndData",
    input_col="feature_path",
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
        df_selected,
        tree_name=tree_name,
        input_col=input_col,
        lumi_col=lumi_col,
    )

    return info, df_selected, chain, lumi_per_file, total_lumi
    

def process_histogram(args, chain, total_lumi, info):
    os.makedirs(args.outdir, exist_ok=True)

    partition = args.partition
    outfile = f"{args.outdir}/hist_{partition}.root"

    tree_name = chain.GetName()
    df = ROOT.RDataFrame(chain)

    base_filter = ""
    filters = []

    if base_filter:
        filters.append(f"({base_filter})")

    neutrino_filter = build_neutrino_filter(info)
    if neutrino_filter:
        filters.append(f"({neutrino_filter})")

    full_filter = " && ".join(filters)
    df_filtered = df.Filter(full_filter, "FinalSelection") if full_filter else df

    n_total = df.Count()
    n_pass = df_filtered.Count()

    opts = ROOT.RDF.RSnapshotOptions()
    opts.fMode = "RECREATE"

    columns = [str(c) for c in df_filtered.GetColumnNames()]
    if not columns:
        raise RuntimeError("No columns available in df_filtered for Snapshot")

    df_filtered.Snapshot(tree_name, outfile, columns, opts)

    total_events = int(n_total.GetValue())
    passed_events = int(n_pass.GetValue())

    print(f"\nFile: {outfile}")
    print(f"Category            : {info.category}")
    print(f"Filter              : {full_filter}")
    print(f"Total events        : {total_events}")
    print(f"Events after filter : {passed_events}")
    print(
        f"Efficiency          : {passed_events / total_events:.6f}"
        if total_events > 0 else
        "Efficiency          : 0.000000"
    )

    fout = ROOT.TFile(outfile, "UPDATE")
    ROOT.TParameter("double")("total_lumi", float(total_lumi)).Write()
    fout.Close()

    
def main(args):
    info, df_selected, chain, lumi_per_file, total_lumi = extract_info_from_partition(args)
    

    process_histogram(args, chain, total_lumi, info)

    
    
    
        


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
