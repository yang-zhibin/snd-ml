import argparse
import csv
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


PYTHON_KEYWORDS = {
    "and",
    "or",
    "not",
    "abs",
    "True",
    "False",
}


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def format_count(value):
    return f"{int(value):,}"


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


TQDM_KWARGS = {
    "dynamic_ncols": True,
    "mininterval": 5,
    "file": sys.stdout,
}


def progress_iter(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **{**TQDM_KWARGS, **kwargs})


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def split_csv_arg(value):
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def resolve_path(base, rel):
    rel = str(rel)
    if rel.startswith("root://") or rel.startswith("/"):
        return rel
    return f"{str(base).rstrip('/')}/{rel.lstrip('/')}"


def safe_float(value, default=0.0):
    if value is None or pd.isna(value):
        return default
    return float(value)


def sanitize_name(value):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value)).strip("_")


def extract_identifiers(expression):
    names = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression))
    return {name for name in names if name not in PYTHON_KEYWORDS}


def expand_macros(expression, macros):
    expanded = str(expression)
    changed = True
    while changed:
        changed = False
        for name, value in macros.items():
            if name in expanded:
                expanded = expanded.replace(name, f"({value})")
                changed = True
    return expanded


def parse_energy_range(value):
    if value is None or pd.isna(value):
        raise ValueError("Missing energy_range")

    text = str(value).strip()
    match = re.fullmatch(r"\(?\s*(\d+)\s*[-,]\s*(\d+)\s*\)?(?:\s*GeV)?", text)
    if not match:
        raise ValueError(f"Could not parse energy_range value: {value}")

    return int(match.group(1)), int(match.group(2))


def row_value(row, key, default=None):
    return row[key] if key in row.index else default


def normalize_metadata_path(path):
    return str(path)


def metadata_row_matches_rule(row, rule):
    data_type = str(row["data_type"])
    subfolder = str(row.get("subfolder", ""))

    if data_type != str(rule["source_data_type"]):
        return False

    allowed_subfolders = rule.get("source_subfolders")
    if allowed_subfolders and subfolder not in {str(v) for v in allowed_subfolders}:
        return False

    return True


def resolve_group_settings_from_rule(region, particle_group, particle_rule_name, partition_config):
    settings = dict(partition_config.get("defaults", {}))
    rule = partition_config["particle_group_rules"][particle_rule_name]
    settings.update({k: v for k, v in rule.items() if k not in {"event_filter"}})

    region_settings = partition_config.get("regions", {}).get(region, {})
    settings.update(region_settings)

    override = partition_config.get("overrides", {}).get(region, {})
    particle_family = str(rule.get("particle_family", ""))
    for key in (particle_group, particle_rule_name, particle_family):
        if key in override:
            settings.update(override[key])
            break

    return settings


def find_particle_rule_for_group(particle_group, partition_config):
    for rule_name, rule in partition_config["particle_group_rules"].items():
        if not rule.get("process", True):
            continue
        if "derive_particle_group" not in rule and particle_group == rule_name:
            return rule_name, rule

    for rule_name, rule in partition_config["particle_group_rules"].items():
        if not rule.get("process", True):
            continue
        if str(rule.get("particle_family", "")) == particle_group:
            return rule_name, rule

    return None, None


def particle_group_from_metadata_row(row, rule_name, rule):
    if not metadata_row_matches_rule(row, rule):
        return None

    if "derive_particle_group" not in rule:
        return rule_name

    energy_column = rule["energy_column"]
    if energy_column not in row.index or pd.isna(row[energy_column]):
        return None

    emin, emax = parse_energy_range(row[energy_column])
    return rule["derive_particle_group"].format(
        particle_family=rule["particle_family"],
        energy_min=emin,
        energy_max=emax,
    )


def select_lumi_metadata_rows(metadata_paths, region, particle_group, part_index, partition_config):
    log(
        f"Selecting whole-file lumi rows for region '{region}', "
        f"particle group '{particle_group}', part {part_index:03d}"
    )

    rows = []
    matched_rule_name = None
    for rule_name, rule in partition_config["particle_group_rules"].items():
        if not rule.get("process", True):
            continue

        for metadata_path in metadata_paths:
            metadata_path = normalize_metadata_path(metadata_path)
            df = pd.read_csv(metadata_path)
            matched_here = 0
            for row_number, row in df.iterrows():
                row_particle_group = particle_group_from_metadata_row(row, rule_name, rule)
                if row_particle_group is None:
                    continue
                particle_family = str(rule.get("particle_family", ""))
                if particle_group not in {row_particle_group, particle_family, rule_name}:
                    continue
                matched_rule_name = rule_name
                matched_here += 1
                rows.append(
                    {
                        "metadata_path": metadata_path,
                        "row_number": int(row_number),
                        "lumi_per_file": safe_float(row_value(row, "lumi_per_file", 0.0)),
                    }
                )
            if matched_here:
                log(
                    f"  {Path(metadata_path).name}: {format_count(matched_here)} candidate rows "
                    f"with rule '{rule_name}'"
                )

    if matched_rule_name is None:
        raise ValueError(f"Could not find particle rule/metadata rows for particle group '{particle_group}'")

    settings = resolve_group_settings_from_rule(region, particle_group, matched_rule_name, partition_config)
    if not settings.get("process", True):
        return {}, settings

    n_partitions = int(settings.get("n_partitions", 1))
    if n_partitions <= 0:
        raise ValueError(f"n_partitions must be positive, got {n_partitions}")
    if part_index < 1 or part_index > n_partitions:
        return {}, settings

    fraction = float(settings.get("fraction", 1.0))
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    total_lumi = sum(row["lumi_per_file"] for row in rows)
    target_lumi = total_lumi * fraction
    log(
        f"  total lumi={total_lumi:.6g}, fraction={fraction:.6g}, "
        f"selected lumi target={target_lumi:.6g}, requested partitions={n_partitions}"
    )
    if target_lumi <= 0.0:
        log("  selected lumi target is zero; this partition will be empty")
        return {}, settings

    selected_rows = []
    selected_lumi = 0.0
    for row in rows:
        if selected_lumi >= target_lumi and selected_rows:
            break
        selected_rows.append(row)
        selected_lumi += row["lumi_per_file"]

    if not selected_rows:
        return {}, settings

    target_per_partition = selected_lumi / n_partitions if n_partitions else selected_lumi
    partition_rows = defaultdict(list)
    current_part = 1
    current_lumi = 0.0

    for row in selected_rows:
        if (
            current_part < n_partitions
            and partition_rows[current_part]
            and current_lumi >= target_per_partition
        ):
            current_part += 1
            current_lumi = 0.0
        partition_rows[current_part].append(row)
        current_lumi += row["lumi_per_file"]

    allowed = defaultdict(set)
    for row in partition_rows.get(part_index, []):
        allowed[row["metadata_path"]].add(row["row_number"])

    selected_for_part = sum(len(row_numbers) for row_numbers in allowed.values())
    lumi_for_part = sum(
        row["lumi_per_file"]
        for row in partition_rows.get(part_index, [])
    )
    log(
        f"  part {part_index:03d}: selected {format_count(selected_for_part)} whole files, "
        f"approx lumi={lumi_for_part:.6g}"
    )
    return dict(allowed), settings


def metadata_paths_to_rows(metadata_paths, allowed_metadata_rows=None, skip_missing_files=False):
    rows = []
    skipped_missing = []
    for metadata_path in metadata_paths:
        metadata_path = normalize_metadata_path(metadata_path)
        df = pd.read_csv(metadata_path)
        allowed_rows = None
        if allowed_metadata_rows is not None:
            allowed_rows = allowed_metadata_rows.get(metadata_path, set())

        for row_number, row in df.iterrows():
            if allowed_rows is not None and int(row_number) not in allowed_rows:
                continue

            feature_path = resolve_path(row["output_base_path"], row["feature_path"])
            hit3d_path = resolve_path(row["output_base_path"], row["hit3d_path"])
            missing_paths = [
                path
                for path in (feature_path, hit3d_path)
                if path.startswith("/") and not os.path.exists(path)
            ]
            if missing_paths:
                if skip_missing_files:
                    skipped_missing.append(
                        {
                            "metadata_path": metadata_path,
                            "row_number": int(row_number),
                            "missing_path": missing_paths[0],
                        }
                    )
                    continue
                raise FileNotFoundError(missing_paths[0])

            rows.append(
                {
                    "metadata_path": metadata_path,
                    "row_number": int(row_number),
                    "row": row,
                    "feature_path": feature_path,
                    "hit3d_path": hit3d_path,
                    "lumi_per_file": safe_float(row_value(row, "lumi_per_file", 0.0)),
                }
            )

    if skipped_missing:
        log(f"Skipped {format_count(len(skipped_missing))} selected metadata rows with missing feature/hit3D files")
        for item in skipped_missing[:5]:
            log(
                "  missing file example: "
                f"{Path(item['metadata_path']).name} row {item['row_number']}: {item['missing_path']}"
            )
        if len(skipped_missing) > 5:
            log(f"  ... {format_count(len(skipped_missing) - 5)} more missing rows not shown")

    return rows


def infer_particle_rule_for_rows(particle_group, metadata_rows, partition_config):
    for item in metadata_rows:
        row = item["row"]
        for rule_name, rule in partition_config["particle_group_rules"].items():
            if not rule.get("process", True):
                continue
            row_particle_group = particle_group_from_metadata_row(row, rule_name, rule)
            if row_particle_group is None:
                continue
            particle_family = str(rule.get("particle_family", ""))
            if particle_group in {row_particle_group, particle_family, rule_name}:
                return rule_name, rule

    rule_name, rule = find_particle_rule_for_group(particle_group, partition_config)
    if rule_name is None:
        raise ValueError(f"Could not infer particle rule for particle group '{particle_group}'")
    return rule_name, rule


def rdf_vector_string(values):
    import ROOT

    result = ROOT.std.vector("string")()
    for value in values:
        result.push_back(str(value))
    return result


def get_tree_branch_names(file_path, tree_name):
    import ROOT

    root_file = ROOT.TFile.Open(file_path)
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open ROOT file: {file_path}")
    try:
        tree = root_file.Get(tree_name)
        if not tree:
            raise KeyError(f"Tree '{tree_name}' not found in {file_path}")
        return [branch.GetName() for branch in tree.GetListOfBranches()]
    finally:
        root_file.Close()


def build_tchain(tree_name, paths, label):
    import ROOT

    chain = ROOT.TChain(tree_name)
    log(f"Building {label} TChain with {format_count(len(paths))} files")
    for path in progress_iter(paths, desc=f"Adding {label} files", unit="file"):
        added = chain.Add(path)
        if added == 0:
            raise OSError(f"Could not add {path} to {label} TChain")
    log(f"  {label} chain entries: {format_count(chain.GetEntries())}")
    return chain


def root_region_expression(region_config, region_name):
    macros = region_config.get("expression_macros", {})
    return expand_macros(region_config["regions"][region_name]["expression"], macros)


def combine_root_expressions(*expressions):
    expressions = [str(expr).strip() for expr in expressions if str(expr).strip()]
    if not expressions:
        return "true"
    return " && ".join(f"({expr})" for expr in expressions)


def alias_expression_identifiers(expression, alias_prefix="feat", name_prefix="feat__"):
    identifiers = sorted(extract_identifiers(expression), key=len, reverse=True)
    aliases = {}
    aliased_expression = str(expression)
    for name in identifiers:
        alias_name = f"{name_prefix}{name}"
        aliases[alias_name] = f"{alias_prefix}.{name}"
        aliased_expression = re.sub(rf"\b{re.escape(name)}\b", alias_name, aliased_expression)
    return aliased_expression, aliases


def apply_metadata_defines(rdf, existing_columns, particle_group, particle_id):
    existing_columns = set(existing_columns)
    escaped_particle_group = str(particle_group).replace("\\", "\\\\").replace('"', '\\"')
    definitions = {
        "event_uid": 'std::string("")',
        "original_entry": "rdfentry_",
        "source_file_index": "-1",
        "particle_group": f'std::string("{escaped_particle_group}")',
        "particle_id": f"int({int(particle_id)})",
    }
    for name, expression in definitions.items():
        if name not in existing_columns:
            rdf = rdf.Define(name, expression)
    return rdf


def configured_snapshot_columns(configured_columns, existing_columns, added_columns, label):
    existing_columns = list(existing_columns)
    existing_set = set(existing_columns)
    if configured_columns is None:
        selected = list(existing_columns)
    else:
        selected = []
        missing = []
        for column in configured_columns:
            column = str(column)
            if column in existing_set:
                selected.append(column)
            else:
                missing.append(column)
        if missing:
            raise KeyError(f"Configured {label} snapshot branches are missing: {missing}")

    for column in added_columns:
        if column not in selected:
            selected.append(column)
    return selected


def require_local_output_path(output_path):
    if str(output_path).startswith("root://"):
        raise ValueError(
            f"ROOT output path must be local for direct script writes: {output_path}. "
            "Use Snakemake staging to copy outputs to EOS."
        )


def snapshot_rdf(rdf, tree_name, output_path, columns, compression_level=None):
    import ROOT

    require_local_output_path(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    log(f"Snapshot {tree_name}: {output_path}")
    options = ROOT.RDF.RSnapshotOptions()
    if compression_level is not None:
        options.fCompressionLevel = int(compression_level)
        log(f"  compression level: {int(compression_level)}")
    result = rdf.Snapshot(tree_name, output_path, rdf_vector_string(columns), options)
    if hasattr(result, "GetValue"):
        result.GetValue()


def tree_entries(path, tree_name):
    import ROOT

    root_file = ROOT.TFile.Open(path)
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open ROOT file: {path}")
    try:
        tree = root_file.Get(tree_name)
        if not tree:
            raise KeyError(f"Tree '{tree_name}' not found in {path}")
        return int(tree.GetEntries())
    finally:
        root_file.Close()


def check_entry_counts(metadata_rows, feature_tree_name, hit3d_tree_name):
    log(f"Checking feature/hit3D entry counts for {format_count(len(metadata_rows))} file pairs")
    for item in progress_iter(metadata_rows, desc="Checking entry counts", unit="file"):
        n_feature = tree_entries(item["feature_path"], feature_tree_name)
        n_hit3d = tree_entries(item["hit3d_path"], hit3d_tree_name)
        if n_feature != n_hit3d:
            raise ValueError(
                "Feature/hit3D entry count mismatch: "
                f"{item['feature_path']} has {format_count(n_feature)} entries, "
                f"{item['hit3d_path']} has {format_count(n_hit3d)} entries"
            )
    log("Feature/hit3D entry count check passed")


def write_empty_metadata_csv(output_metadata, partition_config):
    output_metadata = Path(output_metadata)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metadata, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=partition_config["metadata_output_columns"])
        writer.writeheader()
    log(f"Wrote empty metadata CSV: {output_metadata}")


def write_rdf_metadata_row(
    output_metadata,
    region,
    particle_group,
    particle_family,
    particle_id,
    particle_rule_name,
    n_events,
    lumi_per_partition,
    settings,
    region_config,
    partition_config,
    feature_output,
    hit3d_output,
    metadata_rows,
    part_index,
):
    output_metadata = Path(output_metadata)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    partition_id = partition_config["grouping"]["output_partition_id_template"].format(
        region=sanitize_name(region),
        particle_group=sanitize_name(particle_group),
        part_index=int(part_index),
    )
    particle_rule = partition_config["particle_group_rules"][particle_rule_name]
    row = {
        "partition_version": partition_config["partition_version"],
        "region_version": region_config["region_version"],
        "region": region,
        "particle_group": particle_group,
        "particle_family": particle_family,
        "particle_id": particle_id,
        "partition_id": partition_id,
        "feature_partition_path": feature_output,
        "hit3d_partition_path": hit3d_output,
        "n_events": n_events,
        "lumi_per_partition": lumi_per_partition,
        "fraction": settings.get("fraction"),
        "max_events": settings.get("max_events"),
        "n_partitions_requested": settings.get("n_partitions"),
        "n_partitions_written": settings.get("n_partitions"),
        "selection_expression": region_config["regions"][region]["expression"],
        "particle_filter_expression": particle_rule.get("event_filter", ""),
        "source_metadata_csv": ";".join(sorted({item["metadata_path"] for item in metadata_rows})),
        "source_feature_paths": ";".join(sorted({item["feature_path"] for item in metadata_rows})),
        "source_hit3d_paths": ";".join(sorted({item["hit3d_path"] for item in metadata_rows})),
        "source_file_indices": "",
        "first_original_entry": "",
        "last_original_entry": "",
        "shuffle": partition_config.get("shuffle"),
        "split_method": "lumi_file_chunks_rdf_filter",
        "seed": partition_config.get("seed"),
    }

    with open(output_metadata, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=partition_config["metadata_output_columns"])
        writer.writeheader()
        if n_events > 0:
            writer.writerow({field: row.get(field, "") for field in partition_config["metadata_output_columns"]})
    log(f"Wrote RDF metadata CSV with {1 if n_events > 0 else 0} rows: {output_metadata}")


def run_rdf_partition(
    metadata_paths,
    region_config,
    partition_config,
    output_dir,
    output_metadata,
    only_regions,
    only_particles,
    part_index,
    feature_output=None,
    hit3d_output=None,
    metadata_feature_output=None,
    metadata_hit3d_output=None,
    skip_missing_files=False,
    check_entry_counts_enabled=False,
    rdf_threads=0,
):
    if not only_regions or len(only_regions) != 1:
        raise ValueError("Exactly one --only-regions value is required")
    if not only_particles or len(only_particles) != 1:
        raise ValueError("Exactly one --only-particles value is required")
    if part_index is None:
        raise ValueError("--only-part-index is required")

    import ROOT
    if rdf_threads and int(rdf_threads) > 0:
        ROOT.EnableImplicitMT(int(rdf_threads))
        log(f"Enabled ROOT implicit multithreading with {int(rdf_threads)} threads")

    region = next(iter(only_regions))
    particle_group = next(iter(only_particles))
    log(f"Using ROOT RDataFrame for {region} / {particle_group} / part {part_index:03d}")

    allowed_metadata_rows, lumi_settings = select_lumi_metadata_rows(
        metadata_paths=metadata_paths,
        region=region,
        particle_group=particle_group,
        part_index=part_index,
        partition_config=partition_config,
    )

    metadata_rows = metadata_paths_to_rows(
        metadata_paths=metadata_paths,
        allowed_metadata_rows=allowed_metadata_rows,
        skip_missing_files=skip_missing_files,
    )
    log(f"RDF job will use {format_count(len(metadata_rows))} metadata rows")

    if not metadata_rows:
        write_empty_metadata_csv(output_metadata, partition_config)
        return

    particle_rule_name, particle_rule = infer_particle_rule_for_rows(
        particle_group,
        metadata_rows,
        partition_config,
    )
    settings = lumi_settings or resolve_group_settings_from_rule(
        region,
        particle_group,
        particle_rule_name,
        partition_config,
    )
    particle_family = str(particle_rule["particle_family"])
    particle_id = int(particle_rule["particle_id"])
    if settings.get("max_events") is not None:
        log(
            "Note: max_events is ignored in lumi_file_chunks_rdf_filter mode; "
            f"configured value is {settings.get('max_events')}"
        )

    partition_id = partition_config["grouping"]["output_partition_id_template"].format(
        region=sanitize_name(region),
        particle_group=sanitize_name(particle_group),
        part_index=int(part_index),
    )
    base_dir = (
        Path(output_dir)
        / partition_config["partition_version"]
        / region_config["region_version"]
        / sanitize_name(region)
        / sanitize_name(particle_group)
    )
    feature_output = feature_output or str(base_dir / f"{partition_config['output']['feature_prefix']}_{partition_id}.root")
    hit3d_output = hit3d_output or str(base_dir / f"{partition_config['output']['hit3d_prefix']}_{partition_id}.root")
    metadata_feature_output = metadata_feature_output or feature_output
    metadata_hit3d_output = metadata_hit3d_output or hit3d_output

    feature_paths = [item["feature_path"] for item in metadata_rows]
    hit3d_paths = [item["hit3d_path"] for item in metadata_rows]
    feature_tree_name = partition_config["input"]["feature_tree"]
    hit3d_tree_name = partition_config["input"]["hit3d_tree"]

    if check_entry_counts_enabled:
        check_entry_counts(metadata_rows, feature_tree_name, hit3d_tree_name)

    feature_chain = build_tchain(feature_tree_name, feature_paths, "feature")
    hit3d_chain = build_tchain(hit3d_tree_name, hit3d_paths, "hit3D")

    region_expr = root_region_expression(region_config, region)
    particle_expr = particle_rule.get("event_filter", "")
    selection_expr = combine_root_expressions(region_expr, particle_expr)
    log(f"RDF selection: {selection_expr}")

    feature_columns = get_tree_branch_names(feature_paths[0], feature_tree_name)
    hit3d_columns = get_tree_branch_names(hit3d_paths[0], hit3d_tree_name)
    added_columns = partition_config["output"].get(
        "added_event_branches",
        ["event_uid", "original_entry", "source_file_index", "particle_group", "particle_id"],
    )
    feature_snapshot_columns = configured_snapshot_columns(
        partition_config["output"].get("feature_branches"),
        feature_columns,
        added_columns,
        "feature",
    )
    hit3d_snapshot_columns = configured_snapshot_columns(
        partition_config["output"].get("hit3d_branches"),
        hit3d_columns,
        added_columns,
        "hit3D",
    )
    log(
        f"Snapshot columns: feature={format_count(len(feature_snapshot_columns))}, "
        f"hit3D={format_count(len(hit3d_snapshot_columns))}"
    )

    feature_rdf = ROOT.RDataFrame(feature_chain).Filter(selection_expr)
    feature_rdf = apply_metadata_defines(feature_rdf, feature_columns, particle_group, particle_id)

    hit3d_chain.AddFriend(feature_chain, "feat")
    hit_selection_expr, hit_aliases = alias_expression_identifiers(selection_expr)
    hit3d_rdf = ROOT.RDataFrame(hit3d_chain)
    for alias_name, target_name in hit_aliases.items():
        hit3d_rdf = hit3d_rdf.Alias(alias_name, target_name)
    hit3d_rdf = hit3d_rdf.Filter(hit_selection_expr)
    hit3d_rdf = apply_metadata_defines(hit3d_rdf, hit3d_columns, particle_group, particle_id)

    n_events = int(feature_rdf.Count().GetValue())
    log(f"RDF selected events in this whole-file lumi partition: {format_count(n_events)}")
    lumi_per_partition = sum(item["lumi_per_file"] for item in metadata_rows)
    feature_out_rdf = feature_rdf
    hit3d_out_rdf = hit3d_rdf

    if n_events > 0:
        compression_level = partition_config["output"].get("compression_level")
        snapshot_rdf(
            feature_out_rdf,
            feature_tree_name,
            feature_output,
            feature_snapshot_columns,
            compression_level=compression_level,
        )
        snapshot_rdf(
            hit3d_out_rdf,
            hit3d_tree_name,
            hit3d_output,
            hit3d_snapshot_columns,
            compression_level=compression_level,
        )
    else:
        log("RDF partition has zero selected events; writing metadata header only")

    write_rdf_metadata_row(
        output_metadata=output_metadata,
        region=region,
        particle_group=particle_group,
        particle_family=particle_family,
        particle_id=particle_id,
        particle_rule_name=particle_rule_name,
        n_events=n_events,
        lumi_per_partition=lumi_per_partition,
        settings=settings,
        region_config=region_config,
        partition_config=partition_config,
        feature_output=metadata_feature_output,
        hit3d_output=metadata_hit3d_output,
        metadata_rows=metadata_rows,
        part_index=part_index,
    )


def main():
    script_start = time.time()
    parser = argparse.ArgumentParser(description="Build aligned region/particle feature and hit3D partitions.")
    parser.add_argument("-m", "--metadata", nargs="+", required=True, help="Input metadata CSV file(s)")
    parser.add_argument("-r", "--region-config", required=True, help="Region YAML config")
    parser.add_argument("-p", "--partition-config", required=True, help="Partition YAML config")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory for partition ROOT files")
    parser.add_argument("--output-metadata", default=None, help="Output partition metadata CSV")
    parser.add_argument("--only-regions", default=None, help="Comma-separated region names to process")
    parser.add_argument("--only-particles", default=None, help="Comma-separated particle groups/families/rules to process")
    parser.add_argument("--only-part-index", type=int, default=None, help="Write only this 1-based partition index")
    parser.add_argument("--output-feature", default=None, help="Override feature ROOT output path for a single partition")
    parser.add_argument("--output-hit3d", default=None, help="Override hit3D ROOT output path for a single partition")
    parser.add_argument(
        "--metadata-feature-output",
        default=None,
        help="Feature ROOT path to record in metadata when --output-feature points to a temporary file",
    )
    parser.add_argument(
        "--metadata-hit3d-output",
        default=None,
        help="Hit3D ROOT path to record in metadata when --output-hit3d points to a temporary file",
    )
    parser.add_argument("--skip-missing-files", action="store_true", help="Skip metadata rows whose feature/hit3D files are missing")
    parser.add_argument(
        "--check-entry-counts",
        action="store_true",
        help="Before RDF filtering, verify each selected feature/hit3D file pair has the same number of entries",
    )
    parser.add_argument(
        "--rdf-threads",
        type=int,
        default=0,
        help="Enable ROOT implicit multithreading with this many threads; 0 keeps ROOT single-threaded",
    )
    args = parser.parse_args()

    log("Starting build_region_partitions.py")
    log(f"Metadata inputs: {', '.join(args.metadata)}")
    log(f"Region config: {args.region_config}")
    log(f"Partition config: {args.partition_config}")
    if args.only_regions:
        log(f"Only regions: {args.only_regions}")
    if args.only_particles:
        log(f"Only particles: {args.only_particles}")
    if args.only_part_index is not None:
        log(f"Only partition index: {args.only_part_index:03d}")

    region_config = load_yaml(args.region_config)
    partition_config = load_yaml(args.partition_config)

    output_dir = args.output_dir or partition_config["output"]["root_dir"]
    output_metadata = args.output_metadata or partition_config["output"]["metadata_csv"]
    only_regions = split_csv_arg(args.only_regions)
    only_particles = split_csv_arg(args.only_particles)

    run_rdf_partition(
        metadata_paths=args.metadata,
        region_config=region_config,
        partition_config=partition_config,
        output_dir=output_dir,
        output_metadata=output_metadata,
        only_regions=only_regions,
        only_particles=only_particles,
        part_index=args.only_part_index,
        feature_output=args.output_feature,
        hit3d_output=args.output_hit3d,
        metadata_feature_output=args.metadata_feature_output,
        metadata_hit3d_output=args.metadata_hit3d_output,
        skip_missing_files=args.skip_missing_files,
        check_entry_counts_enabled=args.check_entry_counts,
        rdf_threads=args.rdf_threads,
    )
    log(
        f"Finished build_region_partitions.py in {format_duration(time.time() - script_start)}; "
        f"partition metadata: {output_metadata}"
    )


if __name__ == "__main__":
    main()
