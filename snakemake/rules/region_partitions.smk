REGION_CONFIG = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/regions/region_v1.yaml"
PARTITION_CONFIG = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/partitions/partition_v1.yaml"

with open(REGION_CONFIG, "r", encoding="utf-8") as handle:
    REGION_PARTITION_REGION_CONFIG = yaml.safe_load(handle)

with open(PARTITION_CONFIG, "r", encoding="utf-8") as handle:
    REGION_PARTITION_CONFIG = yaml.safe_load(handle)

REGION_VERSION = REGION_PARTITION_REGION_CONFIG["region_version"]
PARTITION_VERSION = REGION_PARTITION_CONFIG["partition_version"]
REGION_PARTITION_ROOT_DIR = REGION_PARTITION_CONFIG["output"]["root_dir"]
if not os.path.isabs(REGION_PARTITION_ROOT_DIR):
    REGION_PARTITION_ROOT_DIR = f"{PERSONAL_WORK_SPACE}/{REGION_PARTITION_ROOT_DIR}"

REGION_PARTITION_METADATA_DIR = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/region_partitions/"
    f"{PARTITION_VERSION}_{REGION_VERSION}"
)
REGION_PARTITION_FEATURE_ROW_DIR = f"{REGION_PARTITION_METADATA_DIR}/rows_feature"
REGION_PARTITION_HIT3D_ROW_DIR = f"{REGION_PARTITION_METADATA_DIR}/rows_hit3d"
REGION_PARTITION_METADATA_CSV = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/region_partitions/"
    f"{PARTITION_VERSION}_{REGION_VERSION}.csv"
)


def _sanitize_partition_name(value):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value)).strip("_")


def _parse_partition_energy_range(value):
    text = str(value).strip()
    match = re.fullmatch(r"\(?\s*(\d+)\s*[-,]\s*(\d+)\s*\)?(?:\s*GeV)?", text)
    if not match:
        raise ValueError(f"Could not parse energy_range value: {value}")
    return int(match.group(1)), int(match.group(2))


def _region_partition_metadata_files():
    metadata_root = f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated"
    metadata_names = raw_metadata_csv_list + subset_metadata_csv_list + eventbuilder_metadata_csv_list
    return [
        f"{metadata_root}/{metadata_name}"
        for metadata_name in metadata_names
        if os.path.isfile(f"{metadata_root}/{metadata_name}")
    ]


REGION_PARTITION_SOURCE_METADATA = _region_partition_metadata_files()


def _load_region_partition_source_rows():
    rows = []
    for metadata_path in REGION_PARTITION_SOURCE_METADATA:
        df = pd.read_csv(metadata_path)
        for row_number, row in df.iterrows():
            item = row.to_dict()
            item["__metadata_path"] = metadata_path
            item["__row_number"] = int(row_number)
            rows.append(item)
    return rows


REGION_PARTITION_SOURCE_ROWS = _load_region_partition_source_rows()


def _metadata_row_matches_partition_rule(row, particle_rule_config):
    if str(row.get("data_type", "")) != str(particle_rule_config["source_data_type"]):
        return False

    allowed_subfolders = particle_rule_config.get("source_subfolders")
    if allowed_subfolders and str(row.get("subfolder", "")) not in {str(v) for v in allowed_subfolders}:
        return False

    return True


def _particle_group_for_metadata_row(row, rule_name, particle_rule_config):
    if not _metadata_row_matches_partition_rule(row, particle_rule_config):
        return None

    if "derive_particle_group" not in particle_rule_config:
        return rule_name

    energy_column = particle_rule_config["energy_column"]
    if energy_column not in row or pd.isna(row[energy_column]):
        return None

    energy_min, energy_max = _parse_partition_energy_range(row[energy_column])
    return particle_rule_config["derive_particle_group"].format(
        particle_family=particle_rule_config["particle_family"],
        energy_min=energy_min,
        energy_max=energy_max,
    )


def _region_partition_particle_groups():
    groups = {}
    for rule_name, particle_rule_config in REGION_PARTITION_CONFIG["particle_group_rules"].items():
        if not particle_rule_config.get("process", True):
            continue
        for row in REGION_PARTITION_SOURCE_ROWS:
            particle_group = _particle_group_for_metadata_row(row, rule_name, particle_rule_config)
            if particle_group is None:
                continue
            groups[particle_group] = {
                "particle_group": particle_group,
                "particle_rule": rule_name,
                "particle_family": str(particle_rule_config["particle_family"]),
            }
    return groups


REGION_PARTITION_PARTICLE_GROUPS = _region_partition_particle_groups()


def _region_partition_settings(region, particle_group):
    group_info = REGION_PARTITION_PARTICLE_GROUPS[particle_group]
    particle_rule = group_info["particle_rule"]
    particle_family = group_info["particle_family"]
    settings = dict(REGION_PARTITION_CONFIG.get("defaults", {}))
    particle_rule_config = REGION_PARTITION_CONFIG["particle_group_rules"][particle_rule]
    settings.update({k: v for k, v in particle_rule_config.items() if k not in {"event_filter"}})
    settings.update(REGION_PARTITION_CONFIG.get("regions", {}).get(region, {}))

    override = REGION_PARTITION_CONFIG.get("overrides", {}).get(region, {})
    for key in (particle_group, particle_rule, particle_family):
        if key in override:
            settings.update(override[key])
            break

    return settings


def _region_partition_id(region, particle_group, part_index):
    return REGION_PARTITION_CONFIG["grouping"]["output_partition_id_template"].format(
        region=_sanitize_partition_name(region),
        particle_group=_sanitize_partition_name(particle_group),
        part_index=int(part_index),
    )


def _region_partition_root_path(region, particle_group, part_index, prefix):
    partition_id = _region_partition_id(region, particle_group, part_index)
    return (
        f"{REGION_PARTITION_ROOT_DIR}/{PARTITION_VERSION}/{REGION_VERSION}/"
        f"{_sanitize_partition_name(region)}/{_sanitize_partition_name(particle_group)}/"
        f"{prefix}_{partition_id}.root"
    )


def _region_partition_product_enabled(region, particle_group, product):
    product_config = REGION_PARTITION_CONFIG.get("products", {}).get(product, {})
    if not product_config.get("enabled", True):
        return False
    settings = _region_partition_settings(region, particle_group)
    if not settings.get("process", True):
        return False
    default_process = product_config.get("default_process", True)
    return bool(settings.get(f"produce_{product}", default_process))


def _region_partition_product_row_path(region, particle_group, part_index, product):
    partition_id = _region_partition_id(region, particle_group, part_index)
    if product == "feature":
        return f"{REGION_PARTITION_FEATURE_ROW_DIR}/{partition_id}.csv"
    if product == "hit3d":
        return f"{REGION_PARTITION_HIT3D_ROW_DIR}/{partition_id}.csv"
    raise ValueError(f"Unknown region partition product: {product}")


def _region_partition_product_targets(product):
    targets = []
    for region in REGION_PARTITION_REGION_CONFIG["regions"].keys():
        for particle_group in REGION_PARTITION_PARTICLE_GROUPS.keys():
            settings = _region_partition_settings(region, particle_group)
            if not settings.get("process", True):
                continue
            if not _region_partition_product_enabled(region, particle_group, product):
                continue
            for part_index in range(1, int(settings.get("n_partitions", 1)) + 1):
                targets.append(_region_partition_product_row_path(region, particle_group, part_index, product))
    return targets


REGION_PARTITION_FEATURE_ROW_TARGETS = _region_partition_product_targets("feature")
REGION_PARTITION_HIT3D_ROW_TARGETS = _region_partition_product_targets("hit3d")


rule region_partitions:
    input:
        REGION_PARTITION_METADATA_CSV


rule region_feature_partitions:
    input:
        REGION_PARTITION_FEATURE_ROW_TARGETS


rule region_hit3d_partitions:
    input:
        REGION_PARTITION_HIT3D_ROW_TARGETS


rule build_region_feature_partition:
    input:
        metadata=REGION_PARTITION_SOURCE_METADATA,
        region_config=REGION_CONFIG,
        partition_config=PARTITION_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/convertData/build_region_partitions.py",
    output:
        metadata_row=f"{REGION_PARTITION_FEATURE_ROW_DIR}/{{region}}__{{particle_group}}__part{{part_index}}.csv"
    wildcard_constraints:
        part_index=r"\d{3}",
        region=r"[^/]+",
        particle_group=r"[^/]+"
    params:
        output_dir=REGION_PARTITION_ROOT_DIR,
        feature_output=lambda wildcards: _region_partition_root_path(
            wildcards.region,
            wildcards.particle_group,
            wildcards.part_index,
            REGION_PARTITION_CONFIG["output"]["feature_prefix"],
        ),
        check_entry_counts_flag=(
            "--check-entry-counts"
            if REGION_PARTITION_CONFIG.get("validation", {}).get("check_entry_counts", False)
            else ""
        ),
        skip_missing_files_flag=(
            "--skip-missing-files"
            if REGION_PARTITION_CONFIG.get("validation", {}).get("skip_missing_files", False)
            else ""
        ),
    threads: int(REGION_PARTITION_CONFIG.get("performance", {}).get("rdf_threads", 1))
    resources:
        runtime= 60 * 60,
        mem_mb=4000,
        disk_mb=4000,
        nvidia_gpu=0
    shell:
        r"""
        echo "Building region feature partition {wildcards.region} {wildcards.particle_group} part {wildcards.part_index}"
        set +u
        source {env_script_lcg}
        set -euo pipefail

        tmp_parent="${{TMPDIR:-/tmp}}"
        mkdir -p "$tmp_parent"
        tmp_dir=$(mktemp -d "${{tmp_parent}}/region_partition.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT

        tmp_feature="${{tmp_dir}}/$(basename "{params.feature_output}")"
        tmp_metadata="${{tmp_dir}}/$(basename "{output.metadata_row}")"

        copy_output() {{
            src="$1"
            dst="$2"
            if [[ "$dst" == root://* ]]; then
                xrdcp -f "$src" "$dst"
            elif [[ "$dst" == /eos/* ]]; then
                mkdir -p "$(dirname "$dst")"
                xrdcp -f "$src" "$dst"
            else
                mkdir -p "$(dirname "$dst")"
                tmp_dst="${{dst}}.tmp.$$"
                cp -f "$src" "$tmp_dst"
                mv -f "$tmp_dst" "$dst"
            fi
        }}

        python {input.script} \
            -m {input.metadata} \
            -r {input.region_config} \
            -p {input.partition_config} \
            -o {params.output_dir} \
            --output-metadata "$tmp_metadata" \
            --output-feature "$tmp_feature" \
            --metadata-feature-output "{params.feature_output}" \
            --only-regions {wildcards.region} \
            --only-particles {wildcards.particle_group} \
            --only-part-index {wildcards.part_index} \
            --rdf-threads {threads} \
            --product feature \
            {params.skip_missing_files_flag} \
            {params.check_entry_counts_flag}

        if [[ -f "$tmp_feature" ]]; then
            copy_output "$tmp_feature" "{params.feature_output}"
        fi
        copy_output "$tmp_metadata" "{output.metadata_row}"
        """


rule build_region_hit3d_partition:
    input:
        metadata=REGION_PARTITION_SOURCE_METADATA,
        region_config=REGION_CONFIG,
        partition_config=PARTITION_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/convertData/build_region_partitions.py",
        feature_metadata_row=(
            f"{REGION_PARTITION_FEATURE_ROW_DIR}/"
            "{region}__{particle_group}__part{part_index}.csv"
        ),
    output:
        metadata_row=f"{REGION_PARTITION_HIT3D_ROW_DIR}/{{region}}__{{particle_group}}__part{{part_index}}.csv"
    wildcard_constraints:
        part_index=r"\d{3}",
        region=r"[^/]+",
        particle_group=r"[^/]+"
    params:
        output_dir=REGION_PARTITION_ROOT_DIR,
        feature_output=lambda wildcards: _region_partition_root_path(
            wildcards.region,
            wildcards.particle_group,
            wildcards.part_index,
            REGION_PARTITION_CONFIG["output"]["feature_prefix"],
        ),
        hit3d_output=lambda wildcards: _region_partition_root_path(
            wildcards.region,
            wildcards.particle_group,
            wildcards.part_index,
            REGION_PARTITION_CONFIG["output"]["hit3d_prefix"],
        ),
        check_entry_counts_flag=(
            "--check-entry-counts"
            if REGION_PARTITION_CONFIG.get("validation", {}).get("check_entry_counts", False)
            else ""
        ),
        skip_missing_files_flag=(
            "--skip-missing-files"
            if REGION_PARTITION_CONFIG.get("validation", {}).get("skip_missing_files", False)
            else ""
        ),
    threads: int(REGION_PARTITION_CONFIG.get("performance", {}).get("rdf_threads", 1))
    resources:
        runtime=60 * 60,
        mem_mb=4000,
        disk_mb=4000,
        nvidia_gpu=0
    shell:
        r"""
        echo "Building region hit3D partition {wildcards.region} {wildcards.particle_group} part {wildcards.part_index}"
        set +u
        source {env_script_lcg}
        set -euo pipefail

        tmp_parent="${{TMPDIR:-/tmp}}"
        mkdir -p "$tmp_parent"
        tmp_dir=$(mktemp -d "${{tmp_parent}}/region_partition.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT

        tmp_hit3d="${{tmp_dir}}/$(basename "{params.hit3d_output}")"
        tmp_metadata="${{tmp_dir}}/$(basename "{output.metadata_row}")"

        copy_output() {{
            src="$1"
            dst="$2"
            if [[ "$dst" == root://* ]]; then
                xrdcp -f "$src" "$dst"
            elif [[ "$dst" == /eos/* ]]; then
                mkdir -p "$(dirname "$dst")"
                xrdcp -f "$src" "$dst"
            else
                mkdir -p "$(dirname "$dst")"
                tmp_dst="${{dst}}.tmp.$$"
                cp -f "$src" "$tmp_dst"
                mv -f "$tmp_dst" "$dst"
            fi
        }}

        python {input.script} \
            -m {input.metadata} \
            -r {input.region_config} \
            -p {input.partition_config} \
            -o {params.output_dir} \
            --output-metadata "$tmp_metadata" \
            --output-hit3d "$tmp_hit3d" \
            --metadata-feature-output "{params.feature_output}" \
            --metadata-hit3d-output "{params.hit3d_output}" \
            --only-regions {wildcards.region} \
            --only-particles {wildcards.particle_group} \
            --only-part-index {wildcards.part_index} \
            --rdf-threads {threads} \
            --product hit3d \
            {params.skip_missing_files_flag} \
            {params.check_entry_counts_flag}

        if [[ -f "$tmp_hit3d" ]]; then
            copy_output "$tmp_hit3d" "{params.hit3d_output}"
        fi
        copy_output "$tmp_metadata" "{output.metadata_row}"
        """


rule combine_region_partition_metadata:
    input:
        feature_rows=REGION_PARTITION_FEATURE_ROW_TARGETS,
        hit3d_rows=REGION_PARTITION_HIT3D_ROW_TARGETS,
    output:
        REGION_PARTITION_METADATA_CSV
    resources:
        runtime=30 * 60,
        mem_mb=1000,
        disk_mb=1000,
        nvidia_gpu=0
    run:
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)

        fieldnames = REGION_PARTITION_CONFIG["metadata_output_columns"]
        rows_by_partition = {}

        def _read_nonempty_rows(paths):
            rows = []
            for path in paths:
                if os.path.getsize(path) == 0:
                    continue
                frame = pd.read_csv(path)
                if len(frame) > 0:
                    rows.extend(frame.to_dict("records"))
            return rows

        def _truthy(value):
            return str(value).strip().lower() in {"1", "true", "yes"}

        def _clean(value):
            if pd.isna(value):
                return ""
            return value

        for row in _read_nonempty_rows(input.feature_rows):
            partition_id = row["partition_id"]
            rows_by_partition[partition_id] = {
                field: _clean(row.get(field, ""))
                for field in fieldnames
            }

        for row in _read_nonempty_rows(input.hit3d_rows):
            partition_id = row["partition_id"]
            cleaned = {field: _clean(row.get(field, "")) for field in fieldnames}
            if partition_id not in rows_by_partition:
                rows_by_partition[partition_id] = cleaned
                continue

            merged = rows_by_partition[partition_id]
            for field in (
                "hit3d_partition_path",
                "hit3d_available",
                "hit3d_required",
                "source_hit3d_paths",
            ):
                if cleaned.get(field, "") != "":
                    merged[field] = cleaned[field]

            for field in ("n_events", "lumi_per_partition"):
                if merged.get(field, "") == "" and cleaned.get(field, "") != "":
                    merged[field] = cleaned[field]

            products = []
            for value in (merged.get("products_written", ""), cleaned.get("products_written", "")):
                for product in str(value).split(";"):
                    product = product.strip()
                    if product and product not in products:
                        products.append(product)
            merged["products_written"] = ";".join(products)
            merged["feature_available"] = bool(_truthy(merged.get("feature_available", False)))
            merged["hit3d_available"] = bool(_truthy(merged.get("hit3d_available", False)))
            merged["hit3d_required"] = bool(_truthy(merged.get("hit3d_required", False)))

        rows = list(rows_by_partition.values())
        if rows:
            pd.DataFrame(rows, columns=fieldnames).to_csv(output[0], index=False)
        else:
            pd.DataFrame(columns=fieldnames).to_csv(output[0], index=False)
