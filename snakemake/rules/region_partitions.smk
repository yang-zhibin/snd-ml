from pathlib import Path

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
ABCD_INPUT_AUDIT_CONFIG = (
    f"{workflow.basedir}/metadata/configs/region_evaluation/"
    "abcd_input_audit_v1.yaml"
)

with open(ABCD_INPUT_AUDIT_CONFIG, "r", encoding="utf-8") as handle:
    ABCD_INPUT_AUDIT = yaml.safe_load(handle)

ABCD_INPUT_AUDIT_OUTPUT_DIR = ABCD_INPUT_AUDIT["output"]["directory"]
if not os.path.isabs(ABCD_INPUT_AUDIT_OUTPUT_DIR):
    ABCD_INPUT_AUDIT_OUTPUT_DIR = (
        f"{PERSONAL_WORK_SPACE}/{ABCD_INPUT_AUDIT_OUTPUT_DIR}"
    )
ABCD_INPUT_AUDIT_CSV = (
    f"{ABCD_INPUT_AUDIT_OUTPUT_DIR}/{ABCD_INPUT_AUDIT['output']['csv']}"
)
ABCD_INPUT_AUDIT_REPORT = (
    f"{ABCD_INPUT_AUDIT_OUTPUT_DIR}/{ABCD_INPUT_AUDIT['output']['report']}"
)
ABCD_DENSITY_CONFIG = (
    f"{workflow.basedir}/metadata/configs/region_evaluation/"
    "abcd_density_sndsw_scifi_v1.yaml"
)

with open(ABCD_DENSITY_CONFIG, "r", encoding="utf-8") as handle:
    ABCD_DENSITY = yaml.safe_load(handle)

ABCD_DENSITY_OUTPUT_DIR = ABCD_DENSITY["output"]["directory"]
if not os.path.isabs(ABCD_DENSITY_OUTPUT_DIR):
    ABCD_DENSITY_OUTPUT_DIR = f"{PERSONAL_WORK_SPACE}/{ABCD_DENSITY_OUTPUT_DIR}"
ABCD_DENSITY_PARTICLE_GROUP = ABCD_DENSITY["catalogue"]["particle_group"]
ABCD_DENSITY_REGIONS = list(ABCD_DENSITY["regions"])
ABCD_DENSITY_HIST_PATTERN = (
    f"{ABCD_DENSITY_OUTPUT_DIR}/"
    f"{ABCD_DENSITY['output']['histogram_directory']}/"
    f"{{region}}__{ABCD_DENSITY_PARTICLE_GROUP}.root"
)
ABCD_DENSITY_SUMMARY_PATTERN = (
    f"{ABCD_DENSITY_OUTPUT_DIR}/"
    f"{ABCD_DENSITY['output']['summary_directory']}/"
    f"{{region}}__{ABCD_DENSITY_PARTICLE_GROUP}.csv"
)
ABCD_DENSITY_HIST_TARGETS = expand(
    ABCD_DENSITY_HIST_PATTERN,
    region=ABCD_DENSITY_REGIONS,
)
ABCD_DENSITY_SUMMARY_TARGETS = expand(
    ABCD_DENSITY_SUMMARY_PATTERN,
    region=ABCD_DENSITY_REGIONS,
)
ABCD_DENSITY_MC_CONFIG = (
    f"{workflow.basedir}/metadata/configs/region_evaluation/"
    "abcd_density_neutrino_mc_v1.yaml"
)

with open(ABCD_DENSITY_MC_CONFIG, "r", encoding="utf-8") as handle:
    ABCD_DENSITY_MC = yaml.safe_load(handle)

ABCD_DENSITY_MC_OUTPUT_DIR = ABCD_DENSITY_MC["output"]["directory"]
if not os.path.isabs(ABCD_DENSITY_MC_OUTPUT_DIR):
    ABCD_DENSITY_MC_OUTPUT_DIR = (
        f"{PERSONAL_WORK_SPACE}/{ABCD_DENSITY_MC_OUTPUT_DIR}"
    )
ABCD_DENSITY_MC_REGIONS = list(ABCD_DENSITY_MC["regions"])
ABCD_DENSITY_MC_GROUPS = list(ABCD_DENSITY_MC["samples"])
ABCD_DENSITY_MC_HIST_PATTERN = (
    f"{ABCD_DENSITY_MC_OUTPUT_DIR}/"
    f"{ABCD_DENSITY_MC['output']['histogram_directory']}/"
    "{region}__{particle_group}.root"
)
ABCD_DENSITY_MC_SUMMARY_PATTERN = (
    f"{ABCD_DENSITY_MC_OUTPUT_DIR}/"
    f"{ABCD_DENSITY_MC['output']['summary_directory']}/"
    "{region}__{particle_group}.csv"
)
ABCD_DENSITY_MC_HIST_TARGETS = expand(
    ABCD_DENSITY_MC_HIST_PATTERN,
    region=ABCD_DENSITY_MC_REGIONS,
    particle_group=ABCD_DENSITY_MC_GROUPS,
)
ABCD_DENSITY_MC_SUMMARY_TARGETS = expand(
    ABCD_DENSITY_MC_SUMMARY_PATTERN,
    region=ABCD_DENSITY_MC_REGIONS,
    particle_group=ABCD_DENSITY_MC_GROUPS,
)
ABCD_DENSITY_TABLE_CONFIG = (
    f"{workflow.basedir}/metadata/configs/region_evaluation/"
    "abcd_density_analysis_table_v1.yaml"
)

with open(ABCD_DENSITY_TABLE_CONFIG, "r", encoding="utf-8") as handle:
    ABCD_DENSITY_TABLE = yaml.safe_load(handle)

ABCD_DENSITY_TABLE_OUTPUT_DIR = ABCD_DENSITY_TABLE["output"]["directory"]
if not os.path.isabs(ABCD_DENSITY_TABLE_OUTPUT_DIR):
    ABCD_DENSITY_TABLE_OUTPUT_DIR = (
        f"{PERSONAL_WORK_SPACE}/{ABCD_DENSITY_TABLE_OUTPUT_DIR}"
    )
ABCD_DENSITY_TABLE_CSV = (
    f"{ABCD_DENSITY_TABLE_OUTPUT_DIR}/{ABCD_DENSITY_TABLE['output']['csv']}"
)
ABCD_DENSITY_REPORT_CONFIG = (
    f"{workflow.basedir}/metadata/configs/region_evaluation/"
    "abcd_density_analysis_report_v1.yaml"
)

with open(ABCD_DENSITY_REPORT_CONFIG, "r", encoding="utf-8") as handle:
    ABCD_DENSITY_REPORT = yaml.safe_load(handle)

ABCD_DENSITY_REPORT_OUTPUT_DIR = ABCD_DENSITY_REPORT["output"]["directory"]
if not os.path.isabs(ABCD_DENSITY_REPORT_OUTPUT_DIR):
    ABCD_DENSITY_REPORT_OUTPUT_DIR = (
        f"{PERSONAL_WORK_SPACE}/{ABCD_DENSITY_REPORT_OUTPUT_DIR}"
    )
ABCD_DENSITY_REPORT_TEX = (
    f"{ABCD_DENSITY_REPORT_OUTPUT_DIR}/{ABCD_DENSITY_REPORT['output']['tex']}"
)
ABCD_DENSITY_REPORT_PDF = (
    f"{ABCD_DENSITY_REPORT_OUTPUT_DIR}/{ABCD_DENSITY_REPORT['output']['pdf']}"
)
ABCD_DENSITY_REPORT_PLOT_DIR = (
    f"{ABCD_DENSITY_REPORT_OUTPUT_DIR}/"
    f"{ABCD_DENSITY_REPORT['output']['normalized_plot_directory']}"
)
ABCD_DENSITY_MERGE_CONFIG = (
    f"{workflow.basedir}/metadata/configs/region_evaluation/"
    "abcd_density_merged_binning_v1.yaml"
)

with open(ABCD_DENSITY_MERGE_CONFIG, "r", encoding="utf-8") as handle:
    ABCD_DENSITY_MERGE = yaml.safe_load(handle)

ABCD_DENSITY_MERGE_OUTPUT_DIR = ABCD_DENSITY_MERGE["output"]["directory"]
if not os.path.isabs(ABCD_DENSITY_MERGE_OUTPUT_DIR):
    ABCD_DENSITY_MERGE_OUTPUT_DIR = (
        f"{PERSONAL_WORK_SPACE}/{ABCD_DENSITY_MERGE_OUTPUT_DIR}"
    )
ABCD_DENSITY_MERGE_CSV = (
    f"{ABCD_DENSITY_MERGE_OUTPUT_DIR}/{ABCD_DENSITY_MERGE['output']['csv']}"
)
ABCD_DENSITY_MERGE_MARKDOWN = (
    f"{ABCD_DENSITY_MERGE_OUTPUT_DIR}/"
    f"{ABCD_DENSITY_MERGE['output']['markdown']}"
)
ABCD_DENSITY_VALIDATION_CLOSURE_CONFIG = (
    f"{workflow.basedir}/metadata/configs/region_evaluation/"
    "abcd_density_validation_closure_v1.yaml"
)

with open(ABCD_DENSITY_VALIDATION_CLOSURE_CONFIG, "r", encoding="utf-8") as handle:
    ABCD_DENSITY_VALIDATION_CLOSURE = yaml.safe_load(handle)

ABCD_DENSITY_VALIDATION_CLOSURE_OUTPUT_DIR = (
    ABCD_DENSITY_VALIDATION_CLOSURE["output"]["directory"]
)
if not os.path.isabs(ABCD_DENSITY_VALIDATION_CLOSURE_OUTPUT_DIR):
    ABCD_DENSITY_VALIDATION_CLOSURE_OUTPUT_DIR = (
        f"{PERSONAL_WORK_SPACE}/{ABCD_DENSITY_VALIDATION_CLOSURE_OUTPUT_DIR}"
    )
ABCD_DENSITY_VALIDATION_CLOSURE_CSV = (
    f"{ABCD_DENSITY_VALIDATION_CLOSURE_OUTPUT_DIR}/"
    f"{ABCD_DENSITY_VALIDATION_CLOSURE['output']['csv']}"
)
ABCD_DENSITY_VALIDATION_CLOSURE_MARKDOWN = (
    f"{ABCD_DENSITY_VALIDATION_CLOSURE_OUTPUT_DIR}/"
    f"{ABCD_DENSITY_VALIDATION_CLOSURE['output']['markdown']}"
)
ABCD_DENSITY_VALIDATION_CLOSURE_TEX = (
    f"{ABCD_DENSITY_VALIDATION_CLOSURE_OUTPUT_DIR}/"
    f"{ABCD_DENSITY_VALIDATION_CLOSURE['output']['tex']}"
)
ABCD_DENSITY_VALIDATION_CLOSURE_PDF = (
    f"{ABCD_DENSITY_VALIDATION_CLOSURE_OUTPUT_DIR}/"
    f"{ABCD_DENSITY_VALIDATION_CLOSURE['output']['pdf']}"
)
ABCD_DENSITY_VALIDATION_CLOSURE_PLOT_DIR = (
    f"{ABCD_DENSITY_VALIDATION_CLOSURE_OUTPUT_DIR}/"
    f"{ABCD_DENSITY_VALIDATION_CLOSURE['output']['plot_directory']}"
)
ABCD_DENSITY_NOMINAL_CLOSURE_CONFIG = (
    f"{workflow.basedir}/metadata/configs/region_evaluation/"
    "abcd_density_nominal_closure_v1.yaml"
)

with open(ABCD_DENSITY_NOMINAL_CLOSURE_CONFIG, "r", encoding="utf-8") as handle:
    ABCD_DENSITY_NOMINAL_CLOSURE = yaml.safe_load(handle)

ABCD_DENSITY_NOMINAL_CLOSURE_OUTPUT_DIR = (
    ABCD_DENSITY_NOMINAL_CLOSURE["output"]["directory"]
)
if not os.path.isabs(ABCD_DENSITY_NOMINAL_CLOSURE_OUTPUT_DIR):
    ABCD_DENSITY_NOMINAL_CLOSURE_OUTPUT_DIR = (
        f"{PERSONAL_WORK_SPACE}/{ABCD_DENSITY_NOMINAL_CLOSURE_OUTPUT_DIR}"
    )
ABCD_DENSITY_NOMINAL_CLOSURE_CSV = (
    f"{ABCD_DENSITY_NOMINAL_CLOSURE_OUTPUT_DIR}/"
    f"{ABCD_DENSITY_NOMINAL_CLOSURE['output']['csv']}"
)
ABCD_DENSITY_NOMINAL_CLOSURE_MARKDOWN = (
    f"{ABCD_DENSITY_NOMINAL_CLOSURE_OUTPUT_DIR}/"
    f"{ABCD_DENSITY_NOMINAL_CLOSURE['output']['markdown']}"
)
ABCD_DENSITY_NOMINAL_CLOSURE_TEX = (
    f"{ABCD_DENSITY_NOMINAL_CLOSURE_OUTPUT_DIR}/"
    f"{ABCD_DENSITY_NOMINAL_CLOSURE['output']['tex']}"
)
ABCD_DENSITY_NOMINAL_CLOSURE_PDF = (
    f"{ABCD_DENSITY_NOMINAL_CLOSURE_OUTPUT_DIR}/"
    f"{ABCD_DENSITY_NOMINAL_CLOSURE['output']['pdf']}"
)
ABCD_DENSITY_NOMINAL_CLOSURE_PLOT_DIR = (
    f"{ABCD_DENSITY_NOMINAL_CLOSURE_OUTPUT_DIR}/"
    f"{ABCD_DENSITY_NOMINAL_CLOSURE['output']['plot_directory']}"
)
ABCD_DENSITY_SIGNAL_REGION_FIT_CONFIG = (
    f"{workflow.basedir}/metadata/configs/region_evaluation/"
    "abcd_density_signal_region_fit_v1.yaml"
)

with open(ABCD_DENSITY_SIGNAL_REGION_FIT_CONFIG, "r", encoding="utf-8") as handle:
    ABCD_DENSITY_SIGNAL_REGION_FIT = yaml.safe_load(handle)

ABCD_DENSITY_SIGNAL_REGION_FIT_OUTPUT_DIR = (
    ABCD_DENSITY_SIGNAL_REGION_FIT["output"]["directory"]
)
if not os.path.isabs(ABCD_DENSITY_SIGNAL_REGION_FIT_OUTPUT_DIR):
    ABCD_DENSITY_SIGNAL_REGION_FIT_OUTPUT_DIR = (
        f"{PERSONAL_WORK_SPACE}/{ABCD_DENSITY_SIGNAL_REGION_FIT_OUTPUT_DIR}"
    )


def _abcd_density_signal_region_fit_output(key):
    return (
        f"{ABCD_DENSITY_SIGNAL_REGION_FIT_OUTPUT_DIR}/"
        f"{ABCD_DENSITY_SIGNAL_REGION_FIT['output'][key]}"
    )


ABCD_DENSITY_SIGNAL_REGION_FIT_INPUTS_CSV = _abcd_density_signal_region_fit_output(
    "inputs_csv"
)
ABCD_DENSITY_SIGNAL_REGION_FIT_WORKSPACE_JSON = _abcd_density_signal_region_fit_output(
    "workspace_json"
)
ABCD_DENSITY_SIGNAL_REGION_FIT_RESULTS_JSON = _abcd_density_signal_region_fit_output(
    "fit_results_json"
)
ABCD_DENSITY_SIGNAL_REGION_FIT_MU_SCAN_CSV = _abcd_density_signal_region_fit_output(
    "mu_scan_csv"
)
ABCD_DENSITY_SIGNAL_REGION_FIT_ALPHA_SCAN_CSV = _abcd_density_signal_region_fit_output(
    "alpha_scan_csv"
)
ABCD_DENSITY_SIGNAL_REGION_FIT_2D_CSV = _abcd_density_signal_region_fit_output(
    "likelihood_2d_csv"
)
ABCD_DENSITY_SIGNAL_REGION_FIT_REPORT = _abcd_density_signal_region_fit_output(
    "report_markdown"
)
ABCD_DENSITY_SIGNAL_REGION_FIT_REPORT_TEX = _abcd_density_signal_region_fit_output(
    "report_tex"
)
ABCD_DENSITY_SIGNAL_REGION_FIT_REPORT_PDF = _abcd_density_signal_region_fit_output(
    "report_pdf"
)
ABCD_DENSITY_SIGNAL_REGION_FIT_PLOT_DIR = _abcd_density_signal_region_fit_output(
    "plot_directory"
)
ABCD_DENSITY_HIGH_DENSITY_CONFIG = (
    f"{workflow.basedir}/metadata/configs/region_evaluation/"
    "abcd_density_high_density_fits_v1.yaml"
)

with open(ABCD_DENSITY_HIGH_DENSITY_CONFIG, "r", encoding="utf-8") as handle:
    ABCD_DENSITY_HIGH_DENSITY = yaml.safe_load(handle)

ABCD_DENSITY_HIGH_DENSITY_OUTPUT_DIR = ABCD_DENSITY_HIGH_DENSITY["output"]["directory"]
if not os.path.isabs(ABCD_DENSITY_HIGH_DENSITY_OUTPUT_DIR):
    ABCD_DENSITY_HIGH_DENSITY_OUTPUT_DIR = (
        f"{PERSONAL_WORK_SPACE}/{ABCD_DENSITY_HIGH_DENSITY_OUTPUT_DIR}"
    )


def _abcd_density_high_density_output(analysis, key):
    analysis_dir = ABCD_DENSITY_HIGH_DENSITY["analyses"][analysis]["output_directory"]
    return (
        f"{ABCD_DENSITY_HIGH_DENSITY_OUTPUT_DIR}/{analysis_dir}/"
        f"{ABCD_DENSITY_HIGH_DENSITY['output'][key]}"
    )


ABCD_HIGH_CUT_INPUTS = _abcd_density_high_density_output("cutcount", "inputs_csv")
ABCD_HIGH_CUT_WORKSPACE = _abcd_density_high_density_output("cutcount", "workspace_json")
ABCD_HIGH_CUT_RESULTS = _abcd_density_high_density_output("cutcount", "fit_results_json")
ABCD_HIGH_CUT_SCAN = _abcd_density_high_density_output("cutcount", "mu_scan_csv")
ABCD_HIGH_CUT_MARKDOWN = _abcd_density_high_density_output("cutcount", "report_markdown")
ABCD_HIGH_CUT_TEX = _abcd_density_high_density_output("cutcount", "report_tex")
ABCD_HIGH_CUT_PDF = _abcd_density_high_density_output("cutcount", "report_pdf")
ABCD_HIGH_CUT_PLOTS = _abcd_density_high_density_output("cutcount", "plot_directory")
ABCD_HIGH_SHAPE_INPUTS = _abcd_density_high_density_output("shape", "inputs_csv")
ABCD_HIGH_SHAPE_WORKSPACE = _abcd_density_high_density_output("shape", "workspace_json")
ABCD_HIGH_SHAPE_RESULTS = _abcd_density_high_density_output("shape", "fit_results_json")
ABCD_HIGH_SHAPE_SCAN = _abcd_density_high_density_output("shape", "mu_scan_csv")
ABCD_HIGH_SHAPE_MARKDOWN = _abcd_density_high_density_output("shape", "report_markdown")
ABCD_HIGH_SHAPE_TEX = _abcd_density_high_density_output("shape", "report_tex")
ABCD_HIGH_SHAPE_PDF = _abcd_density_high_density_output("shape", "report_pdf")
ABCD_HIGH_SHAPE_PLOTS = _abcd_density_high_density_output("shape", "plot_directory")
ABCD_DENSITY_BLINDED_CONFIG = (
    f"{workflow.basedir}/metadata/configs/region_evaluation/"
    "abcd_density_blinded_counting_v2.yaml"
)
ABCD_DENSITY_UNBLIND_AUTHORIZATION = (
    f"{workflow.basedir}/metadata/configs/region_evaluation/"
    "abcd_density_unblind_authorization_v2.yaml"
)

with open(ABCD_DENSITY_BLINDED_CONFIG, "r", encoding="utf-8") as handle:
    ABCD_DENSITY_BLINDED = yaml.safe_load(handle)

ABCD_DENSITY_BLINDED_OUTPUT_DIR = ABCD_DENSITY_BLINDED["output"]["directory"]
if not os.path.isabs(ABCD_DENSITY_BLINDED_OUTPUT_DIR):
    ABCD_DENSITY_BLINDED_OUTPUT_DIR = (
        f"{PERSONAL_WORK_SPACE}/{ABCD_DENSITY_BLINDED_OUTPUT_DIR}"
    )
ABCD_BLINDED_SANITIZED = (
    f"{ABCD_DENSITY_BLINDED_OUTPUT_DIR}/"
    f"{ABCD_DENSITY_BLINDED['output']['sanitized_inputs']}"
)
ABCD_BLINDED_MANIFEST = (
    f"{ABCD_DENSITY_BLINDED_OUTPUT_DIR}/"
    f"{ABCD_DENSITY_BLINDED['output']['freeze_manifest']}"
)
ABCD_BLINDED_ROLE_REGIONS = ABCD_DENSITY_BLINDED["source"]["roles"]
ABCD_BLINDED_VALIDATION_REGIONS = list(
    ABCD_DENSITY_BLINDED["source"]["validation_roles"].values()
)
ABCD_BLINDED_CONTROL_REGIONS = list(dict.fromkeys(
    [ABCD_BLINDED_ROLE_REGIONS[role] for role in ("B", "C", "D")]
    + ABCD_BLINDED_VALIDATION_REGIONS
))
ABCD_BLINDED_ALL_REGIONS = list(dict.fromkeys(
    [ABCD_BLINDED_ROLE_REGIONS[role] for role in ("A", "B", "C", "D")]
    + ABCD_BLINDED_VALIDATION_REGIONS
))
ABCD_BLINDED_DATA_HISTS = expand(
    ABCD_DENSITY_HIST_PATTERN, region=ABCD_BLINDED_CONTROL_REGIONS
)
ABCD_BLINDED_DATA_SUMMARIES = expand(
    ABCD_DENSITY_SUMMARY_PATTERN, region=ABCD_BLINDED_CONTROL_REGIONS
)
ABCD_BLINDED_MC_HISTS = expand(
    ABCD_DENSITY_MC_HIST_PATTERN,
    region=ABCD_BLINDED_ALL_REGIONS,
    particle_group=ABCD_DENSITY_MC_GROUPS,
)
ABCD_BLINDED_MC_SUMMARIES = expand(
    ABCD_DENSITY_MC_SUMMARY_PATTERN,
    region=ABCD_BLINDED_ALL_REGIONS,
    particle_group=ABCD_DENSITY_MC_GROUPS,
)


def _abcd_density_blinded_output(directory_key, key):
    analysis_dir = ABCD_DENSITY_BLINDED["output"][directory_key]
    return (
        f"{ABCD_DENSITY_BLINDED_OUTPUT_DIR}/{analysis_dir}/"
        f"{ABCD_DENSITY_BLINDED['output'][key]}"
    )


ABCD_BLINDED_OUTPUT_KEYS = [
    "model_inputs", "control_workspace", "expected_background_workspace",
    "expected_signal_workspace", "control_fit", "control_covariance",
    "expected_results", "mu_scan",
    "report_markdown", "report_tex", "report_pdf",
]
ABCD_BLINDED_EXPECTED_OUTPUTS = {
    key: _abcd_density_blinded_output("expected_directory", key)
    for key in ABCD_BLINDED_OUTPUT_KEYS
}
ABCD_BLINDED_EXPECTED_PLOTS = _abcd_density_blinded_output(
    "expected_directory", "plot_directory"
)
ABCD_UNBLINDED_OUTPUT_KEYS = [
    "observed_workspace", "observed_results", "observed_mu_scan", "toy_q0",
    "final_report_markdown", "final_report_tex", "final_report_pdf",
]
ABCD_UNBLINDED_OUTPUTS = {
    key: _abcd_density_blinded_output("unblinded_directory", key)
    for key in ABCD_UNBLINDED_OUTPUT_KEYS
}
ABCD_UNBLINDED_PLOTS = _abcd_density_blinded_output(
    "unblinded_directory", "final_plot_directory"
)
ABCD_SIGNAL_DATA_HIST = ABCD_DENSITY_HIST_PATTERN.format(
    region=ABCD_BLINDED_ROLE_REGIONS["A"]
)
ABCD_SIGNAL_DATA_SUMMARY = ABCD_DENSITY_SUMMARY_PATTERN.format(
    region=ABCD_BLINDED_ROLE_REGIONS["A"]
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


rule region_v1_abcd_input_audit:
    input:
        region_config=REGION_CONFIG,
        audit_config=ABCD_INPUT_AUDIT_CONFIG,
        script=(
            f"{Path(workflow.basedir).parent}/evaluation_region_partitions/"
            "audit_abcd_region_inputs.py"
        ),
    params:
        partition_catalogue=REGION_PARTITION_METADATA_CSV,
    output:
        csv=ABCD_INPUT_AUDIT_CSV,
        report=ABCD_INPUT_AUDIT_REPORT,
    resources:
        runtime=10 * 60,
        mem_mb=1000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1

        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import yaml; print(yaml.__version__)' >&2

        python3 {input.script} \
          --partition-catalogue {params.partition_catalogue} \
          --region-config {input.region_config} \
          --audit-config {input.audit_config} \
          --output-csv {output.csv} \
          --output-report {output.report}
        """


rule region_v1_abcd_density_hists:
    input:
        ABCD_DENSITY_HIST_TARGETS,
        ABCD_DENSITY_SUMMARY_TARGETS,


rule build_region_v1_abcd_density_hist:
    input:
        config=ABCD_DENSITY_CONFIG,
        script=(
            f"{Path(workflow.basedir).parent}/evaluation_region_partitions/"
            "make_abcd_density_hist.py"
        ),
    params:
        partition_catalogue=REGION_PARTITION_METADATA_CSV,
        particle_group=ABCD_DENSITY_PARTICLE_GROUP,
    output:
        hist=ABCD_DENSITY_HIST_PATTERN,
        summary=ABCD_DENSITY_SUMMARY_PATTERN,
    wildcard_constraints:
        region="|".join(re.escape(region) for region in ABCD_DENSITY_REGIONS)
    threads: 2
    resources:
        runtime=60 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1

        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2

        python3 {input.script} \
          --partition-catalogue {params.partition_catalogue} \
          --config {input.config} \
          --region {wildcards.region} \
          --particle-group {params.particle_group} \
          --output {output.hist} \
          --summary {output.summary} \
          --threads {threads}
        """


rule region_v1_abcd_density_mc_hists:
    input:
        ABCD_DENSITY_MC_HIST_TARGETS,
        ABCD_DENSITY_MC_SUMMARY_TARGETS,


rule build_region_v1_abcd_density_mc_hist:
    input:
        config=ABCD_DENSITY_MC_CONFIG,
        script=(
            f"{Path(workflow.basedir).parent}/evaluation_region_partitions/"
            "make_abcd_density_mc_hist.py"
        ),
    params:
        partition_catalogue=REGION_PARTITION_METADATA_CSV,
    output:
        hist=ABCD_DENSITY_MC_HIST_PATTERN,
        summary=ABCD_DENSITY_MC_SUMMARY_PATTERN,
    wildcard_constraints:
        region="|".join(
            re.escape(region) for region in ABCD_DENSITY_MC_REGIONS
        ),
        particle_group="|".join(
            re.escape(group) for group in ABCD_DENSITY_MC_GROUPS
        )
    threads: 2
    resources:
        runtime=60 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1

        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2

        python3 {input.script} \
          --partition-catalogue {params.partition_catalogue} \
          --config {input.config} \
          --region {wildcards.region} \
          --particle-group {wildcards.particle_group} \
          --output {output.hist} \
          --summary {output.summary} \
          --threads {threads}
        """


rule region_v1_abcd_density_analysis_table:
    input:
        config=ABCD_DENSITY_TABLE_CONFIG,
        data_config=ABCD_DENSITY_CONFIG,
        mc_config=ABCD_DENSITY_MC_CONFIG,
        script=(
            f"{Path(workflow.basedir).parent}/evaluation_region_partitions/"
            "make_abcd_density_analysis_table.py"
        ),
        data_hists=ABCD_DENSITY_HIST_TARGETS,
        data_summaries=ABCD_DENSITY_SUMMARY_TARGETS,
        mc_hists=ABCD_DENSITY_MC_HIST_TARGETS,
        mc_summaries=ABCD_DENSITY_MC_SUMMARY_TARGETS,
    params:
        product_directory=ABCD_DENSITY_OUTPUT_DIR,
    output:
        table=ABCD_DENSITY_TABLE_CSV,
    resources:
        runtime=10 * 60,
        mem_mb=2000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1

        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2

        python3 {input.script} \
          --config {input.config} \
          --data-config {input.data_config} \
          --mc-config {input.mc_config} \
          --product-directory {params.product_directory} \
          --output {output.table}
        """


rule region_v1_abcd_density_analysis_report:
    input:
        table=ABCD_DENSITY_TABLE_CSV,
        config=ABCD_DENSITY_REPORT_CONFIG,
        script=(
            f"{Path(workflow.basedir).parent}/evaluation_region_partitions/"
            "make_abcd_density_analysis_report.py"
        ),
    output:
        tex=ABCD_DENSITY_REPORT_TEX,
        pdf=ABCD_DENSITY_REPORT_PDF,
        plot_dir=directory(ABCD_DENSITY_REPORT_PLOT_DIR),
    params:
        pdf_name=ABCD_DENSITY_REPORT["output"]["pdf"],
    resources:
        runtime=10 * 60,
        mem_mb=1000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1

        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import matplotlib, yaml; print(matplotlib.__version__, yaml.__version__)' >&2
        command -v pdflatex >&2

        python3 {input.script} \
          --input-table {input.table} \
          --config {input.config} \
          --output-tex {output.tex} \
          --output-plot-dir {output.plot_dir}

        report_build_dir=$(mktemp -d /tmp/abcd_density_report.XXXXXX)
        trap 'rm -rf "$report_build_dir"' EXIT
        pdflatex -interaction=nonstopmode -halt-on-error \
          -output-directory "$report_build_dir" {output.tex}
        pdflatex -interaction=nonstopmode -halt-on-error \
          -output-directory "$report_build_dir" {output.tex}
        cp "$report_build_dir/{params.pdf_name}" {output.pdf}
        """


rule region_v1_abcd_density_merge_policy:
    input:
        config=ABCD_DENSITY_MERGE_CONFIG,
        script=(
            f"{Path(workflow.basedir).parent}/evaluation_region_partitions/"
            "audit_abcd_density_merge_policy.py"
        ),
    output:
        csv=ABCD_DENSITY_MERGE_CSV,
        markdown=ABCD_DENSITY_MERGE_MARKDOWN,
    params:
        table=ABCD_DENSITY_TABLE_CSV,
    resources:
        runtime=5 * 60,
        mem_mb=500,
        disk_mb=500,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1

        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import yaml; print(yaml.__version__)' >&2

        python3 {input.script} \
          --input-table {params.table} \
          --config {input.config} \
          --output-csv {output.csv} \
          --output-markdown {output.markdown}
        """


rule region_v1_abcd_density_validation_closure:
    input:
        merge_audit=ABCD_DENSITY_MERGE_CSV,
        merge_config=ABCD_DENSITY_MERGE_CONFIG,
        config=ABCD_DENSITY_VALIDATION_CLOSURE_CONFIG,
        script=(
            f"{Path(workflow.basedir).parent}/evaluation_region_partitions/"
            "calculate_abcd_density_validation_closure.py"
        ),
    output:
        csv=ABCD_DENSITY_VALIDATION_CLOSURE_CSV,
        markdown=ABCD_DENSITY_VALIDATION_CLOSURE_MARKDOWN,
        tex=ABCD_DENSITY_VALIDATION_CLOSURE_TEX,
        pdf=ABCD_DENSITY_VALIDATION_CLOSURE_PDF,
        plot_dir=directory(ABCD_DENSITY_VALIDATION_CLOSURE_PLOT_DIR),
    params:
        table=ABCD_DENSITY_TABLE_CSV,
        pdf_name=ABCD_DENSITY_VALIDATION_CLOSURE["output"]["pdf"],
    resources:
        runtime=10 * 60,
        mem_mb=1000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1

        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import matplotlib, yaml; print(matplotlib.__version__, yaml.__version__)' >&2
        command -v pdflatex >&2

        python3 {input.script} \
          --input-table {params.table} \
          --merge-audit {input.merge_audit} \
          --merge-config {input.merge_config} \
          --config {input.config} \
          --output-csv {output.csv} \
          --output-markdown {output.markdown} \
          --output-tex {output.tex} \
          --output-plot-dir {output.plot_dir}

        closure_build_dir=$(mktemp -d /tmp/abcd_validation_closure.XXXXXX)
        trap 'rm -rf "$closure_build_dir"' EXIT
        pdflatex -interaction=nonstopmode -halt-on-error \
          -output-directory "$closure_build_dir" {output.tex}
        pdflatex -interaction=nonstopmode -halt-on-error \
          -output-directory "$closure_build_dir" {output.tex}
        cp "$closure_build_dir/{params.pdf_name}" {output.pdf}
        """


rule region_v1_abcd_density_nominal_closure:
    input:
        merge_audit=ABCD_DENSITY_MERGE_CSV,
        merge_config=ABCD_DENSITY_MERGE_CONFIG,
        config=ABCD_DENSITY_NOMINAL_CLOSURE_CONFIG,
        script=(
            f"{Path(workflow.basedir).parent}/evaluation_region_partitions/"
            "calculate_abcd_density_validation_closure.py"
        ),
    output:
        csv=ABCD_DENSITY_NOMINAL_CLOSURE_CSV,
        markdown=ABCD_DENSITY_NOMINAL_CLOSURE_MARKDOWN,
        tex=ABCD_DENSITY_NOMINAL_CLOSURE_TEX,
        pdf=ABCD_DENSITY_NOMINAL_CLOSURE_PDF,
        plot_dir=directory(ABCD_DENSITY_NOMINAL_CLOSURE_PLOT_DIR),
    params:
        table=ABCD_DENSITY_TABLE_CSV,
        pdf_name=ABCD_DENSITY_NOMINAL_CLOSURE["output"]["pdf"],
    resources:
        runtime=10 * 60,
        mem_mb=1000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1

        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import matplotlib, yaml; print(matplotlib.__version__, yaml.__version__)' >&2
        command -v pdflatex >&2

        python3 {input.script} \
          --input-table {params.table} \
          --merge-audit {input.merge_audit} \
          --merge-config {input.merge_config} \
          --config {input.config} \
          --output-csv {output.csv} \
          --output-markdown {output.markdown} \
          --output-tex {output.tex} \
          --output-plot-dir {output.plot_dir}

        closure_build_dir=$(mktemp -d /tmp/abcd_nominal_closure.XXXXXX)
        trap 'rm -rf "$closure_build_dir"' EXIT
        pdflatex -interaction=nonstopmode -halt-on-error \
          -output-directory "$closure_build_dir" {output.tex}
        pdflatex -interaction=nonstopmode -halt-on-error \
          -output-directory "$closure_build_dir" {output.tex}
        cp "$closure_build_dir/{params.pdf_name}" {output.pdf}
        """


rule region_v1_abcd_density_closures:
    input:
        validation_csv=ABCD_DENSITY_VALIDATION_CLOSURE_CSV,
        validation_pdf=ABCD_DENSITY_VALIDATION_CLOSURE_PDF,
        nominal_csv=ABCD_DENSITY_NOMINAL_CLOSURE_CSV,
        nominal_pdf=ABCD_DENSITY_NOMINAL_CLOSURE_PDF,


rule region_v1_abcd_density_signal_region_fit:
    input:
        closure_csv=ABCD_DENSITY_NOMINAL_CLOSURE_CSV,
        config=ABCD_DENSITY_SIGNAL_REGION_FIT_CONFIG,
        script=(
            f"{Path(workflow.basedir).parent}/evaluation_region_partitions/"
            "fit_abcd_density_signal_region.py"
        ),
    output:
        inputs_csv=ABCD_DENSITY_SIGNAL_REGION_FIT_INPUTS_CSV,
        workspace_json=ABCD_DENSITY_SIGNAL_REGION_FIT_WORKSPACE_JSON,
        fit_json=ABCD_DENSITY_SIGNAL_REGION_FIT_RESULTS_JSON,
        mu_scan_csv=ABCD_DENSITY_SIGNAL_REGION_FIT_MU_SCAN_CSV,
        alpha_scan_csv=ABCD_DENSITY_SIGNAL_REGION_FIT_ALPHA_SCAN_CSV,
        likelihood_2d_csv=ABCD_DENSITY_SIGNAL_REGION_FIT_2D_CSV,
        report=ABCD_DENSITY_SIGNAL_REGION_FIT_REPORT,
        report_tex=ABCD_DENSITY_SIGNAL_REGION_FIT_REPORT_TEX,
        report_pdf=ABCD_DENSITY_SIGNAL_REGION_FIT_REPORT_PDF,
        plot_dir=directory(ABCD_DENSITY_SIGNAL_REGION_FIT_PLOT_DIR),
    params:
        pdf_name=ABCD_DENSITY_SIGNAL_REGION_FIT["output"]["report_pdf"],
    resources:
        runtime=10 * 60,
        mem_mb=1000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1

        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import importlib.metadata as m; import cabinetry, matplotlib, numpy, pyhf, yaml; print("cabinetry", m.version("cabinetry"), "pyhf", pyhf.__version__, "matplotlib", matplotlib.__version__, "numpy", numpy.__version__, "yaml", yaml.__version__)' >&2
        command -v pdflatex >&2

        python3 {input.script} \
          --closure-csv {input.closure_csv} \
          --config {input.config} \
          --output-inputs-csv {output.inputs_csv} \
          --output-workspace-json {output.workspace_json} \
          --output-fit-json {output.fit_json} \
          --output-mu-scan-csv {output.mu_scan_csv} \
          --output-alpha-scan-csv {output.alpha_scan_csv} \
          --output-likelihood-2d-csv {output.likelihood_2d_csv} \
          --output-report-markdown {output.report} \
          --output-report-tex {output.report_tex} \
          --output-plot-dir {output.plot_dir}

        report_build_dir=$(mktemp -d /tmp/abcd_signal_fit_report.XXXXXX)
        trap 'rm -rf "$report_build_dir"' EXIT
        pdflatex -interaction=nonstopmode -halt-on-error \
          -output-directory "$report_build_dir" {output.report_tex}
        pdflatex -interaction=nonstopmode -halt-on-error \
          -output-directory "$report_build_dir" {output.report_tex}
        cp "$report_build_dir/{params.pdf_name}" {output.report_pdf}
        """


rule region_v1_abcd_density_high_density_reports:
    input:
        table=ABCD_DENSITY_TABLE_CSV,
        config=ABCD_DENSITY_HIGH_DENSITY_CONFIG,
        script=(
            f"{Path(workflow.basedir).parent}/evaluation_region_partitions/"
            "fit_abcd_high_density_reports.py"
        ),
    output:
        cut_inputs=ABCD_HIGH_CUT_INPUTS,
        cut_workspace=ABCD_HIGH_CUT_WORKSPACE,
        cut_results=ABCD_HIGH_CUT_RESULTS,
        cut_scan=ABCD_HIGH_CUT_SCAN,
        cut_markdown=ABCD_HIGH_CUT_MARKDOWN,
        cut_tex=ABCD_HIGH_CUT_TEX,
        cut_pdf=ABCD_HIGH_CUT_PDF,
        cut_plots=directory(ABCD_HIGH_CUT_PLOTS),
        shape_inputs=ABCD_HIGH_SHAPE_INPUTS,
        shape_workspace=ABCD_HIGH_SHAPE_WORKSPACE,
        shape_results=ABCD_HIGH_SHAPE_RESULTS,
        shape_scan=ABCD_HIGH_SHAPE_SCAN,
        shape_markdown=ABCD_HIGH_SHAPE_MARKDOWN,
        shape_tex=ABCD_HIGH_SHAPE_TEX,
        shape_pdf=ABCD_HIGH_SHAPE_PDF,
        shape_plots=directory(ABCD_HIGH_SHAPE_PLOTS),
    params:
        output_root=ABCD_DENSITY_HIGH_DENSITY_OUTPUT_DIR,
        cut_pdf_name=ABCD_DENSITY_HIGH_DENSITY["output"]["report_pdf"],
        shape_pdf_name=ABCD_DENSITY_HIGH_DENSITY["output"]["report_pdf"],
    resources:
        runtime=15 * 60,
        mem_mb=1500,
        disk_mb=1500,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1

        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import importlib.metadata as m; import cabinetry, matplotlib, numpy, pyhf, yaml; print("cabinetry", m.version("cabinetry"), "pyhf", pyhf.__version__, "matplotlib", matplotlib.__version__, "numpy", numpy.__version__, "yaml", yaml.__version__)' >&2
        command -v pdflatex >&2

        python3 {input.script} \
          --input-table {input.table} \
          --config {input.config} \
          --output-root {params.output_root}

        report_build_dir=$(mktemp -d /tmp/abcd_high_density_reports.XXXXXX)
        trap 'rm -rf "$report_build_dir"' EXIT
        mkdir -p "$report_build_dir/cutcount" "$report_build_dir/shape"
        pdflatex -interaction=nonstopmode -halt-on-error \
          -output-directory "$report_build_dir/cutcount" {output.cut_tex}
        pdflatex -interaction=nonstopmode -halt-on-error \
          -output-directory "$report_build_dir/cutcount" {output.cut_tex}
        cp "$report_build_dir/cutcount/{params.cut_pdf_name}" {output.cut_pdf}
        pdflatex -interaction=nonstopmode -halt-on-error \
          -output-directory "$report_build_dir/shape" {output.shape_tex}
        pdflatex -interaction=nonstopmode -halt-on-error \
          -output-directory "$report_build_dir/shape" {output.shape_tex}
        cp "$report_build_dir/shape/{params.shape_pdf_name}" {output.shape_pdf}
        """


rule region_v1_abcd_density_blinded_inputs:
    input:
        config=ABCD_DENSITY_BLINDED_CONFIG,
        data_config=ABCD_DENSITY_CONFIG,
        mc_config=ABCD_DENSITY_MC_CONFIG,
        script=(
            f"{Path(workflow.basedir).parent}/evaluation_region_partitions/"
            "make_abcd_blinded_inputs.py"
        ),
        data_hists=ABCD_BLINDED_DATA_HISTS,
        data_summaries=ABCD_BLINDED_DATA_SUMMARIES,
        mc_hists=ABCD_BLINDED_MC_HISTS,
        mc_summaries=ABCD_BLINDED_MC_SUMMARIES,
    output:
        sanitized=ABCD_BLINDED_SANITIZED,
    params:
        product_directory=ABCD_DENSITY_OUTPUT_DIR,
    resources:
        runtime=10 * 60,
        mem_mb=1500,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1

        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2

        python3 {input.script} \
          --config {input.config} \
          --data-config {input.data_config} \
          --mc-config {input.mc_config} \
          --product-directory {params.product_directory} \
          --declared-data-inputs {input.data_hists} {input.data_summaries} \
          --output {output.sanitized}
        """


rule region_v1_abcd_density_blinded_expected_report:
    input:
        sanitized=ABCD_BLINDED_SANITIZED,
        config=ABCD_DENSITY_BLINDED_CONFIG,
        script=(
            f"{Path(workflow.basedir).parent}/evaluation_region_partitions/"
            "run_abcd_blinded_expected.py"
        ),
    output:
        manifest=ABCD_BLINDED_MANIFEST,
        model_inputs=ABCD_BLINDED_EXPECTED_OUTPUTS["model_inputs"],
        control_workspace=ABCD_BLINDED_EXPECTED_OUTPUTS["control_workspace"],
        background_workspace=ABCD_BLINDED_EXPECTED_OUTPUTS["expected_background_workspace"],
        signal_workspace=ABCD_BLINDED_EXPECTED_OUTPUTS["expected_signal_workspace"],
        control_fit=ABCD_BLINDED_EXPECTED_OUTPUTS["control_fit"],
        control_covariance=ABCD_BLINDED_EXPECTED_OUTPUTS["control_covariance"],
        results=ABCD_BLINDED_EXPECTED_OUTPUTS["expected_results"],
        scan=ABCD_BLINDED_EXPECTED_OUTPUTS["mu_scan"],
        markdown=ABCD_BLINDED_EXPECTED_OUTPUTS["report_markdown"],
        tex=ABCD_BLINDED_EXPECTED_OUTPUTS["report_tex"],
        pdf=ABCD_BLINDED_EXPECTED_OUTPUTS["report_pdf"],
        plots=directory(ABCD_BLINDED_EXPECTED_PLOTS),
    params:
        output_root=ABCD_DENSITY_BLINDED_OUTPUT_DIR,
        pdf_name=ABCD_DENSITY_BLINDED["output"]["report_pdf"],
    resources:
        runtime=20 * 60,
        mem_mb=2000,
        disk_mb=2000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1

        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import cabinetry, matplotlib, numpy, pyhf, scipy, yaml; print(pyhf.__version__, matplotlib.__version__, numpy.__version__)' >&2
        command -v pdflatex >&2

        python3 {input.script} \
          --input {input.sanitized} \
          --config {input.config} \
          --output-root {params.output_root}

        report_build_dir=$(mktemp -d /tmp/abcd_blinded_expected.XXXXXX)
        trap 'rm -rf "$report_build_dir"' EXIT
        pdflatex -interaction=nonstopmode -halt-on-error \
          -output-directory "$report_build_dir" {output.tex}
        pdflatex -interaction=nonstopmode -halt-on-error \
          -output-directory "$report_build_dir" {output.tex}
        cp "$report_build_dir/{params.pdf_name}" {output.pdf}
        """


rule region_v1_abcd_density_unblind_one_bin:
    input:
        sanitized=ABCD_BLINDED_SANITIZED,
        manifest=ABCD_BLINDED_MANIFEST,
        config=ABCD_DENSITY_BLINDED_CONFIG,
        authorization=ABCD_DENSITY_UNBLIND_AUTHORIZATION,
        data_config=ABCD_DENSITY_CONFIG,
        signal_data_root=ABCD_SIGNAL_DATA_HIST,
        signal_data_summary=ABCD_SIGNAL_DATA_SUMMARY,
        script=(
            f"{Path(workflow.basedir).parent}/evaluation_region_partitions/"
            "run_abcd_unblinded.py"
        ),
    output:
        workspace=ABCD_UNBLINDED_OUTPUTS["observed_workspace"],
        results=ABCD_UNBLINDED_OUTPUTS["observed_results"],
        scan=ABCD_UNBLINDED_OUTPUTS["observed_mu_scan"],
        toys=ABCD_UNBLINDED_OUTPUTS["toy_q0"],
        markdown=ABCD_UNBLINDED_OUTPUTS["final_report_markdown"],
        tex=ABCD_UNBLINDED_OUTPUTS["final_report_tex"],
        pdf=ABCD_UNBLINDED_OUTPUTS["final_report_pdf"],
        plots=directory(ABCD_UNBLINDED_PLOTS),
    params:
        output_root=ABCD_DENSITY_BLINDED_OUTPUT_DIR,
        pdf_name=ABCD_DENSITY_BLINDED["output"]["final_report_pdf"],
    resources:
        runtime=8 * 60 * 60,
        mem_mb=2500,
        disk_mb=2500,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1

        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, cabinetry, matplotlib, numpy, pyhf, scipy, yaml; print(ROOT.__version__, pyhf.__version__)' >&2
        command -v pdflatex >&2

        python3 {input.script} \
          --input {input.sanitized} \
          --config {input.config} \
          --freeze-manifest {input.manifest} \
          --authorization {input.authorization} \
          --signal-data-root {input.signal_data_root} \
          --signal-data-summary {input.signal_data_summary} \
          --data-config {input.data_config} \
          --output-root {params.output_root}

        report_build_dir=$(mktemp -d /tmp/abcd_unblinded.XXXXXX)
        trap 'rm -rf "$report_build_dir"' EXIT
        pdflatex -interaction=nonstopmode -halt-on-error \
          -output-directory "$report_build_dir" {output.tex}
        pdflatex -interaction=nonstopmode -halt-on-error \
          -output-directory "$report_build_dir" {output.tex}
        cp "$report_build_dir/{params.pdf_name}" {output.pdf}
        """


rule build_region_feature_partition:
    input:
        metadata=REGION_PARTITION_SOURCE_METADATA,
        region_config=REGION_CONFIG,
        partition_config=PARTITION_CONFIG,
        script=str(Path(workflow.basedir).resolve().parent / "convertData/build_region_partitions.py"),
        integrity=str(Path(workflow.basedir).resolve().parent / "convertData/feature_integrity.py"),
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
        script=str(Path(workflow.basedir).resolve().parent / "convertData/build_region_partitions.py"),
        integrity=str(Path(workflow.basedir).resolve().parent / "convertData/feature_integrity.py"),
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
            --feature-reference-metadata {input.feature_metadata_row} \
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


# Isolated Region V1 DS--SciFi passing-muon-like rejection diagnostic.
# This target is intentionally not part of a broad/default aggregate.
ABCD_DS_SCIFI_CONFIG = (
    f"{workflow.basedir}/metadata/configs/region_evaluation/"
    "abcd_ds_scifi_muon_rejection_v1.yaml"
)
with open(ABCD_DS_SCIFI_CONFIG, "r", encoding="utf-8") as handle:
    ABCD_DS_SCIFI = yaml.safe_load(handle)

ABCD_DS_SCIFI_OUTPUT_DIR = ABCD_DS_SCIFI["output"]["directory"]
if not os.path.isabs(ABCD_DS_SCIFI_OUTPUT_DIR):
    ABCD_DS_SCIFI_OUTPUT_DIR = f"{PERSONAL_WORK_SPACE}/{ABCD_DS_SCIFI_OUTPUT_DIR}"
ABCD_DS_SCIFI_DATA_SHARD_COUNT = int(ABCD_DS_SCIFI["input"]["data_shards"])
ABCD_DS_SCIFI_SHARDS = list(range(ABCD_DS_SCIFI_DATA_SHARD_COUNT))
ABCD_DS_SCIFI_REGIONS = list(ABCD_DS_SCIFI["regions"])
ABCD_DS_SCIFI_SAMPLES = list(ABCD_DS_SCIFI["mc_samples"])
ABCD_DS_SCIFI_DATA_SHARD_ROOT = f"{ABCD_DS_SCIFI_OUTPUT_DIR}/data_shards/shard_{{shard}}.root"
ABCD_DS_SCIFI_DATA_SHARD_CSV = f"{ABCD_DS_SCIFI_OUTPUT_DIR}/data_shards/shard_{{shard}}.csv"
ABCD_DS_SCIFI_DATA_ROOTS = expand(ABCD_DS_SCIFI_DATA_SHARD_ROOT, shard=ABCD_DS_SCIFI_SHARDS)
ABCD_DS_SCIFI_DATA_CSVS = expand(ABCD_DS_SCIFI_DATA_SHARD_CSV, shard=ABCD_DS_SCIFI_SHARDS)
ABCD_DS_SCIFI_DATA_ROOT = f"{ABCD_DS_SCIFI_OUTPUT_DIR}/{ABCD_DS_SCIFI['output']['data_root']}"
ABCD_DS_SCIFI_DATA_SUMMARY = f"{ABCD_DS_SCIFI_OUTPUT_DIR}/{ABCD_DS_SCIFI['output']['data_summary']}"
ABCD_DS_SCIFI_MC_ROOT = f"{ABCD_DS_SCIFI_OUTPUT_DIR}/mc/{{region}}__{{sample}}.root"
ABCD_DS_SCIFI_MC_CSV = f"{ABCD_DS_SCIFI_OUTPUT_DIR}/mc/{{region}}__{{sample}}.csv"
ABCD_DS_SCIFI_MC_ROOTS = expand(ABCD_DS_SCIFI_MC_ROOT, region=ABCD_DS_SCIFI_REGIONS, sample=ABCD_DS_SCIFI_SAMPLES)
ABCD_DS_SCIFI_MC_CSVS = expand(ABCD_DS_SCIFI_MC_CSV, region=ABCD_DS_SCIFI_REGIONS, sample=ABCD_DS_SCIFI_SAMPLES)
ABCD_DS_SCIFI_COMBINED_SUMMARY = f"{ABCD_DS_SCIFI_OUTPUT_DIR}/{ABCD_DS_SCIFI['output']['combined_summary']}"
ABCD_DS_SCIFI_PLOT_ROOT = f"{ABCD_DS_SCIFI_OUTPUT_DIR}/{ABCD_DS_SCIFI['output']['plot_root']}"
ABCD_DS_SCIFI_REPORT_PDF = f"{ABCD_DS_SCIFI_OUTPUT_DIR}/{ABCD_DS_SCIFI['output']['report_pdf']}"
ABCD_DS_SCIFI_REPORT_MD = f"{ABCD_DS_SCIFI_OUTPUT_DIR}/{ABCD_DS_SCIFI['output']['report_markdown']}"


rule region_v1_abcd_ds_scifi_muon_rejection_v1:
    input:
        ABCD_DS_SCIFI_COMBINED_SUMMARY,
        ABCD_DS_SCIFI_PLOT_ROOT,
        ABCD_DS_SCIFI_REPORT_PDF,
        ABCD_DS_SCIFI_REPORT_MD,


rule region_v1_abcd_ds_scifi_data_shards_v1:
    input:
        ABCD_DS_SCIFI_DATA_ROOTS,
        ABCD_DS_SCIFI_DATA_CSVS,


rule build_region_v1_abcd_ds_scifi_data_shard:
    input:
        config=ABCD_DS_SCIFI_CONFIG,
        metadata=ABCD_DS_SCIFI["input"]["full_data_metadata"],
        region_config=f"{workflow.basedir}/{ABCD_DS_SCIFI['input']['region_config'].removeprefix('snakemake/')}",
        helper=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/abcd_ds_scifi_selection.py",
        script=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/make_abcd_ds_scifi_rejection_data_shard.py",
    output:
        root=ABCD_DS_SCIFI_DATA_SHARD_ROOT,
        summary=ABCD_DS_SCIFI_DATA_SHARD_CSV,
    wildcard_constraints:
        shard=r"\d+"
    threads: 2
    resources:
        runtime=2 * 60 * 60,
        mem_mb=4000,
        disk_mb=2000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2
        python3 {input.script} \
          --metadata {input.metadata} \
          --config {input.config} \
          --region-config {input.region_config} \
          --shard-index {wildcards.shard} \
          --shard-count {ABCD_DS_SCIFI_DATA_SHARD_COUNT} \
          --output-root {output.root} \
          --output-summary {output.summary} \
          --threads {threads}
        """


rule merge_region_v1_abcd_ds_scifi_data:
    input:
        config=ABCD_DS_SCIFI_CONFIG,
        roots=ABCD_DS_SCIFI_DATA_ROOTS,
        summaries=ABCD_DS_SCIFI_DATA_CSVS,
        helper=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/abcd_ds_scifi_selection.py",
        script=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/merge_abcd_ds_scifi_rejection_data.py",
    output:
        root=ABCD_DS_SCIFI_DATA_ROOT,
        summary=ABCD_DS_SCIFI_DATA_SUMMARY,
    params:
        roots=" ".join(f"--input-root {path}" for path in ABCD_DS_SCIFI_DATA_ROOTS),
        summaries=" ".join(f"--input-summary {path}" for path in ABCD_DS_SCIFI_DATA_CSVS),
    resources:
        runtime=30 * 60,
        mem_mb=3000,
        disk_mb=3000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2
        python3 {input.script} --config {input.config} {params.roots} {params.summaries} \
          --output-root {output.root} --output-summary {output.summary}
        """


rule build_region_v1_abcd_ds_scifi_mc:
    input:
        config=ABCD_DS_SCIFI_CONFIG,
        helper=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/abcd_ds_scifi_selection.py",
        script=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/make_abcd_ds_scifi_rejection_mc.py",
    params:
        # The study consumes this existing frozen catalogue as authoritative
        # state and must never enter its producer DAG.
        catalogue=ABCD_DS_SCIFI["input"]["partition_catalogue"],
    output:
        root=ABCD_DS_SCIFI_MC_ROOT,
        summary=ABCD_DS_SCIFI_MC_CSV,
    wildcard_constraints:
        region="|".join(re.escape(value) for value in ABCD_DS_SCIFI_REGIONS),
        sample="|".join(re.escape(value) for value in ABCD_DS_SCIFI_SAMPLES)
    threads: 2
    resources:
        runtime=60 * 60,
        mem_mb=3000,
        disk_mb=1500,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2
        python3 {input.script} --partition-catalogue {params.catalogue} --config {input.config} \
          --region {wildcards.region} --sample {wildcards.sample} \
          --output-root {output.root} --output-summary {output.summary} --threads {threads}
        """


rule region_v1_abcd_ds_scifi_rejection_report_v1:
    input:
        config=ABCD_DS_SCIFI_CONFIG,
        data_root=ABCD_DS_SCIFI_DATA_ROOT,
        data_summary=ABCD_DS_SCIFI_DATA_SUMMARY,
        mc_roots=ABCD_DS_SCIFI_MC_ROOTS,
        mc_summaries=ABCD_DS_SCIFI_MC_CSVS,
        helper=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/abcd_ds_scifi_selection.py",
        script=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/make_abcd_ds_scifi_rejection_report.py",
    output:
        summary=ABCD_DS_SCIFI_COMBINED_SUMMARY,
        root=ABCD_DS_SCIFI_PLOT_ROOT,
        pdf=ABCD_DS_SCIFI_REPORT_PDF,
        markdown=ABCD_DS_SCIFI_REPORT_MD,
    params:
        mc_roots=" ".join(f"--mc-root {path}" for path in ABCD_DS_SCIFI_MC_ROOTS),
        mc_summaries=" ".join(f"--mc-summary {path}" for path in ABCD_DS_SCIFI_MC_CSVS),
    resources:
        runtime=30 * 60,
        mem_mb=3000,
        disk_mb=2000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2
        python3 {input.script} --config {input.config} \
          --data-root {input.data_root} --data-summary {input.data_summary} \
          {params.mc_roots} {params.mc_summaries} \
          --output-summary {output.summary} --output-root {output.root} \
          --output-pdf {output.pdf} --output-markdown {output.markdown}
        """


# Isolated Region V1 exact V7 SciFi--DS loose-projection diagnostic.
# This target consumes frozen Region V1 partitions and is not broad/default.
ABCD_DS_SCIFI_LOOSE_CONFIG = (
    f"{workflow.basedir}/metadata/configs/region_evaluation/"
    "abcd_ds_scifi_loose_projection_v1.yaml"
)
with open(ABCD_DS_SCIFI_LOOSE_CONFIG, "r", encoding="utf-8") as handle:
    ABCD_DS_SCIFI_LOOSE = yaml.safe_load(handle)

ABCD_DS_SCIFI_LOOSE_OUTPUT_DIR = ABCD_DS_SCIFI_LOOSE["output"]["directory"]
if not os.path.isabs(ABCD_DS_SCIFI_LOOSE_OUTPUT_DIR):
    ABCD_DS_SCIFI_LOOSE_OUTPUT_DIR = f"{PERSONAL_WORK_SPACE}/{ABCD_DS_SCIFI_LOOSE_OUTPUT_DIR}"
ABCD_DS_SCIFI_LOOSE_REGIONS = list(ABCD_DS_SCIFI_LOOSE["regions"])
ABCD_DS_SCIFI_LOOSE_SAMPLES = list(ABCD_DS_SCIFI_LOOSE["mc_samples"])
ABCD_DS_SCIFI_LOOSE_DATA_GROUP = ABCD_DS_SCIFI_LOOSE["input"].get("data_particle_group", "real_data")
ABCD_DS_SCIFI_LOOSE_DATA_REGION_ROOT = f"{ABCD_DS_SCIFI_LOOSE_OUTPUT_DIR}/data_regions/{{region}}.root"
ABCD_DS_SCIFI_LOOSE_DATA_REGION_CSV = f"{ABCD_DS_SCIFI_LOOSE_OUTPUT_DIR}/data_regions/{{region}}.csv"
ABCD_DS_SCIFI_LOOSE_DATA_ROOTS = expand(ABCD_DS_SCIFI_LOOSE_DATA_REGION_ROOT, region=ABCD_DS_SCIFI_LOOSE_REGIONS)
ABCD_DS_SCIFI_LOOSE_DATA_CSVS = expand(ABCD_DS_SCIFI_LOOSE_DATA_REGION_CSV, region=ABCD_DS_SCIFI_LOOSE_REGIONS)
ABCD_DS_SCIFI_LOOSE_DATA_ROOT = f"{ABCD_DS_SCIFI_LOOSE_OUTPUT_DIR}/{ABCD_DS_SCIFI_LOOSE['output']['data_root']}"
ABCD_DS_SCIFI_LOOSE_DATA_SUMMARY = f"{ABCD_DS_SCIFI_LOOSE_OUTPUT_DIR}/{ABCD_DS_SCIFI_LOOSE['output']['data_summary']}"
ABCD_DS_SCIFI_LOOSE_MC_ROOT = f"{ABCD_DS_SCIFI_LOOSE_OUTPUT_DIR}/mc/{{region}}__{{sample}}.root"
ABCD_DS_SCIFI_LOOSE_MC_CSV = f"{ABCD_DS_SCIFI_LOOSE_OUTPUT_DIR}/mc/{{region}}__{{sample}}.csv"
ABCD_DS_SCIFI_LOOSE_MC_ROOTS = expand(ABCD_DS_SCIFI_LOOSE_MC_ROOT, region=ABCD_DS_SCIFI_LOOSE_REGIONS, sample=ABCD_DS_SCIFI_LOOSE_SAMPLES)
ABCD_DS_SCIFI_LOOSE_MC_CSVS = expand(ABCD_DS_SCIFI_LOOSE_MC_CSV, region=ABCD_DS_SCIFI_LOOSE_REGIONS, sample=ABCD_DS_SCIFI_LOOSE_SAMPLES)
ABCD_DS_SCIFI_LOOSE_COMBINED_SUMMARY = f"{ABCD_DS_SCIFI_LOOSE_OUTPUT_DIR}/{ABCD_DS_SCIFI_LOOSE['output']['combined_summary']}"
ABCD_DS_SCIFI_LOOSE_PLOT_ROOT = f"{ABCD_DS_SCIFI_LOOSE_OUTPUT_DIR}/{ABCD_DS_SCIFI_LOOSE['output']['plot_root']}"
ABCD_DS_SCIFI_LOOSE_REPORT_PDF = f"{ABCD_DS_SCIFI_LOOSE_OUTPUT_DIR}/{ABCD_DS_SCIFI_LOOSE['output']['report_pdf']}"
ABCD_DS_SCIFI_LOOSE_REPORT_MD = f"{ABCD_DS_SCIFI_LOOSE_OUTPUT_DIR}/{ABCD_DS_SCIFI_LOOSE['output']['report_markdown']}"


rule region_v1_abcd_ds_scifi_loose_projection_v1:
    input:
        ABCD_DS_SCIFI_LOOSE_COMBINED_SUMMARY,
        ABCD_DS_SCIFI_LOOSE_PLOT_ROOT,
        ABCD_DS_SCIFI_LOOSE_REPORT_PDF,
        ABCD_DS_SCIFI_LOOSE_REPORT_MD,


rule region_v1_abcd_ds_scifi_loose_data_regions_v1:
    input:
        ABCD_DS_SCIFI_LOOSE_DATA_ROOTS,
        ABCD_DS_SCIFI_LOOSE_DATA_CSVS,


rule build_region_v1_abcd_ds_scifi_loose_data_region:
    input:
        config=ABCD_DS_SCIFI_LOOSE_CONFIG,
        helper=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/abcd_ds_scifi_selection.py",
        script=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/make_abcd_ds_scifi_rejection_mc.py",
    params:
        catalogue=ABCD_DS_SCIFI_LOOSE["input"]["partition_catalogue"],
        sample=ABCD_DS_SCIFI_LOOSE_DATA_GROUP,
    output:
        root=ABCD_DS_SCIFI_LOOSE_DATA_REGION_ROOT,
        summary=ABCD_DS_SCIFI_LOOSE_DATA_REGION_CSV,
    wildcard_constraints:
        region="|".join(re.escape(value) for value in ABCD_DS_SCIFI_LOOSE_REGIONS)
    threads: 2
    resources:
        runtime=60 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2
        python3 {input.script} --partition-catalogue {params.catalogue} --config {input.config} \
          --region {wildcards.region} --sample {params.sample} \
          --output-root {output.root} --output-summary {output.summary} --threads {threads}
        """


rule merge_region_v1_abcd_ds_scifi_loose_data:
    input:
        config=ABCD_DS_SCIFI_LOOSE_CONFIG,
        roots=ABCD_DS_SCIFI_LOOSE_DATA_ROOTS,
        summaries=ABCD_DS_SCIFI_LOOSE_DATA_CSVS,
        helper=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/abcd_ds_scifi_selection.py",
        script=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/merge_abcd_ds_scifi_rejection_data_partitions.py",
    output:
        root=ABCD_DS_SCIFI_LOOSE_DATA_ROOT,
        summary=ABCD_DS_SCIFI_LOOSE_DATA_SUMMARY,
    params:
        roots=" ".join(f"--input-root {path}" for path in ABCD_DS_SCIFI_LOOSE_DATA_ROOTS),
        summaries=" ".join(f"--input-summary {path}" for path in ABCD_DS_SCIFI_LOOSE_DATA_CSVS),
    resources:
        runtime=15 * 60,
        mem_mb=3000,
        disk_mb=3000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2
        python3 {input.script} --config {input.config} {params.roots} {params.summaries} \
          --output-root {output.root} --output-summary {output.summary}
        """


rule build_region_v1_abcd_ds_scifi_loose_mc:
    input:
        config=ABCD_DS_SCIFI_LOOSE_CONFIG,
        helper=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/abcd_ds_scifi_selection.py",
        script=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/make_abcd_ds_scifi_rejection_mc.py",
    params:
        catalogue=ABCD_DS_SCIFI_LOOSE["input"]["partition_catalogue"],
    output:
        root=ABCD_DS_SCIFI_LOOSE_MC_ROOT,
        summary=ABCD_DS_SCIFI_LOOSE_MC_CSV,
    wildcard_constraints:
        region="|".join(re.escape(value) for value in ABCD_DS_SCIFI_LOOSE_REGIONS),
        sample="|".join(re.escape(value) for value in ABCD_DS_SCIFI_LOOSE_SAMPLES)
    threads: 1
    resources:
        runtime=60 * 60,
        mem_mb=3000,
        disk_mb=1500,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2
        python3 {input.script} --partition-catalogue {params.catalogue} --config {input.config} \
          --region {wildcards.region} --sample {wildcards.sample} \
          --output-root {output.root} --output-summary {output.summary} --threads {threads}
        """


rule region_v1_abcd_ds_scifi_loose_report_v1:
    input:
        config=ABCD_DS_SCIFI_LOOSE_CONFIG,
        data_root=ABCD_DS_SCIFI_LOOSE_DATA_ROOT,
        data_summary=ABCD_DS_SCIFI_LOOSE_DATA_SUMMARY,
        mc_roots=ABCD_DS_SCIFI_LOOSE_MC_ROOTS,
        mc_summaries=ABCD_DS_SCIFI_LOOSE_MC_CSVS,
        helper=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/abcd_ds_scifi_selection.py",
        script=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/make_abcd_ds_scifi_rejection_report.py",
    output:
        summary=ABCD_DS_SCIFI_LOOSE_COMBINED_SUMMARY,
        root=ABCD_DS_SCIFI_LOOSE_PLOT_ROOT,
        pdf=ABCD_DS_SCIFI_LOOSE_REPORT_PDF,
        markdown=ABCD_DS_SCIFI_LOOSE_REPORT_MD,
    params:
        mc_roots=" ".join(f"--mc-root {path}" for path in ABCD_DS_SCIFI_LOOSE_MC_ROOTS),
        mc_summaries=" ".join(f"--mc-summary {path}" for path in ABCD_DS_SCIFI_LOOSE_MC_CSVS),
    resources:
        runtime=30 * 60,
        mem_mb=3000,
        disk_mb=2000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2
        python3 {input.script} --config {input.config} \
          --data-root {input.data_root} --data-summary {input.data_summary} \
          {params.mc_roots} {params.mc_summaries} \
          --output-summary {output.summary} --output-root {output.root} \
          --output-pdf {output.pdf} --output-markdown {output.markdown}
        """


# Isolated Region V1 Veto edge-muon control-sample diagnostic. The selected
# population is the edge tag itself; the complementary population is retained
# in the backend products only for exact closure and comparison.
ABCD_VETO_EDGE_CONFIG = (
    f"{workflow.basedir}/metadata/configs/region_evaluation/"
    "abcd_veto_edge_muon_selection_v1.yaml"
)
with open(ABCD_VETO_EDGE_CONFIG, "r", encoding="utf-8") as handle:
    ABCD_VETO_EDGE = yaml.safe_load(handle)

ABCD_VETO_EDGE_OUTPUT_DIR = ABCD_VETO_EDGE["output"]["directory"]
if not os.path.isabs(ABCD_VETO_EDGE_OUTPUT_DIR):
    ABCD_VETO_EDGE_OUTPUT_DIR = f"{PERSONAL_WORK_SPACE}/{ABCD_VETO_EDGE_OUTPUT_DIR}"
ABCD_VETO_EDGE_REGIONS = list(ABCD_VETO_EDGE["regions"])
ABCD_VETO_EDGE_SAMPLES = list(ABCD_VETO_EDGE["mc_samples"])
ABCD_VETO_EDGE_DATA_GROUP = ABCD_VETO_EDGE["input"].get("data_particle_group", "real_data")
ABCD_VETO_EDGE_DATA_REGION_ROOT = f"{ABCD_VETO_EDGE_OUTPUT_DIR}/data_regions/{{region}}.root"
ABCD_VETO_EDGE_DATA_REGION_CSV = f"{ABCD_VETO_EDGE_OUTPUT_DIR}/data_regions/{{region}}.csv"
ABCD_VETO_EDGE_DATA_REGION_ROOTS = expand(
    ABCD_VETO_EDGE_DATA_REGION_ROOT, region=ABCD_VETO_EDGE_REGIONS
)
ABCD_VETO_EDGE_DATA_REGION_CSVS = expand(
    ABCD_VETO_EDGE_DATA_REGION_CSV, region=ABCD_VETO_EDGE_REGIONS
)
ABCD_VETO_EDGE_DATA_ROOT = f"{ABCD_VETO_EDGE_OUTPUT_DIR}/{ABCD_VETO_EDGE['output']['data_root']}"
ABCD_VETO_EDGE_DATA_SUMMARY = f"{ABCD_VETO_EDGE_OUTPUT_DIR}/{ABCD_VETO_EDGE['output']['data_summary']}"
ABCD_VETO_EDGE_MC_ROOT = f"{ABCD_VETO_EDGE_OUTPUT_DIR}/mc/{{region}}__{{sample}}.root"
ABCD_VETO_EDGE_MC_CSV = f"{ABCD_VETO_EDGE_OUTPUT_DIR}/mc/{{region}}__{{sample}}.csv"
ABCD_VETO_EDGE_MC_ROOTS = expand(
    ABCD_VETO_EDGE_MC_ROOT,
    region=ABCD_VETO_EDGE_REGIONS,
    sample=ABCD_VETO_EDGE_SAMPLES,
)
ABCD_VETO_EDGE_MC_CSVS = expand(
    ABCD_VETO_EDGE_MC_CSV,
    region=ABCD_VETO_EDGE_REGIONS,
    sample=ABCD_VETO_EDGE_SAMPLES,
)
ABCD_VETO_EDGE_COMBINED_SUMMARY = f"{ABCD_VETO_EDGE_OUTPUT_DIR}/{ABCD_VETO_EDGE['output']['combined_summary']}"
ABCD_VETO_EDGE_PLOT_ROOT = f"{ABCD_VETO_EDGE_OUTPUT_DIR}/{ABCD_VETO_EDGE['output']['plot_root']}"
ABCD_VETO_EDGE_REPORT_PDF = f"{ABCD_VETO_EDGE_OUTPUT_DIR}/{ABCD_VETO_EDGE['output']['report_pdf']}"
ABCD_VETO_EDGE_REPORT_MD = f"{ABCD_VETO_EDGE_OUTPUT_DIR}/{ABCD_VETO_EDGE['output']['report_markdown']}"


rule region_v1_abcd_veto_edge_muon_selection_v1:
    input:
        ABCD_VETO_EDGE_COMBINED_SUMMARY,
        ABCD_VETO_EDGE_PLOT_ROOT,
        ABCD_VETO_EDGE_REPORT_PDF,
        ABCD_VETO_EDGE_REPORT_MD,


rule region_v1_abcd_veto_edge_data_regions_v1:
    input:
        ABCD_VETO_EDGE_DATA_REGION_ROOTS,
        ABCD_VETO_EDGE_DATA_REGION_CSVS,


rule build_region_v1_abcd_veto_edge_data_region:
    input:
        config=ABCD_VETO_EDGE_CONFIG,
        helper=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/abcd_ds_scifi_selection.py",
        script=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/make_abcd_ds_scifi_rejection_mc.py",
    params:
        catalogue=ABCD_VETO_EDGE["input"]["partition_catalogue"],
        sample=ABCD_VETO_EDGE_DATA_GROUP,
    output:
        root=ABCD_VETO_EDGE_DATA_REGION_ROOT,
        summary=ABCD_VETO_EDGE_DATA_REGION_CSV,
    wildcard_constraints:
        region="|".join(re.escape(value) for value in ABCD_VETO_EDGE_REGIONS)
    threads: 1
    resources:
        runtime=60 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2
        python3 {input.script} --partition-catalogue {params.catalogue} --config {input.config} \
          --region {wildcards.region} --sample {params.sample} \
          --output-root {output.root} --output-summary {output.summary} --threads {threads}
        """


rule merge_region_v1_abcd_veto_edge_data:
    input:
        config=ABCD_VETO_EDGE_CONFIG,
        roots=ABCD_VETO_EDGE_DATA_REGION_ROOTS,
        summaries=ABCD_VETO_EDGE_DATA_REGION_CSVS,
        helper=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/abcd_ds_scifi_selection.py",
        script=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/merge_abcd_ds_scifi_rejection_data_partitions.py",
    output:
        root=ABCD_VETO_EDGE_DATA_ROOT,
        summary=ABCD_VETO_EDGE_DATA_SUMMARY,
    params:
        roots=" ".join(f"--input-root {path}" for path in ABCD_VETO_EDGE_DATA_REGION_ROOTS),
        summaries=" ".join(f"--input-summary {path}" for path in ABCD_VETO_EDGE_DATA_REGION_CSVS),
    resources:
        runtime=15 * 60,
        mem_mb=3000,
        disk_mb=3000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2
        python3 {input.script} --config {input.config} {params.roots} {params.summaries} \
          --output-root {output.root} --output-summary {output.summary}
        """


rule build_region_v1_abcd_veto_edge_mc:
    input:
        config=ABCD_VETO_EDGE_CONFIG,
        helper=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/abcd_ds_scifi_selection.py",
        script=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/make_abcd_ds_scifi_rejection_mc.py",
    params:
        catalogue=ABCD_VETO_EDGE["input"]["partition_catalogue"],
    output:
        root=ABCD_VETO_EDGE_MC_ROOT,
        summary=ABCD_VETO_EDGE_MC_CSV,
    wildcard_constraints:
        region="|".join(re.escape(value) for value in ABCD_VETO_EDGE_REGIONS),
        sample="|".join(re.escape(value) for value in ABCD_VETO_EDGE_SAMPLES)
    threads: 1
    resources:
        runtime=60 * 60,
        mem_mb=3000,
        disk_mb=1500,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2
        python3 {input.script} --partition-catalogue {params.catalogue} --config {input.config} \
          --region {wildcards.region} --sample {wildcards.sample} \
          --output-root {output.root} --output-summary {output.summary} --threads {threads}
        """


rule region_v1_abcd_veto_edge_report_v1:
    input:
        config=ABCD_VETO_EDGE_CONFIG,
        data_root=ABCD_VETO_EDGE_DATA_ROOT,
        data_summary=ABCD_VETO_EDGE_DATA_SUMMARY,
        mc_roots=ABCD_VETO_EDGE_MC_ROOTS,
        mc_summaries=ABCD_VETO_EDGE_MC_CSVS,
        helper=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/abcd_ds_scifi_selection.py",
        common_report=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/make_abcd_ds_scifi_rejection_report.py",
        script=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/make_abcd_veto_edge_report.py",
    output:
        summary=ABCD_VETO_EDGE_COMBINED_SUMMARY,
        root=ABCD_VETO_EDGE_PLOT_ROOT,
        pdf=ABCD_VETO_EDGE_REPORT_PDF,
        markdown=ABCD_VETO_EDGE_REPORT_MD,
    params:
        mc_roots=" ".join(f"--mc-root {path}" for path in ABCD_VETO_EDGE_MC_ROOTS),
        mc_summaries=" ".join(f"--mc-summary {path}" for path in ABCD_VETO_EDGE_MC_CSVS),
    resources:
        runtime=30 * 60,
        mem_mb=3000,
        disk_mb=2000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2
        python3 {input.script} --config {input.config} \
          --data-root {input.data_root} --data-summary {input.data_summary} \
          {params.mc_roots} {params.mc_summaries} \
          --output-summary {output.summary} --output-root {output.root} \
          --output-pdf {output.pdf} --output-markdown {output.markdown}
        """


# Isolated Region V1 central-muon rejection. A central muon requires both
# central y information and a valid central x position; the complement is the
# kept analysis population.
ABCD_VETO_CENTRAL_CONFIG = (
    f"{workflow.basedir}/metadata/configs/region_evaluation/"
    "abcd_veto_central_muon_rejection_v1.yaml"
)
with open(ABCD_VETO_CENTRAL_CONFIG, "r", encoding="utf-8") as handle:
    ABCD_VETO_CENTRAL = yaml.safe_load(handle)

ABCD_VETO_CENTRAL_OUTPUT_DIR = ABCD_VETO_CENTRAL["output"]["directory"]
if not os.path.isabs(ABCD_VETO_CENTRAL_OUTPUT_DIR):
    ABCD_VETO_CENTRAL_OUTPUT_DIR = f"{PERSONAL_WORK_SPACE}/{ABCD_VETO_CENTRAL_OUTPUT_DIR}"
ABCD_VETO_CENTRAL_REGIONS = list(ABCD_VETO_CENTRAL["regions"])
ABCD_VETO_CENTRAL_SAMPLES = list(ABCD_VETO_CENTRAL["mc_samples"])
ABCD_VETO_CENTRAL_DATA_GROUP = ABCD_VETO_CENTRAL["input"].get("data_particle_group", "real_data")
ABCD_VETO_CENTRAL_DATA_REGION_ROOT = f"{ABCD_VETO_CENTRAL_OUTPUT_DIR}/data_regions/{{region}}.root"
ABCD_VETO_CENTRAL_DATA_REGION_CSV = f"{ABCD_VETO_CENTRAL_OUTPUT_DIR}/data_regions/{{region}}.csv"
ABCD_VETO_CENTRAL_DATA_REGION_ROOTS = expand(
    ABCD_VETO_CENTRAL_DATA_REGION_ROOT, region=ABCD_VETO_CENTRAL_REGIONS
)
ABCD_VETO_CENTRAL_DATA_REGION_CSVS = expand(
    ABCD_VETO_CENTRAL_DATA_REGION_CSV, region=ABCD_VETO_CENTRAL_REGIONS
)
ABCD_VETO_CENTRAL_DATA_ROOT = f"{ABCD_VETO_CENTRAL_OUTPUT_DIR}/{ABCD_VETO_CENTRAL['output']['data_root']}"
ABCD_VETO_CENTRAL_DATA_SUMMARY = f"{ABCD_VETO_CENTRAL_OUTPUT_DIR}/{ABCD_VETO_CENTRAL['output']['data_summary']}"
ABCD_VETO_CENTRAL_MC_ROOT = f"{ABCD_VETO_CENTRAL_OUTPUT_DIR}/mc/{{region}}__{{sample}}.root"
ABCD_VETO_CENTRAL_MC_CSV = f"{ABCD_VETO_CENTRAL_OUTPUT_DIR}/mc/{{region}}__{{sample}}.csv"
ABCD_VETO_CENTRAL_MC_ROOTS = expand(
    ABCD_VETO_CENTRAL_MC_ROOT,
    region=ABCD_VETO_CENTRAL_REGIONS,
    sample=ABCD_VETO_CENTRAL_SAMPLES,
)
ABCD_VETO_CENTRAL_MC_CSVS = expand(
    ABCD_VETO_CENTRAL_MC_CSV,
    region=ABCD_VETO_CENTRAL_REGIONS,
    sample=ABCD_VETO_CENTRAL_SAMPLES,
)
ABCD_VETO_CENTRAL_COMBINED_SUMMARY = f"{ABCD_VETO_CENTRAL_OUTPUT_DIR}/{ABCD_VETO_CENTRAL['output']['combined_summary']}"
ABCD_VETO_CENTRAL_PLOT_ROOT = f"{ABCD_VETO_CENTRAL_OUTPUT_DIR}/{ABCD_VETO_CENTRAL['output']['plot_root']}"
ABCD_VETO_CENTRAL_REPORT_PDF = f"{ABCD_VETO_CENTRAL_OUTPUT_DIR}/{ABCD_VETO_CENTRAL['output']['report_pdf']}"
ABCD_VETO_CENTRAL_REPORT_MD = f"{ABCD_VETO_CENTRAL_OUTPUT_DIR}/{ABCD_VETO_CENTRAL['output']['report_markdown']}"


rule region_v1_abcd_veto_central_muon_rejection_v1:
    input:
        ABCD_VETO_CENTRAL_COMBINED_SUMMARY,
        ABCD_VETO_CENTRAL_PLOT_ROOT,
        ABCD_VETO_CENTRAL_REPORT_PDF,
        ABCD_VETO_CENTRAL_REPORT_MD,


rule build_region_v1_abcd_veto_central_data_region:
    input:
        config=ABCD_VETO_CENTRAL_CONFIG,
        helper=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/abcd_ds_scifi_selection.py",
        script=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/make_abcd_ds_scifi_rejection_mc.py",
    params:
        catalogue=ABCD_VETO_CENTRAL["input"]["partition_catalogue"],
        sample=ABCD_VETO_CENTRAL_DATA_GROUP,
    output:
        root=ABCD_VETO_CENTRAL_DATA_REGION_ROOT,
        summary=ABCD_VETO_CENTRAL_DATA_REGION_CSV,
    wildcard_constraints:
        region="|".join(re.escape(value) for value in ABCD_VETO_CENTRAL_REGIONS)
    threads: 1
    resources:
        runtime=60 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2
        python3 {input.script} --partition-catalogue {params.catalogue} --config {input.config} \
          --region {wildcards.region} --sample {params.sample} \
          --output-root {output.root} --output-summary {output.summary} --threads {threads}
        """


rule merge_region_v1_abcd_veto_central_data:
    input:
        config=ABCD_VETO_CENTRAL_CONFIG,
        roots=ABCD_VETO_CENTRAL_DATA_REGION_ROOTS,
        summaries=ABCD_VETO_CENTRAL_DATA_REGION_CSVS,
        helper=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/abcd_ds_scifi_selection.py",
        script=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/merge_abcd_ds_scifi_rejection_data_partitions.py",
    output:
        root=ABCD_VETO_CENTRAL_DATA_ROOT,
        summary=ABCD_VETO_CENTRAL_DATA_SUMMARY,
    params:
        roots=" ".join(f"--input-root {path}" for path in ABCD_VETO_CENTRAL_DATA_REGION_ROOTS),
        summaries=" ".join(f"--input-summary {path}" for path in ABCD_VETO_CENTRAL_DATA_REGION_CSVS),
    resources:
        runtime=15 * 60,
        mem_mb=3000,
        disk_mb=3000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2
        python3 {input.script} --config {input.config} {params.roots} {params.summaries} \
          --output-root {output.root} --output-summary {output.summary}
        """


rule build_region_v1_abcd_veto_central_mc:
    input:
        config=ABCD_VETO_CENTRAL_CONFIG,
        helper=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/abcd_ds_scifi_selection.py",
        script=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/make_abcd_ds_scifi_rejection_mc.py",
    params:
        catalogue=ABCD_VETO_CENTRAL["input"]["partition_catalogue"],
    output:
        root=ABCD_VETO_CENTRAL_MC_ROOT,
        summary=ABCD_VETO_CENTRAL_MC_CSV,
    wildcard_constraints:
        region="|".join(re.escape(value) for value in ABCD_VETO_CENTRAL_REGIONS),
        sample="|".join(re.escape(value) for value in ABCD_VETO_CENTRAL_SAMPLES)
    threads: 1
    resources:
        runtime=60 * 60,
        mem_mb=3000,
        disk_mb=1500,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2
        python3 {input.script} --partition-catalogue {params.catalogue} --config {input.config} \
          --region {wildcards.region} --sample {wildcards.sample} \
          --output-root {output.root} --output-summary {output.summary} --threads {threads}
        """


rule region_v1_abcd_veto_central_report_v1:
    input:
        config=ABCD_VETO_CENTRAL_CONFIG,
        data_root=ABCD_VETO_CENTRAL_DATA_ROOT,
        data_summary=ABCD_VETO_CENTRAL_DATA_SUMMARY,
        mc_roots=ABCD_VETO_CENTRAL_MC_ROOTS,
        mc_summaries=ABCD_VETO_CENTRAL_MC_CSVS,
        helper=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/abcd_ds_scifi_selection.py",
        common_report=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/make_abcd_ds_scifi_rejection_report.py",
        script=f"{Path(workflow.basedir).parent}/evaluation_region_partitions/make_abcd_veto_central_report.py",
    output:
        summary=ABCD_VETO_CENTRAL_COMBINED_SUMMARY,
        root=ABCD_VETO_CENTRAL_PLOT_ROOT,
        pdf=ABCD_VETO_CENTRAL_REPORT_PDF,
        markdown=ABCD_VETO_CENTRAL_REPORT_MD,
    params:
        mc_roots=" ".join(f"--mc-root {path}" for path in ABCD_VETO_CENTRAL_MC_ROOTS),
        mc_summaries=" ".join(f"--mc-summary {path}" for path in ABCD_VETO_CENTRAL_MC_CSVS),
    resources:
        runtime=30 * 60,
        mem_mb=3000,
        disk_mb=2000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        hostname >&2
        pwd >&2
        command -v python3 >&2
        python3 --version >&2
        python3 -c 'import ROOT, yaml; print(ROOT.__version__, yaml.__version__)' >&2
        python3 {input.script} --config {input.config} \
          --data-root {input.data_root} --data-summary {input.data_summary} \
          {params.mc_roots} {params.mc_summaries} \
          --output-summary {output.summary} --output-root {output.root} \
          --output-pdf {output.pdf} --output-markdown {output.markdown}
        """
