REGION_EVAL_HIST_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/region_evaluation/"
    "hist_features_v1.yaml"
)
REGION_EVAL_OPTIONS_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/region_evaluation/"
    "evaluation_options_v1.yaml"
)
REGION_EVAL_CUTFLOW_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/region_evaluation/"
    "cutflow_efficiency_v1.yaml"
)
REGION_EVAL_CUTFLOW_QDC_TOTAL_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/region_evaluation/"
    "cutflow_efficiency_qdc_total_v1.yaml"
)
REGION_EVAL_EVENT_DISPLAY_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/region_evaluation/"
    "event_display_v1.yaml"
)

with open(REGION_EVAL_HIST_CONFIG, "r", encoding="utf-8") as handle:
    REGION_EVAL_HIST_FEATURE_CONFIG = yaml.safe_load(handle)

with open(REGION_EVAL_OPTIONS_CONFIG, "r", encoding="utf-8") as handle:
    REGION_EVAL_OPTIONS = yaml.safe_load(handle)

with open(REGION_EVAL_CUTFLOW_CONFIG, "r", encoding="utf-8") as handle:
    REGION_EVAL_CUTFLOW_CONFIG_DATA = yaml.safe_load(handle)

with open(REGION_EVAL_CUTFLOW_QDC_TOTAL_CONFIG, "r", encoding="utf-8") as handle:
    REGION_EVAL_CUTFLOW_QDC_TOTAL_CONFIG_DATA = yaml.safe_load(handle)

with open(REGION_EVAL_EVENT_DISPLAY_CONFIG, "r", encoding="utf-8") as handle:
    REGION_EVAL_EVENT_DISPLAY_CONFIG_DATA = yaml.safe_load(handle)

REGION_EVAL_VERSION = REGION_EVAL_OPTIONS["version"]
REGION_EVAL_HIST_VERSION = REGION_EVAL_HIST_FEATURE_CONFIG["version"]
REGION_EVAL_CUTFLOW_VERSION = REGION_EVAL_CUTFLOW_CONFIG_DATA["version"]
REGION_EVAL_CUTFLOW_QDC_TOTAL_VERSION = REGION_EVAL_CUTFLOW_QDC_TOTAL_CONFIG_DATA["version"]
REGION_EVAL_METADATA_CSV = (
    f"{PERSONAL_WORK_SPACE}/"
    f"{REGION_EVAL_OPTIONS['metadata']['partition_csv']}"
)


def _region_eval_output_path(config_value):
    if os.path.isabs(config_value) or str(config_value).startswith("root://"):
        return str(config_value)
    return f"{PERSONAL_WORK_SPACE}/{config_value}"


REGION_EVAL_OUTPUT_ROOT = _region_eval_output_path(REGION_EVAL_OPTIONS["outputs"]["root_dir"])
REGION_EVAL_HIST_OUTPUT_DIR = _region_eval_output_path(REGION_EVAL_OPTIONS["outputs"]["hist_dir"])
REGION_EVAL_PLOT_OUTPUT_DIR = _region_eval_output_path(REGION_EVAL_OPTIONS["outputs"]["plot_dir"])


def _region_eval_study_output_dir(study_name):
    return (
        f"{REGION_EVAL_OUTPUT_ROOT}/"
        f"{REGION_PARTITION_CONFIG['partition_version']}_{REGION_VERSION}/"
        f"{REGION_EVAL_HIST_VERSION}__{REGION_EVAL_VERSION}/"
        f"{study_name}"
    )


def _region_eval_cutflow_output_dir(cutflow_version=REGION_EVAL_CUTFLOW_VERSION):
    return (
        f"{REGION_EVAL_OUTPUT_ROOT}/"
        f"{REGION_PARTITION_CONFIG['partition_version']}_{REGION_VERSION}/"
        f"{cutflow_version}__{REGION_EVAL_VERSION}/"
        "cutflow_efficiency"
    )


REGION_EVAL_FIDUCIAL_SUMMARY_CSV = (
    f"{_region_eval_study_output_dir('fiducial_efficiency')}/summary.csv"
)
REGION_EVAL_FIDUCIAL_OVERLAP_CSV = (
    f"{_region_eval_study_output_dir('fiducial_efficiency')}/overlap.csv"
)
REGION_EVAL_EVENTBUILDER_VETO_TIMING_CSV = (
    f"{_region_eval_study_output_dir('eventbuilder_veto_timing')}/summary.csv"
)
REGION_EVAL_CUTFLOW_EFFICIENCY_CSV = (
    f"{_region_eval_cutflow_output_dir()}/summary.csv"
)
REGION_EVAL_CUTFLOW_EFFICIENCY_BY_STUDY_CSV = (
    f"{_region_eval_cutflow_output_dir()}/summary_by_study.csv"
)
REGION_EVAL_CUTFLOW_REGIONAL_CSV = (
    f"{_region_eval_cutflow_output_dir()}/regional_cut_efficiency.csv"
)
REGION_EVAL_CUTFLOW_OVERALL_CSV = (
    f"{_region_eval_cutflow_output_dir()}/overall_cut_efficiency.csv"
)
REGION_EVAL_CUTFLOW_VETO_CSV = (
    f"{_region_eval_cutflow_output_dir()}/veto_cut_efficiency.csv"
)
REGION_EVAL_CUTFLOW_US_CSV = (
    f"{_region_eval_cutflow_output_dir()}/us_cut_efficiency.csv"
)
REGION_EVAL_CUTFLOW_EVENTBUILDER_COMPARISON_CSV = (
    f"{_region_eval_cutflow_output_dir()}/eventbuilder_comparison.csv"
)
REGION_EVAL_CUTFLOW_TABLE_PDF = (
    f"{_region_eval_cutflow_output_dir()}/tables.pdf"
)
REGION_EVAL_CUTFLOW_SHARD_DIR = (
    f"{_region_eval_cutflow_output_dir()}/shards"
)
REGION_EVAL_CUTFLOW_QDC_TOTAL_EFFICIENCY_CSV = (
    f"{_region_eval_cutflow_output_dir(REGION_EVAL_CUTFLOW_QDC_TOTAL_VERSION)}/summary.csv"
)
REGION_EVAL_CUTFLOW_QDC_TOTAL_EFFICIENCY_BY_STUDY_CSV = (
    f"{_region_eval_cutflow_output_dir(REGION_EVAL_CUTFLOW_QDC_TOTAL_VERSION)}/summary_by_study.csv"
)
REGION_EVAL_CUTFLOW_QDC_TOTAL_REGIONAL_CSV = (
    f"{_region_eval_cutflow_output_dir(REGION_EVAL_CUTFLOW_QDC_TOTAL_VERSION)}/regional_cut_efficiency.csv"
)
REGION_EVAL_CUTFLOW_QDC_TOTAL_OVERALL_CSV = (
    f"{_region_eval_cutflow_output_dir(REGION_EVAL_CUTFLOW_QDC_TOTAL_VERSION)}/overall_cut_efficiency.csv"
)
REGION_EVAL_CUTFLOW_QDC_TOTAL_VETO_CSV = (
    f"{_region_eval_cutflow_output_dir(REGION_EVAL_CUTFLOW_QDC_TOTAL_VERSION)}/veto_cut_efficiency.csv"
)
REGION_EVAL_CUTFLOW_QDC_TOTAL_US_CSV = (
    f"{_region_eval_cutflow_output_dir(REGION_EVAL_CUTFLOW_QDC_TOTAL_VERSION)}/us_cut_efficiency.csv"
)
REGION_EVAL_CUTFLOW_QDC_TOTAL_EVENTBUILDER_COMPARISON_CSV = (
    f"{_region_eval_cutflow_output_dir(REGION_EVAL_CUTFLOW_QDC_TOTAL_VERSION)}/eventbuilder_comparison.csv"
)
REGION_EVAL_CUTFLOW_QDC_TOTAL_TABLE_PDF = (
    f"{_region_eval_cutflow_output_dir(REGION_EVAL_CUTFLOW_QDC_TOTAL_VERSION)}/tables.pdf"
)
REGION_EVAL_CUTFLOW_QDC_TOTAL_SHARD_DIR = (
    f"{_region_eval_cutflow_output_dir(REGION_EVAL_CUTFLOW_QDC_TOTAL_VERSION)}/shards"
)
REGION_EVAL_EVENT_DISPLAY_DONE = (
    f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/event_display/"
    "selected_event_displays.done"
)
REGION_EVAL_EVENT_DISPLAY_FOLLOW_CONFIG = (
    REGION_EVAL_EVENT_DISPLAY_CONFIG_DATA.get("follow_event", {}) or {}
)
REGION_EVAL_FOLLOW_EVENT_SUMMARY_CSV = _region_eval_output_path(
    REGION_EVAL_EVENT_DISPLAY_FOLLOW_CONFIG.get(
        "summary_csv",
        "evaluation_region_partitions/event_display/summary.csv",
    )
)
REGION_EVAL_FOLLOW_EVENT_DISPLAY_DONE = (
    f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/event_display/follow_events/"
    f"run{REGION_EVAL_EVENT_DISPLAY_FOLLOW_CONFIG.get('runId', 'configured')}_"
    f"event{REGION_EVAL_EVENT_DISPLAY_FOLLOW_CONFIG.get('eventId', 'configured')}/"
    "follow_event_displays.done"
)


def _region_eval_metadata_rows():
    if not os.path.isfile(REGION_EVAL_METADATA_CSV):
        return []
    df = pd.read_csv(REGION_EVAL_METADATA_CSV)
    required = {"region", "particle_group"}
    if not required.issubset(df.columns):
        return []
    return df.to_dict("records")


REGION_EVAL_METADATA_ROWS = _region_eval_metadata_rows()
REGION_EVAL_FEATURES = list(REGION_EVAL_HIST_FEATURE_CONFIG.get("features", {}).keys())
REGION_EVAL_BASE_CUT_KEYS = list(
    REGION_EVAL_OPTIONS.get("cuts", {}).get("base_cut_options", {}).keys()
)
REGION_EVAL_EXTRA_CUT_KEYS = list(
    REGION_EVAL_OPTIONS.get("cuts", {}).get("extra_cut_options", {}).keys()
)
REGION_EVAL_SNAKEMAKE_OPTIONS = REGION_EVAL_OPTIONS.get("snakemake", {})
REGION_EVAL_DEFAULT_FEATURES = REGION_EVAL_SNAKEMAKE_OPTIONS.get(
    "default_features",
    REGION_EVAL_FEATURES,
)
REGION_EVAL_DEFAULT_BASE_CUT_KEYS = REGION_EVAL_SNAKEMAKE_OPTIONS.get(
    "default_base_cut_keys",
    REGION_EVAL_BASE_CUT_KEYS,
)
REGION_EVAL_DEFAULT_EXTRA_CUT_KEYS = REGION_EVAL_SNAKEMAKE_OPTIONS.get(
    "default_extra_cut_keys",
    REGION_EVAL_EXTRA_CUT_KEYS,
)
REGION_EVAL_CONFIG_REGIONS = set(
    REGION_EVAL_OPTIONS.get("comparisons", {}).get("regions", [])
)
REGION_EVAL_CONFIG_PARTICLE_GROUPS = set(
    REGION_EVAL_OPTIONS.get("comparisons", {}).get("particle_groups", [])
)
REGION_EVAL_COMPARISON_MODES = list(
    REGION_EVAL_OPTIONS.get("comparisons", {}).get("modes", {}).keys()
)


def _region_eval_group_pairs():
    pairs = []
    seen = set()
    for row in REGION_EVAL_METADATA_ROWS:
        region = str(row["region"])
        particle_group = str(row["particle_group"])

        if REGION_EVAL_CONFIG_REGIONS and region not in REGION_EVAL_CONFIG_REGIONS:
            continue
        if REGION_EVAL_CONFIG_PARTICLE_GROUPS and particle_group not in REGION_EVAL_CONFIG_PARTICLE_GROUPS:
            continue

        key = (region, particle_group)
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"region": region, "particle_group": particle_group})
    return pairs


REGION_EVAL_GROUP_PAIRS = _region_eval_group_pairs()
REGION_EVAL_GROUP_PAIR_SET = {
    (item["region"], item["particle_group"])
    for item in REGION_EVAL_GROUP_PAIRS
}
REGION_EVAL_REGIONS = sorted({item["region"] for item in REGION_EVAL_GROUP_PAIRS})
REGION_EVAL_PARTICLE_GROUPS = sorted({item["particle_group"] for item in REGION_EVAL_GROUP_PAIRS})


def _region_eval_veto_category(region):
    region = str(region)
    if "_no_veto_" in region:
        return "no_veto"
    if "_has_veto_" in region:
        return "has_veto"
    return "unknown_veto"


def _region_eval_us_category(region):
    region = str(region)
    if "_has_us_" in region:
        return "has_us"
    if "_no_us_" in region:
        return "no_us"
    return "unknown_us"


def _region_eval_fiducial_merged_region(region):
    region = str(region)
    if "_no_veto_" in region and "_has_us_" in region:
        return "merged_01_signal_sb2_no_veto_has_us"
    if "_no_veto_" in region and "_no_us_" in region:
        return "merged_02_sb1_sb3_no_veto_no_us"
    if "_has_veto_" in region and "_has_us_" in region:
        return "merged_03_sb4_sb6_has_veto_has_us"
    if "_has_veto_" in region and "_no_us_" in region:
        return "merged_04_sb5_sb7_has_veto_no_us"
    return "merged_unknown"


def _region_eval_cutflow_enabled_studies_for(config_data):
    studies = []
    for study_name, study_cfg in config_data.get("studies", {}).items():
        if study_cfg.get("enabled", True):
            studies.append(str(study_name))
    return studies


def _region_eval_cutflow_enabled_studies():
    return _region_eval_cutflow_enabled_studies_for(REGION_EVAL_CUTFLOW_CONFIG_DATA)


def _region_eval_cutflow_qdc_total_enabled_studies():
    return _region_eval_cutflow_enabled_studies_for(REGION_EVAL_CUTFLOW_QDC_TOTAL_CONFIG_DATA)


def _region_eval_cutflow_study_aggregation_modes_for(config_data, study_name):
    study_cfg = config_data["studies"][study_name]
    return [
        str(value)
        for value in study_cfg.get(
            "aggregation_modes",
            config_data.get("defaults", {}).get("aggregation_modes", []),
        )
    ]


def _region_eval_cutflow_study_aggregation_modes(study_name):
    return _region_eval_cutflow_study_aggregation_modes_for(
        REGION_EVAL_CUTFLOW_CONFIG_DATA,
        study_name,
    )


def _region_eval_cutflow_qdc_total_study_aggregation_modes(study_name):
    return _region_eval_cutflow_study_aggregation_modes_for(
        REGION_EVAL_CUTFLOW_QDC_TOTAL_CONFIG_DATA,
        study_name,
    )


def _region_eval_cutflow_shard_path(study, aggregation_mode, group_region, group_particle):
    return (
        f"{REGION_EVAL_CUTFLOW_SHARD_DIR}/"
        f"{study}/{aggregation_mode}/{group_region}__{group_particle}.csv"
    )


def _region_eval_cutflow_qdc_total_shard_path(study, aggregation_mode, group_region, group_particle):
    return (
        f"{REGION_EVAL_CUTFLOW_QDC_TOTAL_SHARD_DIR}/"
        f"{study}/{aggregation_mode}/{group_region}__{group_particle}.csv"
    )


def _region_eval_cutflow_shard_specs_for(config_data):
    specs = []
    seen = set()
    group_pairs = [
        (str(item["region"]), str(item["particle_group"]))
        for item in REGION_EVAL_GROUP_PAIRS
    ]
    particles = sorted({particle for _, particle in group_pairs})
    veto_groups = sorted(
        {
            (_region_eval_veto_category(region), particle)
            for region, particle in group_pairs
        }
    )
    us_groups = sorted(
        {
            (_region_eval_us_category(region), particle)
            for region, particle in group_pairs
        }
    )

    for study in _region_eval_cutflow_enabled_studies_for(config_data):
        for aggregation_mode in _region_eval_cutflow_study_aggregation_modes_for(config_data, study):
            if aggregation_mode == "per_region_particle":
                groups = sorted(group_pairs)
            elif aggregation_mode == "fiducial_merged_region_particle":
                groups = sorted(
                    {
                        (_region_eval_fiducial_merged_region(region), particle)
                        for region, particle in group_pairs
                    }
                )
            elif aggregation_mode == "overall_particle":
                groups = [("ALL", particle) for particle in particles]
            elif aggregation_mode == "veto_category_particle":
                groups = veto_groups
            elif aggregation_mode == "us_category_particle":
                groups = us_groups
            elif aggregation_mode == "overall_all":
                groups = [("ALL", "ALL")]
            else:
                continue

            for group_region, group_particle in groups:
                key = (study, aggregation_mode, group_region, group_particle)
                if key in seen:
                    continue
                seen.add(key)
                specs.append(
                    {
                        "study": study,
                        "aggregation_mode": aggregation_mode,
                        "group_region": group_region,
                        "group_particle": group_particle,
                    }
                )
    return specs


def _region_eval_cutflow_shard_specs():
    return _region_eval_cutflow_shard_specs_for(REGION_EVAL_CUTFLOW_CONFIG_DATA)


def _region_eval_cutflow_qdc_total_shard_specs():
    return _region_eval_cutflow_shard_specs_for(REGION_EVAL_CUTFLOW_QDC_TOTAL_CONFIG_DATA)


REGION_EVAL_CUTFLOW_SHARD_SPECS = _region_eval_cutflow_shard_specs()
REGION_EVAL_CUTFLOW_SHARD_TARGETS = [
    _region_eval_cutflow_shard_path(
        spec["study"],
        spec["aggregation_mode"],
        spec["group_region"],
        spec["group_particle"],
    )
    for spec in REGION_EVAL_CUTFLOW_SHARD_SPECS
]
REGION_EVAL_CUTFLOW_QDC_TOTAL_SHARD_SPECS = _region_eval_cutflow_qdc_total_shard_specs()
REGION_EVAL_CUTFLOW_QDC_TOTAL_SHARD_TARGETS = [
    _region_eval_cutflow_qdc_total_shard_path(
        spec["study"],
        spec["aggregation_mode"],
        spec["group_region"],
        spec["group_particle"],
    )
    for spec in REGION_EVAL_CUTFLOW_QDC_TOTAL_SHARD_SPECS
]


def _region_eval_group_hist_path(region, particle_group, feature, base_cut_key, extra_cut_key):
    return (
        f"{REGION_EVAL_HIST_OUTPUT_DIR}/"
        f"{REGION_PARTITION_CONFIG['partition_version']}_{REGION_VERSION}/"
        f"{REGION_EVAL_HIST_VERSION}__{REGION_EVAL_VERSION}/"
        f"{feature}/{base_cut_key}/{extra_cut_key}/{region}/{particle_group}.root"
    )


def _region_eval_feature_pair_features(feature):
    paired_features = []
    for study in REGION_EVAL_OPTIONS.get("feature_pair_comparisons", {}).values():
        if not study.get("enabled", True):
            continue
        for pair in study.get("pairs", []):
            left = str(pair["left"])
            right = str(pair["right"])
            trigger = str(pair.get("trigger", right))
            if str(feature) == trigger:
                paired_features.extend([left, right])
    return sorted(set(paired_features))


def _region_eval_comparison_plot_dir(feature, base_cut_key, extra_cut_key):
    return (
        f"{REGION_EVAL_PLOT_OUTPUT_DIR}/"
        f"{REGION_PARTITION_CONFIG['partition_version']}_{REGION_VERSION}/"
        f"{REGION_EVAL_HIST_VERSION}__{REGION_EVAL_VERSION}/"
        f"{feature}/{base_cut_key}/{extra_cut_key}"
    )


def _region_eval_comparison_plot_pdf(feature, base_cut_key, extra_cut_key):
    return (
        f"{_region_eval_comparison_plot_dir(feature, base_cut_key, extra_cut_key)}"
        "/plot.pdf"
    )


def _region_eval_comparison_summary_csv(feature, base_cut_key, extra_cut_key):
    return (
        f"{_region_eval_comparison_plot_dir(feature, base_cut_key, extra_cut_key)}"
        "/summary.csv"
    )


def _region_eval_hist_inputs_for_comparison(wildcards):
    features = [wildcards.feature]
    for paired_feature in _region_eval_feature_pair_features(wildcards.feature):
        if paired_feature not in features:
            features.append(paired_feature)

    return [
        _region_eval_group_hist_path(
            item["region"],
            item["particle_group"],
            feature,
            wildcards.base_cut_key,
            wildcards.extra_cut_key,
        )
        for item in REGION_EVAL_GROUP_PAIRS
        for feature in features
    ]


def _region_eval_default_comparison_plot_targets():
    targets = []
    for feature in REGION_EVAL_DEFAULT_FEATURES:
        for base_cut_key in REGION_EVAL_DEFAULT_BASE_CUT_KEYS:
            for extra_cut_key in REGION_EVAL_DEFAULT_EXTRA_CUT_KEYS:
                targets.append(
                    _region_eval_comparison_plot_pdf(
                        feature,
                        base_cut_key,
                        extra_cut_key,
                    )
                )
    return targets


def _region_eval_default_group_hist_targets():
    targets = []
    for group in REGION_EVAL_GROUP_PAIRS:
        for feature in REGION_EVAL_DEFAULT_FEATURES:
            for base_cut_key in REGION_EVAL_DEFAULT_BASE_CUT_KEYS:
                for extra_cut_key in REGION_EVAL_DEFAULT_EXTRA_CUT_KEYS:
                    targets.append(
                        _region_eval_group_hist_path(
                            group["region"],
                            group["particle_group"],
                            feature,
                            base_cut_key,
                            extra_cut_key,
                        )
                    )
    return targets


rule region_eval_group_hist:
    input:
        hist_config=REGION_EVAL_HIST_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/make_group_hist.py",
    params:
        eval_config=REGION_EVAL_OPTIONS_CONFIG,
        metadata=REGION_EVAL_METADATA_CSV,
    output:
        hist=(
            f"{REGION_EVAL_HIST_OUTPUT_DIR}/"
            f"{REGION_PARTITION_CONFIG['partition_version']}_{REGION_VERSION}/"
            f"{REGION_EVAL_HIST_VERSION}__{REGION_EVAL_VERSION}/"
            "{feature}/{base_cut_key}/{extra_cut_key}/{region}/{particle_group}.root"
        )
    wildcard_constraints:
        feature="|".join(REGION_EVAL_FEATURES),
        base_cut_key="|".join(REGION_EVAL_BASE_CUT_KEYS),
        extra_cut_key="|".join(REGION_EVAL_EXTRA_CUT_KEYS),
        region="|".join(REGION_EVAL_REGIONS) if REGION_EVAL_REGIONS else "[^/]+",
        particle_group="|".join(REGION_EVAL_PARTICLE_GROUPS) if REGION_EVAL_PARTICLE_GROUPS else "[^/]+",
    threads: 1
    resources:
        runtime=45 * 60,
        mem_mb=3000,
        disk_mb=3000,
        nvidia_gpu=0
    shell:
        r"""
        echo "building region evaluation grouped histogram"
        set +u
        source {env_script_lcg}
        set -euo pipefail

        python {input.script} \
            --metadata-csv {params.metadata} \
            --hist-config {input.hist_config} \
            --eval-config {params.eval_config} \
            --region {wildcards.region} \
            --particle-group {wildcards.particle_group} \
            --feature {wildcards.feature} \
            --base-cut-key {wildcards.base_cut_key} \
            --extra-cut-key {wildcards.extra_cut_key} \
            --output {output.hist} \
            --repo-root {PERSONAL_WORK_SPACE}
        """


rule region_eval_group_hists:
    input:
        lambda wildcards: _region_eval_default_group_hist_targets()


rule region_eval_feature_comparison_plot:
    input:
        hist_config=REGION_EVAL_HIST_CONFIG,
        eval_config=REGION_EVAL_OPTIONS_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/plot_comparison.py",
        hists=_region_eval_hist_inputs_for_comparison,
    output:
        pdf=(
            f"{REGION_EVAL_PLOT_OUTPUT_DIR}/"
            f"{REGION_PARTITION_CONFIG['partition_version']}_{REGION_VERSION}/"
            f"{REGION_EVAL_HIST_VERSION}__{REGION_EVAL_VERSION}/"
            "{feature}/{base_cut_key}/{extra_cut_key}/plot.pdf"
        ),
        summary=(
            f"{REGION_EVAL_PLOT_OUTPUT_DIR}/"
            f"{REGION_PARTITION_CONFIG['partition_version']}_{REGION_VERSION}/"
            f"{REGION_EVAL_HIST_VERSION}__{REGION_EVAL_VERSION}/"
            "{feature}/{base_cut_key}/{extra_cut_key}/summary.csv"
        )
    params:
        partition_version=REGION_PARTITION_CONFIG["partition_version"],
        region_version=REGION_VERSION,
        metadata=REGION_EVAL_METADATA_CSV,
    wildcard_constraints:
        feature="|".join(REGION_EVAL_FEATURES),
        base_cut_key="|".join(REGION_EVAL_BASE_CUT_KEYS),
        extra_cut_key="|".join(REGION_EVAL_EXTRA_CUT_KEYS),
    threads: 1
    resources:
        runtime=20 * 60,
        mem_mb=2000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        echo "plotting region evaluation comparison"
        set +u
        source {env_script_lcg}
        set -euo pipefail

        python {input.script} \
            --metadata-csv {params.metadata} \
            --hist-config {input.hist_config} \
            --eval-config {input.eval_config} \
            --feature {wildcards.feature} \
            --base-cut-key {wildcards.base_cut_key} \
            --extra-cut-key {wildcards.extra_cut_key} \
            --partition-version {params.partition_version} \
            --region-version {params.region_version} \
            --repo-root {PERSONAL_WORK_SPACE}
        """


rule region_eval_comparison_plots:
    input:
        lambda wildcards: _region_eval_default_comparison_plot_targets()


rule region_eval_fiducial_efficiency:
    input:
        eval_config=REGION_EVAL_OPTIONS_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/fiducial_efficiency.py",
    params:
        metadata=REGION_EVAL_METADATA_CSV,
    output:
        summary=REGION_EVAL_FIDUCIAL_SUMMARY_CSV,
        overlap=REGION_EVAL_FIDUCIAL_OVERLAP_CSV,
    threads: 1
    resources:
        runtime=45 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        echo "summarizing region evaluation fiducial efficiencies"
        set +u
        source {env_script_lcg}
        set -euo pipefail

        python {input.script} \
            --metadata-csv {params.metadata} \
            --eval-config {input.eval_config} \
            --summary {output.summary} \
            --overlap {output.overlap} \
            --repo-root {PERSONAL_WORK_SPACE}
        """


rule region_eval_eventbuilder_veto_timing:
    input:
        hist_config=REGION_EVAL_HIST_CONFIG,
        eval_config=REGION_EVAL_OPTIONS_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/eventbuilder_veto_timing_summary.py",
    params:
        metadata=REGION_EVAL_METADATA_CSV,
    output:
        summary=REGION_EVAL_EVENTBUILDER_VETO_TIMING_CSV,
    threads: 1
    resources:
        runtime=45 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        echo "summarizing EventBuilder veto timing comparisons"
        set +u
        source {env_script_lcg}
        set -euo pipefail

        python {input.script} \
            --metadata-csv {params.metadata} \
            --hist-config {input.hist_config} \
            --eval-config {input.eval_config} \
            --output {output.summary} \
            --repo-root {PERSONAL_WORK_SPACE}
        """


rule region_eval_cutflow_efficiency_shard:
    input:
        eval_config=REGION_EVAL_OPTIONS_CONFIG,
        cutflow_config=REGION_EVAL_CUTFLOW_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/cutflow_efficiency.py",
    params:
        metadata=REGION_EVAL_METADATA_CSV,
    output:
        shard=(
            f"{REGION_EVAL_CUTFLOW_SHARD_DIR}/"
            "{study}/{aggregation_mode}/{group_region}__{group_particle}.csv"
        )
    wildcard_constraints:
        study="|".join(_region_eval_cutflow_enabled_studies()),
        aggregation_mode="|".join(sorted(REGION_EVAL_CUTFLOW_CONFIG_DATA.get("defaults", {}).get("aggregation_modes", []) + [
            mode
            for study in _region_eval_cutflow_enabled_studies()
            for mode in _region_eval_cutflow_study_aggregation_modes(study)
        ])),
        group_region="[^/]+",
        group_particle="[^/]+",
    threads: 1
    resources:
        runtime=45 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        echo "summarizing region evaluation cutflow shard {wildcards.study} {wildcards.aggregation_mode} {wildcards.group_region} {wildcards.group_particle}"
        set +u
        source {env_script_lcg}
        set -euo pipefail

        python {input.script} \
            --metadata-csv {params.metadata} \
            --eval-config {input.eval_config} \
            --cutflow-config {input.cutflow_config} \
            --study {wildcards.study} \
            --aggregation-mode {wildcards.aggregation_mode} \
            --group-region {wildcards.group_region} \
            --group-particle {wildcards.group_particle} \
            --output {output.shard} \
            --repo-root {PERSONAL_WORK_SPACE}
        """


rule region_eval_cutflow_efficiency:
    input:
        eval_config=REGION_EVAL_OPTIONS_CONFIG,
        cutflow_config=REGION_EVAL_CUTFLOW_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/cutflow_efficiency.py",
        shards=REGION_EVAL_CUTFLOW_SHARD_TARGETS,
    params:
        metadata=REGION_EVAL_METADATA_CSV,
        merge_inputs=lambda wildcards, input: " ".join(
            f"--merge-input {path}" for path in input.shards
        ),
    output:
        summary=REGION_EVAL_CUTFLOW_EFFICIENCY_CSV,
        summary_by_study=REGION_EVAL_CUTFLOW_EFFICIENCY_BY_STUDY_CSV,
        regional=REGION_EVAL_CUTFLOW_REGIONAL_CSV,
        overall=REGION_EVAL_CUTFLOW_OVERALL_CSV,
        veto=REGION_EVAL_CUTFLOW_VETO_CSV,
        us=REGION_EVAL_CUTFLOW_US_CSV,
        eventbuilder=REGION_EVAL_CUTFLOW_EVENTBUILDER_COMPARISON_CSV,
    threads: 1
    resources:
        runtime=45 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        echo "merging region evaluation cutflow efficiency shards"
        set +u
        source {env_script_lcg}
        set -euo pipefail

        python {input.script} \
            --metadata-csv {params.metadata} \
            --eval-config {input.eval_config} \
            --cutflow-config {input.cutflow_config} \
            --output {output.summary} \
            --summary-by-study {output.summary_by_study} \
            --regional-cut-efficiency {output.regional} \
            --overall-cut-efficiency {output.overall} \
            --veto-cut-efficiency {output.veto} \
            --us-cut-efficiency {output.us} \
            --eventbuilder-comparison {output.eventbuilder} \
            {params.merge_inputs} \
            --repo-root {PERSONAL_WORK_SPACE}
        """


rule region_eval_cutflow_qdc_total_efficiency_shard:
    input:
        eval_config=REGION_EVAL_OPTIONS_CONFIG,
        cutflow_config=REGION_EVAL_CUTFLOW_QDC_TOTAL_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/cutflow_efficiency.py",
    params:
        metadata=REGION_EVAL_METADATA_CSV,
    output:
        shard=(
            f"{REGION_EVAL_CUTFLOW_QDC_TOTAL_SHARD_DIR}/"
            "{study}/{aggregation_mode}/{group_region}__{group_particle}.csv"
        )
    wildcard_constraints:
        study="|".join(_region_eval_cutflow_qdc_total_enabled_studies()),
        aggregation_mode="|".join(sorted(REGION_EVAL_CUTFLOW_QDC_TOTAL_CONFIG_DATA.get("defaults", {}).get("aggregation_modes", []) + [
            mode
            for study in _region_eval_cutflow_qdc_total_enabled_studies()
            for mode in _region_eval_cutflow_qdc_total_study_aggregation_modes(study)
        ])),
        group_region="[^/]+",
        group_particle="[^/]+",
    threads: 1
    resources:
        runtime=45 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        echo "summarizing QDC-total fiducial cutflow shard {wildcards.study} {wildcards.aggregation_mode} {wildcards.group_region} {wildcards.group_particle}"
        set +u
        source {env_script_lcg}
        set -euo pipefail

        python {input.script} \
            --metadata-csv {params.metadata} \
            --eval-config {input.eval_config} \
            --cutflow-config {input.cutflow_config} \
            --study {wildcards.study} \
            --aggregation-mode {wildcards.aggregation_mode} \
            --group-region {wildcards.group_region} \
            --group-particle {wildcards.group_particle} \
            --output {output.shard} \
            --repo-root {PERSONAL_WORK_SPACE}
        """


rule region_eval_cutflow_qdc_total_efficiency:
    input:
        eval_config=REGION_EVAL_OPTIONS_CONFIG,
        cutflow_config=REGION_EVAL_CUTFLOW_QDC_TOTAL_CONFIG,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/cutflow_efficiency.py",
        shards=REGION_EVAL_CUTFLOW_QDC_TOTAL_SHARD_TARGETS,
    params:
        metadata=REGION_EVAL_METADATA_CSV,
        merge_inputs=lambda wildcards, input: " ".join(
            f"--merge-input {path}" for path in input.shards
        ),
    output:
        summary=REGION_EVAL_CUTFLOW_QDC_TOTAL_EFFICIENCY_CSV,
        summary_by_study=REGION_EVAL_CUTFLOW_QDC_TOTAL_EFFICIENCY_BY_STUDY_CSV,
        regional=REGION_EVAL_CUTFLOW_QDC_TOTAL_REGIONAL_CSV,
        overall=REGION_EVAL_CUTFLOW_QDC_TOTAL_OVERALL_CSV,
        veto=REGION_EVAL_CUTFLOW_QDC_TOTAL_VETO_CSV,
        us=REGION_EVAL_CUTFLOW_QDC_TOTAL_US_CSV,
        eventbuilder=REGION_EVAL_CUTFLOW_QDC_TOTAL_EVENTBUILDER_COMPARISON_CSV,
    threads: 1
    resources:
        runtime=45 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        echo "merging QDC-total fiducial cutflow efficiency shards"
        set +u
        source {env_script_lcg}
        set -euo pipefail

        python {input.script} \
            --metadata-csv {params.metadata} \
            --eval-config {input.eval_config} \
            --cutflow-config {input.cutflow_config} \
            --output {output.summary} \
            --summary-by-study {output.summary_by_study} \
            --regional-cut-efficiency {output.regional} \
            --overall-cut-efficiency {output.overall} \
            --veto-cut-efficiency {output.veto} \
            --us-cut-efficiency {output.us} \
            --eventbuilder-comparison {output.eventbuilder} \
            {params.merge_inputs} \
            --repo-root {PERSONAL_WORK_SPACE}
        """


rule region_eval_event_displays:
    input:
        config=REGION_EVAL_EVENT_DISPLAY_CONFIG,
        metadata=REGION_EVAL_METADATA_CSV,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/plot_saved_events_sndsw.py",
    output:
        done=REGION_EVAL_EVENT_DISPLAY_DONE,
    params:
        outdir=lambda wildcards, output: os.path.dirname(output.done),
    threads: 1
    resources:
        runtime=60 * 60,
        mem_mb=4000,
        disk_mb=4000,
        nvidia_gpu=0
    shell:
        r"""
        echo "plotting region partition SNDSW event displays"
        export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u
        source {env_script_sndsw_nue}
        set -euo pipefail
        export EOSSHIP=root://eosuser.cern.ch/

        mkdir -p "{params.outdir}"

        python {input.script} \
            --config {input.config} \
            --metadata-csv {input.metadata} \
            --outdir "{params.outdir}" \
            --repo-root {PERSONAL_WORK_SPACE}

        touch "{output.done}"
        """


rule region_eval_follow_event_display:
    input:
        config=REGION_EVAL_EVENT_DISPLAY_CONFIG,
        selected_summary=REGION_EVAL_FOLLOW_EVENT_SUMMARY_CSV,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/plot_saved_events_sndsw.py",
    output:
        done=REGION_EVAL_FOLLOW_EVENT_DISPLAY_DONE,
    params:
        outdir=lambda wildcards, output: os.path.dirname(os.path.dirname(os.path.dirname(output.done))),
        run_id=REGION_EVAL_EVENT_DISPLAY_FOLLOW_CONFIG.get("runId", ""),
        event_id=REGION_EVAL_EVENT_DISPLAY_FOLLOW_CONFIG.get("eventId", ""),
        n_following=REGION_EVAL_EVENT_DISPLAY_FOLLOW_CONFIG.get("n_following", 1),
    threads: 1
    resources:
        runtime=60 * 60,
        mem_mb=4000,
        disk_mb=4000,
        nvidia_gpu=0
    shell:
        r"""
        echo "plotting follow-up SNDSW event display"
        export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u
        source {env_script_sndsw_nue}
        set -euo pipefail
        export EOSSHIP=root://eosuser.cern.ch/

        mkdir -p "{params.outdir}"

        python {input.script} \
            --config {input.config} \
            --outdir "{params.outdir}" \
            --follow-summary-csv "{input.selected_summary}" \
            --follow-run {params.run_id} \
            --follow-event {params.event_id} \
            --n-following {params.n_following} \
            --repo-root {PERSONAL_WORK_SPACE}

        touch "{output.done}"
        """


rule region_eval_cutflow_table_plots:
    input:
        regional=REGION_EVAL_CUTFLOW_REGIONAL_CSV,
        overall=REGION_EVAL_CUTFLOW_OVERALL_CSV,
        veto=REGION_EVAL_CUTFLOW_VETO_CSV,
        us=REGION_EVAL_CUTFLOW_US_CSV,
        eval_config=REGION_EVAL_OPTIONS_CONFIG,
        metadata=REGION_EVAL_METADATA_CSV,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/plot_cutflow_tables.py",
    output:
        pdf=REGION_EVAL_CUTFLOW_TABLE_PDF,
    threads: 1
    resources:
        runtime=10 * 60,
        mem_mb=1500,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        echo "plotting region evaluation cutflow tables"
        set +u
        source {env_script_lcg}
        set -euo pipefail

        python {input.script} \
            --input-csv regional={input.regional} \
            --input-csv overall={input.overall} \
            --input-csv veto={input.veto} \
            --input-csv us={input.us} \
            --eval-config {input.eval_config} \
            --metadata-csv {input.metadata} \
            --output-pdf {output.pdf}
        """


rule region_eval_cutflow_qdc_total_table_plots:
    input:
        regional=REGION_EVAL_CUTFLOW_QDC_TOTAL_REGIONAL_CSV,
        overall=REGION_EVAL_CUTFLOW_QDC_TOTAL_OVERALL_CSV,
        veto=REGION_EVAL_CUTFLOW_QDC_TOTAL_VETO_CSV,
        us=REGION_EVAL_CUTFLOW_QDC_TOTAL_US_CSV,
        eval_config=REGION_EVAL_OPTIONS_CONFIG,
        metadata=REGION_EVAL_METADATA_CSV,
        script=f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/plot_cutflow_tables.py",
    output:
        pdf=REGION_EVAL_CUTFLOW_QDC_TOTAL_TABLE_PDF,
    threads: 1
    resources:
        runtime=10 * 60,
        mem_mb=1500,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        echo "plotting QDC-total fiducial cutflow tables"
        set +u
        source {env_script_lcg}
        set -euo pipefail

        python {input.script} \
            --input-csv regional={input.regional} \
            --input-csv overall={input.overall} \
            --input-csv veto={input.veto} \
            --input-csv us={input.us} \
            --eval-config {input.eval_config} \
            --metadata-csv {input.metadata} \
            --output-pdf {output.pdf}
        """


rule region_eval_cutflow_efficiency_with_tables:
    input:
        REGION_EVAL_CUTFLOW_EFFICIENCY_CSV,
        REGION_EVAL_CUTFLOW_EFFICIENCY_BY_STUDY_CSV,
        REGION_EVAL_CUTFLOW_REGIONAL_CSV,
        REGION_EVAL_CUTFLOW_OVERALL_CSV,
        REGION_EVAL_CUTFLOW_VETO_CSV,
        REGION_EVAL_CUTFLOW_US_CSV,
        REGION_EVAL_CUTFLOW_EVENTBUILDER_COMPARISON_CSV,
        REGION_EVAL_CUTFLOW_TABLE_PDF,


rule region_eval_cutflow_qdc_total_efficiency_with_tables:
    input:
        REGION_EVAL_CUTFLOW_QDC_TOTAL_EFFICIENCY_CSV,
        REGION_EVAL_CUTFLOW_QDC_TOTAL_EFFICIENCY_BY_STUDY_CSV,
        REGION_EVAL_CUTFLOW_QDC_TOTAL_REGIONAL_CSV,
        REGION_EVAL_CUTFLOW_QDC_TOTAL_OVERALL_CSV,
        REGION_EVAL_CUTFLOW_QDC_TOTAL_VETO_CSV,
        REGION_EVAL_CUTFLOW_QDC_TOTAL_US_CSV,
        REGION_EVAL_CUTFLOW_QDC_TOTAL_EVENTBUILDER_COMPARISON_CSV,
        REGION_EVAL_CUTFLOW_QDC_TOTAL_TABLE_PDF,


rule region_eval_studies:
    input:
        REGION_EVAL_FIDUCIAL_SUMMARY_CSV,
        REGION_EVAL_FIDUCIAL_OVERLAP_CSV,
        REGION_EVAL_EVENTBUILDER_VETO_TIMING_CSV,
        REGION_EVAL_CUTFLOW_EFFICIENCY_CSV,
        REGION_EVAL_CUTFLOW_EFFICIENCY_BY_STUDY_CSV,
        REGION_EVAL_CUTFLOW_REGIONAL_CSV,
        REGION_EVAL_CUTFLOW_OVERALL_CSV,
        REGION_EVAL_CUTFLOW_VETO_CSV,
        REGION_EVAL_CUTFLOW_US_CSV,
        REGION_EVAL_CUTFLOW_EVENTBUILDER_COMPARISON_CSV,
        REGION_EVAL_CUTFLOW_TABLE_PDF,
        REGION_EVAL_EVENT_DISPLAY_DONE,
