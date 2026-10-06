EM_SHOWER_REPO_ROOT = str(Path(workflow.basedir).resolve().parent)
EM_SHOWER_CONFIG_DIR = f"{EM_SHOWER_REPO_ROOT}/snakemake/metadata/configs/em_shower"
EM_SHOWER_PRODUCTION_CONFIG = f"{EM_SHOWER_CONFIG_DIR}/production_v1.yaml"
EM_SHOWER_REGION_CONFIG = f"{EM_SHOWER_CONFIG_DIR}/regions_v1.yaml"
EM_SHOWER_PARTITION_CONFIG = f"{EM_SHOWER_CONFIG_DIR}/partitions_v1.yaml"
EM_SHOWER_HIST_CONFIG = f"{EM_SHOWER_CONFIG_DIR}/hist_features_v1.yaml"
EM_SHOWER_EVAL_CONFIG = f"{EM_SHOWER_CONFIG_DIR}/evaluation_options_v1.yaml"
EM_SHOWER_MUON_CONFIG = f"{EM_SHOWER_CONFIG_DIR}/mc_muon_up_v1.yaml"
EM_SHOWER_MUON_EVAL_CONFIG = f"{EM_SHOWER_CONFIG_DIR}/evaluation_options_muon_up_v1.yaml"

with open(EM_SHOWER_PRODUCTION_CONFIG, "r", encoding="utf-8") as handle:
    EM_SHOWER_PRODUCTION = yaml.safe_load(handle)
with open(EM_SHOWER_REGION_CONFIG, "r", encoding="utf-8") as handle:
    EM_SHOWER_REGIONS_CONFIG = yaml.safe_load(handle)
with open(EM_SHOWER_PARTITION_CONFIG, "r", encoding="utf-8") as handle:
    EM_SHOWER_PARTITIONS_CONFIG = yaml.safe_load(handle)
with open(EM_SHOWER_HIST_CONFIG, "r", encoding="utf-8") as handle:
    EM_SHOWER_HISTS_CONFIG = yaml.safe_load(handle)
with open(EM_SHOWER_EVAL_CONFIG, "r", encoding="utf-8") as handle:
    EM_SHOWER_EVALUATION = yaml.safe_load(handle)
with open(EM_SHOWER_MUON_CONFIG, "r", encoding="utf-8") as handle:
    EM_SHOWER_MUON = yaml.safe_load(handle)
with open(EM_SHOWER_MUON_EVAL_CONFIG, "r", encoding="utf-8") as handle:
    EM_SHOWER_MUON_EVALUATION = yaml.safe_load(handle)


def _em_shower_local_path(value):
    path = Path(str(value))
    return str(path if path.is_absolute() else Path(EM_SHOWER_REPO_ROOT) / path)


EM_SHOWER_VERSION = EM_SHOWER_PRODUCTION["version"]
EM_SHOWER_FEATURE_ROOT = EM_SHOWER_PRODUCTION["outputs"]["feature_root"]
EM_SHOWER_PARTITION_ROOT = EM_SHOWER_PRODUCTION["outputs"]["partition_root"]
EM_SHOWER_HIST_ROOT = EM_SHOWER_PRODUCTION["outputs"]["hist_root"]
EM_SHOWER_WORK_ROOT = _em_shower_local_path(
    EM_SHOWER_PRODUCTION["outputs"]["workspace_root"]
)
EM_SHOWER_SOURCE_METADATA = list(EM_SHOWER_PRODUCTION["source_metadata"])
EM_SHOWER_PREPARED_METADATA = f"{EM_SHOWER_WORK_ROOT}/metadata/source_features.csv"
EM_SHOWER_PARTITION_METADATA = f"{EM_SHOWER_WORK_ROOT}/metadata/partitions.csv"
EM_SHOWER_PARTITION_ROW_DIR = f"{EM_SHOWER_WORK_ROOT}/metadata/partition_rows"
EM_SHOWER_READY_DIR = f"{EM_SHOWER_WORK_ROOT}/metadata/feature_ready"
EM_SHOWER_CLASSES = list(EM_SHOWER_PRODUCTION["classes"])
EM_SHOWER_REGIONS = list(EM_SHOWER_EVALUATION["comparisons"]["regions"])
EM_SHOWER_UNKNOWN_REGIONS = set(EM_SHOWER_REGIONS) - set(EM_SHOWER_REGIONS_CONFIG["regions"])
if EM_SHOWER_UNKNOWN_REGIONS:
    raise ValueError(
        "EM-shower evaluation requests undefined regions: "
        f"{sorted(EM_SHOWER_UNKNOWN_REGIONS)}"
    )
EM_SHOWER_FEATURES = list(EM_SHOWER_HISTS_CONFIG["features"])
EM_SHOWER_PAIRS_2D = list(EM_SHOWER_HISTS_CONFIG["features2d"])
EM_SHOWER_PARTITION_COUNT = int(EM_SHOWER_PRODUCTION["partition_count"])
EM_SHOWER_PARTITION_VERSION = EM_SHOWER_PARTITIONS_CONFIG["partition_version"]
EM_SHOWER_REGION_VERSION = EM_SHOWER_REGIONS_CONFIG["region_version"]
EM_SHOWER_HIST_VERSION = EM_SHOWER_HISTS_CONFIG["version"]
EM_SHOWER_EVAL_VERSION = EM_SHOWER_EVALUATION["version"]
EM_SHOWER_PLOT_ROOT = _em_shower_local_path(
    EM_SHOWER_EVALUATION["outputs"]["plot_dir"]
)
EM_SHOWER_OUTPUT_DIR_MARKER = (
    f"{EM_SHOWER_WORK_ROOT}/metadata/output_dirs/"
    f"{EM_SHOWER_HIST_VERSION}__{EM_SHOWER_EVAL_VERSION}.prepared"
)

EM_SHOWER_MUON_SOURCE_METADATA = str(EM_SHOWER_MUON["source_metadata"])
EM_SHOWER_MUON_FEATURE_ROOT = str(EM_SHOWER_MUON["outputs"]["feature_root"])
EM_SHOWER_MUON_PREPARED_METADATA = _em_shower_local_path(
    EM_SHOWER_MUON["outputs"]["prepared_metadata"]
)
EM_SHOWER_MUON_PARTITION_METADATA = _em_shower_local_path(
    EM_SHOWER_MUON["outputs"]["partition_metadata"]
)
EM_SHOWER_COMPARISON_METADATA = _em_shower_local_path(
    EM_SHOWER_MUON["outputs"]["comparison_metadata"]
)
EM_SHOWER_MUON_PARTITION_ROW_DIR = f"{EM_SHOWER_WORK_ROOT}/metadata/mc_muon_up/partition_rows"
EM_SHOWER_MUON_GROUP = "MC_muon_up"
EM_SHOWER_COMPARISON_GROUPS = list(
    EM_SHOWER_MUON_EVALUATION["comparisons"]["particle_groups"]
)
EM_SHOWER_MUON_HIST_FEATURES = [
    feature for feature in EM_SHOWER_FEATURES
    if feature not in set(EM_SHOWER_MUON["histograms"]["excluded_features"])
]
EM_SHOWER_MUON_HIST_PAIRS = [
    pair for pair in EM_SHOWER_PAIRS_2D
    if pair not in set(EM_SHOWER_MUON["histograms"]["excluded_pairs"])
]
EM_SHOWER_MUON_PLOT_FEATURES = list(EM_SHOWER_MUON["plot_features"])
EM_SHOWER_MUON_EVAL_VERSION = str(EM_SHOWER_MUON_EVALUATION["version"])
EM_SHOWER_MUON_PLOT_ROOT = _em_shower_local_path(
    EM_SHOWER_MUON_EVALUATION["outputs"]["plot_dir"]
)
EM_SHOWER_MUON_OUTPUT_DIR_MARKER = (
    f"{EM_SHOWER_WORK_ROOT}/metadata/output_dirs/"
    f"{EM_SHOWER_HIST_VERSION}__{EM_SHOWER_MUON_EVAL_VERSION}.prepared"
)


def _em_shower_resolve(row, column):
    value = str(row.get(column, ""))
    if value.startswith("root://") or os.path.isabs(value):
        return value
    return f"{str(row['output_base_path']).rstrip('/')}/{value.lstrip('/')}"


def _em_shower_load_sources():
    records = {}
    for metadata_path in EM_SHOWER_SOURCE_METADATA:
        frame = pd.read_csv(metadata_path)
        for _, row in frame.iterrows():
            record = row.to_dict()
            key = (str(record["subfolder"]), str(int(record["partition"])))
            if key in records:
                raise ValueError(f"Duplicate EM-shower source key: {key}")
            record["cutflow_path"] = _em_shower_resolve(record, "nueAnalysisFilter_path")
            records[key] = record
    return records


EM_SHOWER_SOURCE_RECORDS = _em_shower_load_sources()
EM_SHOWER_SUBFOLDERS = sorted({key[0] for key in EM_SHOWER_SOURCE_RECORDS})
EM_SHOWER_CLASS_SUBFOLDER = {
    particle: str(
        EM_SHOWER_PARTITIONS_CONFIG["particle_group_rules"][particle]["source_subfolders"][0]
    )
    for particle in EM_SHOWER_CLASSES
}


def _em_shower_record(wildcards):
    key = (str(wildcards.subfolder), str(int(wildcards.partition)))
    if key not in EM_SHOWER_SOURCE_RECORDS:
        raise ValueError(f"Unknown EM-shower source key: {key}")
    return EM_SHOWER_SOURCE_RECORDS[key]


def _em_shower_feature_path(subfolder, partition):
    filename = EM_SHOWER_PRODUCTION["feature"]["filename_template"].format(
        subfolder=subfolder, partition=int(partition)
    )
    return f"{EM_SHOWER_FEATURE_ROOT}/{subfolder}/{int(partition)}/{filename}"


def _em_shower_marker_path(subfolder, partition):
    return f"{_em_shower_feature_path(subfolder, partition)}.validated"


EM_SHOWER_FEATURE_TARGETS = [
    _em_shower_feature_path(subfolder, partition)
    for subfolder, partition in EM_SHOWER_SOURCE_RECORDS
]
EM_SHOWER_MARKER_TARGETS = [f"{path}.validated" for path in EM_SHOWER_FEATURE_TARGETS]


def _em_shower_partition_id(region, particle_group, part_index):
    return EM_SHOWER_PARTITIONS_CONFIG["grouping"]["output_partition_id_template"].format(
        region=region, particle_group=particle_group, part_index=int(part_index)
    )


def _em_shower_partition_path(region, particle_group, part_index):
    partition_id = _em_shower_partition_id(region, particle_group, part_index)
    return (
        f"{EM_SHOWER_PARTITION_ROOT}/{EM_SHOWER_PARTITION_VERSION}/"
        f"{EM_SHOWER_REGION_VERSION}/{region}/{particle_group}/"
        f"scifi_cluster_partition_{partition_id}.root"
    )


def _em_shower_partition_row(region, particle_group, part_index):
    return f"{EM_SHOWER_PARTITION_ROW_DIR}/{_em_shower_partition_id(region, particle_group, part_index)}.csv"


EM_SHOWER_PARTITION_TARGETS = [
    _em_shower_partition_path(region, particle, part_index)
    for region in EM_SHOWER_REGIONS
    for particle in EM_SHOWER_CLASSES
    for part_index in range(1, EM_SHOWER_PARTITION_COUNT + 1)
]
EM_SHOWER_PARTITION_ROWS = [
    _em_shower_partition_row(region, particle, part_index)
    for region in EM_SHOWER_REGIONS
    for particle in EM_SHOWER_CLASSES
    for part_index in range(1, EM_SHOWER_PARTITION_COUNT + 1)
]


def _em_shower_subfolder_markers(wildcards):
    subfolder = str(wildcards.subfolder)
    return [
        _em_shower_marker_path(folder, partition)
        for folder, partition in EM_SHOWER_SOURCE_RECORDS
        if folder == subfolder
    ]


def _em_shower_class_ready(wildcards):
    subfolder = EM_SHOWER_CLASS_SUBFOLDER[str(wildcards.particle_group)]
    return f"{EM_SHOWER_READY_DIR}/{subfolder}.validated"


def _em_shower_hist_path(region, particle_group, feature):
    return (
        f"{EM_SHOWER_HIST_ROOT}/{EM_SHOWER_PARTITION_VERSION}_{EM_SHOWER_REGION_VERSION}/"
        f"{EM_SHOWER_HIST_VERSION}__{EM_SHOWER_EVAL_VERSION}/{feature}/none/none/"
        f"{region}/{particle_group}.root"
    )


def _em_shower_hist2d_path(region, particle_group, pair):
    return (
        f"{EM_SHOWER_HIST_ROOT}/2d/{EM_SHOWER_HIST_VERSION}__{EM_SHOWER_EVAL_VERSION}/"
        f"{pair}/{region}/{particle_group}.root"
    )


def _em_shower_plot2d_path(pair, region, filename="plot.pdf"):
    return (
        f"{EM_SHOWER_PLOT_ROOT}/2d/{EM_SHOWER_HIST_VERSION}__{EM_SHOWER_EVAL_VERSION}/"
        f"{pair}/{region}/{filename}"
    )


def _em_shower_group_partitions(wildcards):
    return [
        _em_shower_partition_path(wildcards.region, wildcards.particle_group, part_index)
        for part_index in range(1, EM_SHOWER_PARTITION_COUNT + 1)
    ]


def _em_shower_plot_hist_inputs(wildcards):
    return [
        _em_shower_hist_path(region, particle, wildcards.feature)
        for region in EM_SHOWER_REGIONS
        for particle in EM_SHOWER_CLASSES
    ]


def _em_shower_plot_path(feature):
    return (
        f"{EM_SHOWER_PLOT_ROOT}/{EM_SHOWER_PARTITION_VERSION}_{EM_SHOWER_REGION_VERSION}/"
        f"{EM_SHOWER_HIST_VERSION}__{EM_SHOWER_EVAL_VERSION}/{feature}/none/none/plot.pdf"
    )


def _em_shower_output_directories():
    directories = set()
    for feature in EM_SHOWER_FEATURES:
        directories.add(str(Path(_em_shower_plot_path(feature)).parent))
        for region in EM_SHOWER_REGIONS:
            for particle in EM_SHOWER_CLASSES:
                directories.add(
                    str(Path(_em_shower_hist_path(region, particle, feature)).parent)
                )
    for pair in EM_SHOWER_PAIRS_2D:
        for region in EM_SHOWER_REGIONS:
            directories.add(str(Path(_em_shower_plot2d_path(pair, region)).parent))
            for particle in EM_SHOWER_CLASSES:
                directories.add(
                    str(Path(_em_shower_hist2d_path(region, particle, pair)).parent)
                )
    return sorted(directories)


EM_SHOWER_OUTPUT_DIRECTORIES = _em_shower_output_directories()


localrules: em_shower_prepare_evaluation_output_dirs_v1


rule em_shower_prepare_evaluation_output_dirs_v1:
    input:
        partition_config=EM_SHOWER_PARTITION_CONFIG,
        region_config=EM_SHOWER_REGION_CONFIG,
        hist_config=EM_SHOWER_HIST_CONFIG,
        eval_config=EM_SHOWER_EVAL_CONFIG,
    output:
        EM_SHOWER_OUTPUT_DIR_MARKER
    threads: 1
    resources:
        runtime=10 * 60,
        mem_mb=500,
        disk_mb=100,
        nvidia_gpu=0
    run:
        for directory in EM_SHOWER_OUTPUT_DIRECTORIES:
            Path(directory).mkdir(parents=True, exist_ok=True)
        marker = Path(output[0])
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            "\n".join(EM_SHOWER_OUTPUT_DIRECTORIES) + "\n",
            encoding="utf-8",
        )


ruleorder: em_shower_mc_neutrino_feature_v1 > process_features


rule em_shower_mc_neutrino_feature_v1:
    input:
        digi=lambda wc: _em_shower_record(wc)["digi_path"],
        geo=lambda wc: _em_shower_record(wc)["geo_path"],
        source_manifest=EM_SHOWER_PREPARED_METADATA,
        converter=f"{EM_SHOWER_REPO_ROOT}/convertData/digi_2_features.py",
        selection=f"{EM_SHOWER_REPO_ROOT}/convertData/sample_selection.py",
        clustering=f"{EM_SHOWER_REPO_ROOT}/convertData/scifi_plane_clustering.py",
        validator=f"{EM_SHOWER_REPO_ROOT}/convertData/validate_scifi_cluster_feature_file.py",
        config=EM_SHOWER_PRODUCTION_CONFIG,
    params:
        cutflow=lambda wc: _em_shower_record(wc)["cutflow_path"],
    output:
        feature=(
            f"{EM_SHOWER_FEATURE_ROOT}/{{subfolder}}/{{partition}}/"
            "feature_scifi_cluster_v1_MC_neutrino_{subfolder}_{partition}.root"
        ),
        marker=(
            f"{EM_SHOWER_FEATURE_ROOT}/{{subfolder}}/{{partition}}/"
            "feature_scifi_cluster_v1_MC_neutrino_{subfolder}_{partition}.root.validated"
        ),
    wildcard_constraints:
        subfolder="2024_ve|2024_vm",
        partition=r"\d+",
    threads: 1
    resources:
        runtime=60 * 60,
        mem_mb=2500,
        disk_mb=3000,
        nvidia_gpu=0
    shell:
        r"""
        # Avoid Conda/LCG conflicts with the SNDSW Python and ROOT stack.
        export PATH=$(echo "$PATH" | tr ':' '\n' \
            | grep -v 'miniconda3' \
            | grep -v '/cvmfs/sft.cern.ch/lcg/' \
            | tr '\n' ':' | sed 's/:$//')
        unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH

        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        set +u
        source {EM_SHOWER_REPO_ROOT}/snakemake/envs/env_sndsw_nue.sh
        export PYTHONPATH="$SNDSW_ROOT:${{PYTHONPATH:-}}"
        set -euo pipefail

        hostname
        pwd
        command -v python3
        python3 --version
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python -c 'import ROOT, SndlhcGeo; print(ROOT.__version__)'

        tmp_parent="${{TMPDIR:-/tmp}}"
        tmp_dir=$(mktemp -d "${{tmp_parent}}/em_shower_feature.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        tmp_feature="${{tmp_dir}}/feature.root"
        tmp_marker="${{tmp_dir}}/validated"

        test -f "{params.cutflow}"
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.converter} \
            -p {params.cutflow} \
            -d {input.digi} \
            -g {input.geo} \
            -o "$tmp_feature" \
            -t MC_neutrino \
            --tree-name cbmsim
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.validator} "$tmp_feature"
        touch "$tmp_marker"
        mkdir -p "$(dirname "{output.feature}")"
        xrdcp -f "$tmp_feature" "{output.feature}"
        xrdcp -f "$tmp_marker" "{output.marker}"
        """


rule em_shower_mc_neutrino_features_v1:
    input:
        [f"{EM_SHOWER_READY_DIR}/{subfolder}.validated" for subfolder in EM_SHOWER_SUBFOLDERS]


rule em_shower_mc_neutrino_subfolder_ready_v1:
    input:
        _em_shower_subfolder_markers
    output:
        f"{EM_SHOWER_READY_DIR}/{{subfolder}}.validated"
    wildcard_constraints:
        subfolder="2024_ve|2024_vm"
    resources:
        runtime=5 * 60,
        mem_mb=500,
        disk_mb=100,
        nvidia_gpu=0,
    shell:
        r"""
        mkdir -p "$(dirname "{output}")"
        touch {output}
        """


rule em_shower_prepare_source_metadata_v1:
    input:
        sources=EM_SHOWER_SOURCE_METADATA,
        config=EM_SHOWER_PRODUCTION_CONFIG,
        script=f"{EM_SHOWER_REPO_ROOT}/evaluation_region_partitions/prepare_em_shower_neutrino_metadata.py",
    output:
        EM_SHOWER_PREPARED_METADATA
    resources:
        runtime=5 * 60,
        mem_mb=1000,
        disk_mb=1000,
        nvidia_gpu=0,
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {EM_SHOWER_REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        python3 {input.script} --config {input.config} --output {output}
        """


rule em_shower_neutrino_partition_v1:
    input:
        metadata=EM_SHOWER_PREPARED_METADATA,
        ready=_em_shower_class_ready,
        region_config=EM_SHOWER_REGION_CONFIG,
        partition_config=EM_SHOWER_PARTITION_CONFIG,
        script=f"{EM_SHOWER_REPO_ROOT}/convertData/build_region_partitions.py",
    output:
        feature=(
            f"{EM_SHOWER_PARTITION_ROOT}/{EM_SHOWER_PARTITION_VERSION}/"
            f"{EM_SHOWER_REGION_VERSION}/{{region}}/{{particle_group}}/"
            "scifi_cluster_partition_{region}__{particle_group}__part{part_index}.root"
        ),
        row=f"{EM_SHOWER_PARTITION_ROW_DIR}/{{region}}__{{particle_group}}__part{{part_index}}.csv",
    wildcard_constraints:
        region="|".join(EM_SHOWER_REGIONS),
        particle_group="|".join(EM_SHOWER_CLASSES),
        part_index=r"\d{3}",
    threads: int(EM_SHOWER_PARTITIONS_CONFIG["performance"]["rdf_threads"])
    resources:
        runtime=60 * 60,
        mem_mb=5000,
        disk_mb=4000,
        nvidia_gpu=0
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {EM_SHOWER_REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        hostname
        pwd
        command -v python3
        python3 --version
        python3 -c 'import ROOT, pandas, yaml; print(ROOT.__version__)'

        tmp_parent="${{TMPDIR:-/tmp}}"
        tmp_dir=$(mktemp -d "${{tmp_parent}}/em_shower_partition.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        tmp_feature="${{tmp_dir}}/feature.root"
        tmp_row="${{tmp_dir}}/row.csv"
        python3 {input.script} \
            -m {input.metadata} \
            -r {input.region_config} \
            -p {input.partition_config} \
            -o {EM_SHOWER_PARTITION_ROOT} \
            --output-metadata "$tmp_row" \
            --output-feature "$tmp_feature" \
            --metadata-feature-output "{output.feature}" \
            --only-regions {wildcards.region} \
            --only-particles {wildcards.particle_group} \
            --only-part-index {wildcards.part_index} \
            --rdf-threads {threads} \
            --product feature
        mkdir -p "$(dirname "{output.feature}")" "$(dirname "{output.row}")"
        xrdcp -f "$tmp_feature" "{output.feature}"
        cp -f "$tmp_row" "{output.row}"
        """


rule em_shower_neutrino_partitions_v1:
    input:
        EM_SHOWER_PARTITION_TARGETS,
        EM_SHOWER_PARTITION_ROWS,


rule em_shower_combine_partition_metadata_v1:
    input:
        rows=EM_SHOWER_PARTITION_ROWS,
        script=f"{EM_SHOWER_REPO_ROOT}/evaluation_region_partitions/combine_partition_metadata.py",
    output:
        EM_SHOWER_PARTITION_METADATA
    threads: 1
    resources:
        runtime=30 * 60,
        mem_mb=1000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {EM_SHOWER_REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        python3 {input.script} --input {input.rows} --output {output}
        """


rule em_shower_neutrino_hist_v1:
    input:
        output_dirs=EM_SHOWER_OUTPUT_DIR_MARKER,
        metadata=EM_SHOWER_PARTITION_METADATA,
        partitions=_em_shower_group_partitions,
        hist_config=EM_SHOWER_HIST_CONFIG,
        eval_config=EM_SHOWER_EVAL_CONFIG,
        script=f"{EM_SHOWER_REPO_ROOT}/evaluation_region_partitions/make_group_hist.py",
    output:
        hist=(
            f"{EM_SHOWER_HIST_ROOT}/{EM_SHOWER_PARTITION_VERSION}_{EM_SHOWER_REGION_VERSION}/"
            f"{EM_SHOWER_HIST_VERSION}__{EM_SHOWER_EVAL_VERSION}/{{feature}}/none/none/"
            "{region}/{particle_group}.root"
        )
    wildcard_constraints:
        feature="|".join(EM_SHOWER_FEATURES),
        region="|".join(EM_SHOWER_REGIONS),
        particle_group="|".join(EM_SHOWER_CLASSES),
    threads: 1
    resources:
        runtime=45 * 60,
        mem_mb=3000,
        disk_mb=3000,
        nvidia_gpu=0
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {EM_SHOWER_REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        python3 {input.script} \
            --metadata-csv {input.metadata} \
            --hist-config {input.hist_config} \
            --eval-config {input.eval_config} \
            --region {wildcards.region} \
            --particle-group {wildcards.particle_group} \
            --feature {wildcards.feature} \
            --base-cut-key none \
            --extra-cut-key none \
            --output {output.hist} \
            --repo-root {EM_SHOWER_REPO_ROOT}
        """


rule em_shower_neutrino_hists_v1:
    input:
        [
            _em_shower_hist_path(region, particle, feature)
            for region in EM_SHOWER_REGIONS
            for particle in EM_SHOWER_CLASSES
            for feature in EM_SHOWER_FEATURES
        ]


rule em_shower_neutrino_plot_v1:
    input:
        output_dirs=EM_SHOWER_OUTPUT_DIR_MARKER,
        metadata=EM_SHOWER_PARTITION_METADATA,
        hists=_em_shower_plot_hist_inputs,
        hist_config=EM_SHOWER_HIST_CONFIG,
        eval_config=EM_SHOWER_EVAL_CONFIG,
        script=f"{EM_SHOWER_REPO_ROOT}/evaluation_region_partitions/plot_comparison.py",
    output:
        pdf=(
            f"{EM_SHOWER_PLOT_ROOT}/{EM_SHOWER_PARTITION_VERSION}_{EM_SHOWER_REGION_VERSION}/"
            f"{EM_SHOWER_HIST_VERSION}__{EM_SHOWER_EVAL_VERSION}/{{feature}}/none/none/plot.pdf"
        ),
        summary=(
            f"{EM_SHOWER_PLOT_ROOT}/{EM_SHOWER_PARTITION_VERSION}_{EM_SHOWER_REGION_VERSION}/"
            f"{EM_SHOWER_HIST_VERSION}__{EM_SHOWER_EVAL_VERSION}/{{feature}}/none/none/summary.csv"
        ),
    wildcard_constraints:
        feature="|".join(EM_SHOWER_FEATURES),
    threads: 1
    resources:
        runtime=20 * 60,
        mem_mb=2000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {EM_SHOWER_REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        python3 {input.script} \
            --metadata-csv {input.metadata} \
            --hist-config {input.hist_config} \
            --eval-config {input.eval_config} \
            --feature {wildcards.feature} \
            --base-cut-key none \
            --extra-cut-key none \
            --partition-version {EM_SHOWER_PARTITION_VERSION} \
            --region-version {EM_SHOWER_REGION_VERSION} \
            --repo-root {EM_SHOWER_REPO_ROOT}
        """


rule em_shower_neutrino_hist2d_v1:
    input:
        output_dirs=EM_SHOWER_OUTPUT_DIR_MARKER,
        metadata=EM_SHOWER_PARTITION_METADATA,
        partitions=_em_shower_group_partitions,
        hist_config=EM_SHOWER_HIST_CONFIG,
        eval_config=EM_SHOWER_EVAL_CONFIG,
        script=f"{EM_SHOWER_REPO_ROOT}/evaluation_region_partitions/make_group_hist2d.py",
    output:
        hist=(
            f"{EM_SHOWER_HIST_ROOT}/2d/{EM_SHOWER_HIST_VERSION}__{EM_SHOWER_EVAL_VERSION}/"
            "{pair}/{region}/{particle_group}.root"
        )
    wildcard_constraints:
        pair="|".join(EM_SHOWER_PAIRS_2D),
        region="|".join(EM_SHOWER_REGIONS),
        particle_group="|".join(EM_SHOWER_CLASSES),
    threads: 1
    resources:
        runtime=45 * 60,
        mem_mb=3000,
        disk_mb=3000,
        nvidia_gpu=0
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {EM_SHOWER_REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        python3 {input.script} \
            --metadata-csv {input.metadata} \
            --hist-config {input.hist_config} \
            --eval-config {input.eval_config} \
            --region {wildcards.region} \
            --particle-group {wildcards.particle_group} \
            --pair {wildcards.pair} \
            --output {output.hist} \
            --repo-root {EM_SHOWER_REPO_ROOT}
        """


def _em_shower_hist2d_plot_inputs(wildcards):
    return [
        _em_shower_hist2d_path(wildcards.region, particle, wildcards.pair)
        for particle in EM_SHOWER_CLASSES
    ]


rule em_shower_neutrino_plot2d_v1:
    input:
        output_dirs=EM_SHOWER_OUTPUT_DIR_MARKER,
        hists=_em_shower_hist2d_plot_inputs,
        hist_config=EM_SHOWER_HIST_CONFIG,
        script=f"{EM_SHOWER_REPO_ROOT}/evaluation_region_partitions/plot_group_hist2d.py",
    output:
        pdf=(
            f"{EM_SHOWER_PLOT_ROOT}/2d/{EM_SHOWER_HIST_VERSION}__{EM_SHOWER_EVAL_VERSION}/"
            "{pair}/{region}/plot.pdf"
        ),
        summary=(
            f"{EM_SHOWER_PLOT_ROOT}/2d/{EM_SHOWER_HIST_VERSION}__{EM_SHOWER_EVAL_VERSION}/"
            "{pair}/{region}/summary.csv"
        ),
    params:
        hist_args=lambda wc, input: " ".join(
            f"--hist {particle}={path}" for particle, path in zip(EM_SHOWER_CLASSES, input.hists)
        ),
    wildcard_constraints:
        pair="|".join(EM_SHOWER_PAIRS_2D),
        region="|".join(EM_SHOWER_REGIONS),
    threads: 1
    resources:
        runtime=20 * 60,
        mem_mb=2000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {EM_SHOWER_REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        python3 {input.script} \
            --hist-config {input.hist_config} \
            --pair {wildcards.pair} \
            --region {wildcards.region} \
            {params.hist_args} \
            --output {output.pdf} \
            --summary {output.summary}
        """


rule em_shower_neutrino_plots_v1:
    input:
        [_em_shower_plot_path(feature) for feature in EM_SHOWER_FEATURES],
        [
            _em_shower_plot2d_path(pair, region)
            for pair in EM_SHOWER_PAIRS_2D
            for region in EM_SHOWER_REGIONS
        ],


rule em_shower_neutrino_evaluation_v1:
    input:
        rules.em_shower_neutrino_plots_v1.input


# MC-muon-up extension. The source catalogue is a checkpoint because the
# generated MC metadata is intentionally not required for parsing the base
# four-class neutrino workflow.

checkpoint em_shower_prepare_mc_muon_up_metadata_v1:
    input:
        source=EM_SHOWER_MUON_SOURCE_METADATA,
        config=EM_SHOWER_MUON_CONFIG,
        script=(
            f"{EM_SHOWER_REPO_ROOT}/evaluation_region_partitions/"
            "prepare_em_shower_mc_muon_up_metadata.py"
        ),
    output:
        manifest=EM_SHOWER_MUON_PREPARED_METADATA
    threads: 1
    resources:
        runtime=10 * 60,
        mem_mb=1000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {EM_SHOWER_REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        python3 {input.script} --config {input.config} --output {output.manifest}
        """


def _em_shower_muon_manifest(wildcards):
    manifest = checkpoints.em_shower_prepare_mc_muon_up_metadata_v1.get().output.manifest
    return str(manifest)


def _em_shower_muon_records(wildcards):
    frame = pd.read_csv(_em_shower_muon_manifest(wildcards), keep_default_na=False)
    return frame.to_dict("records")


def _em_shower_muon_record(wildcards):
    matches = [
        row for row in _em_shower_muon_records(wildcards)
        if str(row["source_key"]) == str(wildcards.source_key)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one MC-muon source for {wildcards.source_key}, found {len(matches)}")
    record = matches[0]
    if str(record["production"]) != str(wildcards.production):
        raise ValueError(
            f"MC-muon production mismatch for {wildcards.source_key}: "
            f"{record['production']} != {wildcards.production}"
        )
    return record


def _em_shower_muon_feature_path(production, source_key):
    return (
        f"{EM_SHOWER_MUON_FEATURE_ROOT}/{production}/{source_key}/"
        f"feature_scifi_cluster_v1_MC_muon_up_{source_key}.root"
    )


def _em_shower_muon_feature_targets(wildcards):
    return [
        _em_shower_muon_feature_path(row["production"], row["source_key"])
        for row in _em_shower_muon_records(wildcards)
    ]


def _em_shower_muon_marker_targets(wildcards):
    return [f"{path}.validated" for path in _em_shower_muon_feature_targets(wildcards)]


rule em_shower_mc_muon_up_feature_v1:
    input:
        manifest=_em_shower_muon_manifest,
        digi=lambda wc: _em_shower_muon_record(wc)["digi_path"],
        geo=lambda wc: _em_shower_muon_record(wc)["geo_path"],
        converter=f"{EM_SHOWER_REPO_ROOT}/convertData/digi_2_features_mc_muon.py",
        feature_core=f"{EM_SHOWER_REPO_ROOT}/convertData/digi_2_features.py",
        selection=f"{EM_SHOWER_REPO_ROOT}/convertData/sample_selection.py",
        clustering=f"{EM_SHOWER_REPO_ROOT}/convertData/scifi_plane_clustering.py",
        validator=f"{EM_SHOWER_REPO_ROOT}/convertData/validate_scifi_cluster_feature_file.py",
        config=EM_SHOWER_MUON_CONFIG,
    params:
        tree=lambda wc: _em_shower_muon_record(wc)["input_tree"],
    output:
        feature=(
            f"{EM_SHOWER_MUON_FEATURE_ROOT}/{{production}}/{{source_key}}/"
            "feature_scifi_cluster_v1_MC_muon_up_{source_key}.root"
        ),
        marker=(
            f"{EM_SHOWER_MUON_FEATURE_ROOT}/{{production}}/{{source_key}}/"
            "feature_scifi_cluster_v1_MC_muon_up_{source_key}.root.validated"
        ),
    wildcard_constraints:
        production="|".join(EM_SHOWER_MUON["productions"]),
        source_key=r"[0-9a-f]{12}",
    threads: 1
    resources:
        runtime=60 * 60,
        mem_mb=2500,
        disk_mb=3000,
        nvidia_gpu=0
    shell:
        r"""
        export PATH=$(echo "$PATH" | tr ':' '\n' \
            | grep -v 'miniconda3' \
            | grep -v '/cvmfs/sft.cern.ch/lcg/' \
            | tr '\n' ':' | sed 's/:$//')
        unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {EM_SHOWER_REPO_ROOT}/snakemake/envs/env_sndsw_nue.sh
        export PYTHONPATH="$SNDSW_ROOT:${{PYTHONPATH:-}}"
        set -euo pipefail

        hostname
        pwd
        command -v python3
        python3 --version
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python -c 'import ROOT, SndlhcGeo; print(ROOT.__version__)'

        tmp_parent="${{TMPDIR:-/tmp}}"
        tmp_dir=$(mktemp -d "${{tmp_parent}}/em_shower_muon_feature.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        tmp_feature="${{tmp_dir}}/feature.root"
        tmp_marker="${{tmp_dir}}/validated"

        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.converter} \
            -d {input.digi} -g {input.geo} -o "$tmp_feature" -t MC_muon \
            --tree-name {params.tree} --selection-mode direct_input
        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python {input.validator} \
            "$tmp_feature" --required-branch density_sndsw_scifi \
            --required-branch scifi_cluster_qdc_per_hit \
            --required-branch pdgCode --required-branch count_scifi
        touch "$tmp_marker"
        mkdir -p "$(dirname "{output.feature}")"
        xrdcp -f "$tmp_feature" "{output.feature}"
        xrdcp -f "$tmp_marker" "{output.marker}"
        """


rule em_shower_mc_muon_up_features_v1:
    input:
        _em_shower_muon_marker_targets


def _em_shower_muon_partition_path(part_index):
    partition_id = _em_shower_partition_id("scifi35", EM_SHOWER_MUON_GROUP, part_index)
    return (
        f"{EM_SHOWER_PARTITION_ROOT}/{EM_SHOWER_PARTITION_VERSION}/"
        f"{EM_SHOWER_REGION_VERSION}/scifi35/{EM_SHOWER_MUON_GROUP}/"
        f"scifi_cluster_partition_{partition_id}.root"
    )


def _em_shower_muon_partition_row(part_index):
    return (
        f"{EM_SHOWER_MUON_PARTITION_ROW_DIR}/"
        f"scifi35__{EM_SHOWER_MUON_GROUP}__part{int(part_index):03d}.csv"
    )


EM_SHOWER_MUON_PARTITION_TARGETS = [
    _em_shower_muon_partition_path(index)
    for index in range(1, EM_SHOWER_PARTITION_COUNT + 1)
]
EM_SHOWER_MUON_PARTITION_ROWS = [
    _em_shower_muon_partition_row(index)
    for index in range(1, EM_SHOWER_PARTITION_COUNT + 1)
]


rule em_shower_mc_muon_up_partition_v1:
    input:
        metadata=_em_shower_muon_manifest,
        ready=_em_shower_muon_marker_targets,
        region_config=EM_SHOWER_REGION_CONFIG,
        partition_config=EM_SHOWER_PARTITION_CONFIG,
        script=f"{EM_SHOWER_REPO_ROOT}/convertData/build_region_partitions.py",
    output:
        feature=(
            f"{EM_SHOWER_PARTITION_ROOT}/{EM_SHOWER_PARTITION_VERSION}/"
            f"{EM_SHOWER_REGION_VERSION}/scifi35/{EM_SHOWER_MUON_GROUP}/"
            "scifi_cluster_partition_scifi35__MC_muon_up__part{part_index}.root"
        ),
        row=(
            f"{EM_SHOWER_MUON_PARTITION_ROW_DIR}/"
            "scifi35__MC_muon_up__part{part_index}.csv"
        ),
    wildcard_constraints:
        part_index=r"\d{3}",
    threads: int(EM_SHOWER_PARTITIONS_CONFIG["performance"]["rdf_threads"])
    resources:
        runtime=60 * 60,
        mem_mb=5000,
        disk_mb=4000,
        nvidia_gpu=0
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {EM_SHOWER_REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        tmp_parent="${{TMPDIR:-/tmp}}"
        tmp_dir=$(mktemp -d "${{tmp_parent}}/em_shower_muon_partition.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        python3 {input.script} -m {input.metadata} -r {input.region_config} \
            -p {input.partition_config} -o {EM_SHOWER_PARTITION_ROOT} \
            --output-metadata "${{tmp_dir}}/row.csv" \
            --output-feature "${{tmp_dir}}/feature.root" \
            --metadata-feature-output "{output.feature}" \
            --only-regions scifi35 --only-particles {EM_SHOWER_MUON_GROUP} \
            --only-part-index {wildcards.part_index} --rdf-threads {threads} \
            --product feature
        mkdir -p "$(dirname "{output.feature}")" "$(dirname "{output.row}")"
        xrdcp -f "${{tmp_dir}}/feature.root" "{output.feature}"
        cp -f "${{tmp_dir}}/row.csv" "{output.row}"
        """


rule em_shower_mc_muon_up_partitions_v1:
    input:
        EM_SHOWER_MUON_PARTITION_TARGETS,
        EM_SHOWER_MUON_PARTITION_ROWS,


rule em_shower_combine_mc_muon_up_partition_metadata_v1:
    input:
        rows=EM_SHOWER_MUON_PARTITION_ROWS,
        script=f"{EM_SHOWER_REPO_ROOT}/evaluation_region_partitions/combine_partition_metadata.py",
    output:
        EM_SHOWER_MUON_PARTITION_METADATA
    threads: 1
    resources:
        runtime=10 * 60,
        mem_mb=1000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {EM_SHOWER_REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        python3 {input.script} --input {input.rows} --output {output}
        """


rule em_shower_combine_neutrino_muon_up_metadata_v1:
    input:
        neutrino=EM_SHOWER_PARTITION_METADATA,
        muon=EM_SHOWER_MUON_PARTITION_METADATA,
        script=f"{EM_SHOWER_REPO_ROOT}/evaluation_region_partitions/combine_partition_metadata.py",
    output:
        EM_SHOWER_COMPARISON_METADATA
    threads: 1
    resources:
        runtime=10 * 60,
        mem_mb=1000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {EM_SHOWER_REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        python3 {input.script} --input {input.neutrino} {input.muon} --output {output}
        """


def _em_shower_muon_hist_path(feature):
    return _em_shower_hist_path("scifi35", EM_SHOWER_MUON_GROUP, feature)


def _em_shower_muon_hist2d_path(pair):
    return _em_shower_hist2d_path("scifi35", EM_SHOWER_MUON_GROUP, pair)


def _em_shower_muon_plot_path(feature, filename="plot.pdf"):
    return (
        f"{EM_SHOWER_MUON_PLOT_ROOT}/{EM_SHOWER_PARTITION_VERSION}_{EM_SHOWER_REGION_VERSION}/"
        f"{EM_SHOWER_HIST_VERSION}__{EM_SHOWER_MUON_EVAL_VERSION}/{feature}/"
        f"none/none/{filename}"
    )


def _em_shower_muon_output_directories():
    directories = {
        str(Path(_em_shower_muon_hist_path(feature)).parent)
        for feature in EM_SHOWER_MUON_HIST_FEATURES
    }
    directories.update(
        str(Path(_em_shower_muon_hist2d_path(pair)).parent)
        for pair in EM_SHOWER_MUON_HIST_PAIRS
    )
    directories.update(
        str(Path(_em_shower_muon_plot_path(feature)).parent)
        for feature in EM_SHOWER_MUON_PLOT_FEATURES
    )
    return sorted(directories)


EM_SHOWER_MUON_OUTPUT_DIRECTORIES = _em_shower_muon_output_directories()


localrules: em_shower_prepare_mc_muon_up_output_dirs_v1


rule em_shower_prepare_mc_muon_up_output_dirs_v1:
    input:
        muon_config=EM_SHOWER_MUON_CONFIG,
        hist_config=EM_SHOWER_HIST_CONFIG,
        eval_config=EM_SHOWER_MUON_EVAL_CONFIG,
    output:
        EM_SHOWER_MUON_OUTPUT_DIR_MARKER
    threads: 1
    resources:
        runtime=10 * 60,
        mem_mb=500,
        disk_mb=100,
        nvidia_gpu=0
    run:
        for directory in EM_SHOWER_MUON_OUTPUT_DIRECTORIES:
            Path(directory).mkdir(parents=True, exist_ok=True)
        marker = Path(output[0])
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("\n".join(EM_SHOWER_MUON_OUTPUT_DIRECTORIES) + "\n", encoding="utf-8")


rule em_shower_mc_muon_up_hist_v1:
    input:
        output_dirs=EM_SHOWER_MUON_OUTPUT_DIR_MARKER,
        metadata=EM_SHOWER_MUON_PARTITION_METADATA,
        partitions=EM_SHOWER_MUON_PARTITION_TARGETS,
        hist_config=EM_SHOWER_HIST_CONFIG,
        eval_config=EM_SHOWER_EVAL_CONFIG,
        script=f"{EM_SHOWER_REPO_ROOT}/evaluation_region_partitions/make_group_hist.py",
    output:
        hist=(
            f"{EM_SHOWER_HIST_ROOT}/{EM_SHOWER_PARTITION_VERSION}_{EM_SHOWER_REGION_VERSION}/"
            f"{EM_SHOWER_HIST_VERSION}__{EM_SHOWER_EVAL_VERSION}/{{feature}}/none/none/"
            f"scifi35/{EM_SHOWER_MUON_GROUP}.root"
        ),
    wildcard_constraints:
        feature="|".join(EM_SHOWER_MUON_HIST_FEATURES),
    threads: 1
    resources:
        runtime=45 * 60,
        mem_mb=3000,
        disk_mb=3000,
        nvidia_gpu=0
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {EM_SHOWER_REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        python3 {input.script} --metadata-csv {input.metadata} \
            --hist-config {input.hist_config} --eval-config {input.eval_config} \
            --region scifi35 --particle-group {EM_SHOWER_MUON_GROUP} \
            --feature {wildcards.feature} --base-cut-key none --extra-cut-key none \
            --output {output.hist} --repo-root {EM_SHOWER_REPO_ROOT}
        """


rule em_shower_mc_muon_up_hist2d_v1:
    input:
        output_dirs=EM_SHOWER_MUON_OUTPUT_DIR_MARKER,
        metadata=EM_SHOWER_MUON_PARTITION_METADATA,
        partitions=EM_SHOWER_MUON_PARTITION_TARGETS,
        hist_config=EM_SHOWER_HIST_CONFIG,
        eval_config=EM_SHOWER_EVAL_CONFIG,
        script=f"{EM_SHOWER_REPO_ROOT}/evaluation_region_partitions/make_group_hist2d.py",
    output:
        hist=(
            f"{EM_SHOWER_HIST_ROOT}/2d/{EM_SHOWER_HIST_VERSION}__{EM_SHOWER_EVAL_VERSION}/"
            f"{{pair}}/scifi35/{EM_SHOWER_MUON_GROUP}.root"
        ),
    wildcard_constraints:
        pair="|".join(EM_SHOWER_MUON_HIST_PAIRS),
    threads: 1
    resources:
        runtime=45 * 60,
        mem_mb=3000,
        disk_mb=3000,
        nvidia_gpu=0
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {EM_SHOWER_REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        python3 {input.script} --metadata-csv {input.metadata} \
            --hist-config {input.hist_config} --eval-config {input.eval_config} \
            --region scifi35 --particle-group {EM_SHOWER_MUON_GROUP} \
            --pair {wildcards.pair} --output {output.hist} \
            --repo-root {EM_SHOWER_REPO_ROOT}
        """


rule em_shower_mc_muon_up_hists_v1:
    input:
        [_em_shower_muon_hist_path(feature) for feature in EM_SHOWER_MUON_HIST_FEATURES],
        [_em_shower_muon_hist2d_path(pair) for pair in EM_SHOWER_MUON_HIST_PAIRS],


def _em_shower_neutrino_muon_plot_inputs(wildcards):
    return [
        _em_shower_hist_path("scifi35", group, wildcards.feature)
        for group in EM_SHOWER_COMPARISON_GROUPS
    ]


rule em_shower_neutrino_muon_up_plot_v1:
    input:
        output_dirs=EM_SHOWER_MUON_OUTPUT_DIR_MARKER,
        metadata=EM_SHOWER_COMPARISON_METADATA,
        hists=_em_shower_neutrino_muon_plot_inputs,
        hist_config=EM_SHOWER_HIST_CONFIG,
        eval_config=EM_SHOWER_MUON_EVAL_CONFIG,
        script=f"{EM_SHOWER_REPO_ROOT}/evaluation_region_partitions/plot_comparison.py",
    output:
        pdf=(
            f"{EM_SHOWER_MUON_PLOT_ROOT}/{EM_SHOWER_PARTITION_VERSION}_{EM_SHOWER_REGION_VERSION}/"
            f"{EM_SHOWER_HIST_VERSION}__{EM_SHOWER_MUON_EVAL_VERSION}/{{feature}}/"
            "none/none/plot.pdf"
        ),
        summary=(
            f"{EM_SHOWER_MUON_PLOT_ROOT}/{EM_SHOWER_PARTITION_VERSION}_{EM_SHOWER_REGION_VERSION}/"
            f"{EM_SHOWER_HIST_VERSION}__{EM_SHOWER_MUON_EVAL_VERSION}/{{feature}}/"
            "none/none/summary.csv"
        ),
    wildcard_constraints:
        feature="|".join(EM_SHOWER_MUON_PLOT_FEATURES),
    threads: 1
    resources:
        runtime=20 * 60,
        mem_mb=2000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        set +u
        source {EM_SHOWER_REPO_ROOT}/snakemake/envs/env_lcg.sh
        set -euo pipefail
        python3 {input.script} --metadata-csv {input.metadata} \
            --hist-config {input.hist_config} --eval-config {input.eval_config} \
            --feature {wildcards.feature} --base-cut-key none --extra-cut-key none \
            --partition-version {EM_SHOWER_PARTITION_VERSION} \
            --region-version {EM_SHOWER_REGION_VERSION} \
            --repo-root {EM_SHOWER_REPO_ROOT}
        """


rule em_shower_neutrino_muon_up_plots_v1:
    input:
        [_em_shower_muon_plot_path(feature) for feature in EM_SHOWER_MUON_PLOT_FEATURES],
        [_em_shower_muon_plot_path(feature, "summary.csv") for feature in EM_SHOWER_MUON_PLOT_FEATURES],


rule em_shower_neutrino_muon_up_comparison_v1:
    input:
        rules.em_shower_mc_muon_up_features_v1.input,
        rules.em_shower_mc_muon_up_partitions_v1.input,
        rules.em_shower_mc_muon_up_hists_v1.input,
        rules.em_shower_neutrino_muon_up_plots_v1.input,
