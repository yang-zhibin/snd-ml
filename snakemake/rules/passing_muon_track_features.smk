PASSING_MUON_BASE_METADATA = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated/"
    "passing_muon_real_data_2024_metadata.csv"
)
PASSING_MUON_TRACK_METADATA = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated/"
    "passing_muon_track_features_v1_metadata.csv"
)
PASSING_MUON_TRACK_VERSION = "passing_muon_track_features_v1"
PASSING_MUON_TRACK_SCHEMA = "flat_v1"
PASSING_MUON_TRACK_V2_METADATA = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/updated/"
    "passing_muon_track_features_v2_metadata.csv"
)
PASSING_MUON_TRACK_V2_VERSION = "passing_muon_track_features_v2"
PASSING_MUON_TRACK_V2_SCHEMA = "flat_v2"


def _passing_muon_track_table(version, schema):
    if not os.path.isfile(PASSING_MUON_BASE_METADATA):
        return pd.DataFrame()
    table = pd.read_csv(PASSING_MUON_BASE_METADATA)
    table = table.copy()
    table["track_feature_schema"] = schema
    table["track_feature_path"] = table["reco_path"].map(
        lambda path: f"featureTracks_{Path(path).stem}.root"
    )
    table["track_output_base_path"] = table["partition"].map(
        lambda partition: (
            "/eos/experiment/sndlhc/users/zhibin/"
            f"{version}/2024/{partition}"
        )
    )
    table["track_feature_full_path"] = (
        table["track_output_base_path"].str.rstrip("/")
        + "/"
        + table["track_feature_path"]
    )
    return table


PASSING_MUON_TRACK_TABLE = _passing_muon_track_table(
    PASSING_MUON_TRACK_VERSION, PASSING_MUON_TRACK_SCHEMA
)
PASSING_MUON_TRACK_V2_TABLE = _passing_muon_track_table(
    PASSING_MUON_TRACK_V2_VERSION, PASSING_MUON_TRACK_V2_SCHEMA
)
PASSING_MUON_TRACK_ALL_TABLE = pd.concat(
    [PASSING_MUON_TRACK_TABLE, PASSING_MUON_TRACK_V2_TABLE], ignore_index=True
)


def passing_muon_track_targets():
    if PASSING_MUON_TRACK_TABLE.empty:
        return []
    return PASSING_MUON_TRACK_TABLE["track_feature_full_path"].tolist()


def passing_muon_track_v2_targets():
    if PASSING_MUON_TRACK_V2_TABLE.empty:
        return []
    return PASSING_MUON_TRACK_V2_TABLE["track_feature_full_path"].tolist()


def passing_muon_track_value(full_path, column):
    rows = PASSING_MUON_TRACK_ALL_TABLE.loc[
        PASSING_MUON_TRACK_ALL_TABLE["track_feature_full_path"] == str(full_path)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one passing-muon track row for {full_path}, found {len(rows)}"
        )
    return str(rows.iloc[0][column])


rule passing_muon_track_metadata:
    input:
        base=PASSING_MUON_BASE_METADATA,
        script=(
            f"{PERSONAL_WORK_SPACE}/snakemake/metadata/"
            "make_passing_muon_track_metadata.py"
        ),
    output:
        PASSING_MUON_TRACK_METADATA
    shell:
        """
        python3 {input.script} --input {input.base} --output {output}
        """


rule passing_muon_track_v2_metadata:
    input:
        base=PASSING_MUON_BASE_METADATA,
        script=(
            f"{PERSONAL_WORK_SPACE}/snakemake/metadata/"
            "make_passing_muon_track_metadata.py"
        ),
    output:
        PASSING_MUON_TRACK_V2_METADATA
    shell:
        """
        python3 {input.script} --input {input.base} --output {output} \
            --version {PASSING_MUON_TRACK_V2_VERSION} \
            --schema {PASSING_MUON_TRACK_V2_SCHEMA}
        """


rule process_passing_muon_track_features:
    input:
        reco=lambda wildcards: passing_muon_track_value(
            wildcards.track_feature_path, "reco_path"
        ),
        geo=lambda wildcards: passing_muon_track_value(
            wildcards.track_feature_path, "geo_path"
        ),
        converter=f"{PERSONAL_WORK_SPACE}/convertData/digi_2_features.py",
        extractor=f"{PERSONAL_WORK_SPACE}/convertData/reco_track_features.py",
        selection=f"{PERSONAL_WORK_SPACE}/convertData/sample_selection.py",
        scifi_clustering=f"{PERSONAL_WORK_SPACE}/convertData/scifi_plane_clustering.py",
        validator=(
            f"{PERSONAL_WORK_SPACE}/convertData/"
            "validate_reco_track_feature_file.py"
        ),
    params:
        tree=lambda wildcards: passing_muon_track_value(
            wildcards.track_feature_path, "input_tree"
        ),
        schema=lambda wildcards: passing_muon_track_value(
            wildcards.track_feature_path, "track_feature_schema"
        ),
    output:
        track_feature_path="{track_feature_path}"
    threads: 1
    resources:
        runtime=60 * 60,
        mem_mb=2500,
        disk_mb=2500,
        nvidia_gpu=0
    shell:
        r"""
        export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':' | sed 's/:$//')
        set +u
        source {env_script_sndsw_nue}
        export EOSSHIP=root://eosuser.cern.ch/
        export PYTHONPATH="$SNDSW_ROOT:$PYTHONPATH"

        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/passing_muon_tracks.XXXXXX")
        tmp_output="${{tmp_dir}}/$(basename "{output.track_feature_path}")"
        trap 'rm -rf "${{tmp_dir}}"' EXIT

        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python \
            {input.converter} \
            -d {input.reco} \
            -g {input.geo} \
            -o "${{tmp_output}}" \
            -t passing_muon_real_data \
            --tree-name {params.tree} \
            --extract-reco-tracks \
            --reco-track-schema {params.schema}

        /cvmfs/sndlhc.cern.ch/SNDLHC-2024/June25/bin/python \
            {input.validator} "${{tmp_output}}" --schema {params.schema}
        mkdir -p "$(dirname "{output.track_feature_path}")"
        xrdcp -f "${{tmp_output}}" "{output.track_feature_path}"
        """


rule passing_muon_track_features:
    input:
        metadata=PASSING_MUON_TRACK_METADATA,
        features=passing_muon_track_targets()


rule passing_muon_track_features_v2:
    input:
        metadata=PASSING_MUON_TRACK_V2_METADATA,
        features=passing_muon_track_v2_targets()


PASSING_MUON_TRACK_INSPECTION_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/"
    "region_v2_inspection/passing_muon_tracks_v1.yaml"
)
PASSING_MUON_TRACK_INSPECTION_ROOT = (
    "/eos/experiment/sndlhc/users/zhibin/snd-ml/"
    "region_v2_inspection_inputs/passing_muon_tracks_v1/"
    "passing_muon_tracks_run008329.root"
)
PASSING_MUON_TRACK_INSPECTION_DIR = (
    f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
    "region_v2_inspection/passing_muon_tracks_v1"
)
PASSING_MUON_TRACK_HISTS = (
    f"{PASSING_MUON_TRACK_INSPECTION_DIR}/passing_muon_tracks_v1_histograms.root"
)
PASSING_MUON_TRACK_DONE = (
    f"{PASSING_MUON_TRACK_INSPECTION_DIR}/passing_muon_tracks_v1.done"
)
PASSING_MUON_TRACK_PLOT_PREFIXES = [
    "reco_scifi_simple",
    "reco_ds_simple",
    "reco_scifi_hough",
    "reco_ds_hough",
    "reco_combined_hough",
]
PASSING_MUON_TRACK_PLOT_SUFFIXES = [
    "n_all",
    "best_chi2ndf",
    "veto_count_vs_n_all",
    "veto_time_vs_n_all",
    "veto_count_vs_chi2ndf",
    "veto_time_vs_chi2ndf",
    "start_xy_2d",
    "start_xy_chi2ndf_3d",
]
PASSING_MUON_TRACK_PLOTS = [
    f"{PASSING_MUON_TRACK_INSPECTION_DIR}/plots/{prefix}_{suffix}.pdf"
    for prefix in PASSING_MUON_TRACK_PLOT_PREFIXES
    for suffix in PASSING_MUON_TRACK_PLOT_SUFFIXES
] + [
    f"{PASSING_MUON_TRACK_INSPECTION_DIR}/plots/"
    f"reco_{method}_{suffix}.pdf"
    for method in ("simple", "hough")
    for suffix in (
        "ds_n_all_vs_scifi_n_all",
        "ds_best_chi2ndf_vs_scifi_best_chi2ndf",
    )
]


def passing_muon_run8329_track_targets():
    if PASSING_MUON_TRACK_TABLE.empty:
        return []
    return PASSING_MUON_TRACK_TABLE.loc[
        PASSING_MUON_TRACK_TABLE["run"].astype(int) == 8329,
        "track_feature_full_path",
    ].tolist()


rule passing_muon_track_inspection_input:
    input:
        config=PASSING_MUON_TRACK_INSPECTION_CONFIG,
        metadata=PASSING_MUON_TRACK_METADATA,
        features=passing_muon_run8329_track_targets(),
        builder=f"{PERSONAL_WORK_SPACE}/convertData/build_region_v2_inspection_input.py",
    output:
        root=PASSING_MUON_TRACK_INSPECTION_ROOT,
        manifest=(
            f"{PASSING_MUON_TRACK_INSPECTION_DIR}/"
            "passing_muon_tracks_v1_manifest_row.csv"
        ),
        audit=(
            f"{PASSING_MUON_TRACK_INSPECTION_DIR}/"
            "passing_muon_tracks_v1_source_audit.csv"
        ),
    threads: 4
    resources:
        runtime=24 * 60 * 60,
        mem_mb=8000,
        disk_mb=35000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source "{env_script_lcg}"
        set -euo pipefail
        tmp_dir=$(mktemp -d "${{TMPDIR:-/tmp}}/passing_muon_track_inspection.XXXXXX")
        trap 'rm -rf "$tmp_dir"' EXIT
        mkdir -p "$(dirname "{output.root}")" "$(dirname "{output.manifest}")"
        python "{input.builder}" \
          --config "{input.config}" \
          --dataset passing_muon_tracks_run008329 \
          --output "$tmp_dir/output.root" \
          --published-path "{output.root}" \
          --manifest-row "{output.manifest}" \
          --source-audit "{output.audit}" \
          --threads {threads}
        partial="{output.root}.partial.${{RANDOM}}"
        cp "$tmp_dir/output.root" "$partial"
        mv "$partial" "{output.root}"
        """


rule passing_muon_track_inspection_hist:
    input:
        root=PASSING_MUON_TRACK_INSPECTION_ROOT,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "make_reco_track_inspection_hist.py"
        ),
        extractor=f"{PERSONAL_WORK_SPACE}/convertData/reco_track_features.py",
    output:
        PASSING_MUON_TRACK_HISTS
    threads: 2
    shell:
        r"""
        set +u
        source "{env_script_lcg}"
        export PYTHONPATH="{PERSONAL_WORK_SPACE}/convertData:$PYTHONPATH"
        python "{input.script}" --input "{input.root}" --output "{output}" \
          --threads {threads}
        """


rule passing_muon_track_inspection_plots:
    input:
        hists=PASSING_MUON_TRACK_HISTS,
        script=(
            f"{PERSONAL_WORK_SPACE}/evaluation_region_partitions/"
            "plot_reco_track_inspection.py"
        ),
        extractor=f"{PERSONAL_WORK_SPACE}/convertData/reco_track_features.py",
    output:
        plots=PASSING_MUON_TRACK_PLOTS,
        done=PASSING_MUON_TRACK_DONE,
    shell:
        r"""
        set +u
        source "{env_script_lcg}"
        export PYTHONPATH="{PERSONAL_WORK_SPACE}/convertData:$PYTHONPATH"
        python "{input.script}" --input "{input.hists}" \
          --output-dir "{PASSING_MUON_TRACK_INSPECTION_DIR}/plots" \
          --done "{output.done}"
        """


rule passing_muon_track_inspection:
    input:
        PASSING_MUON_TRACK_PLOTS,
        PASSING_MUON_TRACK_DONE
