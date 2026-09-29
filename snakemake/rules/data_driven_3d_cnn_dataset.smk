DD3D_CONFIG = (
    f"{PERSONAL_WORK_SPACE}/snakemake/metadata/configs/ml_dataset/"
    "data_driven_3d_cnn_v1.yaml"
)

with open(DD3D_CONFIG, "r", encoding="utf-8") as _dd3d_stream:
    DD3D = yaml.safe_load(_dd3d_stream)

DD3D_VERSION = DD3D["dataset_version"]
DD3D_META = f"{PERSONAL_WORK_SPACE}/{DD3D['output']['metadata_dir']}"
DD3D_ROOT = DD3D["output"]["root_dir"].rstrip("/")
DD3D_SOURCE_CATALOG = f"{DD3D_META}/{DD3D['output']['source_catalog']}"
DD3D_MUON_AUDIT = f"{DD3D_META}/{DD3D['output']['muon_audit']}"
DD3D_NEUTRINO_AUDIT = f"{DD3D_META}/{DD3D['output']['neutrino_audit']}"
DD3D_DENSITY_PILOT = f"{DD3D_META}/{DD3D['output']['density_sampling_pilot']}"
DD3D_DATASET_CATALOG = f"{DD3D_META}/{DD3D['output']['dataset_catalog']}"
DD3D_FOLDED_CATALOG = f"{DD3D_META}/{DD3D['output']['folded_catalog']}"
DD3D_FOLD_SUMMARY = f"{DD3D_META}/{DD3D['output']['fold_summary']}"
DD3D_DENSITY_MANIFEST = f"{DD3D_META}/{DD3D['output']['density_index_manifest']}"
DD3D_FOLD_VIEWS = f"{DD3D_META}/{DD3D['output']['fold_views']}"
DD3D_VALIDATION = f"{DD3D_META}/{DD3D['output']['validation_report']}"


def _dd3d_repo_path(value):
    value = str(value)
    return value if os.path.isabs(value) or value.startswith("root://") else f"{PERSONAL_WORK_SPACE}/{value}"


DD3D_REAL_SUBSET = DD3D["real_data_subset"]
DD3D_REAL_SUBSET_INPUT = _dd3d_repo_path(DD3D_REAL_SUBSET["input_metadata"])
DD3D_REAL_SUBSET_OUTPUT = _dd3d_repo_path(
    DD3D["source_datasets"]["real_data_2024_skim"]["metadata"]
)
DD3D_REAL_SUBSET_SUMMARY = _dd3d_repo_path(DD3D_REAL_SUBSET["summary"])


DD3D_SOURCE_METADATA = [
    _dd3d_repo_path(definition["metadata"])
    for definition in DD3D["source_datasets"].values()
    if definition.get("enabled", True)
]


rule build_data_driven_3d_cnn_real_data_ml50_subset_v1:
    input:
        config=DD3D_CONFIG,
        metadata=DD3D_REAL_SUBSET_INPUT,
        script=f"{PERSONAL_WORK_SPACE}/convertData/build_real_data_run_subset.py",
        common=f"{PERSONAL_WORK_SPACE}/convertData/data_driven_3d_dataset.py",
    output:
        metadata=DD3D_REAL_SUBSET_OUTPUT,
        summary=DD3D_REAL_SUBSET_SUMMARY,
    threads: 1
    resources:
        runtime=30 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    params:
        files_per_run=int(DD3D_REAL_SUBSET["files_per_run"]),
        seed=int(DD3D_REAL_SUBSET["seed"]),
        workers=int(DD3D.get("availability_workers", 32)),
        expected_runs=int(DD3D_REAL_SUBSET["expected_runs"]),
    shell:
        r"""
        set -euo pipefail
        PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --input {input.metadata} \
          --output {output.metadata} \
          --summary {output.summary} \
          --files-per-run {params.files_per_run} \
          --seed {params.seed} \
          --workers {params.workers} \
          --expected-runs {params.expected_runs}
        """


checkpoint build_data_driven_3d_cnn_source_catalog_v1:
    input:
        config=DD3D_CONFIG,
        metadata=DD3D_SOURCE_METADATA,
        script=f"{PERSONAL_WORK_SPACE}/convertData/build_3d_cnn_source_catalog.py",
        common=f"{PERSONAL_WORK_SPACE}/convertData/data_driven_3d_dataset.py",
    output:
        DD3D_SOURCE_CATALOG
    threads: 1
    resources:
        runtime=30 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --config {input.config} \
          --output {output}
        """


def _dd3d_source_catalog(_wildcards):
    return checkpoints.build_data_driven_3d_cnn_source_catalog_v1.get().output[0]


def _dd3d_muon_source_uids(wildcards):
    catalog = pd.read_csv(_dd3d_source_catalog(wildcards), keep_default_na=False)
    available = catalog["input_available"].astype(str).str.lower().isin(["true", "1", "yes"])
    empty = catalog["input_empty"].astype(str).str.lower().isin(["true", "1", "yes"])
    return catalog.loc[
        available & ~empty & catalog["role"].eq("muon_split"), "source_uid"
    ].astype(str).tolist()


def _dd3d_muon_audit_rows(wildcards):
    return [f"{DD3D_META}/muon_audit_rows/{uid}.csv" for uid in _dd3d_muon_source_uids(wildcards)]


def _dd3d_neutrino_source_uids(wildcards):
    catalog = pd.read_csv(_dd3d_source_catalog(wildcards), keep_default_na=False)
    available = catalog["input_available"].astype(str).str.lower().isin(["true", "1", "yes"])
    empty = catalog["input_empty"].astype(str).str.lower().isin(["true", "1", "yes"])
    return catalog.loc[
        available & ~empty & catalog["role"].eq("neutrino_split"), "source_uid"
    ].astype(str).tolist()


def _dd3d_neutrino_audit_rows(wildcards):
    return [f"{DD3D_META}/neutrino_audit_rows/{uid}.csv" for uid in _dd3d_neutrino_source_uids(wildcards)]


rule build_data_driven_3d_cnn_muon_loose_source_v1:
    input:
        config=DD3D_CONFIG,
        source_catalog=_dd3d_source_catalog,
        script=f"{PERSONAL_WORK_SPACE}/convertData/build_muon_loose_partitions.py",
        selection=f"{PERSONAL_WORK_SPACE}/convertData/passing_muon_selection.py",
        common=f"{PERSONAL_WORK_SPACE}/convertData/data_driven_3d_dataset.py",
    output:
        pass_feature=f"{DD3D_ROOT}/muon/{{source_uid}}/passing_muon_feature.root",
        pass_hit3d=f"{DD3D_ROOT}/muon/{{source_uid}}/passing_muon_hit3d.root",
        fail_feature=f"{DD3D_ROOT}/muon/{{source_uid}}/muon_DIS_feature.root",
        fail_hit3d=f"{DD3D_ROOT}/muon/{{source_uid}}/muon_DIS_hit3d.root",
        signal_feature=f"{DD3D_ROOT}/muon/{{source_uid}}/signal_region_feature.root",
        signal_hit3d=f"{DD3D_ROOT}/muon/{{source_uid}}/signal_region_hit3d.root",
        audit=f"{DD3D_META}/muon_audit_rows/{{source_uid}}.csv",
    wildcard_constraints:
        source_uid=r"src_[0-9a-f]{20}"
    threads: 1
    resources:
        runtime=60 * 60,
        mem_mb=2000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --config {input.config} \
          --source-catalog {input.source_catalog} \
          --source-uid {wildcards.source_uid} \
          --pass-feature {output.pass_feature} \
          --pass-hit3d {output.pass_hit3d} \
          --fail-feature {output.fail_feature} \
          --fail-hit3d {output.fail_hit3d} \
          --signal-feature {output.signal_feature} \
          --signal-hit3d {output.signal_hit3d} \
          --audit {output.audit} \
          --threads {threads}
        """


rule combine_data_driven_3d_cnn_muon_audits_v1:
    input:
        _dd3d_muon_audit_rows
    output:
        DD3D_MUON_AUDIT
    resources:
        runtime=30 * 60,
        mem_mb=4000,
        disk_mb=1000,
        nvidia_gpu=0
    run:
        frames = [pd.read_csv(path, keep_default_na=False) for path in input]
        if not frames or any(len(frame) != 1 for frame in frames):
            raise ValueError("Every muon audit input must contain exactly one row")
        result = pd.concat(frames, ignore_index=True)
        if result["source_uid"].astype(str).duplicated().any():
            raise ValueError("Muon audit source_uid values are not unique")
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output[0], index=False)


rule build_data_driven_3d_cnn_neutrino_source_v1:
    input:
        config=DD3D_CONFIG,
        source_catalog=_dd3d_source_catalog,
        script=f"{PERSONAL_WORK_SPACE}/convertData/build_neutrino_class_partitions.py",
        shared=f"{PERSONAL_WORK_SPACE}/convertData/build_muon_loose_partitions.py",
        common=f"{PERSONAL_WORK_SPACE}/convertData/data_driven_3d_dataset.py",
    output:
        cc_numu_feature=f"{DD3D_ROOT}/neutrino/{{source_uid}}/CC_numu_feature.root",
        cc_numu_hit3d=f"{DD3D_ROOT}/neutrino/{{source_uid}}/CC_numu_hit3d.root",
        cc_nue_feature=f"{DD3D_ROOT}/neutrino/{{source_uid}}/CC_nue_feature.root",
        cc_nue_hit3d=f"{DD3D_ROOT}/neutrino/{{source_uid}}/CC_nue_hit3d.root",
        cc_nutau_feature=f"{DD3D_ROOT}/neutrino/{{source_uid}}/CC_nutau_feature.root",
        cc_nutau_hit3d=f"{DD3D_ROOT}/neutrino/{{source_uid}}/CC_nutau_hit3d.root",
        nc_feature=f"{DD3D_ROOT}/neutrino/{{source_uid}}/NC_feature.root",
        nc_hit3d=f"{DD3D_ROOT}/neutrino/{{source_uid}}/NC_hit3d.root",
        audit=f"{DD3D_META}/neutrino_audit_rows/{{source_uid}}.csv",
    wildcard_constraints:
        source_uid=r"src_[0-9a-f]{20}"
    threads: 1
    resources:
        runtime=60 * 60,
        mem_mb=2000,
        disk_mb=1000,
        nvidia_gpu=0
    params:
        output_root=lambda wildcards: f"{DD3D_ROOT}/neutrino/{wildcards.source_uid}"
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --config {input.config} \
          --source-catalog {input.source_catalog} \
          --source-uid {wildcards.source_uid} \
          --output-root {params.output_root} \
          --audit {output.audit} \
          --threads {threads}
        """


rule combine_data_driven_3d_cnn_neutrino_audits_v1:
    input:
        _dd3d_neutrino_audit_rows
    output:
        DD3D_NEUTRINO_AUDIT
    resources:
        runtime=30 * 60,
        mem_mb=2000,
        disk_mb=1000,
        nvidia_gpu=0
    run:
        frames = [pd.read_csv(path, keep_default_na=False) for path in input]
        if not frames or any(len(frame) != 1 for frame in frames):
            raise ValueError("Every neutrino audit input must contain exactly one row")
        result = pd.concat(frames, ignore_index=True)
        if result["source_uid"].astype(str).duplicated().any():
            raise ValueError("Neutrino audit source_uid values are not unique")
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output[0], index=False)


rule data_driven_3d_cnn_neutrino_partitions_v1:
    input:
        audit=DD3D_NEUTRINO_AUDIT


rule data_driven_3d_cnn_muon_partitions_v1:
    input:
        audit=DD3D_MUON_AUDIT


rule summarize_data_driven_3d_cnn_density_pilot_v1:
    input:
        config=DD3D_CONFIG,
        density_hist=DD3D["sampling_pilot"]["v7_density_hist"],
        selection_hist=DD3D["sampling_pilot"]["v7_selection_hist"],
        script=f"{PERSONAL_WORK_SPACE}/convertData/summarize_muondis_density_sampling.py",
    output:
        DD3D_DENSITY_PILOT
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
        PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --config {input.config} \
          --v7-density-hist {input.density_hist} \
          --v7-selection-hist {input.selection_hist} \
          --output {output}
        """


checkpoint build_data_driven_3d_cnn_dataset_catalog_v1:
    input:
        config=DD3D_CONFIG,
        source_catalog=DD3D_SOURCE_CATALOG,
        audits=DD3D_MUON_AUDIT,
        neutrino_audits=DD3D_NEUTRINO_AUDIT,
        script=f"{PERSONAL_WORK_SPACE}/convertData/build_3d_cnn_dataset_catalog.py",
    output:
        DD3D_DATASET_CATALOG
    threads: 1
    resources:
        runtime=4 * 60 * 60,
        mem_mb=4000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --config {input.config} \
          --source-catalog {input.source_catalog} \
          --muon-audits {input.audits} \
          --neutrino-audits {input.neutrino_audits} \
          --output {output}
        """


rule assign_data_driven_3d_cnn_five_folds_v1:
    input:
        config=DD3D_CONFIG,
        catalog=DD3D_DATASET_CATALOG,
        script=f"{PERSONAL_WORK_SPACE}/convertData/assign_3d_cnn_folds.py",
    output:
        catalog=DD3D_FOLDED_CATALOG,
        summary=DD3D_FOLD_SUMMARY,
    threads: 1
    resources:
        runtime=30 * 60,
        mem_mb=4000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set -euo pipefail
        PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --config {input.config} \
          --input {input.catalog} \
          --output {output.catalog} \
          --summary {output.summary}
        """


def _dd3d_dataset_catalog(_wildcards):
    return checkpoints.build_data_driven_3d_cnn_dataset_catalog_v1.get().output[0]


def _dd3d_muondis_file_uids(wildcards):
    catalog = pd.read_csv(_dd3d_dataset_catalog(wildcards), keep_default_na=False)
    return catalog.loc[catalog["class_name"].eq("muon_DIS"), "dataset_file_uid"].astype(str).tolist()


def _dd3d_density_rows(wildcards):
    return [f"{DD3D_META}/density_index_rows/{uid}.csv" for uid in _dd3d_muondis_file_uids(wildcards)]


rule build_data_driven_3d_cnn_density_index_v1:
    input:
        config=DD3D_CONFIG,
        catalog=_dd3d_dataset_catalog,
        script=f"{PERSONAL_WORK_SPACE}/convertData/build_muondis_density_indices.py",
    output:
        index=f"{DD3D_ROOT}/density_indices/{{dataset_file_uid}}.npz",
        row=f"{DD3D_META}/density_index_rows/{{dataset_file_uid}}.csv",
    wildcard_constraints:
        dataset_file_uid=r"dsf_[0-9a-f]{20}"
    threads: 1
    resources:
        runtime=20 * 60,
        mem_mb=1000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set +u
        source {env_script_lcg}
        set -euo pipefail
        PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --config {input.config} \
          --dataset-catalog {input.catalog} \
          --dataset-file-uid {wildcards.dataset_file_uid} \
          --output-npz {output.index} \
          --output-row {output.row}
        """


rule combine_data_driven_3d_cnn_density_indices_v1:
    input:
        _dd3d_density_rows
    output:
        DD3D_DENSITY_MANIFEST
    resources:
        runtime=30 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    run:
        frames = [pd.read_csv(path, keep_default_na=False) for path in input]
        if not frames or any(len(frame) != 1 for frame in frames):
            raise ValueError("Every density-index row must contain exactly one row")
        result = pd.concat(frames, ignore_index=True)
        if result["dataset_file_uid"].astype(str).duplicated().any():
            raise ValueError("Density-index dataset_file_uid values are not unique")
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output[0], index=False)


rule build_data_driven_3d_cnn_fold_views_v1:
    input:
        config=DD3D_CONFIG,
        folded_catalog=DD3D_FOLDED_CATALOG,
        density_manifest=DD3D_DENSITY_MANIFEST,
        script=f"{PERSONAL_WORK_SPACE}/convertData/build_3d_cnn_fold_views.py",
    output:
        DD3D_FOLD_VIEWS
    resources:
        runtime=30 * 60,
        mem_mb=3000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set -euo pipefail
        PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --config {input.config} \
          --folded-catalog {input.folded_catalog} \
          --density-index-manifest {input.density_manifest} \
          --output {output}
        """


rule validate_data_driven_3d_cnn_dataset_v1:
    input:
        config=DD3D_CONFIG,
        source_catalog=DD3D_SOURCE_CATALOG,
        muon_audit=DD3D_MUON_AUDIT,
        neutrino_audit=DD3D_NEUTRINO_AUDIT,
        dataset_catalog=DD3D_DATASET_CATALOG,
        folded_catalog=DD3D_FOLDED_CATALOG,
        fold_summary=DD3D_FOLD_SUMMARY,
        density_manifest=DD3D_DENSITY_MANIFEST,
        fold_views=DD3D_FOLD_VIEWS,
        # The V7 density pilot is diagnostic-only and is not training data.
        script=f"{PERSONAL_WORK_SPACE}/convertData/validate_3d_cnn_dataset.py",
    output:
        DD3D_VALIDATION
    threads: 1
    resources:
        runtime=30 * 60,
        mem_mb=4000,
        disk_mb=1000,
        nvidia_gpu=0
    shell:
        r"""
        set -euo pipefail
        PYTHONDONTWRITEBYTECODE=1 python3 {input.script} \
          --config {input.config} \
          --source-catalog {input.source_catalog} \
          --muon-audit {input.muon_audit} \
          --neutrino-audit {input.neutrino_audit} \
          --dataset-catalog {input.dataset_catalog} \
          --folded-catalog {input.folded_catalog} \
          --density-index-manifest {input.density_manifest} \
          --fold-views {input.fold_views} \
          --output {output}
        """


rule data_driven_3d_cnn_dataset_v1:
    input:
        real_data_subset_summary=DD3D_REAL_SUBSET_SUMMARY,
        validation=DD3D_VALIDATION,
        catalog=DD3D_FOLDED_CATALOG,
        fold_summary=DD3D_FOLD_SUMMARY,
        density_manifest=DD3D_DENSITY_MANIFEST,
        fold_views=DD3D_FOLD_VIEWS,
